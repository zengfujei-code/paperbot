"""Synthetic test-phase arithmetic and clocks; no admission or market evidence."""
from dataclasses import replace
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[3]),str(Path(__file__).resolve().parents[1])]
from test_common_scores import plan,fits,frame,row
from paperbot_r2 import common_scores
from paperbot_r2.contracts import aware,digest
from paperbot_r2.model_training import FAMILIES
from paperbot_r2.training_data import SYMBOLS
try:
    from paperbot_r2 import test_protocol as protocol
except ImportError:
    protocol=None

TZ=Path(__file__).resolve().parents[5]/'outputs/zoneinfo/America/New_York'
FINAL=aware('2026-12-11T21:02:01+00:00')


class TestProtocolTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(protocol,'prospective test protocol not implemented')
        self.plan=plan();self.slots=protocol.planned_slots(self.plan)
        self.env=patch.dict(os.environ,{},clear=True);self.env.start();self.addCleanup(self.env.stop)
        for target in ('socket.socket','sqlite3.connect'):
            guard=patch(target,side_effect=AssertionError('pure contracts forbid network/SQL'))
            guard.start();self.addCleanup(guard.stop)
    def report(self,frames,*,at=FINAL,training=None,selected='prevalence'):
        return protocol.evaluate_test_frames(self.plan,tuple(frames),as_of=at,fit_statuses=training or fits(),selected_family=selected)
    def full(self,**kwargs):return [frame(s,**kwargs) for s in self.slots]
    def test_exact_test_grid_first_last_and_calendar_rejection(self):
        self.assertEqual(len(self.slots),240)
        for symbol in SYMBOLS:
            found=[s for s in self.slots if s.symbol==symbol]
            self.assertEqual(len(found),80)
            self.assertEqual(found[0].event_at,aware('2026-12-07T16:45:00+00:00'))
            self.assertEqual(found[-1].event_at,aware('2026-12-11T20:30:00+00:00'))
            self.assertEqual(found[-1].target_end,aware('2026-12-11T21:00:00+00:00'))
        changed=plan();changed['calendar']['sessions'][-1]['closes']='2026-12-11T18:00:00+00:00'
        changed['calendar']['sha256']=digest(changed['calendar']['sessions'])
        with self.assertRaises(protocol.TestProtocolError):protocol.planned_slots(changed)
    def test_full_common_metrics_never_reselect_family(self):
        values={f:(.9 if f=='prevalence' else .1) for f in FAMILIES}
        report=self.report(self.full(probabilities=values))
        self.assertTrue(report['engineering_gate_passed']);self.assertEqual(report['selected_family'],'prevalence')
        self.assertAlmostEqual(report['macro']['prevalence']['brier'],.81)
        self.assertAlmostEqual(report['macro']['ewma']['brier'],.01)
        self.assertIsNone(report['proposed_family']);self.assertTrue(report['math_only'])
        for field in ('selection_admission','test_admission','promotion','orders'):self.assertIs(report[field],False)
        self.assertEqual(report['test_sessions'],5);self.assertEqual(report['evidence_class'],'exploratory_risk_forecast_only')
        self.assertEqual(report['trial_units'],15);self.assertEqual(len(report['groups']),15)
    def test_all_five_families_share_one_intersection_including_failure(self):
        values=self.full()
        values=[replace(x,rows=tuple(row(f,status='training_fit_failed') if f=='gaussian_nb' and x.symbol=='SPY' else r
                 for f,r in zip(FAMILIES,x.rows))) for x in values]
        report=self.report(values,training=fits((('SPY','gaussian_nb'),)))
        self.assertFalse(report['engineering_gate_passed']);self.assertEqual(report['selected_family'],'prevalence')
        for f in FAMILIES:
            self.assertIsNone(report['macro'][f]['brier'])
            self.assertEqual(report['symbols']['SPY']['families'][f]['count'],0)
        self.assertEqual(report['symbols']['SPY']['families']['gaussian_nb']['counts']['failed_fit'],80)
        self.assertEqual(report['symbols']['SPY']['families']['ewma']['counts']['settled'],80)
    def test_macro_is_three_equal_symbols_not_row_weight(self):
        values=[frame(s,probabilities={f:float(s.symbol=='SPY') for f in FAMILIES}) for s in self.slots
                if s.symbol!='SPY' or s.event_at<=aware(s.day+'T17:30:00+00:00')]
        report=self.report(values)
        for f in FAMILIES:self.assertAlmostEqual(report['macro'][f]['brier'],1./3.)
        self.assertFalse(report['engineering_gate_passed'])
    def test_no_two_symbol_macro(self):
        report=self.report(frame(s) for s in self.slots if s.symbol!='MSFT')
        for f in FAMILIES:self.assertIsNone(report['macro'][f]['brier'])
    def test_coverage_is_per_day_per_symbol(self):
        values=[]
        for day in sorted({s.day for s in self.slots}):
            for symbol in SYMBOLS:
                values.extend(frame(s) for s in [s for s in self.slots if s.day==day and s.symbol==symbol][:13])
        self.assertTrue(self.report(values)['engineering_gate_passed'])
        self.assertFalse(self.report(values[1:])['engineering_gate_passed'])
    def test_late_missing_quarantine_unissued_counts_and_unfinished_phase(self):
        values=self.full()
        values[0]=replace(values[0],late=True);values[1]=replace(values[1],missing=True)
        values[2]=replace(values[2],quarantined=True)
        values[3]=replace(values[3],issued=False,settled=False,label=None,return_fraction=None)
        group=self.report(values)['groups'][0]
        self.assertEqual(group['counts']['planned'],16);self.assertEqual(group['counts']['common'],12)
        for name in ('late','missing','quarantined','unissued'):self.assertEqual(group['counts'][name],1)
        early=self.report((),at=aware('2026-12-07T14:30:00+00:00'))
        self.assertEqual(sum(g['counts']['not_yet_due'] for g in early['groups']),240)
        self.assertFalse(early['complete']);self.assertFalse(early['engineering_gate_passed'])
    def test_exact_infinite_log_loss_and_calibration_boundaries(self):
        report=self.report(self.full(probabilities={f:0. for f in FAMILIES},label=1))
        self.assertEqual(report['macro']['prevalence']['log_loss'],{'loss':'positive_infinity'})
        self.assertNotIn('Infinity',json.dumps(report,allow_nan=False))
        for p,index in ((0.,0),(.2,1),(.4,2),(.6,3),(.8,4),(1.,4)):
            report=self.report(self.full(probabilities={f:p for f in FAMILIES}))
            self.assertEqual(report['groups'][0]['families']['prevalence']['calibration'][index]['count'],16)
            self.assertAlmostEqual(report['macro']['prevalence']['brier'],p*p)
    def test_centered_variance_and_second_moment_arithmetic(self):
        values=[replace(x,return_fraction=.3,rows=tuple(row(f,.25,mean=.1,variance=.04) for f in FAMILIES)) for x in self.full(label=1)]
        result=self.report(values)['macro']['garch']
        self.assertAlmostEqual(result['brier'],.75**2)
        self.assertAlmostEqual(result['log_loss'],-math.log(.25))
        self.assertAlmostEqual(result['variance_proxy_mse'],0.)
        self.assertAlmostEqual(result['variance_proxy_qlike'],math.log(.04)+1.)
        self.assertAlmostEqual(result['second_moment_mse'],(.05-.09)**2)
        self.assertAlmostEqual(result['second_moment_qlike'],math.log(.05)+.09/.05)
    def test_validation_reserved_off_grid_duplicate_and_future_rejected(self):
        test=frame(self.slots[0]);old=frame(common_scores.planned_slots(self.plan)[0])
        for values,at in (((old,),FINAL),((test,test),FINAL),((test,),test.event_at-timedelta(seconds=1))):
            with self.assertRaises(protocol.TestProtocolError):self.report(values,at=at)
        shifted=replace(test,event_at=test.event_at+timedelta(seconds=1),key=test.symbol+'|'+(test.event_at+timedelta(seconds=1)).isoformat(),
                        target_start=test.target_start+timedelta(seconds=1),target_end=test.target_end+timedelta(seconds=1))
        with self.assertRaises(protocol.TestProtocolError):self.report((shifted,))
    def test_invalid_selection_and_fit_statuses_rejected(self):
        for selected in (None,True,'new_family'):
            with self.assertRaises(protocol.TestProtocolError):self.report((),selected=selected)
        with self.assertRaises(protocol.TestProtocolError):self.report((),training=fits()[:-1])
        with self.assertRaises(protocol.TestProtocolError):self.report(self.full(),training=fits((('SPY','gaussian_nb'),)))
    def test_selection_clock_strict_boundaries_and_monotonicity(self):
        start=aware('2026-12-04T21:02:00+00:00');end=aware('2026-12-07T14:30:00+00:00')
        for at in (start,start-timedelta(seconds=1),end):
            with self.assertRaises(protocol.TestProtocolError):protocol.SelectionClock(TZ,lambda:at)()
        times=iter((start+timedelta(seconds=2),start+timedelta(seconds=1)))
        clock=protocol.SelectionClock(TZ,lambda:next(times));clock()
        with self.assertRaises(protocol.TestProtocolError):clock()
    def test_issue_clock_test_dates_grid_and_120second_limit(self):
        slot=self.slots[0]
        for delta in (0,120):
            at=slot.event_at+timedelta(seconds=delta)
            self.assertEqual(protocol.TestIssueClock(slot.key,TZ,lambda:at)(),at)
        for delta in (-1,121):
            with self.assertRaises(protocol.TestProtocolError):protocol.TestIssueClock(slot.key,TZ,lambda:slot.event_at+timedelta(seconds=delta))()
        for key in ('AAPL|2026-11-24T16:45:00+00:00','AAPL|2026-11-12T16:45:00+00:00',
                    'AAPL|2026-12-07T14:45:00+00:00','AAPL|2026-12-07T20:45:00+00:00','XYZ|2026-12-07T16:45:00+00:00'):
            with self.assertRaises(protocol.TestProtocolError):protocol.TestIssueClock(key,TZ,lambda:slot.event_at)()
    def test_outcome_maturity_and_test_key_guard(self):
        slot=self.slots[-1]
        with self.assertRaises(protocol.TestProtocolError):protocol.TestOutcomeClock(slot.key,TZ,lambda:slot.target_end-timedelta(seconds=1))()
        self.assertEqual(protocol.TestOutcomeClock(slot.key,TZ,lambda:slot.target_end)(),slot.target_end)
        with self.assertRaises(protocol.TestProtocolError):protocol.TestOutcomeClock('AAPL|2026-11-24T16:45:00+00:00',TZ,lambda:FINAL)()
    def test_final_freeze_is_after_last_deadline(self):
        for at in (FINAL-timedelta(seconds=1),aware('2026-12-07T21:02:01+00:00')):
            with self.assertRaises(protocol.TestProtocolError):protocol.TestFreezeClock(TZ,lambda:at)()
        self.assertEqual(protocol.TestFreezeClock(TZ,lambda:FINAL)(),FINAL)
    def test_full_source_commitment_requires_all_real_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            # Hash-only fixture bytes, never imported or claimed as a real protocol.
            for name in protocol.PROTOCOL_FILES:(root/name).write_text('fixture:'+name)
            with patch.object(protocol,'__file__',str(root/'test_protocol.py')):
                first=protocol.source_sha256();self.assertEqual(len(first),64)
                (root/'test_scores.py').write_text('changed fixture')
                self.assertNotEqual(protocol.source_sha256(),first)
                (root/'test_scores.py').unlink()
                with self.assertRaises(protocol.TestProtocolError):protocol.source_sha256()
                (root/'test_scores.py').write_bytes(b'x'*524289)
                with self.assertRaises(protocol.TestProtocolError):protocol.source_sha256()
    def test_partial_real_protocol_has_no_admission_commitment(self):
        root=Path(protocol.__file__).parent
        if all((root/name).is_file() for name in protocol.PROTOCOL_FILES):
            self.assertEqual(len(protocol.source_sha256()),64)
        else:
            with self.assertRaises(protocol.TestProtocolError):protocol.source_sha256()
        self.assertEqual(len(protocol.math_source_sha256()),64)
    def test_whole_source_hash_rejects_an_earlier_file_changed_during_later_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in protocol.PROTOCOL_FILES:(root/name).write_text('fixture:'+name)
            original=protocol._file_hash
            def changed(path):
                result=original(path)
                if path.name=='test_scores.py':(root/'test_protocol.py').write_text('changed during later source read')
                return result
            with patch.object(protocol,'__file__',str(root/'test_protocol.py')),patch.object(protocol,'_file_hash',side_effect=changed):
                with self.assertRaises(protocol.TestProtocolError):protocol.source_sha256()


if __name__=='__main__':unittest.main()
