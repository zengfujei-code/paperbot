"""Synthetic first test outcomes; reducer/storage tests grant no admission."""
from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
import hashlib
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch
import zoneinfo

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[3]),str(Path(__file__).resolve().parents[1])]
from issue_fixture import IssueWorld,Tick,TZ
from test_training_data import ROOT
from paperbot_r2 import common_outcome,forecast_inputs as inputs,candidate_forecasts as maths
from paperbot_r2 import validation_settlements as previous,test_issue_journal as issuer,evaluation_book
from paperbot_r2.contracts import aware,digest
try:
    from paperbot_r2 import test_settlements as settlements
except ImportError:
    settlements=None


class TestSettlementPhaseTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(settlements,'test settlement journal not implemented')

    def test_validation_reserved_future_and_unmatured_keys_fail_before_sql_or_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);book=settlements.TestSettlementJournal(root/'test.sqlite')
            examples=(('AAPL|2026-11-24T16:45:00+00:00','2026-11-24T17:16:00+00:00'),
                ('AAPL|2026-11-12T16:45:00+00:00','2026-11-12T17:16:00+00:00'),
                ('AAPL|2026-12-07T16:45:00+00:00','2026-12-07T17:14:59+00:00'),
                ('AAPL|2026-12-11T20:30:00+00:00','2026-12-07T17:16:00+00:00'),
                ('AAPL|2026-12-07T16:46:00+00:00','2026-12-07T17:16:00+00:00'))
            for key,at in examples:
                with patch('sqlite3.connect',side_effect=AssertionError('phase before SQL')) as connect:
                    with self.assertRaises(settlements.TestSettlementError):
                        book.settle_once(key,timezone_file=TZ,clock=lambda:aware(at))
                    connect.assert_not_called()
            self.assertEqual(list(root.iterdir()),[])

    def test_late_initialization_is_refused_without_sql_or_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);book=settlements.TestSettlementJournal(root/'test.sqlite')
            with patch('sqlite3.connect',side_effect=AssertionError('initialization before SQL')) as connect:
                with self.assertRaises(settlements.TestSettlementError):
                    book.initialize(None,timezone_file=TZ,clock=lambda:aware('2026-11-24T14:30:00+00:00'))
                connect.assert_not_called()
            self.assertEqual(list(root.iterdir()),[])

    def test_initialization_view_cannot_skip_admission_after_validation_open(self):
        with patch('sqlite3.connect',side_effect=AssertionError('no admission bypass SQL')) as connect:
            with self.assertRaises(issuer.TestIssueError):
                with issuer.issue_view(None,timezone_file=TZ,
                    guard=lambda:aware('2026-11-24T14:30:00+00:00'),require_admission=False):
                    self.fail('late initialization context must be refused')
            connect.assert_not_called()

    def test_scalar_witness_or_validation_receipt_is_not_test_publication_authority(self):
        with self.assertRaises(settlements.TestSettlementError):
            settlements.witness_from_issued('AAPL|2026-12-07T16:45:00+00:00',object())


class TestSettlementDomainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if settlements is None:return
        cls.old_tz=zoneinfo.TZPATH;zoneinfo.reset_tzpath([str(ROOT/'outputs/zoneinfo')]);zoneinfo.ZoneInfo.clear_cache()
        cls.addClassCleanup(zoneinfo.reset_tzpath,cls.old_tz);cls.addClassCleanup(zoneinfo.ZoneInfo.clear_cache)
        cls.temp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        with patch.dict(os.environ,{},clear=True),patch.object(socket,'socket',side_effect=AssertionError('no network')):
            cls.w=IssueWorld(Path(cls.temp.name));cls.key=cls.w.receive_until('2026-12-07',9)
            cls.captured=inputs.capture_issue_input(cls.w.training_receipt,cls.w.registration,cls.w.source_pairs(),cls.key,
                timezone_file=TZ,clock=Tick(cls.w.at))
            cls.panel=maths.compute_candidates(cls.w.training_receipt,cls.captured,clock=Tick(cls.w.at+timedelta(seconds=1)))
            cls.w.receive_until('2026-12-07',11,derive=False)
            raw,features=cls.w.pair('2026-12-07')
            cls.result=common_outcome.observe_common_outcome(features,raw,cls.key,clock=Tick(cls.w.at))
        fields=dict(key=cls.key,symbol='AAPL',input_sha256=cls.captured.seal,panel_sha256=cls.panel.seal,
            issue_book_path=str(Path(cls.temp.name)/'explicit-nonauthoritative-witness.sqlite'),
            issue_book_device=0,issue_book_inode=0,issue_book_head_seq=6,issue_book_head_digest='a'*64,
            frame_confirmation_digest='b'*64,issued_at=cls.panel.completed_at+timedelta(microseconds=1),
            issue_commit_observed_at=cls.panel.completed_at+timedelta(microseconds=2),
            available_at=cls.panel.completed_at+timedelta(microseconds=3),event_at=cls.captured.bundle.vector.event_at,
            target_start=cls.captured.bundle.target_start,target_end=cls.captured.bundle.target_end)
        cls.witness=previous.ForecastWitness(**fields,seal=digest(fields))
        cls.binding=dict(issue_binding=dict(validation_issue_binding=dict(plan_json=cls.captured.registration.plan_json)))
        cls.body=dict(key=cls.key,witness=asdict(cls.witness),outcome=asdict(cls.result),missing=False)
        cls.at=cls.result.observed_at+timedelta(microseconds=1)

    def setUp(self):
        self.assertIsNotNone(settlements,'test settlement journal not implemented')
        self.book=settlements.TestSettlementJournal(Path('domain-only-no-file.sqlite'));self.domain=self.book.domain_initial()

    def apply(self,kind,body,at,sha='c'*64):self.book.domain_apply(self.domain,kind,body,at,sha,self.binding)

    def test_first_successor_label_and_actual_result_commit_are_separate_events(self):
        self.apply('observed',deepcopy(self.body),self.at)
        self.apply('confirmed',dict(key=self.key,observed_digest='c'*64),self.at+timedelta(microseconds=1),'d'*64)
        entry=self.domain['frames'][self.key]
        self.assertEqual(entry['status'],'confirmed');self.assertEqual(entry['outcome'].label,1)
        self.assertAlmostEqual(entry['outcome'].return_fraction,99.7/100.4-1.,places=14)
        self.assertFalse(entry['effective_late']);self.assertFalse(entry['current_scoring_permission'])
        self.assertEqual(entry['outcome'].feature_bundle_sha256,self.captured.bundle.seal)
        self.assertEqual(entry['outcome'].raw_plan_sha256,digest(asdict(self.w.next_raw.snapshot()['plan'])))
        self.assertGreater(entry['commit_observed_at'],entry['observed_at'])

    def test_absence_is_not_permanent_missing_until_strictly_after_deadline(self):
        body=dict(self.body,outcome=None,missing=True);deadline=self.witness.target_end+timedelta(seconds=120)
        for when in (self.witness.target_end,deadline):
            with self.assertRaises(settlements.TestSettlementError):self.apply('observed',body,when)
        self.apply('observed',body,deadline+timedelta(microseconds=1))
        self.apply('confirmed',dict(key=self.key,observed_digest='c'*64),deadline+timedelta(microseconds=2))
        entry=self.domain['frames'][self.key]
        self.assertTrue(entry['missing']);self.assertTrue(entry['effective_late']);self.assertIsNone(entry['outcome'])
        with self.assertRaises(settlements.TestSettlementError):self.apply('observed',self.body,deadline+timedelta(seconds=1))

    def test_result_acknowledged_after_deadline_is_late_without_changing_label(self):
        self.apply('observed',self.body,self.at)
        self.apply('confirmed',dict(key=self.key,observed_digest='c'*64),self.witness.target_end+timedelta(seconds=121))
        entry=self.domain['frames'][self.key]
        self.assertEqual(entry['outcome'],self.result);self.assertTrue(entry['effective_late'])

    def test_outcome_key_maturity_missing_flag_and_confirmation_reference_are_checked(self):
        variants=[]
        body=deepcopy(self.body);body['key']=self.key.replace('AAPL|','MSFT|');variants.append((body,self.at))
        body=deepcopy(self.body);body['missing']=True;variants.append((body,self.at))
        variants.append((self.body,self.witness.target_end-timedelta(microseconds=1)))
        for body,when in variants:
            with self.assertRaises(settlements.TestSettlementError):self.apply('observed',body,when)
        self.apply('observed',self.body,self.at)
        for reference,when in (('f'*64,self.at+timedelta(microseconds=1)),('c'*64,self.at)):
            with self.assertRaises(settlements.TestSettlementError):
                self.apply('confirmed',dict(key=self.key,observed_digest=reference),when)

    def test_durable_observation_recovers_only_acknowledgment_and_rejects_copied_history(self):
        with tempfile.TemporaryDirectory() as temp:
            class StorageOnly(settlements.TestSettlementJournal):
                def source_sha256(self):return hashlib.sha256(b'explicit settlement storage fixture; no phase authority').hexdigest()
            book=StorageOnly(Path(temp)/'test.sqlite');clock=Tick(aware('2026-11-24T13:00:00+00:00'))
            book.create(self.binding,clock=clock,check=lambda:None);clock=Tick(self.at);commit=book._commit
            def interrupted(db,kind):
                commit(db,kind)
                if kind=='observed':raise RuntimeError('synthetic interrupt after durable result')
            with issuer._exclusive(book.path),patch.object(book,'_commit',side_effect=interrupted):
                with self.assertRaises(evaluation_book.EvaluationBookError):
                    book.append(book.snapshot(),'observed',self.body,clock=clock,check=lambda:None)
            state=book.snapshot();self.assertEqual(state['head_seq'],3)
            self.assertEqual(state['domain']['frames'][self.key]['status'],'observed')
            with patch.object(common_outcome,'observe_common_outcome',side_effect=AssertionError('no new prices')):
                with issuer._exclusive(book.path):
                    state=book.append(state,'confirmed',dict(key=self.key,observed_digest=state['domain']['frames'][self.key]['observed_digest']),clock=clock,check=lambda:None)
            self.assertEqual(state['domain']['frames'][self.key]['outcome'],self.result)
            self.assertFalse(state['domain']['frames'][self.key]['effective_late'])
            with self.assertRaises(settlements.TestSettlementError):book.settle_once(self.key,timezone_file=TZ,clock=clock)
            other=Path(temp)/'copy.sqlite';shutil.copy2(book.path,other)
            with self.assertRaises(evaluation_book.EvaluationBookError):StorageOnly(other).snapshot()

    def test_confirmation_marker_commit_does_not_replace_result_commit_observation(self):
        with tempfile.TemporaryDirectory() as temp:
            class StorageOnly(settlements.TestSettlementJournal):
                def source_sha256(self):return hashlib.sha256(b'explicit marker storage fixture; no phase authority').hexdigest()
            book=StorageOnly(Path(temp)/'test.sqlite');clock=Tick(aware('2026-11-24T13:00:00+00:00'))
            book.create(self.binding,clock=clock,check=lambda:None);clock=Tick(self.at)
            with issuer._exclusive(book.path):state=book.append(book.snapshot(),'observed',self.body,clock=clock,check=lambda:None)
            commit=book._commit
            def late_marker(db,kind):
                commit(db,kind)
                if kind=='confirmed':clock.value=self.witness.target_end+timedelta(seconds=121)
            with issuer._exclusive(book.path),patch.object(book,'_commit',side_effect=late_marker):
                state=book.append(state,'confirmed',dict(key=self.key,observed_digest=state['domain']['frames'][self.key]['observed_digest']),clock=clock,check=lambda:None)
            entry=state['domain']['frames'][self.key]
            self.assertFalse(entry['effective_late'])
            self.assertLess(entry['commit_observed_at'],self.witness.target_end+timedelta(seconds=120))


if __name__=='__main__':unittest.main()
