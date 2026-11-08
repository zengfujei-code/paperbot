"""Independent synthetic arithmetic; no forecast/selection authority."""
from dataclasses import asdict,replace
from datetime import timedelta
import json
import math
from pathlib import Path
import sys
import unittest

sys.path[:0]=[str(Path(__file__).resolve().parents[3]),str(Path(__file__).resolve().parents[1])]
from paperbot_r2 import common_scores as scores,common_registration as registration
from paperbot_r2.contracts import aware,digest
from paperbot_r2.training_data import TRAINING_DAYS,TIMEZONE_SHA256,SYMBOLS
from paperbot_r2.model_training import FAMILIES


def plan():
    days=registration.VALIDATION_DAYS+registration.TEST_DAYS
    sessions=[dict(day=d,opens=d+'T14:30:00+00:00',
        closes=d+('T18:00:00+00:00' if d=='2026-11-27' else 'T21:00:00+00:00')) for d in days]
    return dict(symbols=list(SYMBOLS),pairs=[list(p) for p in registration.PAIRS],
        split=dict(train=list(TRAINING_DAYS),validation=list(registration.VALIDATION_DAYS),test=list(registration.TEST_DAYS),context=[]),
        calendar=dict(sessions=sessions,sha256=digest(sessions),timezone_sha256=TIMEZONE_SHA256),
        event=dict(formula='abs(target_first_close/gap_first_close-1)>=threshold',threshold_fraction=.005,
                   gap_seconds=900,horizon_seconds=900,feed='iex',adjustment='raw',currency='USD',volume_unit='shares'))


def fits(failed=()):return tuple((s,f,'failed' if (s,f) in failed else 'succeeded') for s,f in registration.PAIRS)


def row(family,p=.2,status='computed',mean=0.,variance=.0001):
    risk=family in ('ewma','garch','hmm')
    return scores.ScoreRow(family,status,float(p) if status=='computed' else None,
        float(mean) if risk and status=='computed' else None,float(variance) if risk and status=='computed' else None)


def frame(slot,*,probabilities=None,label=0,late=False,missing=False,quarantined=False,issued=True,settled=True):
    rows=tuple(row(f,(probabilities or {}).get(f,.2)) for f in FAMILIES)
    return scores.ScoreFrame(slot.key,slot.symbol,slot.event_at,slot.target_start,slot.target_end,rows,
        issued,settled,label if settled else None,.01 if label and settled else 0. if settled else None,
        missing,late,quarantined)


FINAL=aware('2026-12-04T21:02:01+00:00')


class CommonScoreTests(unittest.TestCase):
    def setUp(self):self.plan=plan();self.slots=scores.planned_slots(self.plan)
    def report(self,frames,*,at=FINAL,training=None):
        return scores.evaluate_frames(self.plan,tuple(frames),as_of=at,fit_statuses=training or fits())
    def full(self,**kwargs):return [frame(s,**kwargs) for s in self.slots]

    def test_exact_normal_short_session_grid_and_no_test_admission(self):
        self.assertEqual(len(self.slots),348)
        for symbol in SYMBOLS:
            found=[s for s in self.slots if s.symbol==symbol]
            self.assertEqual(len(found),116)
            self.assertEqual(sum(s.day=='2026-11-27' for s in found),4)
        first=self.slots[0];self.assertEqual(first.event_at,aware('2026-11-24T16:45:00+00:00'))
        self.assertEqual(first.target_end,aware('2026-11-24T17:15:00+00:00'))
        with self.assertRaises(scores.CommonScoreError):scores.planned_slots(self.plan,phase='test')

    def test_point_losses_match_independent_arithmetic_and_exact_infinities(self):
        result=scores.point_losses(row('garch',.25,mean=.1,variance=.04),1,.3)
        self.assertAlmostEqual(result['brier'],.75**2)
        self.assertAlmostEqual(result['log_loss'],-math.log(.25))
        self.assertAlmostEqual(result['variance_proxy_mse'],0.,places=15)
        self.assertAlmostEqual(result['variance_proxy_qlike'],math.log(.04)+1.)
        self.assertAlmostEqual(result['second_moment_mse'],(.05-.09)**2)
        self.assertAlmostEqual(result['second_moment_qlike'],math.log(.05)+.09/.05)
        self.assertEqual(scores.point_losses(row('prevalence',0.),1,0.)['log_loss'],math.inf)
        self.assertEqual(scores.point_losses(row('prevalence',1.),0,0.)['log_loss'],math.inf)
        self.assertEqual(scores.point_losses(row('prevalence',1.),1,0.)['log_loss'],0.)

    def test_calibration_boundaries_do_not_shift_or_clip(self):
        for p,index in ((0.,0),(.2,1),(.4,2),(.6,3),(.8,4),(1.,4)):
            self.assertEqual(scores.point_losses(row('prevalence',p),0,0.)['calibration_bin'],index)

    def test_full_common_metrics_and_deterministic_baseline_tie(self):
        report=self.report(self.full());self.assertTrue(report['engineering_gate_passed'])
        self.assertEqual(report['proposed_family'],'prevalence');self.assertTrue(report['math_only'])
        self.assertFalse(report['selection_admission']);self.assertFalse(report['test_admission'])
        self.assertEqual(report['trial_units'],15)
        for family in FAMILIES:self.assertAlmostEqual(report['macro'][family]['brier'],.04)
        self.assertEqual(len(report['groups']),24)
        for group in report['groups']:
            self.assertEqual(group['counts']['common'],group['counts']['planned'])
            self.assertEqual(group['counts']['missing'],0)
            bins=group['families']['prevalence']['calibration']
            self.assertEqual(bins[1]['count'],group['counts']['planned'])
            self.assertAlmostEqual(bins[1]['mean_probability'],.2)
            self.assertEqual(bins[1]['event_fraction'],0.)

    def test_same_common_intersection_and_failed_fit_does_not_remove_candidate(self):
        frames=self.full();one=frames[0]
        frames[0]=replace(one,rows=tuple(row(f,status='training_fit_failed') if f=='gaussian_nb' else r for f,r in zip(FAMILIES,one.rows)))
        # Declared failure must hold for every published slot of this candidate.
        # The remaining fixture rows contradict that, so reject rather than
        # silently inferring a different fit from the convenient observations.
        with self.assertRaises(scores.CommonScoreError):self.report(frames,training=fits((('AAPL','gaussian_nb'),)))

    def test_failed_family_slots_remain_in_all_denominators(self):
        frames=self.full()
        frames=[replace(x,rows=tuple(row(f,status='training_fit_failed') if f=='gaussian_nb' and x.symbol=='SPY' else r
                for f,r in zip(FAMILIES,x.rows))) for x in frames]
        report=self.report(frames,training=fits((('SPY','gaussian_nb'),)))
        self.assertFalse(report['engineering_gate_passed']);self.assertIsNone(report['proposed_family'])
        for f in FAMILIES:self.assertIsNone(report['macro'][f]['brier'])
        for group in report['groups']:
            if group['symbol']=='SPY':
                self.assertEqual(group['counts']['common'],0)
                self.assertEqual(group['families']['gaussian_nb']['counts']['failed_fit'],group['counts']['planned'])
                self.assertEqual(group['families']['ewma']['counts']['settled'],group['counts']['planned'])
        self.assertEqual(report['trial_units'],15)

    def test_macro_is_equal_symbol_weight_not_equal_row_weight(self):
        frames=[]
        for slot in self.slots:
            if slot.symbol=='SPY' and slot.event_at>aware(slot.day+'T17:30:00+00:00'):continue
            frames.append(frame(slot,probabilities={f:float(slot.symbol=='SPY') for f in FAMILIES}))
        report=self.report(frames)
        for family in FAMILIES:self.assertAlmostEqual(report['macro'][family]['brier'],1./3.)
        self.assertFalse(report['engineering_gate_passed'])

    def test_missing_symbol_never_becomes_a_two_symbol_macro(self):
        report=self.report(frame(s) for s in self.slots if s.symbol!='MSFT')
        for f in FAMILIES:self.assertIsNone(report['macro'][f]['brier'])
        self.assertIsNone(report['proposed_family'])

    def test_each_session_symbol_eighty_percent_gate_including_half_day(self):
        kept=[]
        for day in registration.VALIDATION_DAYS:
            for symbol in SYMBOLS:
                group=[s for s in self.slots if s.day==day and s.symbol==symbol]
                kept.extend(frame(s) for s in group[:4 if day=='2026-11-27' else 13])
        self.assertTrue(self.report(kept)['engineering_gate_passed'])
        half=next(x for x in kept if x.symbol=='AAPL' and x.event_at.date().isoformat()=='2026-11-27')
        self.assertFalse(self.report(x for x in kept if x.key!=half.key)['engineering_gate_passed'])

    def test_late_missing_quarantine_unissued_and_future_counts_stay_visible(self):
        frames=self.full()
        frames[0]=replace(frames[0],late=True)
        frames[1]=replace(frames[1],missing=True)
        frames[2]=replace(frames[2],quarantined=True)
        frames[3]=replace(frames[3],issued=False,settled=False,label=None,return_fraction=None)
        report=self.report(frames);group=report['groups'][0]
        self.assertEqual(group['counts']['planned'],16);self.assertEqual(group['counts']['common'],12)
        for name in ('late','missing','quarantined','unissued'):self.assertEqual(group['counts'][name],1)
        early=self.report((),at=aware('2026-11-24T14:30:00+00:00'))
        self.assertEqual(sum(g['counts']['not_yet_due'] for g in early['groups']),348)
        self.assertFalse(early['engineering_gate_passed']);self.assertIsNone(early['proposed_family'])

    def test_infinite_aggregate_uses_explicit_strict_json_tag(self):
        report=self.report(self.full(probabilities={f:0. for f in FAMILIES},label=1))
        self.assertEqual(report['macro']['prevalence']['log_loss'],{'loss':'positive_infinity'})
        json.dumps(report,allow_nan=False)

    def test_failed_and_late_rows_do_not_inflate_successful_prediction_counts(self):
        frames=self.full();frames[0]=replace(frames[0],late=True)
        late=self.report(frames)['groups'][0]['families']['ewma']['counts']
        self.assertEqual(late['issued'],16);self.assertEqual(late['settled'],16)
        self.assertEqual(late['usable_settled'],15);self.assertEqual(late['late'],1)
        frames=[replace(x,rows=tuple(row(f,status='training_fit_failed') if f=='gaussian_nb' and x.symbol=='SPY' else r
                for f,r in zip(FAMILIES,x.rows))) for x in self.full()]
        report=self.report(frames,training=fits((('SPY','gaussian_nb'),)))
        failed=next(g for g in report['groups'] if g['symbol']=='SPY')['families']['gaussian_nb']['counts']
        self.assertEqual(failed['publication_frames'],16);self.assertEqual(failed['issued'],0)
        self.assertEqual(failed['unissued'],16);self.assertEqual(failed['settled'],0)
        self.assertEqual(failed['failed_fit'],16)

    def test_duplicate_off_grid_future_settled_and_invalid_inputs_reject(self):
        one=frame(self.slots[0])
        with self.assertRaises(scores.CommonScoreError):self.report((one,one))
        offgrid=replace(one,event_at=one.event_at+timedelta(seconds=60),target_start=one.target_start+timedelta(seconds=60),
                        target_end=one.target_end+timedelta(seconds=60),key='AAPL|2026-11-24T16:46:00+00:00')
        with self.assertRaises(scores.CommonScoreError):self.report((offgrid,))
        altered=replace(one,event_at=one.event_at+timedelta(seconds=900),target_start=one.target_start+timedelta(seconds=900),
                        target_end=one.target_end+timedelta(seconds=900),key='AAPL|2026-11-24T17:00:00+00:00')
        with self.assertRaises(scores.CommonScoreError):self.report((altered,),at=aware('2026-11-24T16:45:01+00:00'))
        for p in (-.1,1.1,float('nan'),True):
            with self.assertRaises(scores.CommonScoreError):scores.ScoreRow('prevalence','computed',p,None,None)

    def test_policy_commits_all_rules_and_both_code_roles(self):
        policy=scores.ScorePolicy.build('e'*64,'f'*64)
        self.assertEqual(policy.configuration_sha256,policy.seal)
        self.assertEqual(json.loads(policy.rules_json)['common_families'],list(FAMILIES))
        self.assertIn('prospective_protocol_sha256',asdict(policy))
        with self.assertRaises(scores.CommonScoreError):replace(policy,rules_json='{}')
