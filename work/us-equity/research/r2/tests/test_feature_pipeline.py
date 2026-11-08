from contextlib import contextmanager,closing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict,replace
from datetime import timedelta
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_receipt_book as fixtures
from paperbot_r2.contracts import digest
from paperbot_r2 import receipt_book
from paperbot_r2.feature_pipeline import PipelinePlan,FeatureJournal,PipelineError,derive_once


class FeaturePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixtures.ReceiptBookTests.setUpClass.__func__(cls)
    @classmethod
    def tearDownClass(cls):fixtures.ReceiptBookTests.tearDownClass.__func__(cls)

    def setUp(self):
        environment=patch.dict(os.environ,{},clear=True);environment.start();self.addCleanup(environment.stop)
        fixtures.ReceiptBookTests.setUp(self)
        self.raw_plan=self.plan;self.raw=self.book
        self.raw.initialize(self.raw_plan,now=self.now)
        self.raw.record_calendar(self.calendar(),day='2020-03-09',received_at=self.now)
        self.plan=PipelinePlan(('AAPL',),'2020-03-09','2020-03-09',
            digest(asdict(self.raw_plan)),receipt_book._source(),self.raw_plan.timezone_sha256,
            lookback_returns=2)
        self.journal=FeatureJournal(Path(self.temp.name)/'features.sqlite')
        self.journal.initialize(self.plan,now=self.now)
        self.at=self.now

    def calendar(self,session=None):return fixtures.ReceiptBookTests.calendar(self,session)
    def bar(self,**kwargs):return fixtures.ReceiptBookTests.bar(self,**kwargs)

    def inputs(self):
        for i,price in enumerate((100.,110.,99.)):
            bar=self.bar(offset=i,close=price)
            self.raw.observe([bar],now=bar.received_at+timedelta(seconds=1))
        self.at=bar.ends+timedelta(seconds=3)

    def cycle(self,raw=None):return derive_once(self.journal,raw_book=raw or self.raw,clock=lambda:self.at)
    def entry(self):return next(iter(self.journal.snapshot()['entries'].values()))

    def test_actual_readiness_phases_and_repeat_are_durable_and_deduplicated(self):
        self.inputs();result=self.cycle()
        self.assertEqual(result['active_features'],1);self.assertEqual(result['pending_targets'],1)
        entry=self.entry();self.assertEqual(entry['phase'],'active')
        self.assertEqual(entry['accepted_bundle']['vector']['available_at'],self.at.isoformat())
        head=self.journal.snapshot()['head_seq'];self.at+=timedelta(seconds=10)
        self.cycle();self.assertEqual(self.journal.snapshot()['head_seq'],head)
        self.assertEqual(self.entry()['accepted_bundle'],entry['accepted_bundle'])

    def test_raw_read_time_is_taken_after_the_snapshot_transaction(self):
        self.inputs();owner=self
        class DelayedRaw:
            @contextmanager
            def _transaction(self):
                with owner.raw._transaction() as value:yield value
                owner.at+=timedelta(seconds=1)
        before=self.at;self.cycle(DelayedRaw())
        self.assertEqual(self.entry()['raw_context']['read_at'],(before+timedelta(seconds=1)).isoformat())

    def test_crash_after_prepared_recovers_with_new_time_not_original_time(self):
        self.inputs();initial=self.at;original=self.journal._append
        def crash(kind,body,at):
            result=original(kind,body,at)
            if kind=='prepared':raise RuntimeError('simulated_crash')
            return result
        with patch.object(self.journal,'_append',side_effect=crash):self.cycle()
        self.assertEqual(self.entry()['phase'],'prepared')
        self.at+=timedelta(seconds=10);self.cycle()
        self.assertEqual(self.entry()['phase'],'active')
        self.assertEqual(self.entry()['accepted_bundle']['vector']['available_at'],self.at.isoformat())
        self.assertNotEqual(self.entry()['accepted_bundle']['vector']['available_at'],initial.isoformat())

    def test_crash_after_ready_then_stale_recovery_aborts(self):
        self.inputs();original=self.journal._append
        def crash(kind,body,at):
            result=original(kind,body,at)
            if kind=='ready':raise RuntimeError('simulated_crash')
            return result
        with patch.object(self.journal,'_append',side_effect=crash):self.cycle()
        self.assertEqual(self.entry()['phase'],'ready')
        self.at+=timedelta(seconds=121);result=self.cycle()
        self.assertEqual(self.entry()['phase'],'aborted');self.assertEqual(result['active_features'],0)

    def test_mature_target_label_uses_post_read_time_and_unchanged_features(self):
        self.inputs();self.cycle();before=self.entry()['accepted_bundle']['vector']
        target=self.bar(offset=4,close=100.5)
        self.raw.observe([target],now=target.received_at+timedelta(seconds=1))
        self.at=target.received_at+timedelta(seconds=2);result=self.cycle()
        example=self.entry()['label_example']
        self.assertEqual(result['labels_observed'],1);self.assertEqual(example['label'],1)
        self.assertEqual(example['features'],before)
        self.assertEqual(example['outcome_available_at'],self.at.isoformat())
        self.assertFalse(self.entry()['missing'])
        first=dict(example);self.at+=timedelta(seconds=10);self.cycle()
        self.assertEqual(self.entry()['label_example'],first)

    def test_missing_then_late_target_retains_both_and_never_invents_zero(self):
        self.inputs();self.cycle();target=self.bar(offset=4,delay=130,close=100.4)
        self.at=target.ends+timedelta(seconds=121);self.raw.observe([],now=self.at);self.cycle()
        self.assertTrue(self.entry()['missing']);self.assertIsNone(self.entry()['label_example'])
        self.raw.observe([target],now=target.received_at+timedelta(seconds=1))
        self.at=target.received_at+timedelta(seconds=2);result=self.cycle()
        self.assertTrue(self.entry()['missing']);self.assertTrue(self.entry()['label_late'])
        self.assertEqual(result['late_labels'],1);self.assertEqual(self.entry()['label_example']['label'],0)

    def test_zero_volume_target_cannot_be_replaced_by_revision(self):
        self.inputs();self.cycle();target=self.bar(offset=4,volume=0.)
        self.raw.observe([target],now=target.received_at+timedelta(seconds=1))
        self.at=target.ends+timedelta(seconds=121);self.cycle()
        changed=replace(target,received_at=self.at+timedelta(seconds=1),volume=100.)
        self.raw.observe([changed],now=changed.received_at);self.at=changed.received_at+timedelta(seconds=1);self.cycle()
        self.assertTrue(self.entry()['missing']);self.assertIsNone(self.entry()['label_example'])

    def test_pause_happens_before_any_raw_access(self):
        paused=replace(self.plan,paused_days=('2020-03-09',))
        journal=FeatureJournal(Path(self.temp.name)/'paused.sqlite');journal.initialize(paused,now=self.now)
        class ForbiddenRaw:
            def _transaction(self):raise AssertionError('reserved_raw_must_not_be_read')
        result=derive_once(journal,raw_book=ForbiddenRaw(),clock=lambda:self.now)
        self.assertEqual(result['status'],'paused_reserved_dates');self.assertEqual(journal.snapshot()['head_seq'],1)

    def test_raw_rollback_is_rejected_without_resetting_derived_history(self):
        old=self.raw.path.read_bytes();self.inputs();self.cycle();head=self.journal.snapshot()['head_seq']
        self.raw.path.write_bytes(old);self.at+=timedelta(seconds=5);result=self.cycle()
        self.assertEqual(result['status'],'raw_evidence_unavailable')
        self.assertEqual(self.journal.snapshot()['head_seq'],head)

    def test_no_historical_backfill_for_stale_or_missing_input(self):
        self.inputs();self.at+=timedelta(seconds=121);result=self.cycle()
        self.assertEqual(result['active_features'],0)
        self.assertFalse(any(e['phase']=='active' for e in self.journal.snapshot()['entries'].values()))

    def test_explicit_initialization_missing_journal_and_tampering(self):
        missing=FeatureJournal(Path(self.temp.name)/'absent.sqlite')
        with self.assertRaises(PipelineError):missing.snapshot()
        self.assertFalse(missing.path.exists())
        with self.assertRaises(PipelineError):self.journal.initialize(self.plan,now=self.now)
        with closing(sqlite3.connect(self.journal.path)) as db:
            db.execute("UPDATE observations SET payload='{}' WHERE seq=1");db.commit()
        with self.assertRaises(PipelineError):self.journal.snapshot()

    def test_secret_like_exception_is_not_logged_or_projected(self):
        class BrokenRaw:
            def _transaction(self):raise RuntimeError('SECRET_MUST_NOT_APPEAR')
        result=self.cycle(BrokenRaw())
        self.assertNotIn('SECRET_MUST_NOT_APPEAR',json.dumps(result))
        self.assertEqual(result['orders_submitted'],0);self.assertEqual(result['models_fitted'],0)

    def test_credential_environment_is_rejected_before_raw_access(self):
        class ForbiddenRaw:
            def _transaction(self):raise AssertionError('must_not_read')
        with patch.dict(os.environ,{'ANTHROPIC_AUTH_TOKEN':'synthetic_do_not_project'}):
            result=self.cycle(ForbiddenRaw())
        self.assertEqual(result['status'],'pipeline_unavailable')
        self.assertNotIn('synthetic_do_not_project',json.dumps(result))
        self.assertEqual(self.journal.snapshot()['head_seq'],1)

    def test_concurrent_cycles_never_duplicate_and_deep_replay_matches(self):
        self.inputs()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda _:self.cycle(),range(2)))
        self.assertTrue(any(r['active_features']==1 for r in results))
        state=self.journal.snapshot(deep=True)
        self.assertEqual(len(state['entries']),1)
        self.assertEqual(state,self.journal.snapshot())

    def test_slow_ready_commit_cannot_create_retroactive_active_feature(self):
        self.inputs();original=self.journal._append
        def delayed(kind,body,at):
            result=original(kind,body,at)
            if kind=='ready':self.at+=timedelta(seconds=121)
            return result
        with patch.object(self.journal,'_append',side_effect=delayed):self.cycle()
        self.assertEqual(self.entry()['phase'],'aborted')
        self.assertIsNone(self.entry()['accepted_bundle'])

    def test_slow_label_commit_retains_missing_and_late_status(self):
        self.inputs();self.cycle();target=self.bar(offset=4,close=100.5)
        self.raw.observe([target],now=target.received_at+timedelta(seconds=1))
        self.at=target.received_at+timedelta(seconds=2);original=self.journal._append
        def delayed(kind,body,at):
            result=original(kind,body,at)
            if kind=='label_prepared':self.at+=timedelta(seconds=121)
            return result
        with patch.object(self.journal,'_append',side_effect=delayed):self.cycle()
        entry=self.entry()
        self.assertTrue(entry['missing']);self.assertTrue(entry['label_late'])
        self.assertEqual(entry['label_example']['outcome_available_at'],self.at.isoformat())
        self.journal.snapshot(deep=True)

    def test_failed_window_is_durable_deduplicated_and_retained_after_recovery(self):
        first=self.bar(offset=0);last=self.bar(offset=2)
        self.raw.observe([first],now=first.received_at)
        self.raw.observe([last],now=last.received_at)
        self.at=last.ends+timedelta(seconds=3);result=self.cycle()
        self.assertEqual(result['failed_windows'],1)
        failure=next(iter(self.journal.snapshot()['failed_windows'].values()))
        self.assertEqual(failure['reason'],'missing_input')
        head=self.journal.snapshot()['head_seq'];self.cycle()
        self.assertEqual(self.journal.snapshot()['head_seq'],head)
        # A late first receipt can enable a fresh window, but cannot erase the failure.
        middle=replace(self.bar(offset=1),received_at=self.at)
        self.raw.observe([middle],now=self.at);result=self.cycle()
        self.assertEqual(result['active_features'],1);self.assertEqual(result['failed_windows'],1)

    def test_invalid_and_stale_windows_are_recorded_without_backdating(self):
        self.inputs();self.at+=timedelta(seconds=121);result=self.cycle()
        self.assertEqual(result['failed_windows'],1)
        self.assertEqual(next(iter(self.journal.snapshot()['failed_windows'].values()))['reason'],'stale_input')

    def test_completed_sample_calendar_change_is_sticky_quarantine(self):
        self.inputs();self.cycle();target=self.bar(offset=4,close=100.5)
        self.raw.observe([target],now=target.received_at+timedelta(seconds=1))
        self.at=target.received_at+timedelta(seconds=2);self.cycle()
        original=self.entry()['label_example']
        changed=replace(self.session,closes=self.session.closes-timedelta(minutes=15))
        self.at+=timedelta(seconds=5)
        self.raw.record_calendar(self.calendar(changed),day='2020-03-09',received_at=self.at)
        result=self.cycle()
        self.assertTrue(self.entry()['quarantined']);self.assertEqual(result['usable_labels'],0)
        self.assertEqual(result['quarantined_features'],1);self.assertEqual(self.entry()['label_example'],original)
        self.raw.record_calendar(self.calendar(),day='2020-03-09',received_at=self.at)
        self.cycle();self.assertTrue(self.entry()['quarantined'])
        self.journal.snapshot(deep=True)

    def test_same_head_raw_metadata_rollback_is_rejected(self):
        self.inputs();self.cycle();old=self.raw.path.read_bytes()
        original_head=self.raw.snapshot()['head_seq']
        self.at+=timedelta(seconds=10);self.raw.observe([],now=self.at)
        self.assertEqual(self.raw.snapshot()['head_seq'],original_head)
        self.cycle();head=self.journal.snapshot()['head_seq']
        self.raw.path.write_bytes(old);self.at+=timedelta(seconds=1)
        self.assertEqual(self.cycle()['status'],'raw_evidence_unavailable')
        self.assertEqual(self.journal.snapshot()['head_seq'],head)

    def test_closed_session_reads_are_bounded_but_regular_session_is_not_throttled(self):
        self.at=self.session.closes+timedelta(hours=2)
        self.raw.observe([],now=self.at);self.cycle()
        class ForbiddenRaw:
            def _transaction(self):raise AssertionError('closed_session_read_budget')
        self.at+=timedelta(seconds=30)
        self.assertEqual(self.cycle(ForbiddenRaw())['status'],'market_closed')


if __name__=='__main__':unittest.main()
