"""Synthetic first outcomes and actual commit witnesses, never market proof."""
from datetime import timedelta
from dataclasses import asdict
from contextlib import closing
import os
from pathlib import Path
import socket
import sqlite3
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zoneinfo

sys.path.insert(0,str(Path(__file__).resolve().parent))
from comparison_fixture import ComparisonWorld
from test_training_data import ROOT,TZ,Tick
from test_canonical_history import HANDOFF
from paperbot_r2 import common_outcome,common_registration as common,receipt_book
from paperbot_r2 import validation_settlements as settlements,validation_scores
from paperbot_r2.contracts import aware,digest


class ValidationSettlementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_tz=zoneinfo.TZPATH
        zoneinfo.reset_tzpath([str(ROOT/'outputs/zoneinfo')]);zoneinfo.ZoneInfo.clear_cache()
    @classmethod
    def tearDownClass(cls):zoneinfo.reset_tzpath(cls.old_tz);zoneinfo.ZoneInfo.clear_cache()
    def setUp(self):
        env=patch.dict(os.environ,{},clear=True);env.start();self.addCleanup(env.stop)
        network=patch.object(socket,'create_connection',side_effect=AssertionError('no network'))
        network.start();self.addCleanup(network.stop)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.directory=Path(temp.name);self.c=ComparisonWorld(self.directory)
        self.c.issue();self.book=self.c.settlements
    def result(self):return self.book.snapshot()['domain']['frames'][self.c.key]

    def test_first_close_label_actual_commit_and_failed_candidate_witness(self):
        self.c.outcome();result=self.c.settle()
        self.assertEqual(result['status'],'confirmed');self.assertFalse(result['effective_late'])
        self.assertEqual(result['outcome'].label,1)
        self.assertAlmostEqual(result['outcome'].return_fraction,99.7/100.4-1.,places=14)
        self.assertGreater(result['commit_observed_at'],result['observed_at'])
        self.assertEqual(result['witness'].key,self.c.key)
        self.assertLess(result['witness'].issue_commit_observed_at,result['outcome'].target_start)
        self.assertEqual(self.book.snapshot()['head_seq'],4)
        self.assertFalse(result['current_scoring_permission'])
        with patch.object(common_outcome,'observe_common_outcome',side_effect=AssertionError('first result immutable')):
            again=self.c.settle()
        self.assertEqual(again,result)

    def test_maturity_reserved_and_test_phase_gate_before_any_sql(self):
        clocks=(self.c.clock,Tick(aware('2026-11-12T18:00:00+00:00')),Tick(aware('2026-12-07T14:30:00+00:00')))
        for clock in clocks:
            with patch('sqlite3.connect',side_effect=AssertionError('phase before SQL')) as connect:
                with self.assertRaises(settlements.SettlementError):
                    self.book.settle_once(self.c.key,timezone_file=TZ,clock=clock)
                connect.assert_not_called()

    def test_waiting_is_not_missing_but_post_deadline_absence_is_permanent(self):
        self.c.outcome(count=10)
        end=aware(self.c.key.split('|')[1])+timedelta(seconds=1800)
        self.c.clock=Tick(end+timedelta(seconds=30))
        self.assertEqual(self.c.settle()['status'],'waiting')
        self.assertNotIn(self.c.key,self.book.snapshot()['domain']['frames'])
        self.c.clock=Tick(end+timedelta(seconds=121));result=self.c.settle()
        self.assertTrue(result['missing']);self.assertTrue(result['effective_late']);self.assertIsNone(result['outcome'])
        with patch.object(common_outcome,'observe_common_outcome',side_effect=AssertionError('no retroactive fill')):
            self.assertEqual(self.c.settle(),result)

    def test_result_commit_crossing_deadline_cannot_become_timely(self):
        self.c.outcome();commit=self.book._commit
        end=aware(self.c.key.split('|')[1])+timedelta(seconds=1800)
        def cross(db,kind):
            commit(db,kind)
            if kind=='observed':self.c.clock.value=end+timedelta(seconds=121)
        with patch.object(self.book,'_commit',side_effect=cross):result=self.c.settle()
        self.assertTrue(result['effective_late']);self.assertGreater(result['commit_observed_at'],end+timedelta(seconds=120))

    def test_marker_commit_time_is_distinct_from_observed_result_commit(self):
        self.c.outcome();commit=self.book._commit
        end=aware(self.c.key.split('|')[1])+timedelta(seconds=1800)
        def cross(db,kind):
            commit(db,kind)
            if kind=='confirmed':self.c.clock.value=end+timedelta(seconds=121)
        with patch.object(self.book,'_commit',side_effect=cross):result=self.c.settle()
        self.assertFalse(result['effective_late']);self.assertLessEqual(result['commit_observed_at'],end+timedelta(seconds=120))

    def test_ambiguous_observation_commit_recovers_only_its_acknowledgment(self):
        self.c.outcome();commit=self.book._commit
        def interrupt(db,kind):
            commit(db,kind)
            if kind=='observed':raise RuntimeError('caller interrupted after COMMIT')
        with patch.object(self.book,'_commit',side_effect=interrupt):
            with self.assertRaises(settlements.SettlementError):self.c.settle()
        self.assertEqual(self.result()['status'],'observed')
        with (patch.object(common_outcome,'observe_common_outcome',side_effect=AssertionError('no reobservation')),
              patch.object(receipt_book.ReceiptBook,'_transaction',side_effect=AssertionError('no new prices on recovery'))):
            result=self.c.settle()
        self.assertEqual(result['status'],'confirmed');self.assertFalse(result['effective_late'])

    def test_canonical_invalidation_during_outcome_read_vetoes_persistence(self):
        self.c.outcome();original=common_outcome.observe_common_outcome
        def invalid(*args,**kwargs):
            value=original(*args,**kwargs)
            common.import_exposure(self.c.w.registry,domain='us_equity',days=['2026-11-24'],source_sha256='e'*64,
                timezone_file=TZ,clock=self.c.clock)
            return value
        with patch.object(common_outcome,'observe_common_outcome',side_effect=invalid):
            with self.assertRaises(settlements.SettlementError):self.c.settle()
        self.assertNotIn(self.c.key,self.book.snapshot()['domain']['frames'])

    def test_successor_segment_has_same_first_label_and_explicit_original_sources(self):
        self.c.issue('2026-11-27');self.c.outcome('2026-11-27');result=self.c.settle()
        self.assertEqual(result['outcome'].label,1)
        self.assertEqual(result['outcome'].raw_plan_sha256,digest(asdict(self.c.w.next_raw.snapshot()['plan'])))
        self.assertNotEqual(result['outcome'].raw_plan_sha256,digest(asdict(self.c.w.f.raw.snapshot()['plan'])))
        self.assertEqual(result['witness'].symbol,'AAPL')

    def test_policy_commitment_matches_real_registered_configuration(self):
        self.assertEqual(self.c.w.registration.plan()['configuration_sha256'],self.c.policy.seal)
        self.assertEqual(self.c.policy.evaluation_source_sha256,validation_scores.source_sha256())
        self.assertFalse(self.book.snapshot()['domain']['frames'])

    def test_unconfirmed_publication_cannot_open_outcome_prices(self):
        key=self.c.key.replace('AAPL|','MSFT|');append=self.c.issues._append
        def interrupt(state,kind,*args,**kwargs):
            if kind=='confirmed':raise RuntimeError('unconfirmed issue')
            return append(state,kind,*args,**kwargs)
        with patch.object(self.c.issues,'_append',side_effect=interrupt):
            with self.assertRaises(Exception):
                self.c.issues.issue_once(key,registry=self.c.w.registry,training_journal=self.c.w.training,
                    pairs=self.c.w.source_pairs(),timezone_file=TZ,clock=self.c.clock)
        self.c.key=key;self.c.outcome()
        with patch.object(common_outcome,'observe_common_outcome',side_effect=AssertionError('no unissued outcome')) as observe:
            with self.assertRaises(settlements.SettlementError):self.c.settle()
            observe.assert_not_called()

    def test_recovery_after_deadline_retains_label_but_is_late(self):
        self.c.outcome();commit=self.book._commit
        def interrupt(db,kind):
            commit(db,kind)
            if kind=='observed':raise RuntimeError('first result already durable')
        with patch.object(self.book,'_commit',side_effect=interrupt):
            with self.assertRaises(settlements.SettlementError):self.c.settle()
        first=self.result()['outcome'];self.c.clock=Tick(first.target_end+timedelta(seconds=121))
        with patch.object(common_outcome,'observe_common_outcome',side_effect=AssertionError('do not reobserve')):
            recovered=self.c.settle()
        self.assertEqual(recovered['outcome'],first);self.assertTrue(recovered['effective_late'])

    def test_copied_settlement_book_is_not_another_first_history(self):
        other=self.directory/'copy.sqlite';shutil.copy2(self.book.path,other)
        with self.assertRaises(Exception):settlements.ValidationSettlementJournal(other).snapshot()

    def test_quarantined_feature_never_opens_label_prices(self):
        self.c.outcome();features=self.c.w.f.features
        features._append('quarantined',dict(key=self.c.key,raw_context=features.snapshot()['raw_context'],calendar=None),self.c.clock())
        with patch.object(common_outcome,'observe_common_outcome',side_effect=AssertionError('no quarantined label')) as observe:
            with self.assertRaises(settlements.SettlementError):self.c.settle()
            observe.assert_not_called()

    def test_wal_only_metadata_commit_during_read_invalidates_epoch(self):
        self.c.outcome();raw=self.c.w.f.raw;original=common_outcome.observe_common_outcome
        with closing(sqlite3.connect(raw.path,isolation_level=None)) as keep:
            self.assertEqual(keep.execute('PRAGMA journal_mode=WAL').fetchone()[0].lower(),'wal')
            before=[];after=[]
            def changed(*args,**kwargs):
                result=original(*args,**kwargs);stat=raw.path.stat();before.append((stat.st_size,stat.st_mtime_ns))
                raw.observe([],now=self.c.clock());stat=raw.path.stat();after.append((stat.st_size,stat.st_mtime_ns))
                return result
            with patch.object(common_outcome,'observe_common_outcome',side_effect=changed):
                with self.assertRaises(settlements.SettlementError):self.c.settle()
            self.assertEqual(before,after)
        self.assertNotIn(self.c.key,self.book.snapshot()['domain']['frames'])
