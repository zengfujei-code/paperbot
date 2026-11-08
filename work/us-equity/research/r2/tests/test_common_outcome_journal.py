from contextlib import closing, contextmanager
from dataclasses import replace
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import zoneinfo


ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[1])]

from paperbot_us.data import Bar, Calendar, Session
from paperbot_us.monitor import ObservationArchive
from paperbot_r2.contracts import digest
from paperbot_r2 import receipt_book
from paperbot_r2.feature_pipeline import FeatureJournal, PipelinePlan, derive_once
from paperbot_r2.common_outcome_journal import OutcomeJournal, OutcomeJournalError


class Tick:
    def __init__(self, start):
        self.value = start

    def __call__(self):
        self.value += timedelta(microseconds=100)
        return self.value


class OutcomeJournalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_tzpath = zoneinfo.TZPATH
        zoneinfo.reset_tzpath([str(ROOT / 'outputs/zoneinfo')])
        zoneinfo.ZoneInfo.clear_cache()

    @classmethod
    def tearDownClass(cls):
        zoneinfo.reset_tzpath(cls.old_tzpath)
        zoneinfo.ZoneInfo.clear_cache()

    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.session = Session.from_alpaca(dict(date='2020-03-09', open='09:30', close='16:00'))
        before = self.session.opens - timedelta(minutes=30)
        tzsha = sha256((ROOT / 'outputs/zoneinfo/America/New_York').read_bytes()).hexdigest()
        raw_plan = receipt_book.ReceiptPlan(('AAPL',), '2020-03-09', '2020-03-09', tzsha)
        self.raw = receipt_book.ReceiptBook(self.directory / 'raw.sqlite')
        self.raw.initialize(raw_plan, now=before)
        self.raw.record_calendar(Calendar([self.session], start='2020-03-09', end='2020-03-09'),
                                 day='2020-03-09', received_at=before)
        plan = PipelinePlan(('AAPL',), '2020-03-09', '2020-03-09',
                            digest(raw_plan.__dict__), receipt_book._source(), tzsha,
                            lookback_returns=2)
        self.features = FeatureJournal(self.directory / 'features.sqlite')
        self.features.initialize(plan, now=before)
        self.journal = OutcomeJournal(self.directory / 'common.sqlite')
        self.journal.initialize(self.features, self.raw, clock=lambda: before)
        self.at = before

    def bar(self, offset, close):
        starts = self.session.opens + timedelta(seconds=900 * offset)
        return Bar('AAPL', starts, starts + timedelta(seconds=901),
                   100., max(100., close), min(100., close), close, 100.)

    def receive(self, offset, close):
        value = self.bar(offset, close)
        self.raw.observe([value], now=value.received_at + timedelta(seconds=1))
        self.at = value.received_at + timedelta(seconds=2)
        return value

    def ready_feature(self):
        for offset, close in enumerate((100., 110., 99.)):
            value = self.receive(offset, close)
        derive_once(self.features, raw_book=self.raw, clock=lambda: self.at)
        state = self.features.snapshot()
        self.assertEqual(len(state['entries']), 1)
        self.assertEqual(next(iter(state['entries'].values()))['phase'], 'active')
        return next(iter(state['entries']))

    def record(self, key):
        return self.journal.record_first(self.features, self.raw, key, clock=Tick(self.at))

    def test_first_result_has_separate_post_commit_ack(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 100.5)
        row = self.record(key)
        state = self.journal.snapshot()
        self.assertEqual(state['head_seq'], 3)
        self.assertEqual(row, state['entries'][key])
        self.assertEqual(row['status'], 'acknowledged')
        self.assertFalse(row['score_eligible'])
        self.assertEqual(row['outcome']['label'], 1)
        self.assertEqual(len(row['ack_event_digest']), 64)
        self.assertGreater(datetime.fromisoformat(row['acknowledged_at']),
                           datetime.fromisoformat(row['observed_event_at']))
        self.assertEqual(row['outcome']['feature_key'], key)

    def test_duplicate_cannot_replace_first_label_or_ack_time(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        first = self.receive(4, 100.4)
        original = self.record(key)
        revised = replace(first, close=101., high=101., received_at=self.at + timedelta(seconds=2))
        self.raw.observe([revised], now=revised.received_at + timedelta(seconds=1))
        self.at = revised.received_at + timedelta(seconds=2)
        self.assertEqual(self.record(key), original)
        self.assertEqual(self.journal.snapshot()['head_seq'], 3)

    def test_missing_target_cannot_create_an_outcome(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        with self.assertRaises(OutcomeJournalError):
            self.record(key)
        self.assertEqual(self.journal.snapshot()['head_seq'], 1)
        self.assertEqual(self.journal.snapshot()['entries'], {})

    def test_ack_failure_keeps_pending_first_result_and_recovery_ack_is_later(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 100.5)
        original_append = ObservationArchive.append

        def fail_ack(archive, report):
            if report['kind'] == 'acknowledged':
                raise OSError('simulated crash after observed commit')
            return original_append(archive, report)

        with patch.object(ObservationArchive, 'append', fail_ack):
            with self.assertRaises(OutcomeJournalError):
                self.record(key)
        pending = self.journal.snapshot()['entries'][key]
        self.assertEqual(pending['status'], 'pending_ack')
        self.assertEqual(self.journal.snapshot()['head_seq'], 2)
        self.at += timedelta(minutes=5)
        recovered = self.record(key)
        self.assertEqual(self.journal.snapshot()['head_seq'], 3)
        self.assertEqual(recovered['outcome'], pending['outcome'])
        self.assertTrue(recovered['effective_late'])
        self.assertGreater(datetime.fromisoformat(recovered['acknowledged_at']),
                           datetime.fromisoformat(pending['observed_event_at']))

    def test_corrupt_old_event_prevents_healthy_replay(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 100.5)
        self.record(key)
        with closing(sqlite3.connect(self.journal.path)) as db:
            db.execute('UPDATE observations SET digest=? WHERE seq=2', ('0' * 64,))
            db.commit()
        with self.assertRaises(OutcomeJournalError):
            self.journal.snapshot()

    def test_calendar_revision_keeps_first_record_but_not_a_scoring_claim(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 100.5)
        original = self.record(key)
        changed = replace(self.session, closes=self.session.closes - timedelta(minutes=15))
        self.at += timedelta(seconds=1)
        self.raw.record_calendar(Calendar([changed], start='2020-03-09', end='2020-03-09'),
                                 day='2020-03-09', received_at=self.at)
        self.at += timedelta(seconds=1)
        derive_once(self.features, raw_book=self.raw, clock=lambda: self.at)
        self.assertTrue(self.features.snapshot()['entries'][key]['quarantined'])
        self.assertEqual(self.record(key), original)
        self.assertFalse(self.journal.snapshot()['entries'][key]['score_eligible'])

    def test_mismatched_raw_plan_refuses_without_appending(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 100.5)
        other = receipt_book.ReceiptBook(self.directory / 'other.sqlite')
        plan = receipt_book.ReceiptPlan(('MSFT',), '2020-03-09', '2020-03-09',
                                        self.raw.snapshot()['plan'].timezone_sha256)
        other.initialize(plan, now=self.session.opens - timedelta(minutes=30))
        with self.assertRaises(OutcomeJournalError):
            self.journal.record_first(self.features, other, key, clock=Tick(self.at))
        self.assertEqual(self.journal.snapshot()['head_seq'], 1)

    def test_duplicate_refuses_a_different_feature_book_with_same_plan(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 100.5)
        self.record(key)
        other = FeatureJournal(self.directory / 'empty_same_plan.sqlite')
        other.initialize(self.features.snapshot()['plan'],
                         now=self.session.opens - timedelta(minutes=30))
        with self.assertRaises(OutcomeJournalError):
            self.journal.record_first(other, self.raw, key, clock=Tick(self.at))
        self.assertEqual(self.journal.snapshot()['head_seq'], 3)

    def test_duplicate_refuses_a_different_raw_book_with_same_plan(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 100.5)
        self.record(key)
        other = receipt_book.ReceiptBook(self.directory / 'empty_same_raw_plan.sqlite')
        other.initialize(self.raw.snapshot()['plan'],
                         now=self.session.opens - timedelta(minutes=30))
        with self.assertRaises(OutcomeJournalError):
            self.journal.record_first(self.features, other, key, clock=Tick(self.at))
        self.assertEqual(self.journal.snapshot()['head_seq'], 3)

    def test_copied_same_plan_books_cannot_replace_bound_source_paths(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 100.5)
        self.record(key)
        feature_copy = self.directory / 'feature_copy.sqlite'
        raw_copy = self.directory / 'raw_copy.sqlite'
        shutil.copy2(self.features.path, feature_copy)
        shutil.copy2(self.raw.path, raw_copy)
        with self.assertRaises(OutcomeJournalError):
            self.journal.record_first(FeatureJournal(feature_copy),
                                      receipt_book.ReceiptBook(raw_copy), key,
                                      clock=Tick(self.at))
        self.assertEqual(self.journal.snapshot()['head_seq'], 3)

    def test_reserved_ny_date_blocks_before_any_raw_read(self):
        blocked = datetime.fromisoformat('2026-11-10T15:00:00+00:00')
        with patch.object(self.raw, 'snapshot', side_effect=AssertionError('raw read')) as read:
            with self.assertRaises(OutcomeJournalError):
                self.journal.record_first(self.features, self.raw, 'AAPL|blocked',
                                          clock=lambda: blocked)
            self.assertEqual(read.call_count, 0)
            candidate = OutcomeJournal(self.directory / 'blocked_init.sqlite')
            with self.assertRaises(OutcomeJournalError):
                candidate.initialize(self.features, self.raw, clock=lambda: blocked)
            self.assertEqual(read.call_count, 0)
            self.assertFalse(candidate.path.exists())

    def test_creation_time_is_after_source_reads(self):
        before = self.session.opens - timedelta(minutes=30)
        after = before + timedelta(seconds=1)
        current = [before]
        original = self.raw.snapshot

        def delayed_raw():
            value = original()
            current[0] = after
            return value

        candidate = OutcomeJournal(self.directory / 'post_read_created.sqlite')
        with patch.object(self.raw, 'snapshot', delayed_raw):
            candidate.initialize(self.features, self.raw, clock=lambda: current[0])
        self.assertEqual(candidate.snapshot()['last_at'], after)

    def test_crossing_reserved_midnight_during_feature_read_blocks_next_raw_read(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 100.5)
        current = [datetime.fromisoformat('2026-11-10T04:59:59+00:00')]
        cutoff = datetime.fromisoformat('2026-11-10T05:00:00+00:00')
        original_feature = self.features.snapshot
        original_raw = self.raw._transaction
        calls = [0]
        raw_times = []

        def feature_snapshot(**kwargs):
            value = original_feature(**kwargs)
            calls[0] += 1
            if calls[0] == 2:
                current[0] = cutoff + timedelta(seconds=1)
            return value

        @contextmanager
        def raw_transaction(*args, **kwargs):
            raw_times.append(current[0])
            with original_raw(*args, **kwargs) as value:
                yield value

        with patch.object(self.features, 'snapshot', feature_snapshot), patch.object(self.raw, '_transaction', raw_transaction):
            with self.assertRaises(OutcomeJournalError):
                self.journal.record_first(self.features, self.raw, key, clock=lambda: current[0])
        self.assertGreaterEqual(calls[0], 2)
        self.assertTrue(all(at < cutoff for at in raw_times))

    def test_reserved_r1_dates_must_be_paused_before_journal_creation(self):
        now = datetime.fromisoformat('2026-11-09T00:00:00+00:00')
        tzsha = self.raw.snapshot()['plan'].timezone_sha256
        raw_plan = receipt_book.ReceiptPlan(('AAPL',), '2026-11-09', '2026-11-24', tzsha)
        raw = receipt_book.ReceiptBook(self.directory / 'reserved_raw.sqlite')
        raw.initialize(raw_plan, now=now)
        plan = PipelinePlan(('AAPL',), '2026-11-09', '2026-11-24',
                            digest(raw_plan.__dict__), receipt_book._source(), tzsha,
                            lookback_returns=2, paused_days=())
        features = FeatureJournal(self.directory / 'reserved_features.sqlite')
        features.initialize(plan, now=now)
        candidate = OutcomeJournal(self.directory / 'reserved_common.sqlite')
        with self.assertRaises(OutcomeJournalError):
            candidate.initialize(features, raw, clock=lambda: now)
        self.assertFalse(candidate.path.exists())

    def test_backward_clock_cannot_commit_unacknowledged_result(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 100.5)
        from paperbot_r2 import common_outcome
        current = [self.at + timedelta(seconds=1)]
        original = common_outcome.observe_common_outcome
        completed = []

        def regress_after_observation(*args, **kwargs):
            result = original(*args, **kwargs)
            completed.append(True)
            current[0] = self.at
            return result

        with patch.object(common_outcome, 'observe_common_outcome', regress_after_observation):
            with self.assertRaises(OutcomeJournalError):
                self.journal.record_first(self.features, self.raw, key, clock=lambda: current[0])
        self.assertEqual(completed, [True])
        self.assertEqual(self.journal.snapshot()['head_seq'], 1)


if __name__ == '__main__':
    unittest.main()
