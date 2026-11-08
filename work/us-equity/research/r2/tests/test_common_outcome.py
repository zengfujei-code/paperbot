from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
import os
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import zoneinfo

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[1])]

from paperbot_us.data import Bar, Calendar, Session
from paperbot_r2.contracts import digest
from paperbot_r2 import receipt_book
from paperbot_r2.feature_pipeline import FeatureJournal, PipelinePlan, derive_once
from paperbot_r2.features import label_example
from paperbot_r2.common_outcome import CommonOutcomeError, observe_common_outcome


class CommonOutcomeTests(unittest.TestCase):
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
        self.now = self.session.opens - timedelta(minutes=30)
        tzsha = sha256((ROOT / 'outputs/zoneinfo/America/New_York').read_bytes()).hexdigest()
        raw_plan = receipt_book.ReceiptPlan(('AAPL',), '2020-03-09', '2020-03-09', tzsha)
        self.raw = receipt_book.ReceiptBook(self.directory / 'raw.sqlite')
        self.raw.initialize(raw_plan, now=self.now)
        self.raw.record_calendar(Calendar([self.session], start='2020-03-09', end='2020-03-09'),
                                 day='2020-03-09', received_at=self.now)
        plan = PipelinePlan(('AAPL',), '2020-03-09', '2020-03-09',
                            digest(raw_plan.__dict__), receipt_book._source(), tzsha,
                            lookback_returns=2)
        self.journal = FeatureJournal(self.directory / 'features.sqlite')
        self.journal.initialize(plan, now=self.now)
        self.at = self.now

    def bar(self, offset, close, delay=1):
        start = self.session.opens + timedelta(seconds=900 * offset)
        return Bar('AAPL', start, start + timedelta(seconds=900 + delay),
                   100., max(100., close), min(100., close), close, 100.)

    def ready_feature(self):
        for i, price in enumerate((100., 110., 99.)):
            value = self.bar(i, price)
            self.raw.observe([value], now=value.received_at + timedelta(seconds=1))
        self.at = value.ends + timedelta(seconds=3)
        result = derive_once(self.journal, raw_book=self.raw, clock=lambda: self.at)
        self.assertEqual(result['active_features'], 1)
        return next(iter(self.journal.snapshot()['entries']))

    def receive(self, offset, close, delay=1):
        value = self.bar(offset, close, delay)
        self.raw.observe([value], now=value.received_at + timedelta(seconds=1))
        self.at = value.received_at + timedelta(seconds=2)
        return value

    def observe(self, key):
        return observe_common_outcome(self.journal, self.raw, key, clock=lambda: self.at)

    def test_close_to_close_threshold_is_not_existing_open_to_close_label(self):
        key = self.ready_feature()
        self.receive(3, 99.)
        target = self.receive(4, 99.4)
        result = self.observe(key)
        self.assertEqual(result.label, 0)
        self.assertAlmostEqual(result.return_fraction, 99.4 / 99. - 1.)
        self.assertFalse(result.missing)
        self.assertFalse(result.late)
        self.assertEqual(result.target_start, target.starts)
        self.assertEqual(result.target_end, target.ends)
        entry = next(iter(self.journal.snapshot()['entries'].values()))
        from paperbot_r2.feature_pipeline import _bundle
        old = label_example(_bundle(entry['accepted_bundle']), target,
                            record_sha256=digest('old-label'), threshold_fraction=.005,
                            observed_at=self.at, availability_policy='archived_receipts')
        self.assertEqual(old.label, 1)

    def test_exact_positive_boundary_from_adjacent_closes(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 100.5)
        self.assertEqual(self.observe(key).label, 1)

    def test_exact_negative_boundary_from_adjacent_closes(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 99.5)
        self.assertEqual(self.observe(key).label, 1)

    def test_revision_cannot_replace_first_target_receipt(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        first = self.receive(4, 100.4)
        before = self.observe(key)
        changed = replace(first, close=101., high=101., received_at=self.at + timedelta(seconds=3))
        self.raw.observe([changed], now=changed.received_at + timedelta(seconds=1))
        self.at = changed.received_at + timedelta(seconds=2)
        after = self.observe(key)
        self.assertEqual(before.label, 0)
        self.assertEqual(after.label, 0)
        self.assertEqual(before.target_bar_sha256, after.target_bar_sha256)
        self.assertEqual(len(self.raw.snapshot()['slots']['AAPL|' + first.starts.isoformat()]['revisions']), 1)

    def test_missing_gap_or_target_cannot_be_scored(self):
        key = self.ready_feature()
        with self.assertRaises(CommonOutcomeError):
            self.observe(key)
        self.receive(3, 100.)
        with self.assertRaises(CommonOutcomeError):
            self.observe(key)

    def test_prepared_feature_cannot_be_scored(self):
        for i, price in enumerate((100., 110., 99.)):
            value = self.bar(i, price)
            self.raw.observe([value], now=value.received_at + timedelta(seconds=1))
        self.at = value.ends + timedelta(seconds=3)
        original = self.journal._append

        def stop_after_prepared(kind, body, at):
            result = original(kind, body, at)
            if kind == 'prepared':
                raise RuntimeError('simulated_crash')
            return result

        with patch.object(self.journal, '_append', side_effect=stop_after_prepared):
            derive_once(self.journal, raw_book=self.raw, clock=lambda: self.at)
        key = next(iter(self.journal.snapshot()['entries']))
        self.assertEqual(self.journal.snapshot()['entries'][key]['phase'], 'prepared')
        self.receive(3, 100.)
        self.receive(4, 100.5)
        with self.assertRaises(CommonOutcomeError):
            self.observe(key)

    def test_quarantined_feature_cannot_be_scored(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 100.5)
        changed = replace(self.session, closes=self.session.closes - timedelta(minutes=15))
        self.at += timedelta(seconds=1)
        self.raw.record_calendar(Calendar([changed], start='2020-03-09', end='2020-03-09'),
                                 day='2020-03-09', received_at=self.at)
        self.at += timedelta(seconds=1)
        derive_once(self.journal, raw_book=self.raw, clock=lambda: self.at)
        self.assertTrue(self.journal.snapshot()['entries'][key]['quarantined'])
        with self.assertRaises(CommonOutcomeError):
            self.observe(key)

    def test_late_first_target_keeps_missing_and_late_flags(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        target = self.bar(4, 100.6, delay=130)
        self.at = target.ends + timedelta(seconds=121)
        self.raw.observe([], now=self.at)
        self.raw.observe([target], now=target.received_at + timedelta(seconds=1))
        self.at = target.received_at + timedelta(seconds=2)
        result = self.observe(key)
        self.assertEqual(result.label, 1)
        self.assertTrue(result.missing)
        self.assertTrue(result.late)

    def test_delayed_post_read_observation_is_late_even_if_raw_receipt_was_on_time(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        target = self.receive(4, 100.6)
        self.at = target.ends + timedelta(seconds=121)
        result = self.observe(key)
        self.assertFalse(result.missing)
        self.assertTrue(result.late)

    def test_changed_calendar_refuses(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 100.5)
        self.assertEqual(self.observe(key).label, 1)
        early = Session.from_alpaca(dict(date='2020-03-09', open='09:30', close='13:00'))
        self.raw.record_calendar(Calendar([early], start='2020-03-09', end='2020-03-09'),
                                 day='2020-03-09', received_at=self.at + timedelta(seconds=1))
        self.at += timedelta(seconds=2)
        with self.assertRaises(CommonOutcomeError):
            self.observe(key)

    def test_corrupt_raw_chain_refuses(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 100.5)
        self.assertEqual(self.observe(key).label, 1)
        with closing(sqlite3.connect(self.raw.path)) as db:
            db.execute("UPDATE events SET digest=? WHERE seq=2", ('0' * 64,))
            db.commit()
        with self.assertRaises(CommonOutcomeError):
            self.observe(key)

    def test_clock_before_committed_target_cannot_backdate_observation(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        target = self.receive(4, 100.5)
        self.at = target.ends
        with self.assertRaises(CommonOutcomeError):
            self.observe(key)


if __name__ == '__main__':
    unittest.main()
