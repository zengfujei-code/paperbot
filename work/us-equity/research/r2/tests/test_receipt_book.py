from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
import zoneinfo

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[1])]
from paperbot_us.data import Bar, Calendar, Session
from paperbot_r2.receipt_book import ReceiptPlan, ReceiptBook, ReceiptError


class ReceiptBookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_tzpath = zoneinfo.TZPATH
        zoneinfo.reset_tzpath([str(ROOT / 'outputs/zoneinfo')]);zoneinfo.ZoneInfo.clear_cache()

    @classmethod
    def tearDownClass(cls):
        zoneinfo.reset_tzpath(cls.old_tzpath);zoneinfo.ZoneInfo.clear_cache()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'receipts.sqlite'
        self.session = Session.from_alpaca(dict(date='2020-03-09', open='09:30', close='16:00'))
        self.now = self.session.opens - timedelta(minutes=30)
        self.plan = ReceiptPlan(('AAPL', 'MSFT', 'SPY'), '2020-03-09', '2020-03-09',
            sha256((ROOT / 'outputs/zoneinfo/America/New_York').read_bytes()).hexdigest())
        self.book = ReceiptBook(self.path)

    def start(self):
        self.book.initialize(self.plan, now=self.now)
        self.book.record_calendar(self.calendar(), day='2020-03-09', received_at=self.now)
        return self.book

    def calendar(self, session=None):
        return Calendar([session or self.session], start='2020-03-09', end='2020-03-09')

    def bar(self, symbol='AAPL', offset=0, delay=1, close=100., volume=100.):
        at = self.session.opens + timedelta(seconds=900 * offset)
        return Bar(symbol, at, at + timedelta(seconds=900 + delay), 100., max(100., close), min(100., close), close, volume)

    def slot(self, book=None, symbol='AAPL', offset=0):
        state = (book or self.book).snapshot()
        return state['slots'][symbol + '|' + (self.session.opens + timedelta(seconds=900 * offset)).isoformat()]

    def test_explicit_new_initialization_and_read_only_missing_snapshot(self):
        with self.assertRaises(ReceiptError):self.book.snapshot()
        self.assertFalse(self.path.exists())
        self.book.initialize(self.plan, now=self.now)
        with self.assertRaises(ReceiptError):self.book.initialize(self.plan, now=self.now)
        self.assertEqual(self.book.snapshot()['plan'], self.plan)

    def test_calendar_slots_closed_days_and_half_day(self):
        self.start();self.assertEqual(len(self.book.snapshot()['slots']), 78)
        other = ReceiptBook(Path(self.temp.name) / 'early.sqlite');other.initialize(self.plan, now=self.now)
        early = Session.from_alpaca(dict(date='2020-03-09', open='09:30', close='13:00'))
        other.record_calendar(self.calendar(early), day='2020-03-09', received_at=self.now)
        self.assertEqual(len(other.snapshot()['slots']), 42)
        closed = ReceiptBook(Path(self.temp.name) / 'closed.sqlite');closed.initialize(self.plan, now=self.now)
        closed.record_calendar(Calendar([], start='2020-03-09', end='2020-03-09'), day='2020-03-09', received_at=self.now)
        self.assertEqual(closed.snapshot()['slots'], {})

    def test_on_time_receipt_and_repeat_do_not_replace_first(self):
        self.start();bar = self.bar();now = bar.received_at + timedelta(seconds=1)
        self.book.observe([bar], now=now);before = self.slot()
        self.book.observe([replace(bar, received_at=now + timedelta(seconds=30))], now=now + timedelta(seconds=31))
        after = self.slot()
        self.assertEqual(after['first'], before['first'])
        self.assertEqual(after['first_recorded_at'], now.isoformat())
        self.assertEqual(after['revisions'], [])
        self.assertFalse(after['missed']);self.assertTrue(after['on_time'])

    def test_missing_then_late_is_sticky_after_restart(self):
        self.start();bar = self.bar(delay=130)
        missed = bar.ends + timedelta(seconds=121)
        self.book.observe([], now=missed);self.assertTrue(self.slot()['missed'])
        restarted = ReceiptBook(self.path)
        restarted.observe([bar], now=bar.received_at + timedelta(seconds=1))
        slot = self.slot(restarted)
        self.assertTrue(slot['missed']);self.assertFalse(slot['on_time'])
        self.assertEqual(slot['missed_at'], missed.isoformat())
        self.assertIsNotNone(slot['first'])

    def test_journal_delay_counts_late_even_with_earlier_provider_receipt(self):
        self.start();bar = self.bar(delay=1)
        self.book.observe([bar], now=bar.ends + timedelta(seconds=121))
        self.assertTrue(self.slot()['missed']);self.assertFalse(self.slot()['on_time'])

    def test_inclusive_deadline_boundary_and_clock_floor_without_events(self):
        self.start();bar = self.bar(delay=120)
        self.book.observe([bar], now=bar.received_at)
        self.assertTrue(self.slot()['on_time'])
        self.book.observe([], now=bar.received_at + timedelta(seconds=1))
        with self.assertRaises(ReceiptError):self.book.observe([], now=bar.received_at)

    def test_revisions_retain_first_and_deduplicate_latest_price(self):
        self.start();first = self.bar();self.book.observe([first], now=first.received_at)
        changed = self.bar(delay=20, close=101.)
        self.book.observe([changed], now=changed.received_at)
        self.book.observe([replace(changed, received_at=changed.received_at + timedelta(seconds=1))],
                          now=changed.received_at + timedelta(seconds=1))
        slot = self.slot();self.assertEqual(slot['first']['close'], 100.)
        self.assertEqual(len(slot['revisions']), 1)
        self.assertEqual(slot['revisions'][0]['bar']['close'], 101.)

    def test_changed_calendar_quarantines_without_replacing_original_slots(self):
        self.start();head = self.book.snapshot()['head_seq']
        self.book.record_calendar(self.calendar(), day='2020-03-09', received_at=self.now + timedelta(seconds=1))
        self.assertGreater(self.book.snapshot()['head_seq'], head)
        early = Session.from_alpaca(dict(date='2020-03-09', open='09:30', close='13:00'))
        self.book.record_calendar(self.calendar(early), day='2020-03-09', received_at=self.now + timedelta(seconds=2))
        state = self.book.snapshot();self.assertTrue(state['calendars']['2020-03-09']['changed'])
        self.assertEqual(len(state['slots']), 78)
        with self.assertRaises(ReceiptError):self.book.observe([self.bar()], now=self.bar().received_at)

    def test_malformed_batch_rolls_back_without_expiring_other_slots(self):
        self.start();head = self.book.snapshot()['head_seq']
        valid = self.bar();invalid = replace(valid, symbol='NVDA')
        for bars in ([valid, invalid], [valid, valid], [replace(valid, feed='sip')],
                     [replace(valid, received_at=valid.ends - timedelta(seconds=1))]):
            with self.assertRaises(ReceiptError):self.book.observe(bars, now=valid.received_at)
        self.assertEqual(self.book.snapshot()['head_seq'], head)
        self.assertIsNone(self.slot()['first'])

    def test_zero_volume_is_raw_evidence_not_a_fabricated_usable_signal(self):
        self.start();bar = self.bar(volume=0.)
        self.book.observe([bar], now=bar.received_at)
        self.assertEqual(self.slot()['first']['volume'], 0.)
        self.assertTrue(self.slot()['on_time'])

    def test_chain_head_clock_anchor_and_plan_tampering_fail(self):
        self.start()
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("UPDATE metadata SET last_at='2000-01-01T00:00:00+00:00'");db.commit()
        with self.assertRaises(ReceiptError):self.book.snapshot()

    def test_deleted_tail_is_rejected(self):
        self.start()
        with closing(sqlite3.connect(self.path)) as db:
            db.execute('DELETE FROM events WHERE seq=(SELECT MAX(seq) FROM events)');db.commit()
        with self.assertRaises(ReceiptError):self.book.snapshot()

    def test_concurrent_same_receipt_is_recorded_once(self):
        self.start();bar = self.bar()
        def run():ReceiptBook(self.path).observe([bar], now=bar.received_at)
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda _:run(), range(2)))
        self.assertEqual(self.slot()['revisions'], [])
        self.assertEqual(sum(s['first'] is not None for s in self.book.snapshot()['slots'].values()), 1)

    def test_plan_bounds_and_calendar_outside_explicit_range(self):
        for changes in (dict(symbols=('AAPL',)*2),dict(deadline_seconds=True),dict(max_records=0),
                        dict(end_day='2020-05-01'),dict(timezone_sha256='bad')):
            with self.assertRaises(ReceiptError):replace(self.plan, **changes)
        self.book.initialize(self.plan, now=self.now)
        with self.assertRaises(ReceiptError):
            self.book.record_calendar(self.calendar(), day='2020-03-10', received_at=self.now)

    def test_record_budget_is_explicit_and_failed_update_is_atomic(self):
        limited=replace(self.plan,max_records=2)
        self.book.initialize(limited,now=self.now)
        self.book.record_calendar(self.calendar(),day='2020-03-09',received_at=self.now)
        bar=self.bar()
        with self.assertRaises(ReceiptError) as failure:self.book.observe([bar],now=bar.received_at)
        self.assertEqual(str(failure.exception),'receipt_record_budget_exhausted')
        state=self.book.snapshot();self.assertEqual(state['head_seq'],2)
        self.assertTrue(all(s['first'] is None for s in state['slots'].values()))

    def test_source_binding_covers_runner_transport_and_package_initializers(self):
        self.start()
        import paperbot_r2.receipt_book as module
        from paperbot_us import data,alpaca,regime_baselines
        from paperbot_r2.contracts import digest
        r2=Path(module.__file__).parent;core=Path(data.__file__).parent
        paths={name:r2/name.split('/')[-1] for name in
               ('paperbot_r2/receipt_book.py','paperbot_r2/receipt_runner.py','paperbot_r2/contracts.py','paperbot_r2/__init__.py')}
        paths.update({name:core/name.split('/')[-1] for name in
                      ('paperbot_us/data.py','paperbot_us/alpaca.py','paperbot_us/regime_baselines.py','paperbot_us/__init__.py')})
        expected=digest({key:sha256(path.read_bytes()).hexdigest() for key,path in paths.items()})
        with closing(sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True)) as db:
            actual=db.execute('SELECT source_sha256 FROM metadata').fetchone()[0]
        self.assertEqual(actual,expected)


if __name__ == '__main__':unittest.main()
