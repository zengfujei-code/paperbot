"""Synthetic first-receipt fixtures; these are not historical market results."""
from contextlib import closing
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
import shutil
import socket
import sqlite3
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
import zoneinfo

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[1])]

from paperbot_us.data import Bar, Calendar, Session
from paperbot_us.monitor import ObservationArchive
from paperbot_r2 import receipt_book, training_data as training
from paperbot_r2.contracts import digest
from paperbot_r2.feature_pipeline import FeatureJournal, PipelinePlan, derive_once
from paperbot_r2.common_outcome_journal import OutcomeJournal

TZ = ROOT / 'outputs/zoneinfo/America/New_York'
DAYS = ('2026-10-27', '2026-10-28', '2026-10-29', '2026-10-30',
        '2026-11-02', '2026-11-03', '2026-11-04', '2026-11-05', '2026-11-06', '2026-11-09')
SYMBOLS = ('AAPL', 'MSFT', 'SPY')


class Tick:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        self.value += timedelta(microseconds=100)
        return self.value


class Fixture:
    def __init__(self, directory, *, pending=False):
        self.directory = directory
        self.sessions = tuple(Session.from_alpaca(dict(date=d, open='09:30', close='16:00')) for d in DAYS)
        self.before = self.sessions[0].opens - timedelta(minutes=30)
        raw_plan = receipt_book.ReceiptPlan(SYMBOLS, DAYS[0], '2026-11-25', sha256(TZ.read_bytes()).hexdigest())
        self.raw = receipt_book.ReceiptBook(directory / 'raw.sqlite')
        self.raw.initialize(raw_plan, now=self.before)
        calendar = Calendar(list(self.sessions), start=DAYS[0], end='2026-11-25')
        for day in DAYS:
            self.raw.record_calendar(calendar, day=day, received_at=self.before)
        paused = tuple((datetime(2026, 11, 10) + timedelta(days=i)).date().isoformat() for i in range(14))
        plan = PipelinePlan(SYMBOLS, DAYS[0], '2026-11-25', digest(asdict(raw_plan)),
                            receipt_book._source(), raw_plan.timezone_sha256, paused_days=paused)
        self.features = FeatureJournal(directory / 'features.sqlite')
        self.features.initialize(plan, now=self.before)
        self.outcomes = OutcomeJournal(directory / 'common.sqlite')
        self.outcomes.initialize(self.features, self.raw, clock=lambda: self.before)
        self.at = self.before
        for offset in range(11):
            self.receive(self.sessions[0], offset)
            if offset == 8:
                derive_once(self.features, raw_book=self.raw, clock=Tick(self.at))
        self.keys = tuple(sorted(self.features.snapshot()['entries']))
        if len(self.keys) != 3:
            raise AssertionError('synthetic feature generation did not produce three active entries')
        original = ObservationArchive.append
        def append(archive, event):
            if pending and event['kind'] == 'acknowledged':
                raise OSError('synthetic acknowledgment crash')
            return original(archive, event)
        with patch.object(ObservationArchive, 'append', append):
            outcome_clock = Tick(self.at)
            for key in self.keys:
                if pending:
                    try:
                        self.outcomes.record_first(self.features, self.raw, key, clock=outcome_clock)
                    except ValueError:
                        pass
                else:
                    self.outcomes.record_first(self.features, self.raw, key, clock=outcome_clock)
        self.at += timedelta(seconds=1)

    def bar(self, session, offset, symbol):
        multiplier = {'AAPL': 1., 'MSFT': 2., 'SPY': 4.}[symbol]
        # Offset 9 -> 10 is the common event, exactly +.005; old open-close
        # uses 105 -> 100.5 and has a different value.
        close = (100., 101., 99., 100.2, 100.1, 102., 98., 99.9, 100., 100., 100.5)[offset % 11]
        start = session.opens + timedelta(seconds=900 * offset)
        return Bar(symbol, start, start + timedelta(seconds=901),
                   105. * multiplier, 106. * multiplier, 95. * multiplier,
                   close * multiplier, 100. + offset)

    def receive(self, session, offset):
        bars = [self.bar(session, offset, symbol) for symbol in SYMBOLS]
        self.raw.observe(bars, now=bars[0].received_at + timedelta(seconds=1))
        self.at = bars[0].received_at + timedelta(seconds=2)

    def finish(self):
        # Remaining rows are deliberately recorded late. Their original first
        # prices stay visible; they cannot become on-time scoring evidence.
        rows = [self.bar(s, i, symbol) for s in self.sessions for i in range(26) for symbol in SYMBOLS]
        self.at = self.sessions[-1].closes + timedelta(seconds=3590)
        self.raw.observe(rows, now=self.at)
        self.at = self.sessions[-1].closes + timedelta(seconds=3721)

    def read(self, clock=None):
        return training.read_training_snapshot(self.raw, self.features, self.outcomes,
                                              timezone_file=TZ, clock=clock or Tick(self.at))


class TrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_tz = zoneinfo.TZPATH
        zoneinfo.reset_tzpath([str(ROOT / 'outputs/zoneinfo')]); zoneinfo.ZoneInfo.clear_cache()

    @classmethod
    def tearDownClass(cls):
        zoneinfo.reset_tzpath(cls.old_tz); zoneinfo.ZoneInfo.clear_cache()

    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start(); self.addCleanup(env.stop)
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.fixture = Fixture(self.path)
        network = patch.object(socket, 'create_connection', side_effect=AssertionError('no network'))
        network.start(); self.addCleanup(network.stop)

    def test_first_prices_common_label_and_post_read_availability_are_read_only(self):
        f = self.fixture
        before = {p.name: sha256(p.read_bytes()).hexdigest() for p in self.path.iterdir()}
        snapshot = f.read()
        self.assertFalse(snapshot.ready)
        self.assertEqual(tuple(s.symbol for s in snapshot.symbols), SYMBOLS)
        aapl = snapshot.symbols[0]
        self.assertEqual(len(aapl.returns), 10)
        self.assertEqual(aapl.returns[0].return_fraction, .01)
        self.assertTrue(all(r.available_at == snapshot.available_at for r in aapl.returns))
        self.assertGreater(snapshot.available_at, snapshot.read_started_at)
        self.assertEqual(aapl.examples[0].label, 1)
        self.assertEqual(aapl.examples[0].outcome_id,
                         f.outcomes.snapshot()['entries'][f.keys[0]]['outcome']['example']['outcome_id'])
        self.assertEqual(aapl.coverage[0].planned_returns, 25)
        self.assertEqual(aapl.coverage[0].planned_examples, 16)
        self.assertEqual(aapl.coverage[0].examples, 1)
        self.assertEqual(aapl.coverage[0].missing_examples, 15)
        self.assertEqual(before, {p.name: sha256(p.read_bytes()).hexdigest() for p in self.path.iterdir()})

    def test_late_context_is_counted_without_overnight_returns(self):
        f = self.fixture; f.finish()
        snapshot = f.read()
        self.assertTrue(snapshot.ready)
        for symbol in snapshot.symbols:
            self.assertEqual(len(symbol.returns), 250)
            self.assertEqual(len(symbol.examples), 1)
            self.assertEqual(sum(c.planned_examples for c in symbol.coverage), 160)
            self.assertEqual(symbol.coverage[1].late_returns, 25)
            self.assertTrue(all(r.starts_at.date() == r.ends_at.date() for r in symbol.returns))

    def test_revision_does_not_replace_first_training_price(self):
        f = self.fixture
        old = f.read()
        bar = replace(f.bar(f.sessions[0], 1, 'AAPL'), close=104., received_at=f.at + timedelta(seconds=2))
        f.raw.observe([bar], now=bar.received_at + timedelta(seconds=1)); f.at = bar.received_at + timedelta(seconds=2)
        new = f.read()
        self.assertEqual(tuple(r.return_fraction for r in old.symbols[0].returns),
                         tuple(r.return_fraction for r in new.symbols[0].returns))
        self.assertEqual(old.symbols[0].return_witnesses, new.symbols[0].return_witnesses)

    def test_pending_outcome_keeps_snapshot_unready(self):
        other = self.path / 'pending'; other.mkdir()
        f = Fixture(other, pending=True); f.finish()
        snapshot = f.read()
        self.assertFalse(snapshot.ready)
        self.assertEqual(sum(c.pending_examples for s in snapshot.symbols for c in s.coverage), 3)
        self.assertTrue(all(not s.examples for s in snapshot.symbols))

    def test_calendar_change_quarantines_training_rows(self):
        f = self.fixture
        changed = replace(f.sessions[0], closes=f.sessions[0].closes - timedelta(minutes=15))
        f.raw.record_calendar(Calendar([changed], start=DAYS[0], end=DAYS[0]), day=DAYS[0], received_at=f.at)
        f.at += timedelta(seconds=1)
        derive_once(f.features, raw_book=f.raw, clock=Tick(f.at))
        self.assertEqual(sum(e['quarantined'] for e in f.features.snapshot()['entries'].values()), 3)
        f.at += timedelta(seconds=1)
        snapshot = f.read()
        self.assertFalse(snapshot.ready)
        self.assertEqual(snapshot.symbols[0].coverage[0].calendar_status, 'changed')
        self.assertFalse(snapshot.symbols[0].returns)
        self.assertFalse(snapshot.symbols[0].examples)
        self.assertEqual(snapshot.symbols[0].coverage[0].quarantined_examples, 1)

    def test_copied_same_plan_source_is_rejected(self):
        f = self.fixture
        copied = self.path / 'copy.sqlite'; shutil.copyfile(f.raw.path, copied)
        with self.assertRaises(training.TrainingDataError):
            training.read_training_snapshot(receipt_book.ReceiptBook(copied), f.features, f.outcomes,
                                            timezone_file=TZ, clock=Tick(f.at))

    def test_reserved_cutoff_precedes_any_database_read(self):
        at = datetime(2026, 11, 10, 4, 55, tzinfo=timezone.utc)
        with patch('sqlite3.connect', side_effect=AssertionError('database must remain unread')) as connect:
            with self.assertRaises(training.TrainingDataError):
                self.fixture.read(clock=lambda: at)
            connect.assert_not_called()

    def test_crossing_cutoff_and_backward_clock_are_rejected(self):
        boundary = datetime(2026, 11, 10, 4, 55, tzinfo=timezone.utc)
        values = iter([boundary - timedelta(seconds=1), boundary])
        with self.assertRaises(training.TrainingDataError):
            self.fixture.read(clock=lambda: next(values))
        values = iter([self.fixture.at, self.fixture.at - timedelta(seconds=1)])
        with self.assertRaises(training.TrainingDataError):
            self.fixture.read(clock=lambda: next(values))

    def test_clock_before_source_and_changed_time_zone_are_rejected(self):
        with self.assertRaises(training.TrainingDataError):
            self.fixture.read(clock=lambda: self.fixture.before)
        wrong = self.path / 'wrong-tz'; wrong.write_bytes(b'not the pinned zone')
        with self.assertRaises(training.TrainingDataError):
            training.read_training_snapshot(self.fixture.raw, self.fixture.features, self.fixture.outcomes,
                                            timezone_file=wrong, clock=Tick(self.fixture.at))

    def test_snapshot_replacement_cannot_change_symbol_or_readiness(self):
        value = self.fixture.read()
        with self.assertRaises(training.TrainingDataError):
            replace(value, ready=True)
        with self.assertRaises(training.TrainingDataError):
            replace(value, symbols=tuple(reversed(value.symbols)))

    def test_missing_bar_is_not_bridged_and_no_overnight_row_is_fabricated(self):
        f = self.fixture
        f.receive(f.sessions[1], 0)
        f.receive(f.sessions[1], 2)
        f.receive(f.sessions[1], 3)
        value = f.read()
        self.assertEqual(value.symbols[0].coverage[1].returns, 1)
        self.assertEqual(value.symbols[0].returns[-1].starts_at,
                         f.sessions[1].opens + timedelta(seconds=2700))

    def test_concurrent_raw_change_invalidates_snapshot(self):
        f = self.fixture
        identity = training._identity
        calls = []
        def changed(path):
            calls.append(path)
            if len(calls) == 4:
                f.raw.observe([], now=f.at + timedelta(seconds=1))
            return identity(path)
        with patch.object(training, '_identity', changed):
            with self.assertRaises(training.TrainingDataError):
                f.read()

    def test_changed_adapter_source_invalidates_old_snapshot(self):
        value = self.fixture.read()
        with patch.object(training, 'source_sha256', return_value='0' * 64):
            with self.assertRaises(training.TrainingDataError):
                replace(value)

    def test_wal_raw_revision_during_read_is_detected(self):
        f = self.fixture; f.finish()
        with closing(sqlite3.connect(f.raw.path, isolation_level=None)) as keep_wal:
            self.assertEqual(keep_wal.execute('PRAGMA journal_mode=WAL').fetchone()[0], 'wal')
            keep_wal.execute('PRAGMA user_version=1')
            original = training._identity
            calls = []
            changes = []
            def changed(path):
                calls.append(path)
                if len(calls) == 4:
                    before = original(f.raw.path)
                    bar = replace(f.bar(f.sessions[0], 1, 'AAPL'), close=103.,
                                  received_at=f.at + timedelta(seconds=1))
                    f.raw.observe([bar], now=bar.received_at + timedelta(seconds=1))
                    changes.append(before == original(f.raw.path))
                return original(path)
            with patch.object(training, '_identity', changed):
                accepted = False
                try:
                    f.read()
                    accepted = True
                except training.TrainingDataError:
                    pass
            self.assertEqual(changes, [True], 'fixture must reproduce a WAL-only change')
            self.assertFalse(accepted, 'concurrent WAL revision must invalidate the snapshot')

    def test_wal_outcome_ack_during_read_is_detected(self):
        directory = self.path / 'wal-outcome'; directory.mkdir()
        f = Fixture(directory, pending=True); f.finish()
        with closing(sqlite3.connect(f.outcomes.path, isolation_level=None)) as keep_wal:
            self.assertEqual(keep_wal.execute('PRAGMA journal_mode=WAL').fetchone()[0], 'wal')
            keep_wal.execute('PRAGMA user_version=1')
            original = training._identity
            calls = []
            changes = []
            def changed(path):
                calls.append(path)
                if len(calls) == 4:
                    before = original(f.outcomes.path)
                    f.outcomes.record_first(f.features, f.raw, f.keys[0],
                                            clock=Tick(f.at + timedelta(seconds=1)))
                    changes.append(before == original(f.outcomes.path))
                return original(path)
            with patch.object(training, '_identity', changed):
                accepted = False
                try:
                    f.read()
                    accepted = True
                except training.TrainingDataError:
                    pass
            self.assertEqual(changes, [True], 'fixture must reproduce a WAL-only change')
            self.assertFalse(accepted, 'concurrent WAL acknowledgment must invalidate the snapshot')


if __name__ == '__main__':
    unittest.main()
