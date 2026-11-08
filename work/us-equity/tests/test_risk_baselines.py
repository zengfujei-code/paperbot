from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import importlib
import importlib.util
import sqlite3
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import test_accounting

from paperbot_us.accountbook import AccountBook
from paperbot_us.accounting import ActivityWindow, AccountMark, AccountSnapshot
from paperbot_us.data import Calendar, Session
from paperbot_us.reconciler import ReconciliationCycle


class RiskBaselineTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.risk_baselines'),
                             'durable session risk baselines required')
        self.m = importlib.import_module('paperbot_us.risk_baselines')
        self.f = test_accounting.AccountingTests()
        self.f.setUp()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.book = AccountBook(self.root / 'book.sqlite')
        self.book.initialize(self.f.anchor)
        self.now = self.f.start + timedelta(minutes=1)
        self.receipt = self.book.refresh(
            window=ActivityWindow(self.f.start, self.now, self.now, ()), orders=(),
            account=AccountSnapshot(self.now, '10000', '10000', False, (), self.f.anchor.account_sha256),
            now=self.now)
        self.cycle = self.make_cycle(self.receipt)
        sessions = [Session.from_alpaca(dict(date=day, open='09:30', close='16:00'))
                    for day in ('2026-10-22', '2026-10-23', '2026-10-26', '2026-10-27')]
        self.calendar = Calendar(sessions, start='2026-10-22', end='2026-10-27')
        self.path = self.root / 'risk.sqlite'
        self.store = self.m.RiskBaselineStore(self.path)
        self.store.initialize(book=self.book, cycle=self.cycle, now=self.now)

    def make_cycle(self, receipt):
        inventory = receipt.result.inventory
        return ReconciliationCycle(receipt.as_of, 'reconciled', True, (), receipt,
                                   'b' * 64, 'c' * 64, 2 if inventory else 0, len(inventory))

    def observe(self, **changes):
        return self.store.observe(book=self.book, cycle=changes.get('cycle', self.cycle),
                                  calendar=changes.get('calendar', self.calendar),
                                  now=changes.get('now', self.now))

    def marked(self, at, price='110'):
        self.now = datetime.fromisoformat(at) if isinstance(at, str) else at
        price = Decimal(price)
        args = self.f.inputs()
        mark = AccountMark('AAPL', 3, price, price * 3, self.now - timedelta(seconds=1))
        args['window'] = replace(args['window'], until=self.now - timedelta(seconds=3),
                                 received_at=self.now - timedelta(seconds=2))
        args['account'] = AccountSnapshot(self.now, '9719.98', Decimal('9719.98') + price * 3,
                                          False, (mark,), self.f.anchor.account_sha256)
        self.receipt = self.book.refresh(**{**{k: v for k, v in args.items() if k != 'anchor'}, 'now': self.now})
        self.assertTrue(self.receipt.result.reconciled, self.receipt.result.violations)
        self.cycle = self.make_cycle(self.receipt)
        return self.receipt

    def test_pristine_account_reference_and_repeated_receipt_do_not_start_trading(self):
        value = self.observe()
        self.assertTrue(value.entries_ready)
        self.assertEqual(value.day_start_equity_usd, 10000)
        self.assertEqual(value.high_water_equity_usd, 10000)
        self.assertEqual(value.reference_kind, 'pristine_cash')
        before = self.store.verify()
        self.assertEqual(self.observe(), value)
        self.assertEqual(self.store.verify(), before)
        public = value.public_summary()
        self.assertFalse(public['execution_enabled'])
        self.assertNotIn(self.f.anchor.account_sha256, str(public))

    def test_high_water_survives_losses_restart_and_catch_up_of_intermediate_peaks(self):
        self.observe()
        self.marked('2026-10-23T15:00:00+00:00', '110')
        # Deliberately skip updating the risk store at the higher book snapshot.
        self.marked('2026-10-23T15:01:00+00:00', '90')
        value = self.observe()
        self.assertEqual(value.high_water_equity_usd, Decimal('10049.98'))
        self.assertEqual(value.day_start_equity_usd, 10000)
        self.store = self.m.RiskBaselineStore(self.path)
        self.assertEqual(self.store.replay(), value)
        self.marked('2026-10-23T15:02:00+00:00', '80')
        self.assertEqual(self.observe().high_water_equity_usd, Decimal('10049.98'))

    def test_previous_close_reference_includes_an_overnight_gap_without_resetting_peak(self):
        self.observe()
        self.marked('2026-10-23T19:59:40+00:00', '110')
        self.observe()
        self.marked('2026-10-26T13:31:00+00:00', '75')
        value = self.observe()
        self.assertTrue(value.entries_ready, value.findings)
        self.assertEqual(value.session_day, '2026-10-26')
        self.assertEqual(value.reference_kind, 'previous_observed_close')
        self.assertEqual(value.day_start_equity_usd, Decimal('10049.98'))
        self.assertEqual(value.high_water_equity_usd, Decimal('10049.98'))
        self.assertGreater(value.day_start_equity_usd - self.receipt.result.equity_usd, 100)
        from paperbot_us.risk import Portfolio, Position, OrderIntent, Quote, RiskState, RiskLimits, evaluate
        account = self.book.snapshot().account
        portfolio = Portfolio(self.now, account.cash_usd, account.equity_usd,
                              value.day_start_equity_usd, value.high_water_equity_usd,
                              positions=(Position('AAPL', 3, '75', self.now),))
        intent = OrderIntent('9' * 64, 'AAPL', 'buy', 1, '75.01', '74', self.now + timedelta(seconds=20))
        decision = evaluate(intent, portfolio=portfolio, quote=Quote('AAPL', '75', '75.01', self.now, self.now),
                            state=RiskState(pause_entries=False), limits=RiskLimits(), calendar=self.calendar, now=self.now)
        self.assertIn('daily_loss_limit', decision.violations)

    def test_missing_prior_close_pauses_entries_instead_of_using_morning_equity(self):
        self.observe()
        self.marked('2026-10-23T15:00:00+00:00')
        self.observe()
        self.marked('2026-10-26T13:31:00+00:00', '75')
        value = self.observe()
        self.assertFalse(value.entries_ready)
        self.assertIn('previous_close_reference_missing', value.findings)
        self.assertEqual(value.day_start_equity_usd, 10000)
        self.assertEqual(value.high_water_equity_usd, Decimal('10049.98'))

    def test_latest_valid_near_close_observation_wins_and_day_reference_is_frozen(self):
        self.observe()
        self.marked('2026-10-23T19:59:35+00:00', '100')
        self.observe()
        self.marked('2026-10-23T19:59:55+00:00', '110')
        self.observe()
        self.marked('2026-10-26T13:31:00+00:00', '90')
        self.assertEqual(self.observe().day_start_equity_usd, Decimal('10049.98'))
        self.marked('2026-10-26T14:30:00+00:00', '100')
        self.assertEqual(self.observe().day_start_equity_usd, Decimal('10049.98'))

    def revised_friday_calendar(self):
        sessions = [Session.from_alpaca(dict(date=day, open='09:30', close='13:00' if day == '2026-10-23' else '16:00'))
                    for day in ('2026-10-22', '2026-10-23', '2026-10-26', '2026-10-27')]
        return Calendar(sessions, start='2026-10-22', end='2026-10-27')

    def test_revised_previous_close_cannot_promote_an_old_checkpoint(self):
        self.observe()
        self.marked('2026-10-23T19:59:40+00:00')
        self.observe()
        self.marked('2026-10-26T13:31:00+00:00', '75')
        value = self.observe(calendar=self.revised_friday_calendar())
        self.assertFalse(value.entries_ready)
        self.assertIn('calendar_reference_conflict', value.findings)
        self.assertEqual(value.high_water_equity_usd, Decimal('10049.98'))

    def test_revising_a_frozen_reference_vetoes_same_receipt_and_is_latched_for_the_day(self):
        self.observe()
        self.marked('2026-10-23T19:59:40+00:00')
        self.observe()
        self.marked('2026-10-26T13:31:00+00:00', '75')
        ready = self.observe()
        self.assertTrue(ready.entries_ready)
        revised = self.observe(calendar=self.revised_friday_calendar())
        self.assertFalse(revised.entries_ready)
        self.assertIn('calendar_reference_conflict', revised.findings)
        self.assertEqual(revised.day_start_equity_usd, ready.day_start_equity_usd)
        self.assertFalse(self.observe().entries_ready, 'a same-day calendar flip must not erase the conflict')

    def test_disputed_checkpoint_cannot_be_revalidated_by_replaying_the_original_calendar(self):
        self.observe()
        self.marked('2026-10-23T19:59:40+00:00')
        self.observe()
        self.observe(calendar=self.revised_friday_calendar())
        self.observe()
        self.marked('2026-10-26T13:31:00+00:00', '75')
        self.assertFalse(self.observe().entries_ready)
        self.marked('2026-10-26T19:59:50+00:00', '100')
        self.assertFalse(self.observe().entries_ready)
        self.marked('2026-10-27T13:31:00+00:00', '90')
        next_day = self.observe()
        self.assertTrue(next_day.entries_ready)
        self.assertEqual(next_day.day_start_equity_usd, Decimal('10019.98'))
        self.assertEqual(next_day.high_water_equity_usd, Decimal('10049.98'))

    def test_early_close_and_standard_time_use_supplied_session_boundaries(self):
        self.observe()
        sessions = [Session.from_alpaca(dict(date='2026-11-27', open='09:30', close='13:00')),
                    Session.from_alpaca(dict(date='2026-11-30', open='09:30', close='16:00'))]
        self.calendar = Calendar(sessions, start='2026-11-26', end='2026-11-30')
        self.marked('2026-11-27T17:59:50+00:00')
        self.observe()
        self.marked('2026-11-30T14:31:00+00:00', '90')
        value = self.observe()
        self.assertTrue(value.entries_ready)
        self.assertEqual(value.day_start_equity_usd, Decimal('10049.98'))

    def test_forged_unready_stale_or_regressed_receipts_do_not_change_risk_history(self):
        self.observe()
        before = self.store.verify()
        forged = replace(self.cycle.book_receipt, digest='d' * 64)
        cases = [replace(self.cycle, book_receipt=forged), replace(self.cycle, ready_for_risk=False),
                 replace(self.cycle, findings=('unrecognized_broker_order',))]
        for cycle in cases:
            with self.subTest(cycle=cycle.status), self.assertRaises(self.m.BaselineError):
                self.observe(cycle=cycle)
        for at in (self.now + timedelta(seconds=31), self.now - timedelta(seconds=1)):
            with self.subTest(at=at), self.assertRaises(self.m.BaselineError):
                self.observe(now=at)
        self.assertEqual(self.store.verify(), before)

    def test_risk_state_is_not_reinitialized_and_nonpristine_bootstrap_is_rejected(self):
        with self.assertRaises(self.m.BaselineError):
            self.store.initialize(book=self.book, cycle=self.cycle, now=self.now)
        self.marked('2026-10-23T15:00:00+00:00')
        other = self.m.RiskBaselineStore(self.root / 'other.sqlite')
        with self.assertRaises(self.m.BaselineError):
            other.initialize(book=self.book, cycle=self.cycle, now=self.now)
        self.assertFalse(other.path.exists())

    def test_account_binding_and_current_snapshot_must_match_the_recorded_risk_book(self):
        previous = self.observe()
        other = AccountBook(self.root / 'changed-account.sqlite')
        anchor = replace(self.f.anchor, account_sha256='7' * 64)
        other.initialize(anchor)
        receipt = other.refresh(window=ActivityWindow(anchor.as_of, self.now, self.now, ()), orders=(),
                                account=AccountSnapshot(self.now, '10000', '10000', False, (), anchor.account_sha256),
                                now=self.now)
        with self.assertRaises(self.m.BaselineError):
            self.store.observe(book=other, cycle=self.make_cycle(receipt), calendar=self.calendar, now=self.now)
        self.marked('2026-10-23T15:00:00+00:00')
        with self.assertRaises(self.m.BaselineError):
            self.observe(cycle=replace(self.cycle, positions_count=True))
        self.assertEqual(self.store.replay(), previous)

    def test_missing_calendar_disk_failure_and_source_version_mismatch_never_reduce_the_peak(self):
        previous = self.observe()
        self.marked('2026-10-23T15:00:00+00:00')
        with patch.object(self.m.ObservationArchive, 'append', side_effect=OSError('PRIVATE_DISK')):
            with self.assertRaises(self.m.BaselineError) as caught:
                self.observe()
        self.assertNotIn('PRIVATE', str(caught.exception))
        self.assertEqual(self.store.replay(), previous)
        outside = Calendar([], start='2026-10-26', end='2026-10-26')
        with self.assertRaises(self.m.BaselineError):
            self.observe(calendar=outside)
        with patch.object(self.m, '_version', return_value='f' * 64):
            with self.assertRaises(self.m.BaselineError):
                self.store.replay()
        self.assertEqual(self.observe().high_water_equity_usd, Decimal('10049.98'))

    def test_corrupt_or_missing_history_is_not_a_new_normal_baseline(self):
        self.observe()
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("UPDATE observations SET payload='{}' WHERE seq=1")
            db.commit()
        with self.assertRaises(self.m.BaselineError):
            self.store.replay()
        absent = self.m.RiskBaselineStore(self.root / 'missing.sqlite')
        with self.assertRaises(self.m.BaselineError):
            absent.observe(book=self.book, cycle=self.cycle, calendar=self.calendar, now=self.now)
        self.assertFalse(absent.path.exists())


if __name__ == '__main__':
    unittest.main()
