from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
import importlib
import unittest

import test_orders
import test_native_reconciler
import test_decision_journal
from paperbot_us import risk
from paperbot_us.data import DataQualityError
from paperbot_us.orders import OrderLedger, recheck_ticket
from paperbot_us.risk_baselines import RiskBaselineStore
from paperbot_us.risk_view import prepare_risk_view, RiskViewError


class StopReservationRiskTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(risk, 'StopReservation'), 'stop shares need an explicit risk contract')
        self.R = risk.StopReservation
        self.f = test_orders.OrderLedgerTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.now = self.f.now
        self.reservation = self.R('a' * 64, 'b' * 64, 'AAPL', 5, self.now)
        self.portfolio = risk.Portfolio(self.now, '9250', '10000', '10000', '10000',
            positions=(risk.Position('AAPL', 5, '150', self.now),),
            valuation_quotes=(self.f.quote,), stop_reservations=(self.reservation,))
        self.intent = risk.OrderIntent('c' * 64, 'AAPL', 'sell', 1, '150', None,
            self.now + timedelta(seconds=20))

    def evaluate(self, **changes):
        return risk.evaluate(self.intent, **{**self.f.args, 'portfolio': self.portfolio, **changes})

    def test_active_stop_blocks_a_second_sell_without_reducing_gross_exposure(self):
        result = self.evaluate()
        self.assertIn('insufficient_unreserved_shares', result.violations)
        self.assertEqual(result.gross_exposure_after_usd, Decimal('750.5'))
        self.assertFalse(result.allowed)

    def test_stop_and_simple_sell_reservations_are_added_instead_of_overwritten(self):
        pending = risk.PendingOrder('d' * 64, 'AAPL', 'sell', 1, '150')
        result = self.evaluate(portfolio=replace(self.portfolio, pending_orders=(pending,)))
        self.assertIn('inconsistent_sell_reservations', result.violations)
        self.assertIn('insufficient_unreserved_shares', result.violations)

    def test_old_or_future_stop_evidence_vetoes_even_if_shares_are_otherwise_free(self):
        for shift in (-31, 1):
            with self.subTest(shift=shift):
                old = replace(self.reservation, remaining_quantity=1, observed_at=self.now + timedelta(seconds=shift))
                result = self.evaluate(portfolio=replace(self.portfolio, stop_reservations=(old,)))
                self.assertIn('stale_stop_reservation', result.violations)

    def test_reservation_contract_refuses_duplicates_and_invalid_quantities(self):
        for changes in ({'remaining_quantity': True}, {'remaining_quantity': 0},
                        {'child_order_sha256': 'private-order-id'}, {'symbol': 'invalid'}):
            with self.subTest(changes=changes), self.assertRaises(DataQualityError):
                replace(self.reservation, **changes)
        for duplicate in (self.reservation, replace(self.reservation, parent_intent_id='e' * 64),
                          replace(self.reservation, child_order_sha256='e' * 64)):
            with self.assertRaises(DataQualityError):
                replace(self.portfolio, stop_reservations=(self.reservation, duplicate))

    def test_durable_ticket_recheck_does_not_drop_protective_share_reservations(self):
        record = self.f.ledger.admit(self.intent, **{**self.f.args, 'portfolio': self.portfolio})
        self.assertEqual(record['state'], 'rejected_by_risk')
        checked = recheck_ticket(record, now=self.now)
        self.assertIn('insufficient_unreserved_shares', checked.violations)
        self.assertEqual(len(record['context']['portfolio']['stop_reservations']), 1)

    def test_parent_identity_cannot_be_reused_as_an_unseen_intent(self):
        result = risk.evaluate(replace(self.intent, intent_id='a' * 64),
            **{**self.f.args, 'portfolio': self.portfolio})
        self.assertIn('duplicate_intent', result.violations)

    def test_decision_replay_retains_stop_reservation_and_its_sell_veto(self):
        d = test_decision_journal.DecisionJournalTests(); d.setUp(); self.addCleanup(d.doCleanups)
        reservation = self.R('b' * 64, 'c' * 64, d.symbol, 5, d.now)
        portfolio = risk.Portfolio(d.now, '9490.75', '10000', '10000', '10000',
            positions=(risk.Position(d.symbol, 5, '101.85', d.now),),
            valuation_quotes=(d.quote,), stop_reservations=(reservation,))
        view = replace(d.view, portfolio=portfolio)
        sell = risk.OrderIntent('d' * 64, d.symbol, 'sell', 1, '101.84', None, d.now + timedelta(seconds=20))
        receipt = d.append(d.frame(prepared=view, intent=sell, requested_action='sell', basis='risk_reduction'))
        replayed = d.store.replay(receipt.sequence, sources=d.sources).read()
        self.assertIn('insufficient_unreserved_shares', replayed['outcome']['risk']['violations'])
        self.assertEqual(replayed['inputs']['risk']['portfolio']['stop_reservations'][0]['remaining_quantity'], 5)


class NativeRiskPreparationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.native_risk_view'),
            'verified group/account preparation required')
        self.m = importlib.import_module('paperbot_us.native_risk_view')
        self.n = test_native_reconciler.NativeReconcilerTests(); self.n.setUp(); self.addCleanup(self.n.doCleanups)
        self.ledger = self.n.f.ledger; self.book = self.n.book; self.calendar = self.n.f.f.calendar
        self.baselines = RiskBaselineStore(self.n.f.f.path.with_name('native-risk.sqlite'))
        # Establish the same account's pristine baseline before the fixture's
        # later entry. This separate empty ledger is not substituted after fills.
        original = (self.n.now, self.n.positions, self.n.fills, deepcopy(self.n.account), self.n.visible)
        self.n.now = self.n.f.now; self.n.positions = []; self.n.fills = []; self.n.visible = []
        self.n.account.update(cash='10000', equity='10000')
        empty = OrderLedger(self.n.f.f.path.with_name('initial-empty-orders.sqlite')); empty.initialize()
        reconciler = self.n.m.NativePaperReconciler(broker=self.n.broker, ledger=empty, book=self.book,
            journal=self.n.f.f.path.with_name('initial-native-reconciliation.sqlite'),
            calendar=self.calendar, clock=lambda: self.n.now)
        initial = reconciler.refresh(); self.assertTrue(initial.ready_for_risk, initial.findings)
        self.baselines.initialize(book=self.book, cycle=initial, now=self.n.now)
        self.baselines.observe(book=self.book, cycle=initial, calendar=self.calendar, now=self.n.now)
        self.n.now, self.n.positions, self.n.fills, self.n.account, self.n.visible = original

    def prepare(self, **changes):
        cycle = changes.pop('cycle', None) or self.n.reconciler.refresh()
        quote = risk.Quote('AAPL', '150', '150.1', self.n.now, self.n.now)
        return self.m.prepare_native_risk_view(**{**dict(book=self.book, ledger=self.ledger,
            baselines=self.baselines, cycle=cycle, quotes=(quote,), symbol='AAPL',
            operator_state=risk.RiskState(pause_entries=False), calendar=self.calendar, now=self.n.now), **changes})

    def test_filled_entry_uses_visible_child_reservation_and_provider_available_shares(self):
        view = self.prepare()
        self.assertEqual(view.portfolio.positions[0].quantity, 5)
        self.assertEqual(view.portfolio.pending_orders, ())
        self.assertEqual(view.portfolio.stop_reservations[0].remaining_quantity, 5)
        sell = risk.OrderIntent('c' * 64, 'AAPL', 'sell', 1, '150', None, self.n.now + timedelta(seconds=20))
        decision = risk.evaluate(sell, portfolio=view.portfolio, quote=view.quote, state=view.state,
            limits=risk.RiskLimits(), calendar=self.calendar, now=self.n.now)
        self.assertIn('insufficient_unreserved_shares', decision.violations)
        self.assertFalse(view.public_summary()['execution_enabled'])

    def test_partial_stop_fill_reserves_only_remaining_shares_without_erasing_realized_loss(self):
        self.n.f.child.update(status='partially_filled', filled_qty='2', filled_avg_price='130',
            updated_at=(self.n.f.now + timedelta(seconds=4)).isoformat())
        self.n.visible = [deepcopy(self.n.f.child)]
        self.n.fills.append(self.n.fill('00000000-0000-4000-8000-000000000011', 'sell', '2', '130', self.n.f.child_id, 4, '2', '3'))
        self.n.positions[0].update(qty='3', qty_available='0', market_value='450')
        self.n.account.update(cash='9510', equity='9960')
        view = self.prepare()
        self.assertEqual(view.portfolio.stop_reservations[0].remaining_quantity, 3)
        self.assertEqual(view.portfolio.positions[0].quantity, 3)
        self.assertEqual(self.book.snapshot().receipt.result.realized_usd, Decimal('-40'))

    def test_held_child_of_unfilled_parent_is_not_a_phantom_sell_reservation(self):
        self.n.f.parent.update(status='new', filled_qty='0', filled_avg_price=None, filled_at=None)
        self.n.f.child['status'] = 'held'; self.n.visible = [{**self.n.f.parent, 'legs': None}]
        self.n.positions = []; self.n.fills = []; self.n.account.update(cash='10000', equity='10000')
        view = self.prepare()
        self.assertEqual(view.portfolio.stop_reservations, ())
        self.assertEqual(view.portfolio.pending_orders[0].remaining_quantity, 5)
        self.assertEqual(view.portfolio.pending_orders[0].side, 'buy')

    def test_fully_closed_group_releases_reservation_only_after_actual_fill_accounting(self):
        self.n.f.child.update(status='filled', filled_qty='5', filled_avg_price='130',
            updated_at=(self.n.f.now + timedelta(seconds=4)).isoformat(),
            filled_at=(self.n.f.now + timedelta(seconds=4)).isoformat())
        self.n.visible = []; self.n.positions = []; self.n.account.update(cash='9900', equity='9900')
        self.n.fills.append(self.n.fill('00000000-0000-4000-8000-000000000011', 'sell', '5', '130', self.n.f.child_id, 4, '5', '0'))
        view = self.prepare()
        self.assertEqual(view.portfolio.stop_reservations, ())
        self.assertEqual(view.portfolio.positions, ())
        self.assertEqual(view.portfolio.equity_usd, 9900)

    def test_unprotected_group_cannot_be_upgraded_by_forging_ready_flag(self):
        self.n.f.child['status'] = 'held'; self.n.visible = []; self.n.positions[0]['qty_available'] = '5'
        cycle = self.n.reconciler.refresh(); self.assertFalse(cycle.ready_for_risk)
        before = self.baselines.verify()
        forged = replace(cycle, ready_for_risk=True, status='reconciled', findings=())
        with self.assertRaises(RiskViewError): self.prepare(cycle=forged)
        self.assertEqual(self.baselines.verify(), before)

    def test_stale_group_or_changed_ledger_cannot_produce_current_risk_view(self):
        cycle = self.n.reconciler.refresh()
        self.n.now += timedelta(seconds=31)
        with self.assertRaises(RiskViewError): self.prepare(cycle=cycle)

    def test_missing_or_mismatched_available_share_proof_vetoes_preparation(self):
        cycle = self.n.reconciler.refresh()
        wrong = replace(cycle, positions_available=tuple(replace(p, available_quantity=5) for p in cycle.positions_available))
        with self.assertRaises(RiskViewError): self.prepare(cycle=wrong)

    def test_legacy_view_and_admission_still_require_group_coordinator(self):
        cycle = self.n.reconciler.refresh()
        view = self.prepare(cycle=cycle)
        args = dict(book=self.book, ledger=self.ledger, baselines=self.baselines, cycle=cycle,
            quotes=view.portfolio.valuation_quotes, symbol='AAPL', operator_state=view.state,
            calendar=self.calendar, now=self.n.now)
        with self.assertRaises(RiskViewError): prepare_risk_view(**args)
        intent = risk.OrderIntent('d' * 64, 'AAPL', 'buy', 1, '150.1', '145', self.n.now + timedelta(seconds=20))
        record = self.ledger.admit(intent, portfolio=view.portfolio, quote=view.quote, state=view.state,
            limits=risk.RiskLimits(), calendar=self.calendar, now=self.n.now)
        self.assertEqual(record['state'], 'rejected_by_risk')
        self.assertIn('native_group_coordinator_required', record['context']['reconciliation_findings'])


if __name__ == '__main__':
    unittest.main()
