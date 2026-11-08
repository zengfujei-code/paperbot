from dataclasses import replace
from datetime import timedelta
import importlib
import importlib.util
import unittest
import test_reconciler

from paperbot_us.orders import OrderLedger
from paperbot_us.reconciler import PaperReconciler
from paperbot_us.risk import Portfolio, Quote, RiskState, RiskLimits, OrderIntent, evaluate
from paperbot_us.risk_baselines import RiskBaselineStore


class RiskViewTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.risk_view'),
                             'verified portfolio preparation required')
        self.m = importlib.import_module('paperbot_us.risk_view')
        self.f = test_reconciler.ReconcilerTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.order_template = dict(self.f.order)
        self.f.ledger = OrderLedger(self.f.root / 'view-orders.sqlite')
        self.f.ledger.initialize()
        self.f.order = None
        self.f.fills = []
        self.f.positions = []
        self.f.account.update(cash='10000', equity='10000', buying_power='40000')
        self.f.coordinator = PaperReconciler(broker=self.f.broker, ledger=self.f.ledger,
                                             book=self.f.book, journal=self.f.root / 'view-reconciliation.sqlite',
                                             clock=lambda: self.f.now)
        from paperbot_us.data import Calendar, Session
        self.calendar = Calendar([Session.from_alpaca(dict(date=day, open='09:30', close='16:00'))
                                   for day in ('2026-10-22', '2026-10-23', '2026-10-26')],
                                  start='2026-10-22', end='2026-10-26')
        self.cycle = self.f.coordinator.refresh()
        self.assertTrue(self.cycle.ready_for_risk)
        self.baselines = RiskBaselineStore(self.f.root / 'view-risk.sqlite')
        self.baselines.initialize(book=self.f.book, cycle=self.cycle, now=self.f.now)
        self.baselines.observe(book=self.f.book, cycle=self.cycle, calendar=self.calendar, now=self.f.now)
        self.quote = Quote('AAPL', '100', '100.01', self.f.now, self.f.now)

    def prepare(self, **changes):
        return self.m.prepare_risk_view(book=self.f.book, ledger=self.f.ledger, baselines=self.baselines,
                                        cycle=changes.get('cycle', self.cycle), quotes=changes.get('quotes', (self.quote,)),
                                        symbol='AAPL', operator_state=changes.get('state', RiskState(pause_entries=False)),
                                        calendar=self.calendar, now=self.f.now)

    def risk_inputs(self, portfolio=None):
        return dict(portfolio=portfolio or Portfolio(self.f.now, '10000', '10000', '10000', '10000'),
                    quote=self.quote, state=RiskState(pause_entries=False), limits=RiskLimits(),
                    calendar=self.calendar, now=self.f.now)

    def reserve(self):
        self.f.ledger.admit(self.f.intent, **self.risk_inputs())

    def fill(self):
        self.reserve()
        self.f.ledger.prepare_submission(self.f.intent.intent_id, **self.risk_inputs())
        created, filled = self.f.now + timedelta(seconds=1), self.f.now + timedelta(seconds=2)
        self.f.order = {**self.order_template, 'created_at': created.isoformat(), 'updated_at': filled.isoformat()}
        self.f.fills = [dict(id='20261023150002000::00000000-0000-4000-8000-000000000010',
                             activity_type='FILL', symbol='AAPL', side='buy', qty='2', price='100',
                             cum_qty='2', leaves_qty='0', order_id=self.f.uuid, type='fill',
                             transaction_time=filled.isoformat())]
        self.f.positions = [dict(asset_class='us_equity', symbol='AAPL', side='long', qty='2', qty_available='2',
                                 current_price='100', market_value='200')]
        self.f.account.update(cash='9800', equity='10000')
        self.f.now += timedelta(seconds=3)
        self.quote = Quote('AAPL', '100', '100.01', self.f.now, self.f.now)
        self.cycle = self.f.coordinator.refresh()
        self.assertTrue(self.cycle.ready_for_risk, self.cycle.findings)

    def test_empty_verified_view_uses_cash_and_records_binding_not_margin_buying_power(self):
        view = self.prepare()
        self.assertEqual(view.portfolio.cash_usd, 10000)
        self.assertEqual(view.portfolio.equity_usd, 10000)
        self.assertEqual(view.quote, self.quote)
        self.assertFalse(view.state.pause_entries)
        self.assertFalse(view.public_summary()['execution_enabled'])
        self.assertEqual(view.book_digest, self.cycle.book_receipt.digest)
        self.assertNotIn(self.f.account['id'], str(view.public_summary()))

    def test_realistic_mock_fill_translates_into_marks_and_independent_fresh_quote_loss_veto(self):
        self.fill()
        self.quote = Quote('AAPL', '50', '50.01', self.f.now, self.f.now)
        view = self.prepare()
        self.assertEqual(view.portfolio.positions[0].quantity, 2)
        self.assertEqual(view.portfolio.positions[0].mark_price_usd, 100)
        self.assertEqual(view.portfolio.valuation_quotes, (self.quote,))
        intent = OrderIntent('9' * 64, 'AAPL', 'buy', 1, '50.01', '49', self.f.now + timedelta(seconds=20))
        result = evaluate(intent, portfolio=view.portfolio, quote=view.quote, state=view.state,
                          limits=RiskLimits(), calendar=self.calendar, now=self.f.now)
        self.assertIn('daily_loss_limit', result.violations)

    def test_reserved_intent_stays_local_and_does_not_masquerade_as_an_existing_broker_order(self):
        self.reserve()
        self.cycle = self.f.coordinator.refresh()
        view = self.prepare()
        self.assertEqual(view.portfolio.pending_orders, ())
        ticket = self.f.ledger.prepare_submission(self.f.intent.intent_id,
                                                   **{**self.risk_inputs(portfolio=view.portfolio), 'state': view.state})
        self.assertIsNotNone(ticket)
        self.assertEqual(ticket['state'], 'submission_unknown')

    def test_changed_ledger_unready_cycle_or_missing_stale_and_duplicate_quotes_cannot_prepare(self):
        for quotes in ((), (replace(self.quote, as_of=self.f.now - timedelta(seconds=31)),), (self.quote, self.quote)):
            with self.subTest(quotes=len(quotes)), self.assertRaises(self.m.RiskViewError):
                self.prepare(quotes=quotes)
        with self.assertRaises(self.m.RiskViewError):
            self.prepare(cycle=replace(self.cycle, ready_for_risk=False))
        self.reserve()
        with self.assertRaises(self.m.RiskViewError):
            self.prepare()

    def test_operator_pause_and_halt_are_preserved_and_history_ids_are_not_forgotten(self):
        self.fill()
        paused = self.prepare(state=RiskState(pause_entries=True))
        self.assertTrue(paused.state.pause_entries)
        self.assertIn(self.f.intent.intent_id, paused.state.seen_intent_ids)
        halted = self.prepare(state=RiskState(pause_entries=False, halt_orders=True))
        self.assertTrue(halted.state.halt_orders)

    def test_missing_daily_reference_pauses_buys_but_keeps_a_verified_reducing_exit_possible(self):
        self.fill()
        self.prepare()
        self.f.now = self.f.now.replace(day=26)
        self.quote = Quote('AAPL', '100', '100.01', self.f.now, self.f.now)
        self.cycle = self.f.coordinator.refresh()
        self.assertTrue(self.cycle.ready_for_risk)
        view = self.prepare()
        self.assertTrue(view.state.pause_entries)
        self.assertIn('previous_close_reference_missing', view.findings)
        sell = OrderIntent('9' * 64, 'AAPL', 'sell', 1, '100', None, self.f.now + timedelta(seconds=20))
        result = evaluate(sell, portfolio=view.portfolio, quote=view.quote, state=view.state,
                          limits=RiskLimits(), calendar=self.calendar, now=self.f.now)
        self.assertTrue(result.allowed, result.violations)


if __name__ == '__main__':
    unittest.main()
