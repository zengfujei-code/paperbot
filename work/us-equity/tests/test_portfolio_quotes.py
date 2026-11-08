from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from paperbot_us.data import Calendar, Session
from paperbot_us.orders import OrderLedger, recheck_ticket
from paperbot_us.risk import Portfolio, Position, PendingOrder, Quote, OrderIntent, RiskState, RiskLimits, evaluate


class PortfolioQuoteTests(unittest.TestCase):
    def setUp(self):
        self.assertIn('valuation_quotes', Portfolio.__dataclass_fields__,
                      'risk checks need fresh quotes for the whole held portfolio')
        self.now = datetime(2026, 10, 23, 15, tzinfo=timezone.utc)
        self.calendar = Calendar([Session.from_alpaca(dict(date='2026-10-23', open='09:30', close='16:00'))],
                                 start='2026-10-23', end='2026-10-23')
        self.quote = Quote('AAPL', '150', '150.1', self.now, self.now)
        self.intent = OrderIntent('7' * 64, 'AAPL', 'buy', 1, '150.1', '145', self.now + timedelta(seconds=20))
        self.portfolio = Portfolio(self.now, '9500', '10000', '10000', '10000',
                                   positions=(Position('MSFT', 5, '100', self.now),))
        self.state = RiskState(pause_entries=False)

    def assess(self, portfolio=None):
        return evaluate(self.intent, portfolio=portfolio or self.portfolio, quote=self.quote,
                        state=self.state, limits=RiskLimits(), calendar=self.calendar, now=self.now)

    def test_other_held_symbol_cannot_hide_a_drawdown_behind_an_accounting_mark(self):
        quote = Quote('MSFT', '75', '75.01', self.now, self.now)
        result = self.assess(replace(self.portfolio, valuation_quotes=(quote,)))
        self.assertIn('daily_loss_limit', result.violations)
        self.assertFalse(result.allowed)

    def test_missing_stale_future_and_duplicate_portfolio_quotes_are_refused(self):
        self.assertIn('portfolio_quotes_missing', self.assess().violations)
        for at in (self.now - timedelta(seconds=31), self.now + timedelta(seconds=1)):
            quote = Quote('MSFT', '100', '100.01', at, at)
            result = self.assess(replace(self.portfolio, valuation_quotes=(quote,)))
            self.assertIn('stale_portfolio_quote', result.violations)
        quote = Quote('MSFT', '100', '100.01', self.now, self.now)
        with self.assertRaises(ValueError):
            replace(self.portfolio, valuation_quotes=(quote, quote))

    def test_other_held_and_pending_prices_count_toward_gross_exposure(self):
        held = Position('MSFT', 10, '200', self.now)
        quote = Quote('MSFT', '249.9', '250', self.now, self.now)
        account = replace(self.portfolio, cash_usd=Decimal('8000'), positions=(held,), valuation_quotes=(quote,))
        result = self.assess(account)
        self.assertIn('portfolio_exposure_limit', result.violations)
        self.assertEqual(result.gross_exposure_after_usd, Decimal('2650.1'))
        account = replace(self.portfolio, cash_usd=Decimal('10000'), positions=(),
                          pending_orders=(PendingOrder('8' * 64, 'MSFT', 'buy', 10, '200'),), valuation_quotes=(quote,))
        self.assertIn('portfolio_exposure_limit', self.assess(account).violations)

    def test_an_already_overweight_other_symbol_blocks_additional_entries(self):
        quote = Quote('MSFT', '299.9', '300', self.now, self.now)
        result = self.assess(replace(self.portfolio, valuation_quotes=(quote,)))
        self.assertLess(result.gross_exposure_after_usd, 2500)
        self.assertIn('existing_symbol_exposure_limit', result.violations)

    def test_new_entry_also_uses_the_more_conservative_same_symbol_quote(self):
        competing = Quote('AAPL', '200', '200.01', self.now, self.now)
        account = Portfolio(self.now, '10000', '10000', '10000', '10000', valuation_quotes=(competing,))
        intent = replace(self.intent, quantity=6)
        inputs = dict(portfolio=account, quote=self.quote, state=self.state,
                      limits=RiskLimits(), calendar=self.calendar, now=self.now)
        result = evaluate(intent, **inputs)
        self.assertFalse(result.allowed)
        self.assertEqual(result.gross_exposure_after_usd, Decimal('1200.06'))
        self.assertIn('symbol_exposure_limit', result.violations)
        with tempfile.TemporaryDirectory() as folder:
            ledger = OrderLedger(Path(folder) / 'orders.sqlite')
            ledger.initialize()
            record = ledger.admit(intent, **inputs)
            self.assertEqual(record['state'], 'rejected_by_risk')
            self.assertIsNone(ledger.prepare_submission(intent.intent_id, **inputs))
            final = recheck_ticket(record, now=self.now)
            self.assertFalse(final.allowed)
            self.assertEqual(final.gross_exposure_after_usd, Decimal('1200.06'))

    def test_recent_preopen_quotes_are_not_regular_session_execution_inputs(self):
        opened=self.now.replace(hour=13,minute=30,second=5)
        preopen=opened-timedelta(seconds=10)
        candidate=Quote('AAPL','150','150.1',preopen,opened)
        held=Quote('MSFT','100','100.01',preopen,opened)
        account=replace(self.portfolio,as_of=opened,positions=(Position('MSFT',5,'100',opened),),
                        valuation_quotes=(held,))
        intent=replace(self.intent,expires_at=opened+timedelta(seconds=20))
        result=evaluate(intent,portfolio=account,quote=candidate,state=self.state,limits=RiskLimits(),
                        calendar=self.calendar,now=opened)
        self.assertFalse(result.allowed)
        self.assertIn('off_session_quote',result.violations)
        self.assertIn('off_session_portfolio_quote',result.violations)

    def test_portfolio_quotes_are_durable_and_rechecked_at_the_final_network_boundary(self):
        held_quote = Quote('MSFT', '100', '100.01', self.now - timedelta(seconds=29), self.now)
        account = replace(self.portfolio, valuation_quotes=(held_quote,))
        self.assertTrue(self.assess(account).allowed)
        with tempfile.TemporaryDirectory() as folder:
            ledger = OrderLedger(Path(folder) / 'orders.sqlite')
            ledger.initialize()
            inputs = dict(portfolio=account, quote=self.quote, state=self.state,
                          limits=RiskLimits(), calendar=self.calendar, now=self.now)
            record = ledger.admit(self.intent, **inputs)
            self.assertEqual(record['state'], 'reserved')
            ticket = ledger.prepare_submission(self.intent.intent_id, **inputs)
            self.assertIsNotNone(ticket)
            self.assertEqual(len(ticket['context']['portfolio']['valuation_quotes']), 1)
            result = recheck_ticket(ticket, now=self.now + timedelta(seconds=2))
            self.assertIn('stale_portfolio_quote', result.violations)
            self.assertFalse(result.allowed)

    def test_expired_held_quote_after_commit_prevents_the_mock_broker_post(self):
        from paperbot_us.broker import PaperBroker
        held_quote = Quote('MSFT', '100', '100.01', self.now - timedelta(seconds=29), self.now)
        account = replace(self.portfolio, valuation_quotes=(held_quote,))
        calls = []
        times = iter((self.now, self.now, self.now + timedelta(seconds=2)))
        broker = PaperBroker('PK' + 'T' * 20, 'T' * 32, clock=lambda: next(times, self.now + timedelta(seconds=2)),
                             transport=lambda *args: calls.append(args))
        with tempfile.TemporaryDirectory() as folder:
            ledger = OrderLedger(Path(folder) / 'orders.sqlite')
            ledger.initialize()
            inputs = dict(portfolio=account, quote=self.quote, state=self.state,
                          limits=RiskLimits(), calendar=self.calendar, now=self.now)
            ledger.admit(self.intent, **inputs)
            from paperbot_us.controls import ControlJournal
            controls=ControlJournal(Path(folder)/'controls.sqlite');controls.initialize(now=self.now)
            snapshot=controls.update(expected_revision=1,pause_entries=False,halt_orders=False,reason='operator_resume',now=self.now)
            self.assertIsNone(broker.submit(ledger, self.intent.intent_id, controls=controls, control_snapshot=snapshot, **inputs))
            self.assertEqual(calls, [])
            record = ledger.load(self.intent.intent_id)
            self.assertEqual(record['state'], 'rejected_by_risk')
            self.assertIn('stale_portfolio_quote', record['risk']['violations'])


if __name__ == '__main__':
    unittest.main()
