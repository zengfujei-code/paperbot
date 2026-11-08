from dataclasses import replace
from datetime import datetime,timedelta,timezone
from decimal import Decimal
import unittest
from paperbot_us.data import Calendar,Session,DataQualityError


class RiskTests(unittest.TestCase):
    def setUp(self):
        from paperbot_us.risk import RiskLimits,Portfolio,Position,PendingOrder,OrderIntent,Quote,RiskState,evaluate
        self.Limits,self.Portfolio,self.Position,self.Pending=RiskLimits,Portfolio,Position,PendingOrder
        self.Intent,self.Quote,self.State,self.evaluate=OrderIntent,Quote,RiskState,evaluate
        self.now=datetime(2026,10,23,15,0,tzinfo=timezone.utc)
        self.calendar=Calendar([Session.from_alpaca({'date':'2026-10-23','open':'09:30','close':'16:00'})],start='2026-10-23',end='2026-10-23')
        self.portfolio=Portfolio(self.now,'10000','10000','10000','10000')
        self.intent=OrderIntent('a'*64,'AAPL','buy',5,'150.1','145',self.now+timedelta(seconds=20))
        self.quote=Quote('AAPL','150','150.1',self.now,self.now)
        self.state=RiskState(pause_entries=False)
        self.limits=RiskLimits()

    def assess(self,**changes):
        return self.evaluate(changes.get('intent',self.intent),portfolio=changes.get('portfolio',self.portfolio),
            quote=changes.get('quote',self.quote),state=changes.get('state',self.state),limits=changes.get('limits',self.limits),
            calendar=self.calendar,now=changes.get('now',self.now))

    def test_valid_cash_only_paper_buy_has_explicit_cost_reservation(self):
        result=self.assess()
        self.assertTrue(result.allowed)
        self.assertEqual(result.violations,())
        self.assertEqual(result.reserved_cash_usd,Decimal('754.2525'))
        self.assertEqual(result.mode,'paper')

    def test_fixed_point_money_roundtrip_includes_the_existing_inclusive_maximum(self):
        from paperbot_us.risk import _money
        for typed in (Decimal('1E+4'),Decimal('1E+12'),Decimal('0E+12'),Decimal('0.00000001')):
            with self.subTest(value=typed):self.assertEqual(_money(format(_money(typed),'f')),typed)
        with self.assertRaises(DataQualityError):_money('1000000000001')

    def test_invalid_typed_or_nonfinite_inputs_are_never_valid_contracts(self):
        for value in (True,float('nan'),float('inf'),'NaN','1e999',{},-1):
            with self.subTest(value=type(value).__name__),self.assertRaises(DataQualityError):
                replace(self.quote,bid=value)
        for value in (True,0,1.5,-1):
            with self.subTest(qty=value),self.assertRaises(DataQualityError): replace(self.intent,quantity=value)
        with self.assertRaises(DataQualityError): replace(self.intent,stop_price_usd=None)

    def test_stale_and_future_account_quote_and_position_marks_are_vetoes(self):
        for obj,field in ((self.quote,'as_of'),(self.portfolio,'as_of')):
            for seconds in (-31,1):
                changed=replace(obj,**{field:self.now+timedelta(seconds=seconds)})
                result=self.assess(**({'quote':changed} if obj is self.quote else {'portfolio':changed}))
                self.assertFalse(result.allowed)
        pos=self.Position('AAPL',1,'150',self.now-timedelta(seconds=31))
        account=replace(self.portfolio,cash_usd='9850',positions=(pos,))
        self.assertIn('stale_position_mark',self.assess(portfolio=account).violations)

    def test_duplicate_signal_and_unsupported_market_state_block_orders(self):
        self.assertIn('duplicate_intent',self.assess(state=self.State(False,False,frozenset({'a'*64}))).violations)
        self.assertIn('market_closed',self.assess(now=self.now.replace(hour=21)).violations)
        self.assertIn('intent_expired',self.assess(intent=replace(self.intent,expires_at=self.now)).violations)

    def test_spread_limit_price_and_symbol_mismatch_cannot_be_overridden(self):
        cases=({'quote':replace(self.quote,ask='151')}, {'intent':replace(self.intent,limit_price_usd='152')},
               {'quote':replace(self.quote,symbol='MSFT')})
        for case in cases:
            with self.subTest(case=tuple(case)): self.assertFalse(self.assess(**case).allowed)

    def test_order_symbol_portfolio_and_modeled_trade_risk_caps(self):
        self.assertIn('order_notional_limit',self.assess(intent=replace(self.intent,quantity=10)).violations)
        self.assertIn('trade_risk_limit',self.assess(intent=replace(self.intent,stop_price_usd='100')).violations)
        position=self.Position('AAPL',4,'150',self.now)
        account=replace(self.portfolio,cash_usd='9400',positions=(position,))
        self.assertIn('symbol_exposure_limit',self.assess(portfolio=account).violations)
        pending=self.Pending('b'*64,'MSFT','buy',12,'150')
        account=replace(self.portfolio,pending_orders=(pending,))
        self.assertIn('portfolio_exposure_limit',self.assess(portfolio=account).violations)

    def test_pending_cash_and_same_symbol_reservations_count_before_new_buy(self):
        pending=self.Pending('b'*64,'AAPL','buy',4,'150.1')
        account=replace(self.portfolio,pending_orders=(pending,))
        self.assertIn('symbol_exposure_limit',self.assess(portfolio=account).violations)
        account=replace(self.portfolio,cash_usd='800',equity_usd='800',high_water_equity_usd='800',day_start_equity_usd='800')
        result=self.assess(portfolio=account)
        self.assertIn('cash_reserve_limit',result.violations)

    def test_modeled_trade_loss_reserves_friction_for_both_entry_and_exit(self):
        result=self.assess(intent=replace(self.intent,stop_price_usd='141'))
        self.assertIn('trade_risk_limit',result.violations)
        self.assertEqual(result.modeled_trade_loss_usd,Decimal('52.7775'))

    def test_loss_drawdown_reconciliation_and_pause_veto_new_entries(self):
        for account,reason in ((replace(self.portfolio,cash_usd='9800',equity_usd='9800'),'daily_loss_limit'),
                               (replace(self.portfolio,cash_usd='9400',equity_usd='9400'),'drawdown_limit'),
                               (replace(self.portfolio,reconciled=False),'account_unreconciled'),
                               (replace(self.portfolio,equity_usd='9990'),'equity_mismatch')):
            self.assertIn(reason,self.assess(portfolio=account).violations)
        self.assertIn('entries_paused',self.assess(state=self.State()).violations)

    def test_risk_reducing_sell_survives_entry_pause_and_loss_limits_but_not_halt(self):
        pos=self.Position('AAPL',5,'150',self.now)
        account=replace(self.portfolio,cash_usd='8650',equity_usd='9400',positions=(pos,))
        sell=replace(self.intent,side='sell',limit_price_usd='150',stop_price_usd=None)
        result=self.assess(intent=sell,portfolio=account,state=self.State())
        self.assertTrue(result.allowed)
        self.assertIn('all_orders_halted',self.assess(intent=sell,portfolio=account,state=self.State(halt_orders=True)).violations)

    def test_pending_sells_prevent_overselling_a_partially_reserved_position(self):
        pos=self.Position('AAPL',5,'150',self.now)
        pending=self.Pending('b'*64,'AAPL','sell',2,'150')
        account=replace(self.portfolio,cash_usd='9250',positions=(pos,),pending_orders=(pending,))
        sell=replace(self.intent,side='sell',limit_price_usd='150',stop_price_usd=None)
        self.assertIn('insufficient_unreserved_shares',self.assess(intent=sell,portfolio=account).violations)

    def test_unrecorded_behavior_subclasses_and_false_boolean_flags_are_rejected(self):
        with self.assertRaises(DataQualityError): replace(self.state,pause_entries='false')
        class OtherLimits(self.Limits): pass
        with self.assertRaises(DataQualityError): self.assess(limits=OtherLimits())

    def test_current_quote_revalues_existing_symbol_for_exposure_and_loss_veto(self):
        pos=self.Position('AAPL',5,'150',self.now-timedelta(seconds=20))
        account=replace(self.portfolio,cash_usd='9250',positions=(pos,))
        up=self.Quote('AAPL','199.9','200',self.now,self.now)
        buy=replace(self.intent,quantity=1,limit_price_usd='200',stop_price_usd='195')
        self.assertIn('symbol_exposure_limit',self.assess(portfolio=account,quote=up,intent=buy).violations)
        down=self.Quote('AAPL','99.9','100',self.now,self.now)
        buy=replace(buy,limit_price_usd='100',stop_price_usd='95')
        self.assertIn('daily_loss_limit',self.assess(portfolio=account,quote=down,intent=buy).violations)

    def test_preexisting_impossible_sell_reservations_block_even_an_unrelated_buy(self):
        impossible=self.Pending('b'*64,'MSFT','sell',10,'150')
        self.assertIn('inconsistent_sell_reservations',self.assess(portfolio=replace(self.portfolio,pending_orders=(impossible,))).violations)

    def test_per_order_cap_also_applies_to_risk_reducing_sells(self):
        pos=self.Position('AAPL',5,'150',self.now)
        account=replace(self.portfolio,cash_usd='9250',positions=(pos,))
        sell=replace(self.intent,side='sell',limit_price_usd='150',stop_price_usd=None)
        self.assertIn('order_notional_limit',self.assess(intent=sell,portfolio=account,
                      limits=replace(self.limits,max_order_notional_usd='200')).violations)


if __name__=='__main__': unittest.main()
