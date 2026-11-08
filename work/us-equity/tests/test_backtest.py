from dataclasses import replace
from datetime import datetime,timedelta,timezone
import unittest
from paperbot_us.data import Bar,Calendar,Session,DataQualityError
from paperbot_us.research import ResearchDataset


class BacktestTests(unittest.TestCase):
    def setUp(self):
        from paperbot_us.backtest import BacktestError,CostModel,ReferencePolicy,backtest
        self.Error,self.Cost,self.Policy,self.run_test=BacktestError,CostModel,ReferencePolicy,backtest
        self.start=datetime(2026,10,23,13,30,tzinfo=timezone.utc)
        self.download=self.start+timedelta(days=1)
        self.calendar=Calendar([Session.from_alpaca({'date':'2026-10-23','open':'09:30','close':'16:00'})],
                               start='2026-10-23',end='2026-10-23')
        self.bars=[Bar('AAPL',self.start+timedelta(minutes=15*i),self.download,
                       100+i,102+i,99+i,101+i,100000) for i in range(26)]
    def run_case(self,bars=None,cost=None):
        dataset=ResearchDataset.from_bars(self.bars if bars is None else bars,calendar=self.calendar,
            downloaded_at=self.download,availability_policy='vendor_history_assumed_at_close')
        return self.run_test(dataset,policy=self.Policy(entry_rsi_min=0,entry_rsi_max=100,exit_rsi=100),
                             costs=cost or self.Cost(commission_per_share_usd=0.01,slippage_bps=10))

    def test_first_fill_uses_next_bar_open_and_actual_gap_price(self):
        bars=list(self.bars)
        bars[21]=replace(bars[21],open=180,high=181,low=179,close=180)
        result=self.run_case(bars)
        first=result['fills'][0]
        self.assertEqual((first['side'],first['at'],first['price_usd']),('buy',bars[21].starts.isoformat(),180))
        self.assertEqual(result['decisions'][21]['features']['last_price_usd'],121)
        self.assertLessEqual(first['gross_usd']+first['cost_usd'],1000)

    def test_cash_and_realized_pnl_reconcile_after_daily_liquidation(self):
        result=self.run_case()
        self.assertEqual(result['final_position_shares'],0)
        self.assertEqual(len(result['trades']),1)
        self.assertAlmostEqual(result['ending_equity_usd'],10000+sum(t['net_pnl_usd'] for t in result['trades']))
        self.assertGreater(result['total_cost_usd'],0)
        self.assertTrue(all(e['cash_usd']>=0 for e in result['equity_curve']))
        self.assertEqual(result['drawdown_basis'],'15min_close_equity')
        self.assertEqual(result['validation_status'],'research_only_not_strategy_acceptance')

    def test_explicit_costs_reduce_the_same_synthetic_trade_result(self):
        with_costs=self.run_case()
        zero_costs=self.run_case(cost=self.Cost(commission_per_share_usd=0,slippage_bps=0))
        self.assertLess(with_costs['ending_equity_usd'],zero_costs['ending_equity_usd'])

    def test_missing_liquidity_does_not_fabricate_daily_exit(self):
        bars=list(self.bars)
        bars[-1]=replace(bars[-1],volume=0)
        with self.assertRaises(self.Error): self.run_case(bars)

    def test_fixed_symbols_are_not_reported_as_market_wide_validation(self):
        result=self.run_case()
        self.assertEqual(result['universe_policy'],'fixed_symbol_research_only')
        self.assertEqual(result['availability_policy'],'vendor_history_assumed_at_close')
        self.assertIn('uncalibrated',result['cost_assumption'])
        self.assertEqual(result['broker_orders_submitted'],0)

    def test_unrecorded_strategy_or_cost_overrides_are_rejected(self):
        class DifferentRule(self.Policy):
            def action(self,features,position_open): return 'hold' if position_open else 'buy'
        class DifferentCost(self.Cost):
            def components(self,price,shares): return super().components(price,shares)
        dataset=ResearchDataset.from_bars(self.bars,calendar=self.calendar,downloaded_at=self.download,
                                         availability_policy='vendor_history_assumed_at_close')
        for policy,costs in ((DifferentRule(),self.Cost(0,0)),(self.Policy(),DifferentCost(0,0))):
            with self.subTest(kind=type(policy).__name__),self.assertRaises(self.Error):
                self.run_test(dataset,policy=policy,costs=costs)


if __name__=='__main__': unittest.main()
