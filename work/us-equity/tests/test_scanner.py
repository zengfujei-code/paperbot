import unittest
from datetime import datetime, timezone, timedelta
from dataclasses import replace
from paperbot_us.data import Calendar, Session, Snapshot, DataQualityError
from paperbot_us.alpaca import Asset, UniverseSnapshot


class ScannerTests(unittest.TestCase):
    def setUp(self):
        from paperbot_us.scanner import scan, ScanPolicy
        self.scan, self.Policy = scan, ScanPolicy
        self.now = datetime(2026,10,21,15,tzinfo=timezone.utc)
        self.calendar = Calendar([Session.from_alpaca({'date':'2026-10-21','open':'09:30','close':'16:00'})],
                                 start='2026-10-21',end='2026-10-21')
        self.universe = UniverseSnapshot(tuple(Asset(s,s,'NASDAQ',True) for s in ('AAPL','MSFT','XYZ')),self.now,0)

    def quote(self,symbol,volume=100000):
        return Snapshot(symbol,100,self.now-timedelta(seconds=1),99.99,100.01,10,10,
                        self.now-timedelta(seconds=2),volume,self.now.date(),self.now)

    def test_deterministic_liquidity_ranking_and_explicit_missing_reason(self):
        values = {'MSFT':self.quote('MSFT',200000),'AAPL':self.quote('AAPL')}
        result = self.scan(self.universe,values,calendar=self.calendar,as_of=self.now,policy=self.Policy())
        self.assertEqual([v.symbol for v in result.candidates],['MSFT','AAPL'])
        self.assertEqual(result.rejected,{'XYZ':'snapshot_missing'})
        self.assertEqual(result.feed,'iex')
        self.assertEqual(result.universe_coverage,'current_snapshot_only')
        self.assertEqual(result,self.scan(self.universe,dict(reversed(list(values.items()))),
                         calendar=self.calendar,as_of=self.now,policy=self.Policy()))

    def test_quality_and_policy_rejections_never_become_candidates(self):
        values = {'AAPL':self.quote('MSFT'), 'MSFT':self.quote('MSFT',1)}
        result = self.scan(self.universe,values,calendar=self.calendar,as_of=self.now,policy=self.Policy())
        self.assertEqual(result.candidates,())
        self.assertEqual(result.rejected['AAPL'],'snapshot_identity')
        self.assertEqual(result.rejected['MSFT'],'volume_below_minimum')
        stale = self.scan(self.universe,{'AAPL':self.quote('AAPL')},calendar=self.calendar,
                          as_of=self.now+timedelta(seconds=31),policy=self.Policy())
        self.assertEqual(stale.rejected['AAPL'],'snapshot_invalid_or_stale')

    def test_unavailable_universe_and_invalid_policies_fail_closed(self):
        for changes in ({'received_at':self.now+timedelta(seconds=1)},
                        {'received_at':self.now-timedelta(days=2)},
                        {'assets':self.universe.assets*2}):
            with self.subTest(changes=changes),self.assertRaises(DataQualityError):
                self.scan(replace(self.universe,**changes),{},calendar=self.calendar,as_of=self.now,policy=self.Policy())
        for change in ({'min_price':float('nan')},{'max_spread_bps':True},{'max_candidates':0}):
            with self.subTest(change=change),self.assertRaises(DataQualityError):
                self.Policy(**change)

    def test_closed_market_returns_no_candidates(self):
        result = self.scan(self.universe,{'AAPL':self.quote('AAPL')},calendar=self.calendar,
                           as_of=self.now.replace(hour=20),policy=self.Policy())
        self.assertEqual(result.status,'market_closed')
        self.assertEqual(result.candidates,())

    def test_boundaries_and_overflow_have_explicit_reasons(self):
        cases = (
            (replace(self.quote('AAPL'),price=4),'price_below_minimum'),
            (replace(self.quote('AAPL'),ask=101),'spread_above_maximum'),
            (replace(self.quote('AAPL'),bid_size=0),'empty_quote_size'),
            (replace(self.quote('AAPL'),price=1e308,daily_volume=1e308),'numeric_overflow'))
        for quote,reason in cases:
            with self.subTest(reason=reason):
                result = self.scan(self.universe,{'AAPL':quote},calendar=self.calendar,
                                   as_of=self.now,policy=self.Policy())
                self.assertEqual(result.rejected['AAPL'],reason)
                self.assertEqual(result.candidates,())
        quotes = {s:self.quote(s) for s in ('AAPL','MSFT','XYZ')}
        result = self.scan(self.universe,quotes,calendar=self.calendar,as_of=self.now,
                           policy=self.Policy(max_candidates=1))
        self.assertEqual([c.symbol for c in result.candidates],['AAPL'])
        self.assertEqual(result.rejected,{'MSFT':'candidate_budget','XYZ':'candidate_budget'})

    def test_subnormal_quotes_do_not_crash_the_scan(self):
        quote = replace(self.quote('AAPL'),bid=5e-324,ask=5e-324)
        result = self.scan(self.universe,{'AAPL':quote},calendar=self.calendar,
                           as_of=self.now,policy=self.Policy())
        self.assertEqual(result.rejected['AAPL'],'quote_below_price_floor')

    def test_asset_constructor_rejects_bypassed_eligibility(self):
        for changes in ({'security_id':''},{'symbol':'bad symbol'},{'exchange':'OTC'},
                        {'fractionable':'not-bool'}):
            with self.subTest(changes=changes),self.assertRaises(DataQualityError):
                replace(self.universe.assets[0],**changes)


if __name__ == '__main__': unittest.main()
