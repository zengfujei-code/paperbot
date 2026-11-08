import math
import unittest
from datetime import datetime,timedelta,timezone
from paperbot_us.data import Bar,Calendar,Session,DataQualityError


class QuantTests(unittest.TestCase):
    def setUp(self):
        from paperbot_us.quant import IndicatorSpec,indicators,live_indicators
        self.Spec,self.calculate,self.live=IndicatorSpec,indicators,live_indicators

    def test_reference_vector_has_explicit_fraction_and_price_units(self):
        result=self.calculate(list(range(1,22)),[100]*20+[200],spec=self.Spec())
        self.assertEqual((result.sma_fast_usd,result.sma_slow_usd,result.rsi_sma), (19,11.5,100))
        self.assertAlmostEqual(result.momentum_fraction,21/17-1)
        returns=[1/i for i in range(1,21)]
        mean=sum(returns)/20
        expected=math.sqrt(sum((r-mean)**2 for r in returns)/20)
        self.assertAlmostEqual(result.volatility_15m_fraction,expected)
        self.assertEqual(result.volume_ratio,2)
        self.assertEqual((result.interval_seconds,result.volatility_annualized),(900,False))

    def test_constant_and_declining_prices_have_defined_rsi(self):
        flat=self.calculate([100]*21,[100]*21,spec=self.Spec())
        self.assertEqual((flat.rsi_sma,flat.momentum_fraction,flat.volatility_15m_fraction),(50,0,0))
        down=self.calculate(list(range(21,0,-1)),[100]*21,spec=self.Spec())
        self.assertEqual(down.rsi_sma,0)

    def test_mixed_rsi_uses_fourteen_changes_and_simple_average(self):
        closes=[100]*7
        for change in [1,-2]*7: closes.append(closes[-1]+change)
        result=self.calculate(closes,[100]*21,spec=self.Spec())
        self.assertAlmostEqual(result.rsi_sma,100*7/(7+14))

    def test_invalid_inputs_never_produce_neutral_defaults(self):
        for prices,volumes in (([100]*20,[100]*20),([100]*21,[100]*20),([0]+[100]*20,[100]*21),
                ([float('nan')]+[100]*20,[100]*21),([True]+[100]*20,[100]*21),
                ([100]*21,[0]*21),([100]*21,[-1]+[100]*20)):
            with self.subTest(prices=prices[:1]),self.assertRaises(DataQualityError):
                self.calculate(prices,volumes,spec=self.Spec())
        for values in ({'fast':True},{'fast':20},{'rsi':0},{'slow':10001}):
            with self.subTest(values=values),self.assertRaises(DataQualityError): self.Spec(**values)

    def test_live_boundary_rejects_unavailable_data_and_missing_session_bar(self):
        start=datetime(2026,10,23,13,30,tzinfo=timezone.utc)
        bars=[Bar('AAPL',start+timedelta(minutes=15*i),start+timedelta(minutes=15*(i+1)),
                  100,101,99,100,100) for i in range(21)]
        calendar=Calendar([Session.from_alpaca({'date':'2026-10-23','open':'09:30','close':'16:00'})],
                          start='2026-10-23',end='2026-10-23')
        result=self.live(bars,calendar=calendar,as_of=bars[-1].ends,spec=self.Spec())
        self.assertEqual(result.rsi_sma,50)
        with self.assertRaises(DataQualityError):
            self.live(bars,calendar=calendar,as_of=bars[-1].ends-timedelta(seconds=1),spec=self.Spec())
        with self.assertRaises(DataQualityError):
            self.live(bars[:10]+bars[11:],calendar=calendar,as_of=bars[-1].ends,spec=self.Spec())


if __name__=='__main__': unittest.main()
