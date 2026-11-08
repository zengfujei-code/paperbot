from dataclasses import replace
from datetime import datetime,timedelta,timezone,date
import unittest
from paperbot_us.data import Bar,Calendar,Session,DataQualityError
from paperbot_us.quant import IndicatorSpec


class ResearchTests(unittest.TestCase):
    def setUp(self):
        from paperbot_us.research import ResearchDataset,time_split
        self.Dataset,self.split=ResearchDataset,time_split
        self.start=datetime(2026,10,23,13,30,tzinfo=timezone.utc)
        self.download=self.start+timedelta(days=1)
        self.calendar=Calendar([Session.from_alpaca({'date':'2026-10-23','open':'09:30','close':'16:00'})],
                                start='2026-10-23',end='2026-10-23')
        self.bars=[Bar('AAPL',self.start+timedelta(minutes=15*i),self.download,
                       100+i,102+i,99+i,101+i,100) for i in range(26)]
    def dataset(self,bars=None,policy='vendor_history_assumed_at_close'):
        return self.Dataset.from_bars(self.bars if bars is None else bars,calendar=self.calendar,
                                      downloaded_at=self.download,availability_policy=policy)

    def test_historical_assumption_is_explicit_and_not_an_archived_receipt(self):
        historical=self.dataset()
        features=historical.indicators_for_open(21,spec=IndicatorSpec())
        self.assertEqual(features.last_price_usd,121)
        recorded=self.dataset(policy='archived_receipts')
        with self.assertRaises(DataQualityError): recorded.indicators_for_open(21,spec=IndicatorSpec())
        self.assertEqual(historical.manifest()['availability_policy'],'vendor_history_assumed_at_close')
        self.assertEqual(historical.manifest()['universe_policy'],'fixed_symbol_research_only')

    def test_future_prices_cannot_change_prior_features(self):
        original=self.dataset().indicators_for_open(21,spec=IndicatorSpec())
        changed=list(self.bars)
        changed[21]=replace(changed[21],close=999,high=999)
        self.assertEqual(original,self.dataset(changed).indicators_for_open(21,spec=IndicatorSpec()))
        self.assertNotEqual(self.dataset().manifest()['data_sha256'],self.dataset(changed).manifest()['data_sha256'])

    def test_missing_duplicate_unordered_and_future_data_are_rejected(self):
        for bars in (self.bars[:10]+self.bars[11:],self.bars+[self.bars[-1]],list(reversed(self.bars)),
                     [replace(self.bars[0],received_at=self.download+timedelta(seconds=1))]+self.bars[1:]):
            with self.subTest(count=len(bars)),self.assertRaises(DataQualityError): self.dataset(bars)

    def test_recorded_receipt_can_only_affect_a_later_open(self):
        actual=[replace(b,received_at=b.ends+timedelta(seconds=1)) for b in self.bars]
        dataset=self.dataset(actual,policy='archived_receipts')
        with self.assertRaises(DataQualityError): dataset.indicators_for_open(21,spec=IndicatorSpec())
        self.assertEqual(dataset.indicators_for_open(22,spec=IndicatorSpec()).last_price_usd,121)

    def test_split_preserves_whole_sessions_and_embargo(self):
        days=tuple(date(2026,1,1)+timedelta(days=i) for i in range(10))
        split=self.split(days,train_sessions=3,validation_sessions=2,test_sessions=3,embargo_sessions=1)
        self.assertEqual((split.train,split.validation,split.test),(days[:3],days[4:6],days[7:]))
        self.assertEqual(split.embargo,(days[3],days[6]))
        with self.assertRaises(DataQualityError):
            self.split(tuple(reversed(days)),train_sessions=3,validation_sessions=2,test_sessions=3)


if __name__=='__main__': unittest.main()
