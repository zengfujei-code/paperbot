import importlib
import importlib.util
import math
import unittest
from dataclasses import replace
from datetime import datetime, timezone, timedelta


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.data'), 'data contract not implemented')
        self.m = importlib.import_module('paperbot_us.data')
        self.now = datetime(2026, 10, 21, 15, 0, tzinfo=timezone.utc)

    def session(self, date='2026-10-21', close='16:00'):
        return self.m.Session.from_alpaca({'date':date,'open':'09:30','close':close})

    def bar(self, **changes):
        value = dict(t='2026-10-21T14:30:00Z',o=100,h=102,l=99,c=101,v=500,n=10)
        value.update(changes)
        return self.m.Bar.from_alpaca('AAPL',value,received_at=self.now)

    def test_timestamp_requires_an_explicit_timezone(self):
        self.assertEqual(self.m.utc_time('2026-10-21T23:00:00+08:00'), self.now)
        for value in ('2026-10-21T15:00:00',True,0,'bad'):
            with self.subTest(value=value),self.assertRaises(self.m.DataQualityError):
                self.m.utc_time(value)

    def test_calendar_uses_new_york_dst_and_half_day(self):
        before = self.session('2026-03-06')
        after = self.session('2026-03-09')
        self.assertEqual(before.opens.hour,14)
        self.assertEqual(after.opens.hour,13)
        early = self.session('2026-11-27','13:00')
        self.assertEqual(early.closes.hour,18)

    def test_calendar_coverage_is_explicit_not_an_assumed_weekday_rule(self):
        calendar = self.m.Calendar([self.session()],start='2026-10-21',end='2026-10-22')
        self.assertIsNotNone(calendar.at(self.now))
        self.assertIsNone(calendar.at(self.now+timedelta(days=1)))
        with self.assertRaises(self.m.DataQualityError):
            calendar.at(self.now+timedelta(days=2))

    def test_duplicate_calendar_dates_are_rejected(self):
        with self.assertRaises(self.m.DataQualityError):
            self.m.Calendar([self.session(),self.session()],start='2026-10-21',end='2026-10-21')

    def test_valid_closed_bar_has_explicit_feed_units_and_adjustment(self):
        bar = self.bar()
        bar.validate(as_of=self.now,session=self.session())
        self.assertEqual((bar.feed,bar.adjustment,bar.currency,bar.volume_unit),('iex','raw','USD','shares'))
        self.assertEqual(bar.ends-bar.starts,timedelta(minutes=15))

    def test_bad_prices_volumes_and_nonfinite_values_are_rejected(self):
        for changes in ({'o':0},{'c':-1},{'h':99},{'l':102},{'v':-1},
                        {'c':float('nan')},{'h':float('inf')},{'v':True},{'o':10**400},{'c':'101'}):
            with self.subTest(changes=changes),self.assertRaises(self.m.DataQualityError):
                self.bar(**changes)

    def test_unclosed_future_or_off_session_bars_are_rejected(self):
        for stamp in ('2026-10-21T15:00:00Z','2026-10-21T12:00:00Z','2026-10-21T20:00:00Z'):
            with self.subTest(stamp=stamp),self.assertRaises(self.m.DataQualityError):
                self.bar(t=stamp).validate(as_of=self.now,session=self.session())

    def test_receipt_cannot_precede_close_or_follow_decision_time(self):
        with self.assertRaises(self.m.DataQualityError):
            self.m.Bar.from_alpaca('AAPL',dict(t='2026-10-21T14:30:00Z',o=100,h=102,l=99,c=101,v=500),
                                  received_at=self.now-timedelta(hours=1)).validate(as_of=self.now,session=self.session())
        with self.assertRaises(self.m.DataQualityError):
            self.bar().validate(as_of=self.now-timedelta(seconds=1),session=self.session())

    def test_duplicate_gap_and_mixed_feed_series_are_rejected(self):
        first = self.bar(t='2026-10-21T14:15:00Z')
        second = self.bar(t='2026-10-21T14:30:00Z')
        calendar = self.m.Calendar([self.session()],start='2026-10-21',end='2026-10-21')
        self.m.validate_series([first,second],as_of=self.now,calendar=calendar)
        for bars in ([first,first],[second,first],[self.bar(t='2026-10-21T13:45:00Z'),second],
                     [first,self.m.Bar.from_alpaca('AAPL',dict(t='2026-10-21T14:30:00Z',o=100,h=102,l=99,c=101,v=500),received_at=self.now,feed='sip')]):
            with self.subTest(bars=bars),self.assertRaises(self.m.DataQualityError):
                self.m.validate_series(bars,as_of=self.now,calendar=calendar)

    def test_empty_or_stale_series_is_not_a_trading_input(self):
        calendar = self.m.Calendar([self.session()],start='2026-10-21',end='2026-10-21')
        with self.assertRaises(self.m.DataQualityError):
            self.m.validate_series([],as_of=self.now,calendar=calendar)
        with self.assertRaises(self.m.DataQualityError):
            self.m.validate_series([self.bar()],as_of=self.now+timedelta(hours=1),calendar=calendar)

    def test_final_series_boundary_requires_session_calendar(self):
        with self.assertRaises(self.m.DataQualityError):
            self.m.validate_series([self.bar()],as_of=self.now)
        calendar = self.m.Calendar([self.session()],start='2026-10-21',end='2026-10-21')
        early = self.m.Bar.from_alpaca('AAPL',dict(t='2026-10-21T12:00:00Z',o=100,h=102,l=99,c=101,v=500),
                                     received_at=datetime(2026,10,21,12,15,tzinfo=timezone.utc))
        with self.assertRaises(self.m.DataQualityError):
            self.m.validate_series([early],as_of=early.received_at,calendar=calendar)
        misaligned = self.bar(t='2026-10-21T14:31:00Z')
        with self.assertRaises(self.m.DataQualityError):
            self.m.validate_series([misaligned],as_of=self.now,calendar=calendar)

    def test_unrepresentable_bar_end_is_a_typed_data_error(self):
        with self.assertRaises(self.m.DataQualityError):
            self.bar(t='9999-12-31T23:59:00Z')

    def snapshot(self):
        return self.m.Snapshot.from_alpaca('AAPL', {
            'latestTrade': {'p':100,'t':'2026-10-21T14:59:59Z'},
            'latestQuote': {'bp':99.99,'ap':100.01,'bs':10,'as':10,'t':'2026-10-21T14:59:58Z'},
            'dailyBar': {'v':10000,'t':'2026-10-21T04:00:00Z'}},received_at=self.now)

    def test_independent_trade_and_quote_clocks_need_no_ordering(self):
        calendar = self.m.Calendar([self.session()],start='2026-10-21',end='2026-10-21')
        self.snapshot().validate(as_of=self.now,calendar=calendar)

    def test_direct_snapshot_construction_cannot_bypass_validation(self):
        snapshot = self.snapshot()
        for change in ({'price':float('nan')},{'bid':101},{'daily_volume':-1},
                       {'symbol':'invalid symbol'},{'source':'unknown'},{'feed':'sip'},
                       {'currency':'USDT'},{'volume_unit':'lots'},{'volume_period':'unknown'},
                       {'trade_at':self.now+timedelta(seconds=1)},
                       {'daily_session_day':self.now.date()-timedelta(days=1)}):
            with self.subTest(change=change),self.assertRaises(self.m.DataQualityError):
                replace(snapshot,**change)


if __name__ == '__main__':
    unittest.main()
