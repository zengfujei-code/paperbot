"""Causal issue-input tests on explicitly synthetic two-segment journals."""
from contextlib import closing,contextmanager
from dataclasses import replace
from datetime import datetime,timedelta,timezone
from pathlib import Path
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import zoneinfo

sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_training_data import ROOT,TZ,Tick
from issue_fixture import IssueWorld
from paperbot_r2 import forecast_inputs as inputs,receipt_book
from paperbot_r2.contracts import digest


class ForecastInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_tz=zoneinfo.TZPATH
        zoneinfo.reset_tzpath([str(ROOT/'outputs/zoneinfo')]);zoneinfo.ZoneInfo.clear_cache()

    @classmethod
    def tearDownClass(cls):
        zoneinfo.reset_tzpath(cls.old_tz);zoneinfo.ZoneInfo.clear_cache()

    def setUp(self):
        env=patch.dict(os.environ,{},clear=True);env.start();self.addCleanup(env.stop)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name);self.w=IssueWorld(self.path)

    def capture(self,key,clock=None,pairs=None):
        return inputs.capture_issue_input(self.w.training_receipt,self.w.registration,
            pairs or self.w.source_pairs(),key,timezone_file=TZ,clock=clock or Tick(self.w.at))

    def test_current_feature_first_prices_and_actual_availability_are_read_only(self):
        key=self.w.receive_until('2026-11-24',9)
        paths=[self.w.f.raw.path,self.w.f.features.path,self.w.next_raw.path,self.w.next_features.path]
        before=[p.read_bytes() for p in paths]
        value=self.capture(key)
        self.assertEqual(value.symbol,'AAPL');self.assertEqual(len(value.returns),8)
        self.assertTrue(all(r.available_at==value.available_at for r in value.returns))
        self.assertEqual(value.returns[-1].ends_at,value.bundle.vector.event_at)
        self.assertGreater(value.available_at,value.read_started_at)
        self.assertEqual(value.expected_return_slots,8);self.assertEqual(value.missing_return_slots,0)
        self.assertEqual(len(value.bar_witnesses),9)
        self.assertEqual(before,[p.read_bytes() for p in paths])
        self.assertEqual(inputs.decode_input(inputs.encode_input(value)),value)

    def test_original_and_successor_are_joined_without_overnight_returns(self):
        self.w.receive_until('2026-11-24',26,derive=False)
        self.w.receive_until('2026-11-25',26,derive=False)
        key=self.w.receive_until('2026-11-27',9)
        value=self.capture(key)
        self.assertEqual(len(value.returns),58)
        self.assertEqual(value.expected_return_slots,58)
        self.assertTrue(all(r.starts_at.date()==r.ends_at.date() for r in value.returns))
        self.assertEqual(len({w.segment for w in value.bar_witnesses}),2)

    def test_missing_pair_is_not_fabricated(self):
        self.w.receive_until('2026-11-24',9)
        key=self.w.receive_until('2026-11-25',12,omit=(1,))
        value=self.capture(key)
        self.assertEqual(len(value.returns),17)
        self.assertEqual(value.expected_return_slots,36)
        self.assertEqual(value.missing_return_slots,19)

    def test_stale_future_and_reserved_clock_reject_before_source_reads(self):
        key=self.w.receive_until('2026-11-24',9)
        event=self.w.session('2026-11-24').opens+timedelta(seconds=8100)
        for at in (event-timedelta(seconds=1),event+timedelta(seconds=121),
                   datetime(2026,11,12,15,tzinfo=timezone.utc)):
            with patch('sqlite3.connect',side_effect=AssertionError('no source read')) as connect:
                with self.assertRaises(inputs.ForecastInputError):self.capture(key,clock=lambda:at)
                connect.assert_not_called()

    def test_copied_original_book_cannot_replace_training_ancestor(self):
        key=self.w.receive_until('2026-11-24',9)
        path=self.path/'copy.sqlite';shutil.copyfile(self.w.f.raw.path,path)
        pairs=list(self.w.source_pairs());pairs[0]=inputs.SourcePair(receipt_book.ReceiptBook(path),self.w.f.features)
        with self.assertRaises(inputs.ForecastInputError):self.capture(key,pairs=tuple(pairs))

    def test_reserved_price_timestamp_vetoes_before_deep_raw_replay(self):
        self.w.receive_until('2026-11-12',1,derive=False)
        key=self.w.receive_until('2026-11-24',9)
        with patch.object(self.w.f.raw,'_transaction',side_effect=AssertionError('no deep reserved prices')) as deep:
            with self.assertRaises(inputs.ForecastInputError):self.capture(key)
            deep.assert_not_called()

    def test_calendar_change_vetoes_current_frame(self):
        key=self.w.receive_until('2026-11-24',9)
        session=self.w.session('2026-11-24');changed=replace(session,closes=session.closes-timedelta(minutes=15))
        from paperbot_us.data import Calendar
        self.w.f.raw.record_calendar(Calendar([changed],start='2026-11-24',end='2026-11-24'),
            day='2026-11-24',received_at=self.w.at)
        self.w.at+=timedelta(seconds=1)
        with self.assertRaises(inputs.ForecastInputError):self.capture(key)

    def test_wal_commit_during_capture_is_detected(self):
        key=self.w.receive_until('2026-11-24',9);raw=self.w.f.raw
        with closing(sqlite3.connect(raw.path,isolation_level=None)) as keep:
            keep.execute('PRAGMA journal_mode=WAL');keep.execute('PRAGMA user_version=1')
            original=raw._transaction;changed=[]
            @contextmanager
            def transaction(write=False):
                with original(write) as values:yield values
                if not write and not changed:
                    changed.append(True);raw.observe([],now=self.w.at+timedelta(seconds=1))
            with patch.object(raw,'_transaction',transaction):
                with self.assertRaises(inputs.ForecastInputError):
                    self.capture(key,clock=Tick(self.w.at+timedelta(seconds=2)))
            self.assertEqual(changed,[True])

    def test_future_raw_metadata_rejects_before_deep_prices(self):
        key=self.w.receive_until('2026-11-24',9)
        at=self.w.at
        self.w.receive_until('2026-11-24',10,derive=False)
        with patch.object(self.w.f.raw,'_transaction',side_effect=AssertionError('no future prices')) as deep:
            with self.assertRaises(inputs.ForecastInputError):self.capture(key,clock=Tick(at))
            deep.assert_not_called()

    def test_missing_successor_cannot_shrink_the_declared_campaign(self):
        key=self.w.receive_until('2026-11-24',9)
        with patch('sqlite3.connect',side_effect=AssertionError('no incomplete source read')) as connect:
            with self.assertRaises(inputs.ForecastInputError):self.capture(key,pairs=(self.w.source_pairs()[0],))
            connect.assert_not_called()


    def test_late_prior_session_return_uses_current_read_availability(self):
        from paperbot_us.data import Bar
        self.w.receive_until('2026-11-24',10,omit=(9,),derive=False)
        session=self.w.session('2026-11-24');start=session.opens+timedelta(seconds=8100)
        received=session.closes+timedelta(minutes=1)
        bars=[Bar(s,start,received,100.*m,102.*m,98.*m,100.4*m,110.)
              for s,m in zip(inputs.SYMBOLS,(1.,2.,4.))]
        self.w.f.raw.observe(bars,now=received+timedelta(seconds=1))
        key=self.w.receive_until('2026-11-25',9);value=self.capture(key)
        self.assertEqual(value.expected_return_slots,33);self.assertEqual(len(value.returns),17)
        self.assertEqual(value.late_return_slots,1);self.assertEqual(value.missing_return_slots,16)
        self.assertTrue(all(r.available_at==value.available_at>received for r in value.returns))
        self.assertEqual(sum(w.late for w in value.bar_witnesses),1)


if __name__=='__main__':unittest.main()
