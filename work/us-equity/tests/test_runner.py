from dataclasses import replace
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import tempfile
import unittest
from paperbot_us.alpaca import Asset,UniverseSnapshot
from paperbot_us.data import Calendar,Session,Snapshot
from paperbot_us.audit import ScanArchive,ArchiveError


class FakeClient:
    def __init__(self):
        self.now=datetime(2026,10,23,15,tzinfo=timezone.utc)
        self.batches=[]
        self.asset_calls=0
    def assets(self):
        self.asset_calls+=1
        return UniverseSnapshot(tuple(Asset(s,s,'NASDAQ',True) for s in ('AAPL','MSFT','XYZ')),self.now,0)
    def calendar(self,start,end):
        return Calendar([Session.from_alpaca({'date':start,'open':'09:30','close':'16:00'})],start=start,end=end)
    def snapshots(self,symbols):
        self.batches.append(symbols)
        return {s:Snapshot(s,100,self.now-timedelta(seconds=1),99.99,100.01,10,10,
                    self.now-timedelta(seconds=2),100000,self.now.date(),self.now) for s in symbols}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        from paperbot_us.runner import collect_once,RunnerError
        self.collect,self.Error=collect_once,RunnerError
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.client=FakeClient()
        self.archive=ScanArchive(self.root/'scans.sqlite')
        self.checkpoint=self.root/'checkpoint.json'
    def run_once(self,**options):
        return self.collect(self.client,self.archive,checkpoint=self.checkpoint,
                            clock=lambda:self.client.now,batch_size=2,**options)

    def test_restart_resumes_and_cycle_covers_full_current_pool(self):
        first=self.run_once()
        self.assertEqual((first['next_offset'],first['universe_count'],first['checked_count']),(2,3,2))
        second=self.run_once()
        self.assertEqual(self.client.batches,[['AAPL','MSFT'],['XYZ']])
        self.assertTrue(second['cycle_finished'])
        self.assertEqual(second['next_offset'],0)
        self.assertEqual(second['mode'],'data_only_no_orders')
        self.assertEqual(self.archive.load(first['run_id'])['result']['status'],'ok')

    def test_archive_failure_never_advances_progress(self):
        class FailedArchive:
            def append(self,record,**kwargs): raise ArchiveError('fixture failure')
        with self.assertRaises(self.Error):
            self.collect(self.client,FailedArchive(),checkpoint=self.checkpoint,
                         clock=lambda:self.client.now,batch_size=2)
        self.assertFalse(self.checkpoint.exists())

    def test_closed_market_does_not_fetch_quotes_or_advance(self):
        self.client.now=self.client.now.replace(hour=20)
        result=self.run_once()
        self.assertEqual(result['status'],'market_closed')
        self.assertEqual(self.client.batches,[])
        self.assertEqual(result['next_offset'],0)

    def test_corrupt_checkpoint_is_not_silently_reset(self):
        self.checkpoint.write_text('{invalid')
        with self.assertRaises(self.Error): self.run_once()

    def test_current_universe_is_cached_but_expired_data_is_refreshed(self):
        self.run_once()
        self.run_once()
        self.assertEqual(self.client.asset_calls,1)
        self.client.now+=timedelta(days=1)
        self.run_once()
        self.assertEqual(self.client.asset_calls,2)

    def test_corrupt_universe_cache_is_not_silently_replaced(self):
        self.run_once()
        cache=self.checkpoint.with_name(self.checkpoint.name+'.universe.json')
        cache.write_text('{invalid')
        with self.assertRaises(self.Error): self.run_once()

    def test_concurrent_collector_is_refused_before_provider_calls(self):
        from paperbot_us.runner import _exclusive
        with _exclusive(self.checkpoint):
            with self.assertRaises(self.Error): self.run_once()
        self.assertEqual(self.client.asset_calls,0)
        self.assertEqual(self.run_once()['next_offset'],2)

    def test_request_crossing_market_close_never_advances_progress(self):
        self.client.now=self.client.now.replace(hour=19,minute=59,second=59)
        original=self.client.snapshots
        def late_snapshots(symbols):
            values=original(symbols)
            self.client.now+=timedelta(seconds=2)
            return values
        self.client.snapshots=late_snapshots
        result=self.run_once()
        self.assertEqual(result['status'],'market_closed')
        self.assertEqual((result['checked_count'],result['next_offset']),(0,0))
        self.assertFalse(self.checkpoint.exists())

    def test_invalid_provider_snapshot_reason_survives_archive_and_replay(self):
        def snapshots(symbols):
            self.client.snapshot_rejections={'AAPL':'invalid_snapshot','MSFT':'missing_snapshot'}
            return {}
        self.client.snapshots=snapshots
        result=self.run_once()
        record=self.archive.load(result['run_id'])
        self.assertEqual(record['result']['rejected'],{'AAPL':'snapshot_invalid_or_stale','MSFT':'snapshot_missing'})
        self.assertEqual(record['inputs']['snapshot_rejections']['AAPL'],'invalid_snapshot')


if __name__=='__main__': unittest.main()
