from datetime import datetime,timedelta,timezone
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import sqlite3
from contextlib import closing
from unittest.mock import patch
from paperbot_us.alpaca import Asset,UniverseSnapshot
from paperbot_us.data import Bar,Calendar,Session,Snapshot
from paperbot_us.audit import ScanArchive,make_scan_record
from paperbot_us.scanner import ScanPolicy


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.m=importlib.import_module('paperbot_us.shadow')
        self.now=datetime(2026,10,23,19,tzinfo=timezone.utc)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.calendar=Calendar([Session.from_alpaca(dict(date='2026-10-23',open='09:30',close='16:00'))],start='2026-10-23',end='2026-10-23')
        universe=UniverseSnapshot(tuple(Asset(s,s,'NASDAQ',True) for s in ('AAPL','MSFT','SPY')),self.now,0)
        snapshots={a.symbol:Snapshot(a.symbol,100,self.now,99.99,100.01,10,10,self.now,100000,self.now.date(),self.now) for a in universe.assets}
        self.scan=ScanArchive(self.root/'scans.sqlite')
        record=make_scan_record(universe=universe,snapshots=snapshots,calendar=self.calendar,as_of=self.now,policy=ScanPolicy())
        self.scan.append(record,full_universe=universe)
        owner=self
        class Client:
            calls=0
            def calendar(self,*args): return owner.calendar
            def bars(self,symbols,*,start,end):
                self.calls+=1;return {s:owner.bars(s) for s in symbols}
        self.client=Client()

    def bars(self,symbol):
        start=self.calendar.sessions[self.now.date()].opens
        count=int((self.now-start).total_seconds()//900)
        return [Bar(symbol,start+timedelta(minutes=15*i),self.now,100,105,95,
            100+i*.1+(-.25 if i%3==0 else 0),200 if i==count-1 else 100) for i in range(count)]

    def run_once(self):
        return self.m.analyze_once(client=self.client,scan_archive=self.scan.path,
            archive=self.root/'decisions.sqlite',publish=self.root/'analysis.json',clock=lambda:self.now)

    def test_real_indicators_are_journalled_and_never_misrepresented_as_model_or_orders(self):
        result=self.run_once();self.assertEqual(result['status'],'analyzing')
        self.assertFalse(result['execution_enabled']);self.assertEqual(result['model_status'],'not_configured')
        self.assertEqual(len(result['decisions']),3)
        for d in result['decisions']:
            self.assertEqual(d['final_action'],'hold');self.assertIn('execution_not_enabled',d['vetoes'])
            self.assertTrue(d['features']['sma_fast_usd']>0);self.assertNotIn('confidence',d)
        self.assertGreater(self.m.ObservationArchive(self.root/'decisions.sqlite').verify()['records'],0)
        self.assertEqual(json.loads((self.root/'analysis.json').read_text()),result)

    def test_repeated_bar_is_deduplicated_and_preserves_first_decision_time(self):
        first=self.run_once();calls=self.client.calls;self.now+=timedelta(seconds=10)
        second=self.run_once();self.assertEqual(self.client.calls,calls)
        self.assertEqual(first['decisions'],second['decisions'])
        self.assertEqual(first['analysis_started_at'],second['analysis_started_at'])

    def test_successful_decision_can_replay_and_modified_features_are_rejected(self):
        self.run_once()
        with closing(sqlite3.connect(self.root/'decisions.sqlite')) as db:
            rows=[json.loads(row[0]) for row in db.execute('SELECT payload FROM observations')]
        decision=next(r for r in rows if r['kind']=='decision')
        self.assertEqual(self.m.replay_decision(decision)['final_action'],'hold')
        decision['features']['rsi_sma']=0
        with self.assertRaisesRegex(ValueError,'mismatch'): self.m.replay_decision(decision)

    def test_data_failure_does_not_create_a_buy_and_audit_failure_clears_decisions(self):
        self.client.bars=lambda *a,**k:(_ for _ in ()).throw(OSError('PRIVATE_FAILURE'))
        value=self.run_once();self.assertTrue(all(d['final_action']=='hold' for d in value['decisions']))
        self.assertNotIn('PRIVATE',json.dumps(value))
        with patch.object(self.m.ObservationArchive,'append',side_effect=OSError('PRIVATE_DISK')):
            value=self.m.analyze_once(client=self.client,scan_archive=self.scan.path,archive=self.root/'new.sqlite',
                publish=self.root/'analysis.json',clock=lambda:self.now)
        self.assertEqual(value['status'],'unavailable');self.assertEqual(value['decisions'],[])

    def test_expired_scan_and_market_close_do_not_fetch_bars(self):
        self.now+=timedelta(seconds=181)
        self.assertEqual(self.run_once()['status'],'unavailable');self.assertEqual(self.client.calls,0)
        self.now=self.now.replace(hour=20)
        self.assertEqual(self.run_once()['status'],'market_closed');self.assertEqual(self.client.calls,0)

    def test_transient_data_failure_can_retry_next_minute_without_rewriting_old_decisions(self):
        real=self.client.bars
        self.client.bars=lambda *a,**k:(_ for _ in ()).throw(OSError('offline'))
        first=self.run_once();self.assertTrue(all(d['reason']=='input_unavailable' for d in first['decisions']))
        self.client.bars=real;self.now+=timedelta(seconds=61)
        second=self.run_once();self.assertTrue(all(d['reason']=='rules_evaluated' for d in second['decisions']))
        self.assertNotEqual(first['decisions'][0]['identity'],second['decisions'][0]['identity'])

    def test_incomplete_or_gapped_series_is_explained_as_warmup_or_bad_data(self):
        original=self.bars
        self.bars=lambda symbol:original(symbol)[-5:]
        result=self.run_once();self.assertTrue(all(d['reason']=='insufficient_bars' for d in result['decisions']))
        self.bars=lambda symbol:original(symbol)[:10]+original(symbol)[11:]
        # New bar identity permits a genuinely new decision, rather than rewriting history.
        self.now+=timedelta(minutes=15)
        universe=UniverseSnapshot((Asset('AAPL','AAPL','NASDAQ',True),),self.now,0)
        snapshot=Snapshot('AAPL',100,self.now,99.99,100.01,10,10,self.now,100000,self.now.date(),self.now)
        self.scan.append(make_scan_record(universe=universe,snapshots={'AAPL':snapshot},calendar=self.calendar,
                         as_of=self.now,policy=ScanPolicy()),full_universe=universe)
        value=self.run_once();self.assertEqual(len(value['decisions']),1)
        self.assertEqual(value['decisions'][0]['reason'],'invalid_or_stale_bars')


if __name__=='__main__': unittest.main()
