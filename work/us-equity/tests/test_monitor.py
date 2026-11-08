from datetime import datetime,timezone
import importlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.m=importlib.import_module('paperbot_us.monitor')
        self.now=datetime(2026,10,23,18,tzinfo=timezone.utc)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.calls=[]
        self.account=dict(id='PRIVATE_ACCOUNT_FIXTURE',currency='USD',status='ACTIVE',cash='10000',equity='10000',
                          trading_blocked=False,account_blocked=False)
        self.positions=[];self.orders=[];self.history=[]

    def reader(self):
        def get(url,headers):
            self.calls.append(url)
            if '/v2/account' in url: return self.account
            if '/v2/positions' in url: return self.positions
            if '/v2/orders?status=open' in url: return self.orders
            if '/v2/orders?status=all' in url: return self.history
            raise AssertionError('unexpected endpoint')
        return self.m.AccountReader('PK'+'T'*20,'T'*32,transport=get,clock=lambda:self.now)

    def run_once(self,**changes):
        args=dict(reader=self.reader(),archive=self.m.ObservationArchive(self.root/'observations.sqlite'),
                  publish=self.root/'status.json',scan_archive=None,clock=lambda:self.now)
        return self.m.observe_once(**{**args,**changes})

    def test_empty_paper_baseline_is_verified_and_journalled_before_publication(self):
        result=self.run_once()
        self.assertTrue(result['accounting_ok']);self.assertEqual(result['equity_usd'],10000)
        self.assertEqual(result['phase'],'observing');self.assertFalse(result['execution_enabled'])
        self.assertIsNone(result['paper_started_at']);self.assertEqual(result['profit_usd'],0)
        self.assertNotIn('PRIVATE',json.dumps(result))
        self.assertEqual(self.m.ObservationArchive(self.root/'observations.sqlite').verify()['records'],1)
        self.assertEqual(json.loads((self.root/'status.json').read_text()),result)

    def test_balance_change_positions_orders_and_blocks_require_reconciliation(self):
        for change in ('cash','position','order','blocked'):
            self.account.update(cash='10000',equity='10000',trading_blocked=False);self.positions=[];self.orders=[]
            if change=='cash': self.account.update(cash='10001',equity='10001')
            if change=='position': self.positions=[{'symbol':'PRIVATE_FIXTURE'}]
            if change=='order': self.orders=[{'id':'PRIVATE_FIXTURE'}]
            if change=='blocked': self.account['trading_blocked']=True
            with self.subTest(change=change):
                result=self.run_once()
                self.assertFalse(result['accounting_ok']);self.assertIsNone(result['profit_usd'])
                self.assertIsNone(result['equity_usd']);self.assertNotIn('PRIVATE',json.dumps(result))

    def test_nonfinite_boolean_overflow_and_account_identifiers_are_not_published(self):
        for value in ('NaN','Infinity',True,'1e100000',-1):
            self.account['cash']=value
            with self.subTest(value=value):
                result=self.run_once()
                self.assertFalse(result['accounting_ok']);self.assertIsNone(result['equity_usd'])
                self.assertNotIn('PRIVATE',json.dumps(result))

    def test_failed_network_and_audit_write_never_publish_normal_zero_profit(self):
        class Broken:
            def read(self): raise OSError('PRIVATE_EXCEPTION_FIXTURE')
        result=self.run_once(reader=Broken())
        self.assertIsNone(result['profit_usd']);self.assertNotIn('PRIVATE',json.dumps(result))
        with patch.object(self.m.ObservationArchive,'append',side_effect=OSError('PRIVATE_DISK_FIXTURE')):
            result=self.run_once()
        self.assertIsNone(result['equity_usd']);self.assertEqual(result['status'],'audit_unavailable')

    def test_restarts_keep_observation_start_without_starting_paper_clock(self):
        first=self.run_once();self.now=self.now.replace(minute=1)
        second=self.run_once()
        self.assertEqual(first['observing_started_at'],second['observing_started_at'])
        self.assertIsNone(second['paper_started_at'])
        self.assertEqual(self.m.ObservationArchive(self.root/'observations.sqlite').verify()['records'],2)

    def test_archive_tampering_prevents_publication_of_available_financial_metrics(self):
        self.run_once()
        from contextlib import closing
        with closing(sqlite3.connect(self.root/'observations.sqlite')) as db:
            db.execute("UPDATE observations SET payload='{}'");db.commit()
        result=self.run_once()
        self.assertFalse(result['accounting_ok']);self.assertEqual(result['status'],'audit_unavailable')

    def test_corruption_in_a_committed_prefix_blocks_later_healthy_publication(self):
        self.run_once();self.now=self.now.replace(minute=1);self.run_once()
        from contextlib import closing
        with closing(sqlite3.connect(self.root/'observations.sqlite')) as db:
            db.execute("UPDATE observations SET payload='{}' WHERE seq=1");db.commit()
        self.now=self.now.replace(minute=2)
        result=self.run_once()
        self.assertFalse(result['accounting_ok']);self.assertIsNone(result['equity_usd'])
        self.assertEqual(result['status'],'audit_unavailable')

    def test_get_scope_and_credential_repr_are_bounded(self):
        result=self.reader().read()
        self.assertNotIn('PRIVATE',repr(result));self.assertNotIn('TTTT',repr(self.reader()))
        self.assertEqual(len(self.calls),4)
        self.assertTrue(all(url.startswith('https://paper-api.alpaca.markets/v2/') for url in self.calls))
        for key in ('AK'+'T'*20,'PK\nBAD'):
            with self.assertRaises(self.m.MonitorError): self.m.AccountReader(key,'T'*32)

    def test_real_holdings_and_filled_orders_are_allowlisted_without_claiming_reconciled_pnl(self):
        self.positions=[dict(asset_id='PRIVATE_ASSET',account_id='PRIVATE_ACCOUNT',asset_class='us_equity',
            symbol='AAPL',side='long',qty='2',avg_entry_price='200',current_price='202',unrealized_pl='4')]
        self.history=[dict(id='PRIVATE_ORDER',client_order_id='PRIVATE_CLIENT',asset_class='us_equity',symbol='AAPL',
            side='buy',qty='2',filled_qty='2',filled_avg_price='200',status='filled',
            submitted_at='2026-10-23T17:00:00Z',filled_at='2026-10-23T17:00:02Z',updated_at='2026-10-23T17:00:02Z')]
        result=self.run_once()
        self.assertTrue(result['details_ok']);self.assertIsNone(result['profit_usd'])
        self.assertEqual(result['holdings'],[dict(symbol='AAPL',quantity=2,average_cost_usd=200,
            reference_price_usd=202,unrealized_usd=4)])
        trade=result['recent_trades'][0]
        self.assertEqual((trade['symbol'],trade['side'],trade['quantity'],trade['average_price_usd']),('AAPL','buy',2,200))
        self.assertEqual(trade['time_kind'],'filled_at')
        self.assertNotIn('PRIVATE',json.dumps(result))
        self.assertEqual(self.m.ObservationArchive(self.root/'observations.sqlite').verify()['records'],1)

    def test_invalid_or_stale_details_are_unavailable_and_do_not_claim_empty_holdings(self):
        base=dict(asset_class='us_equity',symbol='AAPL',side='long',qty='2',avg_entry_price='200',current_price='202',unrealized_pl='4')
        for change in (dict(symbol='<img>'),dict(qty='NaN'),dict(unrealized_pl='-Infinity'),dict(side='short')):
            self.positions=[{**base,**change}]
            with self.subTest(change=change):
                value=self.run_once();self.assertFalse(value['details_ok']);self.assertEqual(value['holdings'],[])
                self.assertIsNone(value['positions_count'])
        self.positions=[base,base]
        self.assertFalse(self.run_once()['details_ok'])

    def test_negative_unrealized_and_partial_fill_use_reported_update_time(self):
        self.positions=[dict(asset_class='us_equity',symbol='MSFT',side='long',qty='1',avg_entry_price='500',current_price='499',unrealized_pl='-1')]
        self.history=[dict(id='PRIVATE_ORDER',asset_class='us_equity',symbol='MSFT',side='sell',qty='2',filled_qty='1',
            filled_avg_price='499',status='partially_filled',submitted_at='2026-10-23T17:00:00Z',filled_at=None,
            updated_at='2026-10-23T17:05:00.123456789Z')]
        result=self.run_once();self.assertTrue(result['details_ok'])
        self.assertEqual(result['holdings'][0]['unrealized_usd'],-1)
        self.assertEqual(result['recent_trades'][0]['time_kind'],'updated_at')
        self.assertEqual(result['recent_trades'][0]['status'],'partially_filled')

    def test_historical_fills_or_bounded_history_cannot_revalidate_empty_account_baseline(self):
        canceled=dict(id='PRIVATE_ORDER',asset_class='us_equity',symbol='AAPL',side='buy',qty='1',filled_qty='0',
            filled_avg_price=None,status='canceled',submitted_at='2026-10-23T17:00:00Z',filled_at=None,updated_at='2026-10-23T17:01:00Z')
        self.history=[{**canceled,'id':'PRIVATE_'+str(i)} for i in range(50)]
        result=self.run_once();self.assertTrue(result['details_ok']);self.assertTrue(result['history_window_limited'])
        self.assertFalse(result['accounting_ok']);self.assertIsNone(result['profit_usd'])
        self.history=[{**canceled,'status':'filled','filled_qty':'1','filled_avg_price':'200','filled_at':'2026-10-23T17:00:01Z'}]
        self.assertFalse(self.run_once()['accounting_ok'])


if __name__=='__main__': unittest.main()
