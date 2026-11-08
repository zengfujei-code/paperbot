from contextlib import closing
from datetime import datetime,timedelta,timezone
from decimal import Decimal
import importlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit,parse_qs

from paperbot_us.accountbook import AccountBook
from paperbot_us.accounting import AccountAnchor
from paperbot_us.broker import PaperBroker,HttpReply,_account_fingerprint
from paperbot_us.data import Calendar,Session
from paperbot_us.orders import OrderLedger
from paperbot_us.risk import OrderIntent,Portfolio,Quote,RiskState,RiskLimits


class ReconcilerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.reconciler'),'paper reconciliation coordinator required')
        self.m=importlib.import_module('paperbot_us.reconciler')
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.now=datetime(2026,10,23,15,tzinfo=timezone.utc);self.claim=self.now-timedelta(seconds=10)
        self.anchor=AccountAnchor(self.now.replace(hour=13,minute=30),'10000','a'*64,
            _account_fingerprint('00000000-0000-4000-8000-000000000050'))
        self.book=AccountBook(self.root/'book.sqlite');self.book.initialize(self.anchor)
        self.ledger=OrderLedger(self.root/'orders.sqlite');self.ledger.initialize()
        calendar=Calendar([Session.from_alpaca(dict(date='2026-10-23',open='09:30',close='16:00'))],start='2026-10-23',end='2026-10-23')
        self.intent=OrderIntent('b'*64,'AAPL','buy',2,'100','98',self.now+timedelta(seconds=30))
        args=dict(portfolio=Portfolio(self.claim,'10000','10000','10000','10000'),quote=Quote('AAPL','100','100.01',self.claim,self.claim),
            state=RiskState(pause_entries=False),limits=RiskLimits(),calendar=calendar,now=self.claim)
        self.ledger.admit(self.intent,**args);self.ledger.prepare_submission(self.intent.intent_id,**args)
        self.uuid='00000000-0000-4000-8000-000000000001';self.calls=[];self.failure_path=None
        self.order=dict(id=self.uuid,client_order_id='pb3-'+'b'*40,asset_class='us_equity',symbol='AAPL',side='buy',
            qty='2',filled_qty='2',filled_avg_price='100',type='limit',order_class='simple',time_in_force='day',
            extended_hours=False,limit_price='100',status='filled',created_at=(self.claim+timedelta(seconds=1)).isoformat(),
            updated_at=(self.claim+timedelta(seconds=5)).isoformat())
        self.fills=[dict(id='20261023145955000::00000000-0000-4000-8000-000000000010',activity_type='FILL',
            symbol='AAPL',side='buy',qty='2',price='100',cum_qty='2',leaves_qty='0',order_id=self.uuid,type='fill',
            transaction_time=self.order['updated_at'],account_id='PRIVATE_ACCOUNT')]
        self.open=[];self.positions=[dict(asset_class='us_equity',symbol='AAPL',side='long',qty='2',qty_available='2',
            current_price='100',market_value='200')]
        self.account=dict(status='ACTIVE',currency='USD',cash='9800',equity='10000',trading_blocked=False,
            account_blocked=False,trade_suspended_by_user=False,id='00000000-0000-4000-8000-000000000050')
        self.broker=PaperBroker('PK'+'T'*20,'T'*32,transport=self.transport,clock=lambda:self.now)
        self.coordinator=self.m.PaperReconciler(broker=self.broker,ledger=self.ledger,book=self.book,
            journal=self.root/'runtime.sqlite',clock=lambda:self.now)

    def transport(self,method,url,headers,body):
        self.assertEqual(method,'GET','reconciliation must never mutate broker state')
        self.calls.append((method,urlsplit(url).path));path=urlsplit(url).path
        if path==self.failure_path: raise OSError('PRIVATE_PROVIDER_ERROR')
        if path=='/v2/account/activities': return HttpReply(200,self.fills)
        if path=='/v2/orders': return HttpReply(200,self.open)
        if path=='/v2/orders:by_client_order_id':
            self.assertEqual(parse_qs(urlsplit(url).query)['client_order_id'],[self.order['client_order_id']] if self.order else ['pb3-'+'b'*40])
            return HttpReply(200,self.order) if self.order else HttpReply(404)
        if path=='/v2/positions': return HttpReply(200,self.positions)
        if path=='/v2/account': return HttpReply(200,self.account)
        raise AssertionError('unsupported read')

    def test_restart_recovers_fill_and_commits_replayable_cash_and_positions_using_only_get(self):
        result=self.coordinator.refresh()
        self.assertTrue(result.ready_for_risk,result.findings);self.assertEqual(result.status,'reconciled')
        self.assertEqual(self.ledger.load(self.intent.intent_id)['state'],'filled')
        self.assertEqual(self.book.replay(result.book_receipt.sequence).expected_cash_usd,Decimal('9800'))
        public=result.public_summary();self.assertEqual(public['equity_usd'],10000);self.assertEqual(public['positions_count'],1)
        self.assertNotIn('PRIVATE',json.dumps(public));self.assertNotIn(self.uuid,json.dumps(public))
        self.assertIsNotNone(result.audit_sha256);self.assertFalse(public['execution_enabled'])
        self.now+=timedelta(seconds=2);again=self.coordinator.refresh()
        self.assertTrue(again.ready_for_risk);self.assertEqual(self.book.verify()['activities'],1)

    def test_missing_fill_blocks_then_late_activity_recovers_without_rewriting_failure(self):
        original=self.fills;self.fills=[];first=self.coordinator.refresh()
        self.assertFalse(first.ready_for_risk);self.assertIn('missing_or_extra_order_fills',first.findings)
        self.assertIsNone(first.public_summary()['profit_usd'])
        self.fills=original;self.now+=timedelta(seconds=2);second=self.coordinator.refresh()
        self.assertTrue(second.ready_for_risk);self.assertFalse(self.book.replay(first.book_receipt.sequence).reconciled)

    def test_unknown_submission_is_not_cleared_by_404_or_resent(self):
        self.order=None;self.fills=[];self.positions=[];self.account['cash']='10000'
        for _ in range(2):
            result=self.coordinator.refresh();self.assertFalse(result.ready_for_risk)
            self.assertIn('unconfirmed_submission',result.findings);self.now+=timedelta(seconds=1)
        self.assertEqual(self.ledger.load(self.intent.intent_id)['state'],'submission_unknown')
        self.assertTrue(all(method=='GET' for method,_ in self.calls))

    def test_foreign_open_order_and_inconsistent_available_shares_block_risk(self):
        foreign={**self.order,'id':'00000000-0000-4000-8000-000000000002','client_order_id':'manual-fixture',
            'filled_qty':'0','filled_avg_price':None,'status':'new'}
        self.open=[foreign];first=self.coordinator.refresh()
        self.assertFalse(first.ready_for_risk);self.assertIn('unrecognized_broker_order',first.findings)
        self.open=[];self.positions[0]['qty_available']='1';self.now+=timedelta(seconds=1)
        second=self.coordinator.refresh();self.assertFalse(second.ready_for_risk)
        self.assertIn('available_share_mismatch',second.findings)

    def test_lookup_payload_mismatch_never_updates_owned_order(self):
        self.order['limit_price']='101';result=self.coordinator.refresh()
        self.assertFalse(result.ready_for_risk);self.assertIn('broker_order_mismatch',result.findings)
        self.assertEqual(self.ledger.load(self.intent.intent_id)['state'],'submission_unknown')

    def test_fill_before_actual_broker_creation_is_not_valid_accounting(self):
        self.fills[0]['transaction_time']=(self.claim+timedelta(milliseconds=500)).isoformat()
        value=self.coordinator.refresh();self.assertFalse(value.ready_for_risk)
        self.assertIn('fill_order_time_mismatch',value.findings)

    def test_same_filled_shares_with_conflicting_value_or_watermark_is_not_a_consistent_snapshot(self):
        self.order.update(status='partially_filled',filled_qty='1',filled_avg_price='99')
        self.fills[0].update(qty='1',price='99',cum_qty='1',leaves_qty='1')
        self.positions[0].update(qty='1',qty_available='1',market_value='100')
        self.account.update(cash='9901',equity='10001')
        self.open=[{**self.order,'filled_avg_price':'100'}]
        value=self.coordinator.refresh();self.assertFalse(value.ready_for_risk)
        self.assertIn('non_atomic_order_snapshot',value.findings)
        self.open=[{**self.order,'updated_at':(self.claim+timedelta(seconds=4)).isoformat()}]
        self.now+=timedelta(seconds=1);value=self.coordinator.refresh()
        self.assertFalse(value.ready_for_risk);self.assertIn('non_atomic_order_snapshot',value.findings)

    def test_persistence_delay_cannot_return_stale_inputs_as_ready(self):
        original=self.book.refresh
        def delayed(**kwargs):
            result=original(**kwargs);self.now+=timedelta(seconds=31);return result
        with patch.object(self.book,'refresh',side_effect=delayed): value=self.coordinator.refresh()
        self.assertFalse(value.ready_for_risk);self.assertIn('stale_reconciliation',value.findings)
        self.assertIsNone(value.public_summary()['profit_usd'])

    def test_clock_regression_after_book_commit_cannot_return_future_account_evidence_as_ready(self):
        original=self.transport;refresh=self.book.refresh
        def transport(method,url,headers,body):
            value=original(method,url,headers,body)
            if urlsplit(url).path=='/v2/account': self.now+=timedelta(seconds=2)
            return value
        def regressed(**kwargs):
            value=refresh(**kwargs);self.now-=timedelta(seconds=1);return value
        self.broker._transport=transport
        with patch.object(self.book,'refresh',side_effect=regressed): value=self.coordinator.refresh()
        self.assertFalse(value.ready_for_risk);self.assertIsNone(value.public_summary()['profit_usd'])

    def test_changed_paper_account_cannot_reuse_an_old_experiment_anchor(self):
        self.account['id']='00000000-0000-4000-8000-000000000099'
        value=self.coordinator.refresh();self.assertFalse(value.ready_for_risk)
        self.assertIn('account_identity_mismatch',value.findings)
        self.assertIsNone(value.public_summary()['profit_usd'])

    def test_lookup_receipt_is_included_in_clock_floor_even_after_later_time_regression(self):
        original=self.transport;rewound=False
        def transport(method,url,headers,body):
            nonlocal rewound
            value=original(method,url,headers,body);path=urlsplit(url).path
            if path=='/v2/orders:by_client_order_id': self.now+=timedelta(seconds=2)
            if path=='/v2/positions' and not rewound: self.now-=timedelta(seconds=1);rewound=True
            return value
        self.broker._transport=transport;value=self.coordinator.refresh()
        self.assertFalse(value.ready_for_risk);self.assertIsNone(value.public_summary()['profit_usd'])

    def test_provider_disk_and_corruption_failures_never_publish_verified_money(self):
        self.failure_path='/v2/account';value=self.coordinator.refresh()
        self.assertEqual(value.status,'unavailable');self.assertFalse(value.ready_for_risk)
        self.assertIsNone(value.public_summary()['cash_usd']);self.assertNotIn('PRIVATE',json.dumps(value.public_summary()))
        self.failure_path=None
        with patch.object(self.book,'_append_snapshot',side_effect=OSError('PRIVATE_DISK')):
            value=self.coordinator.refresh();self.assertFalse(value.ready_for_risk);self.assertIsNone(value.book_receipt)
        with closing(sqlite3.connect(self.ledger.path)) as db:
            db.execute("UPDATE events SET digest='broken' WHERE sequence=1");db.commit()
        before=len(self.calls);self.assertEqual(self.coordinator.refresh().status,'unavailable');self.assertEqual(len(self.calls),before)

    def test_journal_failure_hides_even_a_committed_book_receipt(self):
        with patch.object(self.m.ObservationArchive,'append',side_effect=OSError('PRIVATE_JOURNAL')):
            result=self.coordinator.refresh()
        self.assertFalse(result.ready_for_risk);self.assertIsNone(result.book_receipt)
        self.assertIsNone(result.public_summary()['profit_usd']);self.assertEqual(self.book.verify()['snapshots'],1)

    def test_mid_read_position_change_or_user_suspension_is_not_ready(self):
        original=self.transport;count=0
        def transport(method,url,headers,body):
            nonlocal count
            if urlsplit(url).path=='/v2/positions':
                count+=1
                if count==2: return HttpReply(200,[{**self.positions[0],'qty':'1','qty_available':'1'}])
            return original(method,url,headers,body)
        self.broker._transport=transport;first=self.coordinator.refresh()
        self.assertFalse(first.ready_for_risk);self.assertIn('position_snapshot_mismatch',first.findings)
        self.broker._transport=original;self.account['trade_suspended_by_user']=True;self.now+=timedelta(seconds=2)
        self.assertIn('account_blocked',self.coordinator.refresh().findings)

    def test_private_audit_retains_book_and_order_snapshot_links_without_raw_identifiers(self):
        value=self.coordinator.refresh()
        with closing(sqlite3.connect(self.root/'runtime.sqlite')) as db:
            row=json.loads(db.execute('SELECT payload FROM observations ORDER BY seq DESC LIMIT 1').fetchone()[0])
        self.assertEqual(row['book_sha256'],value.book_receipt.digest)
        self.assertEqual(row['orders_sha256'],value.orders_sha256)
        self.assertNotIn('PRIVATE',json.dumps(row));self.assertNotIn(self.uuid,json.dumps(row))


if __name__=='__main__': unittest.main()
