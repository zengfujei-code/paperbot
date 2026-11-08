from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
import importlib
import importlib.util
from urllib.parse import urlsplit
import unittest
from unittest.mock import patch

import test_native_orders
from paperbot_us.accountbook import AccountBook
from paperbot_us.accounting import AccountAnchor
from paperbot_us.broker import PaperBroker,HttpReply,_account_fingerprint


class NativeReconcilerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.native_reconciler'),
                             'group-aware GET reconciliation required')
        self.m=importlib.import_module('paperbot_us.native_reconciler')
        self.f=test_native_orders.NativeOrderLedgerTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.f.claim();self.f.full_parent();self.now=self.f.now+timedelta(seconds=10)
        self.account_id='00000000-0000-4000-8000-000000000050'
        self.book=AccountBook(self.f.f.path.with_name('book.sqlite'))
        self.book.initialize(AccountAnchor(self.f.now-timedelta(seconds=1),'10000','1'*64,_account_fingerprint(self.account_id)))
        self.calls=[];self.visible=[deepcopy(self.f.child)];self.missing=False
        self.positions=[dict(asset_class='us_equity',symbol='AAPL',side='long',qty='5',qty_available='0',
            current_price='150',market_value='750')]
        self.account=dict(id=self.account_id,status='ACTIVE',currency='USD',cash='9250',equity='10000',
            trading_blocked=False,account_blocked=False)
        self.fills=[self.fill('00000000-0000-4000-8000-000000000010','buy','5','150',self.f.parent_id,1,'5','0')]
        self.broker=PaperBroker('PK'+'T'*20,'T'*32,transport=self.transport,clock=lambda:self.now)
        self.reconciler=self.m.NativePaperReconciler(broker=self.broker,ledger=self.f.ledger,book=self.book,
            journal=self.f.f.path.with_name('reconciliation.sqlite'),calendar=self.f.f.calendar,clock=lambda:self.now)

    def fill(self,identity,side,qty,price,oid,second,cum,leaves):
        return dict(id='20261023150000000::'+identity,activity_type='FILL',symbol='AAPL',side=side,qty=qty,
            price=price,cum_qty=cum,leaves_qty=leaves,order_id=oid,type='fill',
            transaction_time=(self.f.now+timedelta(seconds=second)).isoformat())

    def transport(self,method,url,headers,body):
        self.assertEqual(method,'GET','group reconciliation must never send/cancel')
        path=urlsplit(url).path;self.calls.append((method,path))
        if path=='/v2/account/activities': return HttpReply(200,self.fills)
        if path=='/v2/orders': return HttpReply(200,self.visible)
        if path=='/v2/orders:by_client_order_id':
            return HttpReply(404) if self.missing else HttpReply(200,{**self.f.parent,'legs':None})
        if path=='/v2/orders/'+self.f.parent_id: return HttpReply(200,self.f.parent)
        if path=='/v2/positions': return HttpReply(200,self.positions)
        if path=='/v2/account': return HttpReply(200,self.account)
        raise AssertionError('unexpected path')

    def test_completed_entry_and_live_stop_match_accounting_available_shares_and_replay(self):
        cycle=self.reconciler.refresh()
        self.assertTrue(cycle.ready_for_risk,cycle.findings)
        self.assertEqual(cycle.book_receipt.result.expected_cash_usd,Decimal('9250'))
        self.assertEqual(self.book.replay(cycle.book_receipt.sequence),cycle.book_receipt.result)
        self.assertEqual(cycle.positions_available[0].available_quantity,0)
        self.assertFalse(cycle.public_summary()['execution_enabled'])
        self.assertTrue(all(method=='GET' for method,_ in self.calls))

    def test_missing_or_held_stop_never_passes_protection_even_when_money_reconciles(self):
        for status in ('held','pending_cancel','canceled','rejected'):
            with self.subTest(status=status):
                case=NativeReconcilerTests();case.setUp();self.addCleanup(case.doCleanups)
                case.f.child['status']=status
                case.visible=[] if status in ('canceled','rejected','held') else [deepcopy(case.f.child)]
                case.positions[0]['qty_available']='5' if not case.visible else '0'
                cycle=case.reconciler.refresh()
                self.assertFalse(cycle.ready_for_risk)
                self.assertTrue(any(s.startswith('native_protection_') for s in cycle.findings),cycle.findings)

    def test_missing_child_is_not_protection_despite_a_reconciled_entry(self):
        self.f.parent['legs']=[];self.visible=[];self.positions[0]['qty_available']='5'
        cycle=self.reconciler.refresh()
        self.assertFalse(cycle.ready_for_risk)
        self.assertIn('native_protection_protection_evidence_missing',cycle.findings)
        self.assertTrue(cycle.book_receipt.result.reconciled)

    def test_closed_group_without_exit_completion_time_remains_unresolved(self):
        self.f.child.update(status='filled',filled_qty='5',filled_avg_price='130',
            updated_at=(self.f.now+timedelta(seconds=4)).isoformat(),filled_at=None)
        self.visible=[];self.positions=[];self.account.update(cash='9900',equity='9900')
        self.fills.append(self.fill('00000000-0000-4000-8000-000000000011','sell','5','130',self.f.child_id,4,'5','0'))
        cycle=self.reconciler.refresh()
        self.assertFalse(cycle.ready_for_risk)
        self.assertIn('native_protection_exit_fill_time_unavailable',cycle.findings)

    def test_flat_pending_parent_cannot_change_its_native_order_class(self):
        self.f.parent.update(status='new',filled_qty='0',filled_avg_price=None,filled_at=None)
        self.f.child['status']='held'
        self.visible=[{**self.f.parent,'order_class':'simple','legs':None}]
        self.positions=[];self.fills=[];self.account.update(cash='10000',equity='10000')
        cycle=self.reconciler.refresh()
        self.assertFalse(cycle.ready_for_risk)
        self.assertIn('non_atomic_order_snapshot',cycle.findings)

    def test_partial_entry_is_accounted_but_not_treated_as_protected(self):
        self.f.parent.update(status='partially_filled',filled_qty='2',filled_avg_price='150',filled_at=None)
        self.f.child['status']='held';self.visible=[{**self.f.parent,'legs':None}]
        self.positions[0].update(qty='2',qty_available='2',market_value='300')
        self.fills[0].update(qty='2',cum_qty='2',leaves_qty='3')
        self.account.update(cash='9700',equity='10000')
        cycle=self.reconciler.refresh()
        self.assertFalse(cycle.ready_for_risk)
        self.assertIn('native_protection_unprotected_partial_entry',cycle.findings)
        self.assertTrue(cycle.book_receipt.result.reconciled,cycle.book_receipt.result.violations)
        self.assertEqual(self.f.ledger.load(self.f.intent.intent_id)['state'],'partially_filled')

    def test_partial_and_final_gap_stop_fills_reconcile_realized_loss_without_double_sell(self):
        self.f.child.update(status='partially_filled',filled_qty='2',filled_avg_price='130',
            updated_at=(self.f.now+timedelta(seconds=4)).isoformat())
        self.visible=[deepcopy(self.f.child)]
        self.fills.append(self.fill('00000000-0000-4000-8000-000000000011','sell','2','130',self.f.child_id,4,'2','3'))
        self.positions[0].update(qty='3',qty_available='0',market_value='450')
        self.account.update(cash='9510',equity='9960')
        first=self.reconciler.refresh();self.assertTrue(first.ready_for_risk,first.findings)
        self.assertEqual(first.book_receipt.result.realized_usd,Decimal('-40'))
        self.f.child.update(status='filled',filled_qty='5',filled_avg_price='130',
            filled_at=(self.f.now+timedelta(seconds=5)).isoformat(),updated_at=(self.f.now+timedelta(seconds=5)).isoformat())
        self.visible=[];self.positions=[];self.account.update(cash='9900',equity='9900')
        self.fills.append(self.fill('00000000-0000-4000-8000-000000000012','sell','3','130',self.f.child_id,5,'5','0'))
        self.now+=timedelta(seconds=1)
        final=self.reconciler.refresh();self.assertTrue(final.ready_for_risk,final.findings)
        self.assertEqual(final.book_receipt.result.profit_usd,Decimal('-100'))
        self.assertEqual(self.book.replay(first.book_receipt.sequence).realized_usd,Decimal('-40'))

    def test_flat_order_disagreement_foreign_orders_or_reserved_share_mismatch_veto(self):
        self.visible[0]['updated_at']=(self.f.now+timedelta(seconds=1)).isoformat()
        first=self.reconciler.refresh();self.assertIn('non_atomic_order_snapshot',first.findings)
        self.visible=[deepcopy(self.f.child),{**self.f.child,'id':'00000000-0000-4000-8000-000000000099'}]
        self.now+=timedelta(seconds=1);second=self.reconciler.refresh()
        self.assertIn('unrecognized_broker_order',second.findings)
        self.visible=[deepcopy(self.f.child)];self.positions[0]['qty_available']='5'
        self.now+=timedelta(seconds=1);third=self.reconciler.refresh()
        self.assertIn('available_share_mismatch',third.findings)

    def test_404_does_not_erase_unknown_parent_or_create_fills(self):
        self.missing=True;self.visible=[];self.positions=[];self.fills=[]
        self.account.update(cash='10000',equity='10000')
        cycle=self.reconciler.refresh()
        self.assertFalse(cycle.ready_for_risk);self.assertIn('unconfirmed_submission',cycle.findings)
        self.assertEqual(self.f.ledger.load(self.f.intent.intent_id)['state'],'submission_unknown')

    def test_disk_failure_and_post_commit_clock_delay_do_not_publish_ready_evidence(self):
        with patch.object(self.book,'_append_snapshot',side_effect=OSError('PRIVATE_DISK')):
            bad=self.reconciler.refresh()
        self.assertFalse(bad.ready_for_risk);self.assertIsNone(bad.book_receipt)
        original=self.book.refresh
        def delayed(**args):
            receipt=original(**args);self.now+=timedelta(seconds=31);return receipt
        with patch.object(self.book,'refresh',side_effect=delayed): bad=self.reconciler.refresh()
        self.assertFalse(bad.ready_for_risk);self.assertIn('stale_reconciliation',bad.findings)
        self.assertIsNone(bad.public_summary()['profit_usd'])

    def test_account_read_crossing_session_close_rechecks_protection_at_return_time(self):
        self.now=self.now.replace(hour=19,minute=59,second=59)
        original=self.transport
        def transport(method,url,headers,body):
            reply=original(method,url,headers,body)
            if urlsplit(url).path=='/v2/account': self.now+=timedelta(seconds=2)
            return reply
        self.broker._transport=transport
        cycle=self.reconciler.refresh()
        self.assertFalse(cycle.ready_for_risk)
        self.assertIn('native_protection_out_of_session_exposure',cycle.findings)

    def test_journal_commit_crossing_session_close_cannot_return_old_protected_state(self):
        self.now=self.now.replace(hour=19,minute=59,second=59)
        original=self.reconciler.journal.append
        def append(report):
            result=original(report);self.now+=timedelta(seconds=2);return result
        with patch.object(self.reconciler.journal,'append',side_effect=append): cycle=self.reconciler.refresh()
        self.assertFalse(cycle.ready_for_risk)
        self.assertIn('native_protection_out_of_session_exposure',cycle.findings)

    def test_pending_entry_expiring_during_reads_requests_cancellation_and_vetoes(self):
        self.f.parent.update(status='new',filled_qty='0',filled_avg_price=None,filled_at=None)
        self.f.child['status']='held';self.visible=[{**self.f.parent,'legs':None}]
        self.fills=[];self.positions=[];self.account.update(cash='10000',equity='10000')
        self.now=self.f.now+timedelta(seconds=19)
        original=self.transport
        def transport(method,url,headers,body):
            result=original(method,url,headers,body)
            if urlsplit(url).path=='/v2/account': self.now+=timedelta(seconds=2)
            return result
        self.broker._transport=transport;cycle=self.reconciler.refresh()
        self.assertFalse(cycle.ready_for_risk)
        self.assertIn('native_protection_entry_requires_cancellation',cycle.findings)
        self.assertIn('cancel_unfilled_entry',cycle.native_states[0]['required_actions'])


if __name__=='__main__': unittest.main()
