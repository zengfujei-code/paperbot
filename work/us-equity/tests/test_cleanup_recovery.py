from copy import deepcopy
from datetime import timedelta
import importlib,importlib.util,unittest
from unittest.mock import patch
from urllib.parse import urlsplit,parse_qs

import test_cleanup_execution
from paperbot_us.broker import BrokerError,HttpReply
from paperbot_us.orders import _hash,_json


class CleanupRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.cleanup_recovery'),
            'owned cleanup cancellation and pending-state recovery required')
        self.m=importlib.import_module('paperbot_us.cleanup_recovery')
        self.e=test_cleanup_execution.CleanupExecutionTests();self.e.setUp();self.addCleanup(self.e.doCleanups)
        self.e.stage();self.e.submit();self.n=self.e.n;self.ledger=self.e.ledger;self.identity=self.e.intent.intent_id
        self.n.now+=timedelta(seconds=25);self.deletes=[];self.settle=True;self.failure=None
        self.after_delete=None;self.missing=False;self.reply_status=204
        self.old_transport=self.n.broker._transport;self.n.broker._transport=self.transport
        self.gateway=self.m.CleanupRecovery(self.n.broker)

    def transport(self,method,url,headers,body):
        parsed=urlsplit(url)
        self.assertIn(method,('GET','DELETE'),'recovery cannot POST a replacement')
        if method=='DELETE':
            self.assertEqual(parsed.path,'/v2/orders/'+self.e.row['id']);self.deletes.append(parsed.path)
            self.assertEqual(self.ledger.load(self.identity)['cleanup_cancel']['outcome'],'claimed')
            if self.after_delete:self.after_delete()
            elif self.settle:
                self.e.row.update(status='canceled',updated_at=self.n.now.isoformat())
                self.n.visible=[]
                if self.n.positions:self.n.positions[0]['qty_available']=self.n.positions[0]['qty']
            if self.failure:raise self.failure
            return HttpReply(self.reply_status)
        if (self.missing and parsed.path=='/v2/orders:by_client_order_id'
                and parse_qs(parsed.query).get('client_order_id')==[self.e.row['client_order_id']]):return HttpReply(404)
        return self.old_transport(method,url,headers,body)

    def cancel(self,**changes):
        return self.gateway.cancel(self.ledger,self.identity,controls=self.e.control_store,
            control_snapshot=changes.get('control_snapshot',self.e.control))

    def partial(self,status='partially_filled',qty=2):
        self.e.row.update(status=status,filled_qty=str(qty),filled_avg_price='150',updated_at=self.n.now.isoformat())
        self.n.visible=[deepcopy(self.e.row)]
        remaining=5-qty
        self.n.positions=[] if not remaining else [{**self.n.positions[0],
            'qty':str(remaining),'qty_available':'0','market_value':str(150*remaining)}]
        self.n.account.update(cash=str(9250+150*qty),equity='10000')
        self.n.fills.append(self.n.fill('00000000-0000-4000-8000-000000000083','sell',str(qty),'150',
            self.e.row['id'],int((self.n.now-self.n.f.now).total_seconds()),str(qty),str(5-qty)))

    def test_expired_cleanup_cancel_requires_actual_terminal_observation(self):
        result=self.cancel()
        self.assertEqual(result.status,'cleanup_terminal');self.assertEqual(result.remaining_order_quantity,5)
        self.assertEqual(self.ledger.load(self.identity)['state'],'canceled');self.assertEqual(len(self.deletes),1)
        self.assertFalse(result.public_summary()['ready_for_execution'])
        self.assertIn('reconcile_account',result.required_actions)

    def test_ack_without_terminal_observation_never_frees_outstanding_shares(self):
        self.settle=False;result=self.cancel()
        self.assertEqual(result.status,'cleanup_cancel_pending')
        self.gateway=self.m.CleanupRecovery(self.n.broker);self.cancel()
        self.assertEqual(len(self.deletes),1)
        self.assertEqual(self.e.f.m.project_groups(self.ledger.all_orders())[self.e.f.family].reserved_quantity,5)

    def test_partial_fill_cancel_keeps_fills_and_full_get_accounting_proves_remaining(self):
        self.partial();result=self.cancel()
        self.assertEqual(result.filled_quantity,2);self.assertEqual(result.remaining_order_quantity,3)
        self.assertEqual(self.ledger.load(self.identity)['filled_quantity'],2)
        view=self.e.f.prepare()
        self.assertEqual(view.remaining_quantity,3);self.assertEqual(view.available_quantity,3)
        self.assertEqual(self.e.f.book.snapshot().receipt.result.profit_usd,0)

    def test_older_terminal_response_cannot_replace_a_newer_working_observation(self):
        self.partial();self.gateway.recover(self.ledger,self.identity);before=self.ledger.all_orders()
        self.n.now+=timedelta(seconds=1)
        self.e.row.update(status='canceled',filled_qty='1',updated_at=(self.n.now-timedelta(seconds=2)).isoformat())
        result=self.gateway.recover(self.ledger,self.identity)
        self.assertEqual(result.status,'cleanup_unknown')
        after=self.ledger.load(self.identity);old=next(r for r in before if r['intent_id']==self.identity)
        for key in ('state','filled_quantity','filled_notional_usd','broker_updated_at','cleanup_observed_at'):
            self.assertEqual(after[key],old[key])
        self.assertEqual(after['cleanup_lookup_state'],'unknown')

    def test_late_terminal_fill_is_reconciled_without_resurrecting_the_canceled_order(self):
        self.partial();self.cancel();self.e.f.prepare()
        self.n.now+=timedelta(seconds=1)
        self.e.row.update(status='canceled',filled_qty='3',updated_at=self.n.now.isoformat())
        self.n.positions[0].update(qty='2',qty_available='2',market_value='300')
        self.n.account.update(cash='9700',equity='10000')
        self.n.fills.append(self.n.fill('00000000-0000-4000-8000-000000000084','sell','1','150',
            self.e.row['id'],int((self.n.now-self.n.f.now).total_seconds()),'3','2'))
        cycle=self.n.reconciler.refresh()
        self.assertTrue(cycle.book_receipt.result.reconciled,cycle.findings)
        self.assertEqual(self.ledger.load(self.identity)['filled_quantity'],3)
        self.assertEqual(self.ledger.load(self.identity)['state'],'canceled')
        view=self.e.f.prepare(cycle=cycle)
        self.assertEqual(view.remaining_quantity,2);self.assertEqual(view.available_quantity,2)

    def test_provider_pending_cancel_is_not_repeated_or_reported_terminal(self):
        self.e.row.update(status='pending_cancel',updated_at=self.n.now.isoformat())
        self.n.visible=[deepcopy(self.e.row)]
        result=self.cancel()
        self.assertEqual(result.status,'cleanup_cancel_pending');self.assertEqual(self.deletes,[])
        view=self.e.f.prepare()
        self.assertEqual(view.available_quantity,0)
        self.assertEqual(self.ledger.load(self.identity)['cleanup_broker_status'],'pending_cancel')

    def test_full_quantity_with_nonterminal_pending_cancel_remains_unconfirmed(self):
        self.partial(status='pending_cancel',qty=5)
        result=self.gateway.recover(self.ledger,self.identity)
        self.assertEqual(result.status,'cleanup_cancel_pending');self.assertEqual(result.filled_quantity,5)
        cycle=self.n.reconciler.refresh();self.assertFalse(cycle.ready_for_risk)
        self.assertTrue(cycle.book_receipt.result.reconciled)

    def test_unknown_delete_keeps_claim_and_restart_only_reads(self):
        self.settle=False;self.failure=OSError('PRIVATE_PROVIDER');result=self.cancel()
        self.assertEqual(result.status,'cleanup_cancel_pending')
        self.assertNotIn('PRIVATE',str(result.public_summary()))
        self.failure=None;self.gateway=self.m.CleanupRecovery(self.n.broker)
        self.gateway.recover(self.ledger,self.identity);self.cancel()
        self.assertEqual(len(self.deletes),1)

    def test_latest_404_does_not_promote_cached_order_or_free_its_reservation(self):
        before=self.ledger.all_orders();self.missing=True;result=self.cancel()
        self.assertEqual(result.status,'cleanup_unknown');self.assertIsNone(result.filled_quantity)
        self.assertEqual(self.deletes,[])
        after=self.ledger.load(self.identity);old=next(r for r in before if r['intent_id']==self.identity)
        for key in ('state','filled_quantity','filled_notional_usd','broker_order_id','native_cleanup'):
            self.assertEqual(after[key],old[key])
        self.assertEqual(after['cleanup_lookup_state'],'unknown')
        self.assertEqual(self.e.f.m.project_groups(self.ledger.all_orders())[self.e.f.family].reserved_quantity,5)

    def test_overlapping_terminal_success_cannot_clear_a_newer_unknown_lookup(self):
        self.cancel();original=self.n.broker._transport;entered=False
        def transport(method,url,headers,body):
            nonlocal entered
            parsed=urlsplit(url)
            if (not entered and parsed.path=='/v2/orders:by_client_order_id'
                    and parse_qs(parsed.query).get('client_order_id')==[self.e.row['client_order_id']]):
                entered=True;response=original(method,url,headers,body)
                self.n.now+=timedelta(seconds=1);self.missing=True
                self.assertEqual(self.gateway.recover(self.ledger,self.identity).status,'cleanup_unknown')
                self.missing=False;return response
            return original(method,url,headers,body)
        self.n.broker._transport=transport
        self.assertEqual(self.gateway.recover(self.ledger,self.identity).status,'cleanup_unknown')
        self.assertEqual(self.ledger.load(self.identity)['cleanup_lookup_state'],'unknown')

    def test_valid_working_cleanup_survives_entry_pause_but_current_halt_can_cancel(self):
        self.n.now-=timedelta(seconds=20);result=self.cancel()
        self.assertEqual(result.status,'cleanup_working');self.assertEqual(self.deletes,[])
        self.e.control=self.e.control_store.update(expected_revision=1,pause_entries=True,halt_orders=True,
            reason='operator_halt',now=self.n.now)
        self.cancel();self.assertEqual(len(self.deletes),1)

    def test_stale_control_revision_cannot_cancel_even_an_expired_cleanup(self):
        self.e.control_store.update(expected_revision=1,pause_entries=True,halt_orders=True,
            reason='operator_halt',now=self.n.now)
        result=self.cancel();self.assertEqual(result.status,'control_changed');self.assertEqual(self.deletes,[])

    def test_full_fill_race_is_preserved_without_any_replacement_order(self):
        self.after_delete=lambda:self.partial(status='filled',qty=5)
        result=self.cancel()
        self.assertEqual(result.status,'cleanup_terminal');self.assertEqual(result.filled_quantity,5)
        self.assertEqual(self.e.f.m.project_groups(self.ledger.all_orders())[self.e.f.family].remaining_quantity,0)
        self.assertEqual(len(self.deletes),1)

    def test_delay_after_claim_is_known_unsent_but_order_itself_stays_reserved(self):
        original=self.m.claim_cancel
        def delayed(*args,**kwargs):
            result=original(*args,**kwargs);self.n.now+=timedelta(seconds=31);return result
        with patch.object(self.m,'claim_cancel',side_effect=delayed):self.cancel()
        record=self.ledger.load(self.identity)
        self.assertEqual(self.deletes,[]);self.assertEqual(record['cleanup_cancel']['outcome'],'not_sent')
        self.assertIsNone(record['cancel_requested_at']);self.assertEqual(record['state'],'open')
        self.assertEqual(self.e.f.m.project_groups(self.ledger.all_orders())[self.e.f.family].reserved_quantity,5)

    def test_result_write_failure_retains_one_shot_claim_then_get_resolves_terminal(self):
        original=self.ledger._persist
        def persist(db,record,kind,at):
            if kind=='cleanup_cancel_result':raise OSError('PRIVATE_DISK')
            return original(db,record,kind,at)
        with patch.object(self.ledger,'_persist',side_effect=persist):
            with self.assertRaises(BrokerError):self.cancel()
        result=self.gateway.recover(self.ledger,self.identity)
        self.assertEqual(result.status,'cleanup_terminal');self.cancel();self.assertEqual(len(self.deletes),1)

    def test_claim_write_failure_sends_no_cancel_and_keeps_original_order(self):
        original=self.ledger._persist
        def persist(db,record,kind,at):
            if kind=='cleanup_cancel_claim':raise OSError('PRIVATE_DISK')
            return original(db,record,kind,at)
        with patch.object(self.ledger,'_persist',side_effect=persist):
            with self.assertRaises(BrokerError):self.cancel()
        self.assertEqual(self.deletes,[]);self.assertIsNone(self.ledger.load(self.identity)['cancel_requested_at'])

    def test_concurrent_submission_cannot_be_reported_as_retired_unsent(self):
        fresh=test_cleanup_execution.CleanupExecutionTests();fresh.setUp();self.addCleanup(fresh.doCleanups)
        fresh.stage();original=fresh.ledger.load;first=True
        def load(identity):
            nonlocal first
            record=original(identity)
            if first:
                first=False;fresh.submit();fresh.n.now+=timedelta(seconds=25)
            return record
        with patch.object(fresh.ledger,'load',side_effect=load):
            result=self.m.CleanupRecovery(fresh.n.broker).cancel(fresh.ledger,fresh.intent.intent_id,
                controls=fresh.control_store,control_snapshot=fresh.control)
        self.assertEqual(original(fresh.intent.intent_id)['state'],'open')
        self.assertEqual(result.status,'cleanup_unknown')
        self.assertIsNone(result.remaining_order_quantity)


if __name__=='__main__':unittest.main()
