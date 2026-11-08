from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta
import importlib
import importlib.util
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

import test_native_orders
from paperbot_us.broker import PaperBroker,HttpReply,BrokerError
from paperbot_us.controls import ControlJournal


class NativeEntryRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.native_recovery'),
                             'durable native entry cancellation/recovery required')
        self.m=importlib.import_module('paperbot_us.native_recovery')
        self.f=test_native_orders.NativeOrderLedgerTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.f.claim();self.now=self.f.now+timedelta(seconds=25)
        self.calls=[];self.delete_reply=HttpReply(204);self.settle=True;self.after_delete=None
        self.missing=False;self.get_failure_after_delete=False
        self.store=ControlJournal(self.f.f.path.with_name('controls.sqlite'))
        self.snapshot=self.store.initialize(now=self.f.now)
        self.broker=PaperBroker('PK'+'T'*20,'T'*32,transport=self.transport,clock=lambda:self.now)
        self.gateway=self.m.NativeEntryRecovery(self.broker)

    def transport(self,method,url,headers,body):
        path=urlsplit(url).path;self.calls.append((method,path))
        self.assertIsNone(body)
        self.now+=timedelta(milliseconds=100)
        if method=='GET':
            if self.get_failure_after_delete and any(m=='DELETE' for m,_ in self.calls):raise OSError('PRIVATE_READ')
            if self.missing:return HttpReply(404)
            if path=='/v2/orders:by_client_order_id':return HttpReply(200,{**deepcopy(self.f.parent),'legs':None})
            self.assertEqual(path,'/v2/orders/'+self.f.parent_id)
            return HttpReply(200,deepcopy(self.f.parent))
        self.assertEqual(method,'DELETE','recovery must not buy or submit a replacement sell')
        self.assertEqual(path,'/v2/orders/'+self.f.parent_id,'never cancel a protective child here')
        record=self.f.ledger.load(self.f.intent.intent_id)
        self.assertIsNotNone(record['cancel_requested_at'])
        self.assertEqual(record['native_stop']['entry_cancel']['outcome'],'claimed')
        if self.after_delete:self.after_delete()
        elif self.settle:
            self.f.parent.update(status='canceled',updated_at=self.now.isoformat())
            if self.f.child['status'] not in ('canceled','expired','rejected','filled','replaced'):
                self.f.child.update(status='canceled',updated_at=self.now.isoformat())
        if isinstance(self.delete_reply,Exception):raise self.delete_reply
        return self.delete_reply

    def recover(self,**changes):
        return self.gateway.cancel_entry(self.f.ledger,self.f.intent.intent_id,controls=self.store,
            control_snapshot=changes.get('control_snapshot',self.snapshot),calendar=self.f.f.calendar)

    def deletes(self):return [p for m,p in self.calls if m=='DELETE']

    def partial(self):
        self.f.parent.update(status='partially_filled',filled_qty='2',filled_avg_price='150',
                             updated_at=(self.f.now+timedelta(seconds=3)).isoformat())

    def test_expired_parent_is_claimed_once_and_ack_requires_actual_terminal_group_observation(self):
        result=self.recover()
        self.assertEqual(result.status,'entry_canceled')
        self.assertEqual(result.remaining_quantity,0)
        self.assertEqual(len(self.deletes()),1)
        record=self.f.ledger.load(self.f.intent.intent_id)
        self.assertEqual(record['state'],'canceled')
        self.assertEqual(record['native_stop']['entry_cancel']['outcome'],'acknowledged')
        self.assertEqual(record['native_stop']['entry_cancel']['control_digest'],self.snapshot.digest)
        self.assertFalse(result.public_summary()['execution_enabled'])
        self.assertNotIn(self.f.parent_id,str(result.public_summary()))
        self.recover();self.assertEqual(len(self.deletes()),1)

    def test_partial_entry_cancellation_keeps_filled_shares_and_requires_account_cleanup(self):
        self.partial();result=self.recover()
        self.assertEqual(result.status,'partial_cleanup_required')
        self.assertEqual(result.remaining_quantity,2)
        self.assertIn('reconcile_account',result.required_actions)
        self.assertEqual(self.f.ledger.load(self.f.intent.intent_id)['filled_quantity'],2)
        self.assertEqual(len(self.deletes()),1)
        self.assertTrue(all(m in ('GET','DELETE') for m,_ in self.calls))

    def test_acknowledgement_or_restart_never_releases_a_still_open_entry_or_retries_delete(self):
        self.settle=False;result=self.recover()
        self.assertEqual(result.status,'entry_cancel_pending')
        self.assertEqual(self.f.ledger.load(self.f.intent.intent_id)['state'],'open')
        self.assertIsNotNone(self.f.ledger.load(self.f.intent.intent_id)['cancel_requested_at'])
        self.gateway=self.m.NativeEntryRecovery(self.broker);self.recover()
        self.assertEqual(len(self.deletes()),1)

    def test_unknown_delete_and_failed_followup_get_keep_claim_without_retry_or_error_leak(self):
        self.settle=False;self.delete_reply=OSError('PRIVATE_PROVIDER_BODY');self.get_failure_after_delete=True
        result=self.recover()
        self.assertEqual(result.status,'entry_cancel_unknown')
        self.assertNotIn('PRIVATE',str(result.public_summary()))
        self.get_failure_after_delete=False;self.recover()
        self.assertEqual(len(self.deletes()),1)
        self.assertEqual(self.f.ledger.load(self.f.intent.intent_id)['native_stop']['entry_cancel']['outcome'],'unknown')

    def test_full_fill_race_reads_active_stop_and_never_cancels_it_or_invents_an_exit(self):
        def filled():
            self.f.parent.update(status='filled',filled_qty='5',filled_avg_price='150',
                filled_at=self.now.isoformat(),updated_at=self.now.isoformat())
            self.f.child.update(status='new',updated_at=self.now.isoformat())
        self.after_delete=filled;self.delete_reply=HttpReply(422)
        result=self.recover()
        self.assertEqual(result.status,'entry_filled')
        self.assertEqual(result.protection_status,'protected')
        self.assertEqual(result.remaining_quantity,5)
        self.recover();self.assertEqual(len(self.deletes()),1)

    def test_already_filled_parent_is_observed_without_any_cancellation(self):
        self.f.full_parent();result=self.recover()
        self.assertEqual(result.status,'entry_filled')
        self.assertEqual(self.deletes(),[])

    def test_unexpired_entry_under_pause_is_left_alone_but_current_halt_can_cancel_incoming_buy(self):
        self.now=self.f.now+timedelta(seconds=5)
        result=self.recover();self.assertEqual(result.status,'recovery_not_required')
        self.assertEqual(self.deletes(),[])
        halt=self.store.update(expected_revision=self.snapshot.revision,pause_entries=True,halt_orders=True,
                               reason='operator_halt',now=self.now)
        result=self.recover(control_snapshot=halt)
        self.assertEqual(result.status,'entry_canceled');self.assertEqual(len(self.deletes()),1)

    def test_unknown_submission_or_changed_controls_cannot_cancel_an_unverified_order(self):
        self.missing=True;result=self.recover()
        self.assertEqual(result.status,'unconfirmed_submission');self.assertEqual(self.deletes(),[])
        self.missing=False
        self.store.update(expected_revision=self.snapshot.revision,pause_entries=True,halt_orders=True,
                          reason='operator_halt',now=self.now)
        result=self.recover();self.assertEqual(result.status,'control_changed')
        self.assertEqual(self.deletes(),[])

    def test_slow_claim_or_clock_rollback_is_known_unsent_without_releasing_reservations(self):
        self.assertTrue(hasattr(self.f.ledger,'prepare_native_entry_cancellation'))
        claim=self.f.ledger.prepare_native_entry_cancellation
        for delta in (31,-1):
            with self.subTest(delta=delta):
                def delayed(*args,**kwargs):
                    result=claim(*args,**kwargs);self.now+=timedelta(seconds=delta);return result
                with patch.object(self.f.ledger,'prepare_native_entry_cancellation',delayed):
                    result=self.recover()
                self.assertEqual(result.status,'entry_cancel_not_sent')
                record=self.f.ledger.load(self.f.intent.intent_id)
                self.assertEqual(record['native_stop']['entry_cancel']['outcome'],'not_sent')
                self.assertIsNone(record['cancel_requested_at'])
                self.assertEqual(record['state'],'open');self.assertEqual(self.deletes(),[])
                self.now+=timedelta(seconds=2)

    def test_claim_or_after_request_write_failure_never_turns_into_an_unsafe_retry(self):
        persist=self.f.ledger._persist
        def fail_claim(db,record,kind,at):
            if kind=='native_entry_cancel_claim':raise OSError('PRIVATE_DISK')
            return persist(db,record,kind,at)
        with patch.object(self.f.ledger,'_persist',fail_claim):
            with self.assertRaises(BrokerError):self.recover()
        self.assertEqual(self.deletes(),[])
        def fail_result(db,record,kind,at):
            if kind=='native_entry_cancel_result':raise OSError('PRIVATE_DISK')
            return persist(db,record,kind,at)
        self.settle=False
        with patch.object(self.f.ledger,'_persist',fail_result):
            with self.assertRaises(BrokerError):self.recover()
        self.recover();self.assertEqual(len(self.deletes()),1)

    def test_simple_broker_cancel_cannot_bypass_native_group_recovery(self):
        self.f.observe(second=4)
        with self.assertRaises(Exception):self.broker.cancel(self.f.ledger,self.f.intent.intent_id)
        self.assertEqual(self.deletes(),[])

    def test_control_writer_cannot_commit_in_the_middle_of_native_delete(self):
        entered=threading.Event();committed=threading.Event();errors=[];writers=[]
        def update():
            try:
                entered.set();self.store.update(expected_revision=self.snapshot.revision,pause_entries=True,
                    halt_orders=True,reason='operator_halt',now=self.now);committed.set()
            except Exception as error:errors.append(type(error).__name__)
        original=self.broker._transport
        def held(method,url,headers,body):
            if method=='DELETE':
                writer=threading.Thread(target=update);writers.append(writer);writer.start()
                self.assertTrue(entered.wait(2));self.assertFalse(committed.wait(.05))
            return original(method,url,headers,body)
        self.broker._transport=held;self.recover()
        for writer in writers:writer.join(2)
        self.assertEqual(errors,[]);self.assertTrue(committed.is_set())

    def test_missing_latest_group_cannot_upgrade_cached_protection_into_current_safety(self):
        self.f.full_parent();self.f.observe(second=4)
        self.missing=True;before=deepcopy(self.f.ledger.load(self.f.intent.intent_id)['native_stop'])
        result=self.recover()
        self.assertEqual(result.status,'unconfirmed_submission')
        self.assertNotEqual(result.protection_status,'protected')
        self.assertIsNone(result.remaining_quantity)
        self.assertIn('pause_entries',result.required_actions)
        self.assertIn('reconcile_group',result.required_actions)
        self.assertEqual(before,self.f.ledger.load(self.f.intent.intent_id)['native_stop'])
        self.assertEqual(self.deletes(),[])

    def test_inconsistent_active_missing_or_uncertain_child_blocks_parent_cancellation(self):
        for status in ('new','pending_cancel','pending_new','pending_replace'):
            with self.subTest(status=status):
                case=NativeEntryRecoveryTests();case.setUp();self.addCleanup(case.doCleanups)
                case.f.child['status']=status
                result=case.recover()
                self.assertEqual(result.status,'reconciliation_pending')
                self.assertEqual(case.deletes(),[])
                self.assertIsNone(case.f.ledger.load(case.f.intent.intent_id)['cancel_requested_at'])
        self.f.parent['legs']=[]
        result=self.recover();self.assertEqual(result.status,'reconciliation_pending')
        self.assertEqual(self.deletes(),[])

    def test_known_failed_stop_cancels_an_unexpired_incoming_entry(self):
        self.now=self.f.now+timedelta(seconds=5);self.f.child['status']='rejected'
        result=self.recover()
        self.assertEqual(result.status,'entry_canceled')
        self.assertEqual(len(self.deletes()),1)
        self.assertEqual(self.f.ledger.load(self.f.intent.intent_id)['native_stop']['entry_cancel']['reason'],'protection_failed')

    def test_source_cas_rejects_a_full_fill_observed_between_lookup_and_claim(self):
        claim=self.f.ledger.prepare_native_entry_cancellation
        def changed(*args,**kwargs):
            self.f.full_parent();self.now+=timedelta(seconds=1)
            self.f.ledger.observe_native_group(self.f.intent.intent_id,row=self.f.parent,received_at=self.now)
            return claim(*args,**kwargs)
        with patch.object(self.f.ledger,'prepare_native_entry_cancellation',changed):
            with self.assertRaises(BrokerError):self.recover()
        self.assertEqual(self.deletes(),[])
        self.assertEqual(self.f.ledger.load(self.f.intent.intent_id)['state'],'filled')

    def test_known_unsent_retries_are_bounded_and_preserved_in_order_events(self):
        claim=self.f.ledger.prepare_native_entry_cancellation
        def delayed(*args,**kwargs):
            result=claim(*args,**kwargs)
            if result is not None:self.now+=timedelta(seconds=31)
            return result
        with patch.object(self.f.ledger,'prepare_native_entry_cancellation',delayed):
            for _ in range(3):self.assertEqual(self.recover().status,'entry_cancel_not_sent')
            self.assertEqual(self.recover().status,'reconciliation_pending')
        record=self.f.ledger.load(self.f.intent.intent_id)
        self.assertEqual(record['native_stop']['entry_cancel']['generation'],3)
        self.assertEqual(self.deletes(),[])

    def test_group_activated_after_claim_is_known_unsent_and_preserves_new_full_fill(self):
        claim=self.f.ledger.prepare_native_entry_cancellation
        def changed(*args,**kwargs):
            result=claim(*args,**kwargs)
            self.now+=timedelta(seconds=1)
            self.f.parent.update(status='filled',filled_qty='5',filled_avg_price='150',
                filled_at=self.now.isoformat(),updated_at=self.now.isoformat())
            self.f.child.update(status='new',updated_at=self.now.isoformat())
            self.f.ledger.observe_native_group(self.f.intent.intent_id,row=self.f.parent,received_at=self.now)
            return result
        with patch.object(self.f.ledger,'prepare_native_entry_cancellation',changed):result=self.recover()
        self.assertEqual(result.status,'entry_cancel_not_sent')
        self.assertEqual(self.deletes(),[])
        record=self.f.ledger.load(self.f.intent.intent_id)
        self.assertEqual(record['state'],'filled');self.assertEqual(record['filled_quantity'],5)
        self.assertEqual(record['native_stop']['stop']['status'],'new')
        self.assertEqual(record['native_stop']['entry_cancel']['veto_reason'],'group_changed')

    def test_stale_controls_cannot_make_an_unexpired_entry_look_like_a_benign_noop(self):
        self.now=self.f.now+timedelta(seconds=5)
        self.store.update(expected_revision=self.snapshot.revision,pause_entries=True,halt_orders=True,
                          reason='operator_halt',now=self.now)
        result=self.recover()
        self.assertEqual(result.status,'control_changed')
        self.assertIn('pause_entries',result.required_actions)
        self.assertIn('reconcile_group',result.required_actions)
        self.assertEqual(self.deletes(),[])


if __name__=='__main__':unittest.main()
