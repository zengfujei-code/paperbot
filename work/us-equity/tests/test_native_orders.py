from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
import sqlite3
import unittest
from unittest.mock import patch

import test_orders
from paperbot_us.protection import NativeStopPlan


class NativeOrderLedgerTests(unittest.TestCase):
    def setUp(self):
        self.f=test_orders.OrderLedgerTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.ledger=self.f.ledger;self.now=self.f.now;self.intent=self.f.intent
        self.plan=NativeStopPlan.from_intent(self.intent)
        self.parent_id=self.f.broker_id;self.child_id='00000000-0000-4000-8000-000000000002'
        common=dict(asset_class='us_equity',symbol='AAPL',qty='5',filled_qty='0',filled_avg_price=None,
            order_class='oto',time_in_force='day',extended_hours=False,created_at=self.now.isoformat(),
            updated_at=(self.now+timedelta(seconds=1)).isoformat())
        self.child={**common,'id':self.child_id,'client_order_id':'private-provider-child',
            'side':'sell','type':'stop','limit_price':None,'stop_price':'145','status':'held'}
        self.parent={**common,'id':self.parent_id,'client_order_id':'pb3-'+'a'*40,
            'side':'buy','type':'limit','limit_price':'150.1','status':'new','legs':[self.child]}

    def attach(self):
        self.assertTrue(hasattr(self.ledger,'attach_native_stop'),'durable native stop ownership required')
        self.f.admit()
        return self.ledger.attach_native_stop(self.intent.intent_id,plan=self.plan,at=self.now)

    def claim(self):
        self.attach();return self.ledger.prepare_native_submission(self.intent.intent_id,**self.f.args)

    def observe(self,second=2):
        return self.ledger.observe_native_group(self.intent.intent_id,row=self.parent,
            received_at=self.now+timedelta(seconds=second))

    def full_parent(self,second=2):
        self.parent.update(status='filled',filled_qty='5',filled_avg_price='150',
            filled_at=(self.now+timedelta(seconds=1)).isoformat(),
            updated_at=(self.now+timedelta(seconds=second)).isoformat())
        self.child.update(status='new',updated_at=(self.now+timedelta(seconds=second)).isoformat())

    def test_plan_is_durable_immutable_and_can_only_attach_before_claim(self):
        first=self.attach();self.assertIn('native_stop',first)
        self.assertEqual(self.f.Ledger(self.f.path).load(self.intent.intent_id),first)
        self.assertEqual(self.ledger.attach_native_stop(self.intent.intent_id,plan=self.plan,at=self.now),first)
        with self.assertRaises(self.f.Error):
            self.ledger.attach_native_stop(self.intent.intent_id,plan=replace(self.plan,stop_price_usd='144'),at=self.now)
        self.ledger.prepare_native_submission(self.intent.intent_id,**self.f.args)
        with self.assertRaises(self.f.Error): self.f.observe()
        self.assertEqual(self.ledger.load(self.intent.intent_id)['state'],'submission_unknown')

    def test_parent_and_child_commit_together_and_restart_preserves_ownership(self):
        self.claim();record=self.observe()
        self.assertEqual(record['state'],'open')
        self.assertEqual(record['native_stop']['stop']['broker_order_id'],self.child_id)
        self.assertEqual(self.f.Ledger(self.f.path).all_orders(),[record])
        before=self.ledger.all_orders()
        self.full_parent()
        with patch.object(self.ledger,'_persist',side_effect=OSError('PRIVATE_DISK')):
            with self.assertRaises(self.f.Error) as caught: self.observe(3)
        self.assertNotIn('PRIVATE',str(caught.exception));self.assertEqual(self.ledger.all_orders(),before)
        self.assertEqual(self.observe(3)['state'],'filled')

    def test_partial_parent_and_missing_child_never_release_known_child_evidence(self):
        self.claim();self.observe()
        self.parent.update(status='partially_filled',filled_qty='2',filled_avg_price='150',
            updated_at=(self.now+timedelta(seconds=3)).isoformat())
        partial=self.observe(4);self.assertEqual(partial['filled_quantity'],2)
        self.parent['legs']=[]
        missing=self.observe(5)
        self.assertFalse(missing['native_stop']['child_visible'])
        self.assertEqual(missing['native_stop']['stop']['broker_order_id'],self.child_id)
        self.assertEqual(missing['native_stop']['stop']['filled_quantity'],0)

    def test_gap_stop_fills_produce_owned_accounting_evidence_and_replay(self):
        self.claim();self.full_parent();self.observe(3)
        self.child.update(status='filled',filled_qty='5',filled_avg_price='130',
            filled_at=(self.now+timedelta(seconds=4)).isoformat(),updated_at=(self.now+timedelta(seconds=4)).isoformat())
        self.observe(5)
        from paperbot_us.native_orders import accounting_evidence
        from paperbot_us.accounting import AccountAnchor,Activity,ActivityWindow,AccountSnapshot
        from paperbot_us.accountbook import AccountBook
        orders=accounting_evidence(self.ledger.all_orders())
        self.assertEqual(len(orders),2)
        stop=next(o for o in orders if o.side=='sell')
        self.assertEqual((stop.order_type,stop.limit_price_usd,stop.parent_order_id),('stop',None,self.parent_id))
        anchor=AccountAnchor(self.now-timedelta(seconds=1),'10000','1'*64,'2'*64)
        fills=tuple(Activity(identity,'FILL',symbol='AAPL',side=side,quantity=5,price_usd=price,
            cumulative_quantity=5,leaves_quantity=0,broker_order_id=oid,occurred_at=self.now+timedelta(seconds=at))
            for identity,side,price,oid,at in [('3'*64,'buy','150',self.parent_id,1),('4'*64,'sell','130',self.child_id,4)])
        at=self.now+timedelta(seconds=6);book=AccountBook(self.f.path.with_name('native-book.sqlite'));book.initialize(anchor)
        receipt=book.refresh(window=ActivityWindow(anchor.as_of,at,at,fills),orders=orders,
            account=AccountSnapshot(at,'9900','9900',False,(),anchor.account_sha256),now=at)
        self.assertTrue(receipt.result.reconciled,receipt.result.violations)
        self.assertEqual(receipt.result.realized_usd,Decimal('-100'))
        self.assertEqual(AccountBook(book.path).replay(receipt.sequence),receipt.result)

    def test_cancellation_claim_never_erases_partial_fills_or_confirms_cancellation(self):
        self.claim();self.full_parent();self.observe(3)
        self.child.update(status='partially_filled',filled_qty='2',filled_avg_price='130',
            updated_at=(self.now+timedelta(seconds=4)).isoformat())
        self.observe(5)
        ticket=self.ledger.prepare_stop_cancellation(self.intent.intent_id,at=self.now+timedelta(seconds=6))
        self.assertEqual(ticket['broker_order_id'],self.child_id)
        self.assertIsNone(self.ledger.prepare_stop_cancellation(self.intent.intent_id,at=self.now+timedelta(seconds=7)))
        record=self.ledger.load(self.intent.intent_id)
        self.assertEqual(record['native_stop']['stop']['status'],'partially_filled')
        self.child.update(status='canceled',updated_at=(self.now+timedelta(seconds=7)).isoformat())
        record=self.observe(8)
        self.assertEqual(record['native_stop']['stop']['filled_quantity'],2)
        self.assertEqual(record['native_stop']['stop']['status'],'canceled')
        self.assertEqual(record['filled_quantity'],5)

    def test_child_replacement_fill_regression_or_old_changed_event_is_not_silently_accepted(self):
        self.claim();self.full_parent();self.observe(3)
        before=deepcopy(self.child)
        for changes in (dict(id='00000000-0000-4000-8000-000000000003'),
                        dict(status='partially_filled',filled_qty='2',filled_avg_price='130')):
            with self.subTest(changes=changes):
                self.child.update(changes)
                with self.assertRaises(self.f.Error): self.observe(4)
                self.child.clear();self.child.update(before)
        self.child.update(status='partially_filled',filled_qty='2',filled_avg_price='130',
            updated_at=(self.now+timedelta(seconds=4)).isoformat())
        self.observe(5)
        self.child.update(filled_qty='1',updated_at=(self.now+timedelta(seconds=5)).isoformat())
        with self.assertRaises(self.f.Error): self.observe(6)

    def test_simple_broker_submit_cannot_strip_attached_protection(self):
        self.attach()
        from paperbot_us.broker import PaperBroker,BrokerError
        from paperbot_us.controls import ControlJournal
        controls=ControlJournal(self.f.path.with_name('controls.sqlite'));controls.initialize(now=self.now)
        snap=controls.update(expected_revision=1,pause_entries=False,halt_orders=False,reason='operator_resume',now=self.now)
        calls=[]
        broker=PaperBroker('PK'+'T'*20,'T'*32,transport=lambda *a:calls.append(a),clock=lambda:self.now)
        with self.assertRaises(BrokerError):
            broker.submit(self.ledger,self.intent.intent_id,controls=controls,control_snapshot=snap,**self.f.args)
        self.assertEqual(calls,[]);self.assertEqual(self.ledger.load(self.intent.intent_id)['state'],'reserved')

    def _submit_with_concurrent_native_attachment(self,**extra):
        self.f.admit()
        from paperbot_us.broker import PaperBroker,BrokerError,HttpReply
        from paperbot_us.controls import ControlJournal
        controls=ControlJournal(self.f.path.with_name('controls.sqlite'));controls.initialize(now=self.now)
        snap=controls.update(expected_revision=1,pause_entries=False,halt_orders=False,reason='operator_resume',now=self.now)
        original=self.ledger.load
        def concurrent_attach(identity):
            old=original(identity)
            self.ledger.attach_native_stop(identity,plan=self.plan,at=self.now)
            return old
        calls=[]
        def transport(*args):
            calls.append(args[0])
            row={**self.parent,'order_class':'simple','legs':[],'updated_at':self.now.isoformat()}
            return HttpReply(201,row)
        broker=PaperBroker('PK'+'T'*20,'T'*32,transport=transport,clock=lambda:self.now)
        with patch.object(self.ledger,'load',side_effect=concurrent_attach):
            with self.assertRaises((BrokerError,self.f.Error)):
                broker.submit(self.ledger,self.intent.intent_id,controls=controls,control_snapshot=snap,**{**self.f.args,**extra})
        self.assertEqual(len(calls),0,'an atomic gateway claim must prevent a simple POST after native attachment')
        self.assertEqual(original(self.intent.intent_id)['state'],'reserved')

    def test_native_attachment_between_simple_precheck_and_claim_cannot_send_a_naked_entry(self):
        self._submit_with_concurrent_native_attachment()

    def test_simple_broker_gateway_cannot_be_overridden_by_risk_input_options(self):
        self._submit_with_concurrent_native_attachment(gateway='native_oto_stop')

    def test_confirmed_unsent_risk_veto_cannot_be_revived_by_a_native_observation(self):
        ticket=self.claim()
        from paperbot_us.orders import recheck_ticket
        decision=replace(recheck_ticket(ticket,now=self.now),allowed=False,violations=('entries_paused',))
        self.ledger.reject_unsubmitted(ticket,decision,at=self.now+timedelta(seconds=1))
        with self.assertRaises(self.f.Error): self.observe(2)
        record=self.ledger.load(self.intent.intent_id)
        self.assertEqual(record['state'],'rejected_by_risk')
        self.assertIsNone(record['broker_order_id'])

    def test_legacy_risk_path_does_not_treat_native_stop_reserved_shares_as_free(self):
        self.claim();self.full_parent();self.observe(3)
        from paperbot_us.risk import Position
        at=self.now+timedelta(seconds=5)
        portfolio=replace(self.f.portfolio,as_of=at,cash_usd='9250',equity_usd='10000',
            positions=(Position('AAPL',5,'150',at),))
        sell=replace(self.intent,intent_id='b'*64,side='sell',stop_price_usd=None,expires_at=at+timedelta(seconds=20))
        result=self.f.admit(sell,portfolio=portfolio,quote=replace(self.f.quote,as_of=at,received_at=at),now=at)
        self.assertEqual(result['state'],'rejected_by_risk')
        self.assertIn('native_group_coordinator_required',result['context']['reconciliation_findings'])

    def test_child_cannot_reuse_a_broker_order_already_owned_by_a_different_intent(self):
        from paperbot_us.risk import Position
        earlier=replace(self.intent,intent_id='b'*64,quantity=1)
        self.f.admit(earlier)
        self.ledger.prepare_submission(earlier.intent_id,**self.f.args)
        self.ledger.observe(earlier.intent_id,status='filled',filled_quantity=1,average_fill_price_usd='150',
            broker_order_id=self.child_id,broker_created_at=self.now,broker_event_at=self.now,received_at=self.now)
        self.f.args['portfolio']=replace(self.f.portfolio,cash_usd='9850',
            positions=(Position('AAPL',1,'150',self.now),))
        self.claim()
        with self.assertRaises(self.f.Error): self.observe(2)
        self.assertIsNone(self.ledger.load(self.intent.intent_id)['broker_order_id'])

    def test_cumulative_entry_average_cannot_hide_a_later_fill_above_the_entry_limit(self):
        self.claim()
        self.parent.update(status='partially_filled',filled_qty='2',filled_avg_price='149')
        self.observe(2)
        self.full_parent(second=3)
        # Overall average 150 is legal, but incremental (750-298)/3 exceeds 150.1.
        with self.assertRaises(self.f.Error): self.observe(4)
        self.assertEqual(self.ledger.load(self.intent.intent_id)['filled_notional_usd'],'298')


if __name__=='__main__': unittest.main()
