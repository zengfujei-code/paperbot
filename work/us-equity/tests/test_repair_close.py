from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

import test_repair_execution
import paperbot_us.native_close as close
from paperbot_us.broker import HttpReply,BrokerError
from paperbot_us.decision_journal import DecisionJournal,capture_close_decision
from paperbot_us.risk import Quote
from paperbot_us.risk_view import RiskViewError


class RepairCloseTests(unittest.TestCase):
    def setUp(self):
        self.assertIn('target_repair_id',close.ClosePreparation.__dataclass_fields__,
            'close must select the actual owned current stop leg')
        self.e=test_repair_execution.RepairExecutionTests();self.e.setUp();self.addCleanup(self.e.doCleanups)
        self.e.stage();self.e.submit();self.r=self.e.r;self.n=self.e.n;self.ledger=self.e.ledger
        self.f=self.e.f.f;self.family=self.r.family;self.target=self.r.token.repair_id
        self.control=self.e.control;self.controls=self.e.controls;self.policy=self.e.e.policy
        self.journal=DecisionJournal(self.n.f.f.path.with_name('repair-close-decisions.sqlite'),clock=lambda:self.n.now)
        self.journal.initialize();self.gateway=close.NativeCloseGateway(self.n.broker)
        self.deletes=[];self.settle=True;self.failure=None;self.reply_status=204;self.after_delete=None
        original=self.n.broker._transport
        def transport(method,url,headers,body):
            if method=='DELETE':
                self.assertEqual(urlsplit(url).path,'/v2/orders/'+self.r.row['id'])
                self.deletes.append(urlsplit(url).path)
                row=self.r.m.repair_rows(self.ledger.load(self.family))[-1]
                self.assertEqual(row['close']['outcome'],'claimed')
                self.n.now+=timedelta(seconds=1)
                if self.after_delete:self.after_delete()
                elif self.settle:
                    self.r.row.update(status='canceled',updated_at=self.n.now.isoformat())
                    self.n.visible=[]
                    if self.n.positions:self.n.positions[0]['qty_available']=self.n.positions[0]['qty']
                if self.failure:raise self.failure
                return HttpReply(self.reply_status)
            self.assertEqual(method,'GET','closing preparation cannot POST a replacement')
            return original(method,url,headers,body)
        self.n.broker._transport=transport

    def prepare(self):
        view=self.f.prepare(quotes=(Quote('AAPL','120','120.1',self.n.now,self.n.now),))
        return close.prepare_close(ledger=self.ledger,prepared=view,family_id=self.family,
            calendar=self.f.calendar,now=self.n.now,reason='account_risk')

    def record(self):
        self.plan=self.prepare()
        frame=capture_close_decision(ledger=self.ledger,prepared=self.plan,source=None,policy=self.policy,
            controls=self.control,limits=self.r.f.limits,calendar=self.f.calendar,now=self.n.now)
        self.receipt=self.journal.append(frame,ledger=self.ledger)
        return self.plan

    def request(self):
        return self.gateway.request(self.ledger,self.plan,decision=self.receipt,journal=self.journal,
            controls=self.controls,control_snapshot=self.control,limits=self.r.f.limits,calendar=self.f.calendar,
            book=self.f.book,baselines=self.f.baselines)

    def test_preparation_binds_current_repair_and_preserves_original_stop(self):
        original=deepcopy(self.ledger.load(self.family)['native_stop']['stop']);self.record()
        self.assertEqual(self.plan.target_repair_id,self.target)
        self.assertEqual(self.plan.original.portfolio.stop_reservations[0].remaining_quantity,5)
        self.assertEqual(self.plan.preview.portfolio.stop_reservations,())
        self.assertEqual(self.receipt.read()['inputs']['close']['target_repair_id'],self.target)
        self.assertEqual(self.journal.replay(self.receipt.sequence,ledger=self.ledger),self.receipt)
        self.assertEqual(self.ledger.load(self.family)['native_stop']['stop'],original)

    def test_exact_repair_uuid_is_canceled_once_then_new_account_proof_is_required(self):
        self.record();result=self.request()
        self.assertEqual(result.status,'cleanup_required');self.assertEqual(result.remaining_quantity,5)
        self.assertEqual(len(self.deletes),1);self.assertEqual(len(self.e.posts),1)
        self.request();self.assertEqual(len(self.deletes),1)
        self.assertFalse(result.public_summary()['ready_for_execution'])

    def test_acknowledgement_with_active_repair_keeps_reservation_and_pauses(self):
        self.settle=False;self.record();result=self.request()
        self.assertEqual(result.status,'awaiting_stop_terminal')
        self.assertEqual(self.r.f.m.project_groups(self.ledger.all_orders())[self.family].reserved_quantity,5)
        with self.assertRaises(RiskViewError):self.f.prepare()
        self.request();self.assertEqual(len(self.deletes),1)

    def test_rejected_cancel_with_current_active_stop_is_still_protected(self):
        self.settle=False;self.reply_status=422;self.record();result=self.request()
        self.assertEqual(result.status,'close_rejected_protected')
        self.assertTrue(self.n.reconciler.refresh().ready_for_risk)

    def test_unknown_delete_never_repeats_or_sends_an_alternative_sell(self):
        self.settle=False;self.failure=OSError('PRIVATE_PROVIDER');self.record();result=self.request()
        self.assertEqual(result.status,'awaiting_stop_terminal')
        self.failure=None;self.request();self.assertEqual(len(self.deletes),1)
        self.assertEqual(len(self.e.posts),1)
        self.assertNotIn('PRIVATE',str(result.public_summary()))

    def test_latest_missing_repair_is_unconfirmed_not_cached_cleanup_permission(self):
        self.settle=False;self.record();self.request();self.r.missing=True
        result=self.gateway.recover(self.ledger,self.family,target_repair_id=self.target)
        self.assertEqual(result.status,'unconfirmed_group');self.assertIsNone(result.remaining_quantity)
        self.assertEqual(self.r.f.m.project_groups(self.ledger.all_orders())[self.family].available_quantity,0)

    def test_global_halt_preserves_the_active_repair(self):
        self.record();self.controls.update(expected_revision=self.control.revision,pause_entries=True,
            halt_orders=True,reason='operator_halt',now=self.n.now)
        self.request();self.assertEqual(self.deletes,[])
        self.assertEqual(self.r.row['status'],'new')

    def test_partial_stop_fill_closes_only_remaining_owned_quantity(self):
        self.r.fill(2);self.record();self.assertEqual(self.plan.intent.quantity,3)
        result=self.request();self.assertEqual(result.remaining_quantity,3)
        cycle=self.n.reconciler.refresh();self.assertTrue(cycle.book_receipt.result.reconciled)
        self.assertEqual(cycle.book_receipt.result.profit_usd,-20)

    def test_full_fill_race_needs_no_cleanup_or_second_sell(self):
        self.record();self.after_delete=lambda:self.r.fill(5,'filled')
        result=self.request();self.assertEqual(result.status,'group_closed');self.assertEqual(result.remaining_quantity,0)
        cycle=self.n.reconciler.refresh();self.assertTrue(cycle.book_receipt.result.reconciled)
        self.assertEqual(cycle.book_receipt.result.profit_usd,-50);self.assertEqual(len(self.e.posts),1)

    def test_overdue_nonterminal_repair_never_authorizes_replacement(self):
        self.settle=False;self.record();self.request();self.n.now+=timedelta(seconds=31)
        result=self.gateway.recover(self.ledger,self.family,target_repair_id=self.target)
        self.assertEqual(result.status,'recovery_overdue')
        self.assertEqual(self.r.f.m.project_groups(self.ledger.all_orders())[self.family].available_quantity,0)
        self.assertEqual(len(self.deletes),1);self.assertEqual(len(self.e.posts),1)


if __name__=='__main__':unittest.main()
