from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import importlib,importlib.util,unittest
from unittest.mock import patch
from urllib.parse import urlsplit

import test_repair_orders
from paperbot_us.broker import BrokerError,HttpReply
from paperbot_us.orders import LedgerError


class RepairExecutionTests(unittest.TestCase):
    def setUp(self):
        for name in ('repair_decisions','repair_execution'):
            self.assertIsNotNone(importlib.util.find_spec('paperbot_us.'+name),
                'decision-bound repair sending must be explicit')
        self.m=importlib.import_module('paperbot_us.repair_execution')
        self.d=importlib.import_module('paperbot_us.repair_decisions')
        self.r=test_repair_orders.RepairOrderTests();self.r.setUp();self.addCleanup(self.r.doCleanups)
        self.n=self.r.n;self.f=self.r.f;self.ledger=self.r.ledger;self.plan=self.r.plan
        self.e=self.r.r.r.e;self.controls=self.e.control_store;self.control=self.e.control
        self.journal=self.d.RepairDecisionJournal(self.n.f.f.path.with_name('repair-decisions.sqlite'),clock=lambda:self.n.now)
        self.gateway=self.m.RepairPaperGateway(self.n.broker)
        self.posts=[];self.failure=None;self.bad_reply=False
        original=self.n.broker._transport
        def transport(method,url,headers,body):
            if method=='POST':
                self.assertEqual(urlsplit(url).netloc,'paper-api.alpaca.markets')
                self.assertEqual(urlsplit(url).path,'/v2/orders');self.posts.append(deepcopy(body))
                row=self.r.m.repair_rows(self.ledger.load(self.r.family))[-1]
                self.assertEqual(row['state'],'submission_unknown')
                self.assertEqual(row['dispatch']['outcome'],'claimed')
                self.assertEqual(row['decision_link']['digest'],self.receipt.digest)
                self.r.row=dict(id='00000000-0000-4000-8000-000000000096',asset_class='us_equity',
                    **body,filled_qty='0',filled_avg_price=None,status='new',created_at=self.n.now.isoformat(),
                    updated_at=self.n.now.isoformat(),filled_at=None,legs=None,limit_price=None)
                self.n.visible=[deepcopy(self.r.row)];self.n.positions[0]['qty_available']='0'
                if self.failure:raise self.failure
                return HttpReply(201,{} if self.bad_reply else deepcopy(self.r.row))
            return original(method,url,headers,body)
        self.n.broker._transport=transport

    def record(self,**changes):
        self.receipt=self.journal.record(self.plan,**{**dict(ledger=self.ledger,book=self.f.book,
            baselines=self.f.f.baselines,controls=self.control,calendar=self.f.f.calendar,limits=self.f.limits),**changes})
        return self.receipt

    def stage(self):
        self.record()
        self.r.token=self.r.m.reserve_repair(ledger=self.ledger,plan=self.plan,book=self.f.book,
            baselines=self.f.f.baselines,calendar=self.f.f.calendar,limits=self.f.limits,now=self.n.now,
            decision=self.receipt,journal=self.journal)
        return self.r.token

    def submit(self,**changes):
        return self.gateway.submit(self.ledger,self.r.family,self.r.token.repair_id,**{**dict(
            journal=self.journal,book=self.f.book,baselines=self.f.f.baselines,controls=self.controls,
            control_snapshot=self.control,calendar=self.f.f.calendar,limits=self.f.limits),**changes})

    def row(self):return self.r.m.repair_rows(self.ledger.load(self.r.family))[-1]

    def test_private_decision_replays_original_prefix_after_admission(self):
        self.stage();record=self.journal.replay(self.receipt.sequence,ledger=self.ledger).read()
        self.assertTrue(record['outcome']['allowed'])
        self.assertEqual(record['inputs']['body']['type'],'stop')
        self.assertNotIn('limit_price',record['inputs']['body'])
        self.assertFalse(self.receipt.public_summary()['execution_enabled'])

    def test_one_stop_post_follows_durable_decision_and_claim(self):
        self.stage();result=self.submit()
        self.assertEqual(len(self.posts),1);self.assertEqual(self.posts[0]['type'],'stop')
        self.assertEqual(self.posts[0]['qty'],'5');self.assertEqual(self.posts[0]['time_in_force'],'gtc')
        self.assertFalse(result.public_summary()['protection_active'])
        self.submit();self.assertEqual(len(self.posts),1)

    def test_legacy_allocation_cannot_be_given_dispatch_authority_later(self):
        self.r.token=self.r.reserve()
        with self.assertRaises(BrokerError):self.submit()
        self.assertEqual(self.posts,[]);self.assertIsNone(self.row()['submission_claimed_at'])

    def test_current_global_halt_records_a_veto_and_cannot_admit(self):
        control=self.controls.update(expected_revision=self.control.revision,pause_entries=True,halt_orders=True,
            reason='operator_halt',now=self.n.now)
        receipt=self.record(controls=control)
        self.assertFalse(receipt.read()['outcome']['allowed'])
        self.assertIn('all_orders_halted',receipt.read()['outcome']['vetoes'])
        with self.assertRaises(LedgerError):self.r.m.reserve_repair(ledger=self.ledger,plan=self.plan,
            book=self.f.book,baselines=self.f.f.baselines,calendar=self.f.f.calendar,limits=self.f.limits,
            now=self.n.now,decision=receipt,journal=self.journal)

    def test_changed_control_after_claim_is_known_unsent_and_releases_only_this_allocation(self):
        self.stage();original=self.m.claim_dispatch
        def changed(*args,**kwargs):
            ticket=original(*args,**kwargs)
            self.controls.update(expected_revision=self.control.revision,pause_entries=True,halt_orders=True,
                reason='operator_halt',now=self.n.now)
            return ticket
        with patch.object(self.m,'claim_dispatch',side_effect=changed):result=self.submit()
        self.assertEqual(self.posts,[]);self.assertEqual(self.row()['dispatch']['outcome'],'not_sent')
        self.assertIsNotNone(self.row()['submission_claimed_at'])
        self.assertEqual(self.f.m.project_groups(self.ledger.all_orders())[self.r.family].available_quantity,5)
        self.submit();self.assertEqual(self.posts,[])

    def test_delayed_expired_claim_cannot_post_or_be_retried(self):
        self.stage();original=self.m.claim_dispatch
        def delayed(*args,**kwargs):
            ticket=original(*args,**kwargs);self.n.now+=timedelta(seconds=21);return ticket
        with patch.object(self.m,'claim_dispatch',side_effect=delayed):self.submit()
        self.assertEqual(self.posts,[]);self.assertEqual(self.row()['dispatch']['outcome'],'not_sent')
        self.submit();self.assertEqual(self.posts,[])

    def test_relaxed_limits_cannot_replace_the_recorded_limits(self):
        self.stage()
        with self.assertRaises(BrokerError):self.submit(limits=replace(self.f.limits,max_spread_bps='100'))
        self.assertEqual(self.posts,[]);self.assertIsNone(self.row()['submission_claimed_at'])

    def test_calendar_revision_after_claim_cannot_use_the_old_session(self):
        self.stage();original=self.m.claim_dispatch
        def changed(*args,**kwargs):
            ticket=original(*args,**kwargs);calendar=self.f.f.calendar
            day=next(iter(calendar.sessions));session=calendar.sessions[day]
            calendar.sessions[day]=replace(session,closes=self.n.now)
            return ticket
        with patch.object(self.m,'claim_dispatch',side_effect=changed):self.submit()
        self.assertEqual(self.posts,[]);self.assertEqual(self.row()['dispatch']['outcome'],'not_sent')

    def test_new_account_snapshot_after_claim_requires_a_new_decision(self):
        self.stage();original=self.m.claim_dispatch
        def refreshed(*args,**kwargs):
            ticket=original(*args,**kwargs);self.n.now+=timedelta(seconds=1)
            snapshot=self.f.book.snapshot();account=replace(snapshot.account,received_at=self.n.now,
                positions=tuple(replace(p,received_at=self.n.now) for p in snapshot.account.positions))
            from paperbot_us.native_orders import accounting_evidence
            self.f.book.refresh(window=self.n.broker.activities(after=snapshot.anchor.as_of,until=self.n.now),
                orders=accounting_evidence(self.ledger.all_orders()),account=account,now=self.n.now)
            return ticket
        with patch.object(self.m,'claim_dispatch',side_effect=refreshed):self.submit()
        self.assertEqual(self.posts,[]);self.assertEqual(self.row()['dispatch']['outcome'],'not_sent')

    def test_unknown_post_is_not_retried_and_exact_get_recovers_it(self):
        self.stage();self.failure=OSError('PRIVATE_PROVIDER')
        result=self.submit();self.assertEqual(len(self.posts),1)
        self.assertNotIn('PRIVATE',str(result.public_summary()))
        self.n.now+=timedelta(seconds=21);self.submit();self.assertEqual(len(self.posts),1)
        self.assertEqual(self.f.m.project_groups(self.ledger.all_orders())[self.r.family].available_quantity,0)
        self.failure=None;self.r.recover()
        self.assertTrue(self.n.reconciler.refresh().ready_for_risk)

    def test_invalid_success_payload_keeps_unknown_claim_and_reservation(self):
        self.stage();self.bad_reply=True;self.submit()
        self.assertEqual(len(self.posts),1);self.assertEqual(self.row()['state'],'submission_unknown')
        self.submit();self.assertEqual(len(self.posts),1)
        self.r.recover();self.assertTrue(self.n.reconciler.refresh().ready_for_risk)

    def test_claim_disk_failure_sends_no_request(self):
        self.stage();original=self.ledger._persist
        def persist(db,record,kind,at):
            if kind=='repair_dispatch_claim':raise OSError('PRIVATE_DISK')
            return original(db,record,kind,at)
        with patch.object(self.ledger,'_persist',side_effect=persist):
            with self.assertRaises(BrokerError):self.submit()
        self.assertEqual(self.posts,[]);self.assertIsNone(self.row()['submission_claimed_at'])

    def test_result_disk_failure_never_repeats_the_post(self):
        self.stage();original=self.ledger._persist
        def persist(db,record,kind,at):
            if kind=='repair_dispatch_result':raise OSError('PRIVATE_DISK')
            return original(db,record,kind,at)
        with patch.object(self.ledger,'_persist',side_effect=persist):
            with self.assertRaises(BrokerError):self.submit()
        self.submit();self.assertEqual(len(self.posts),1)
        self.r.recover();self.assertTrue(self.n.reconciler.refresh().ready_for_risk)

    def test_clock_before_committed_decision_cannot_claim_or_send(self):
        self.stage();self.n.now-=timedelta(seconds=1)
        with self.assertRaises(BrokerError):self.submit()
        self.assertEqual(self.posts,[]);self.assertIsNone(self.row()['submission_claimed_at'])

    def test_final_time_is_sampled_after_slow_validation(self):
        self.stage();original=self.m._calendar;calls=0;elapsed=[0]
        self.gateway=self.m.RepairPaperGateway(self.n.broker,monotonic=lambda:elapsed[0])
        def slow(calendar):
            nonlocal calls
            value=original(calendar);calls+=1
            if calls==2:self.n.now+=timedelta(seconds=21);elapsed[0]+=21
            return value
        with patch.object(self.m,'_calendar',side_effect=slow):self.submit()
        self.assertEqual(self.posts,[]);self.assertEqual(self.row()['dispatch']['outcome'],'not_sent')

    def test_final_time_detects_rollback_during_validation(self):
        self.stage();original=self.m._calendar;calls=0;elapsed=[0]
        self.gateway=self.m.RepairPaperGateway(self.n.broker,monotonic=lambda:elapsed[0])
        def rolled(calendar):
            nonlocal calls
            value=original(calendar);calls+=1
            if calls==2:self.n.now-=timedelta(seconds=1);elapsed[0]+=1
            return value
        with patch.object(self.m,'_calendar',side_effect=rolled):self.submit()
        self.assertEqual(self.posts,[]);self.assertEqual(self.row()['dispatch']['outcome'],'not_sent')
        self.assertTrue(self.row()['dispatch']['clock_regressed'])

    def test_submitted_stop_partial_fill_reconciles_actual_loss(self):
        self.stage();self.submit();self.r.fill(2)
        cycle=self.n.reconciler.refresh();self.assertTrue(cycle.ready_for_risk,cycle.findings)
        self.assertEqual(cycle.book_receipt.result.profit_usd,-20)
        self.assertEqual(self.f.m.project_groups(self.ledger.all_orders())[self.r.family].remaining_quantity,3)


if __name__=='__main__':unittest.main()
