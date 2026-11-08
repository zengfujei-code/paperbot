from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import importlib, importlib.util, unittest
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from urllib.parse import urlsplit,parse_qs

import test_native_cleanup
from paperbot_us.broker import BrokerError,HttpReply
from paperbot_us.controls import ControlJournal
from paperbot_us.decision_journal import DecisionJournal,DecisionError,DecisionFrame
from paperbot_us.execution_policy import ExecutionPolicy,PolicySnapshot
from paperbot_us.orders import LedgerError,_base,_json,_hash
from paperbot_us.risk import RiskLimits,Quote


class CleanupExecutionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.cleanup_execution'),
            'decision-bound cleanup dispatch required')
        self.m=importlib.import_module('paperbot_us.cleanup_execution')
        self.f=test_native_cleanup.NativeCleanupTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.n=self.f.n;self.ledger=self.f.ledger;self.intent=self.f.intent
        self.control_store=ControlJournal(self.n.f.f.path.with_name('dispatch-controls.sqlite'))
        self.control=self.control_store.initialize(now=self.n.now)
        self.journal=DecisionJournal(self.n.f.f.path.with_name('dispatch-decisions.sqlite'),clock=lambda:self.n.now)
        self.journal.initialize()
        limits_sha=_hash(_json(_base(self.f.limits,RiskLimits)))
        self.policy=PolicySnapshot(ExecutionPolicy(1,10000,True,'1'*64,'2'*64,limits_sha,5,True),'3'*64)
        self.posts=[];self.row=None;self.failure=None;self.post_hook=None
        self.old_transport=self.n.broker._transport;self.n.broker._transport=self.transport
        self.gateway=self.m.CleanupPaperGateway(self.n.broker)

    def transport(self,method,url,headers,body):
        parsed=urlsplit(url)
        if method=='POST':
            self.assertEqual(parsed.path,'/v2/orders');self.posts.append(deepcopy(body))
            record=self.ledger.load(self.intent.intent_id)
            self.assertEqual(record['state'],'submission_unknown')
            self.assertEqual(record['cleanup_decision']['digest'],self.receipt.digest)
            if self.post_hook:self.post_hook()
            if self.failure:raise self.failure
            self.row=dict(id='00000000-0000-4000-8000-000000000080',
                client_order_id='pb3-'+self.intent.intent_id[:40],asset_class='us_equity',symbol='AAPL',side='sell',
                type='limit',qty=str(self.intent.quantity),filled_qty='0',filled_avg_price=None,limit_price='150',
                status='new',time_in_force='day',extended_hours=False,order_class='simple',
                created_at=self.n.now.isoformat(),updated_at=self.n.now.isoformat(),legs=None)
            self.n.visible=[deepcopy(self.row)];self.n.positions[0]['qty_available']='0'
            return HttpReply(201,deepcopy(self.row))
        if (parsed.path=='/v2/orders:by_client_order_id' and self.row is not None
                and parse_qs(parsed.query).get('client_order_id')==[self.row['client_order_id']]):
            return HttpReply(200,deepcopy(self.row))
        return self.old_transport(method,url,headers,body)

    def record(self,**changes):
        import paperbot_us.decision_journal as d
        self.assertTrue(hasattr(d,'capture_cleanup_decision'),'explicit complete cleanup decision required')
        view=self.f.prepare()
        frame=d.capture_cleanup_decision(ledger=self.ledger,prepared=view,intent=self.intent,
            policy=changes.get('policy',self.policy),controls=self.control,limits=self.f.limits,
            calendar=self.f.f.calendar,now=self.n.now)
        self.receipt=self.journal.append(frame,ledger=self.ledger)
        return view

    def stage(self):
        view=self.record()
        return self.f.m.admit_cleanup(self.ledger,self.intent,prepared=view,limits=self.f.limits,
            calendar=self.f.f.calendar,now=self.n.now,decision=self.receipt,journal=self.journal)

    def submit(self,prepared=None):
        return self.gateway.submit(self.ledger,self.intent.intent_id,prepared=prepared or self.f.prepare(),
            limits=self.f.limits,calendar=self.f.f.calendar,controls=self.control_store,
            control_snapshot=self.control,journal=self.journal,book=self.f.book,baselines=self.f.f.baselines)

    def test_cleanup_decision_retains_warnings_and_replays_original_prefix_after_admission(self):
        self.stage();record=self.journal.replay(self.receipt.sequence,ledger=self.ledger).read()
        self.assertEqual(record['outcome']['action'],'sell')
        self.assertIn('native_protection_unprotected_position',record['inputs']['prepared']['findings'])
        self.assertEqual(record['inputs']['cleanup']['remaining_quantity'],5)
        self.assertEqual(record['inputs']['basis'],'native_cleanup')
        self.assertFalse(self.receipt.public_summary()['execution_enabled'])

    def test_one_owned_simple_sell_is_sent_after_durable_decision_and_claim(self):
        self.stage();view=self.f.prepare();result=self.submit(view)
        self.assertEqual(result['state'],'open');self.assertEqual(len(self.posts),1)
        self.assertEqual(self.posts[0]['side'],'sell');self.assertEqual(self.posts[0]['qty'],'5')
        self.assertEqual(self.posts[0]['order_class'],'simple');self.assertFalse(self.posts[0]['extended_hours'])
        self.assertIsNone(self.submit(view));self.assertEqual(len(self.posts),1)

    def test_recorded_prefix_requires_original_ledger_and_rejects_a_fork(self):
        self.record()
        with self.assertRaises(DecisionError):self.journal.replay(self.receipt.sequence)
        raw=self.receipt.read()['inputs'];raw['cleanup']['prefix']['digest']='f'*64
        with self.assertRaises(DecisionError):self.journal.append(DecisionFrame(_json(raw)),ledger=self.ledger)
        self.assertEqual(self.journal.verify()['decisions'],1)

    def test_unavailable_final_risk_is_known_unsent_with_previous_measures_preserved(self):
        self.stage();view=self.f.prepare()
        with patch.object(self.m,'recheck_ticket',side_effect=OSError('PRIVATE_RISK')):
            self.assertIsNone(self.submit(view))
        record=self.ledger.load(self.intent.intent_id)
        self.assertEqual(record['state'],'rejected_by_risk')
        self.assertIn('risk_recheck_unavailable',record['risk']['violations'])
        self.assertEqual(record['risk']['gross_exposure_after_usd'],'750.5')
        self.assertEqual(self.posts,[])

    def test_caller_cannot_relax_the_committed_decision_risk_limits(self):
        self.stage()
        view=self.f.prepare(quotes=(Quote('AAPL','150','151',self.n.now,self.n.now),))
        self.f.limits=replace(self.f.limits,max_spread_bps='100')
        with self.assertRaises(BrokerError):self.submit(view)
        self.assertEqual(self.posts,[])
        self.assertEqual(self.ledger.load(self.intent.intent_id)['state'],'reserved')

    def test_fixed_transport_budget_refuses_before_claim_and_remains_retirable(self):
        self.f.limits=replace(self.f.limits,max_order_notional_usd='2000',max_symbol_notional_usd='2000')
        self.policy=replace(self.policy,policy=replace(self.policy.policy,
            risk_limits_sha256=_hash(_json(_base(self.f.limits,RiskLimits)))))
        self.intent=replace(self.intent,limit_price_usd='201')
        original=self.f.prepare
        def prepare(**kwargs):
            kwargs.setdefault('quotes',(Quote('AAPL','201','201.1',self.n.now,self.n.now),))
            return original(**kwargs)
        self.f.prepare=prepare
        self.stage();view=self.f.prepare()
        with self.assertRaises(BrokerError):self.submit(view)
        record=self.ledger.load(self.intent.intent_id)
        self.assertEqual(self.posts,[]);self.assertEqual(record['state'],'reserved')
        retired=self.f.m.retire_cleanup_reservation(self.ledger,self.intent.intent_id,
            expected_sha256=_hash(_json(record)),reason='pre_dispatch_abort',now=self.n.now)
        self.assertEqual(retired['state'],'rejected_by_risk')

    def test_later_caller_calendar_cannot_extend_the_recorded_session_close(self):
        from paperbot_us.data import Calendar,Session
        regular=self.f.f.calendar
        early=Calendar([Session.from_alpaca(dict(date='2026-10-23',open='09:30',close='13:00'))],
            start='2026-10-23',end='2026-10-23')
        self.f.f.calendar=early;self.n.reconciler.calendar=early
        self.n.now=self.n.now.replace(hour=16,minute=59,second=50)
        self.intent=replace(self.intent,expires_at=self.n.now+timedelta(seconds=20))
        self.stage();view=self.f.prepare()
        self.n.now+=timedelta(seconds=15);self.f.f.calendar=regular
        with self.assertRaises(BrokerError):self.submit(view)
        self.assertEqual(self.posts,[])

    def test_account_book_change_after_claim_cannot_use_old_share_proof(self):
        self.stage();view=self.f.prepare();original=self.f.m.prepare_cleanup_dispatch
        def claim(*args,**kwargs):
            result=original(*args,**kwargs);self.n.now+=timedelta(seconds=1)
            from paperbot_us.accounting import AccountSnapshot
            from paperbot_us.native_orders import accounting_evidence
            previous=self.f.book.snapshot();anchor=self.f.book.read_anchor()
            window=self.n.broker.activities(after=anchor.as_of,until=self.n.now)
            account=AccountSnapshot(self.n.now,previous.account.cash_usd,previous.account.equity_usd,
                False,previous.account.positions,previous.account.account_sha256)
            self.f.book.refresh(window=window,orders=accounting_evidence(self.ledger.all_orders()),
                account=account,now=self.n.now)
            return result
        with patch.object(self.f.m,'prepare_cleanup_dispatch',side_effect=claim):self.assertIsNone(self.submit(view))
        self.assertEqual(self.posts,[])

    def test_unrecorded_legacy_allocation_cannot_be_upgraded_into_sendable_order(self):
        self.f.admit();view=self.f.prepare()
        with self.assertRaises(BrokerError):self.submit(view)
        self.assertEqual(self.posts,[])
        self.record()
        with self.assertRaises(LedgerError):self.f.m.admit_cleanup(self.ledger,self.intent,prepared=view,
            limits=self.f.limits,calendar=self.f.f.calendar,now=self.n.now,decision=self.receipt,journal=self.journal)

    def test_disabled_execution_policy_remains_a_veto_for_cleanup(self):
        disabled=replace(self.policy,policy=replace(self.policy.policy,execution_enabled=False))
        view=self.record(policy=disabled)
        self.assertIn('execution_disabled',self.receipt.read()['outcome']['vetoes'])
        with self.assertRaises(LedgerError):self.f.m.admit_cleanup(self.ledger,self.intent,prepared=view,
            limits=self.f.limits,calendar=self.f.f.calendar,now=self.n.now,decision=self.receipt,journal=self.journal)
        self.assertEqual(self.posts,[])

    def test_unknown_transport_keeps_one_shot_claim_across_gateway_restart(self):
        self.stage();view=self.f.prepare();self.failure=OSError('PRIVATE_TRANSPORT')
        with self.assertRaises(BrokerError) as caught:self.submit(view)
        self.assertNotIn('PRIVATE',str(caught.exception))
        self.assertEqual(self.ledger.load(self.intent.intent_id)['state'],'submission_unknown')
        self.gateway=self.m.CleanupPaperGateway(self.n.broker)
        self.assertIsNone(self.submit(view));self.assertEqual(len(self.posts),1)

    def test_changed_controls_before_dispatch_persist_known_unsent_veto(self):
        self.stage();view=self.f.prepare()
        self.control_store.update(expected_revision=1,pause_entries=True,halt_orders=True,
            reason='operator_halt',now=self.n.now)
        self.assertIsNone(self.submit(view));self.assertEqual(self.posts,[])
        self.assertEqual(self.ledger.load(self.intent.intent_id)['state'],'rejected_by_risk')

    def test_clock_delay_after_atomic_claim_prevents_network_and_releases_only_known_unsent(self):
        self.stage();view=self.f.prepare();original=self.f.m.prepare_cleanup_dispatch
        def claim(*args,**kwargs):
            result=original(*args,**kwargs);self.n.now+=timedelta(seconds=31);return result
        with patch.object(self.f.m,'prepare_cleanup_dispatch',side_effect=claim):self.assertIsNone(self.submit(view))
        self.assertEqual(self.posts,[])
        self.assertEqual(self.ledger.load(self.intent.intent_id)['state'],'rejected_by_risk')

    def test_family_change_after_claim_vetoes_without_overwriting_new_stop_fills(self):
        self.stage();view=self.f.prepare();original=self.f.m.prepare_cleanup_dispatch
        def claim(*args,**kwargs):
            result=original(*args,**kwargs);self.n.now+=timedelta(seconds=3)
            self.n.f.child.update(filled_qty='2',filled_avg_price='145',updated_at=self.n.now.isoformat())
            self.ledger.observe_native_group(self.f.family,row=self.n.f.parent,received_at=self.n.now)
            return result
        with patch.object(self.f.m,'prepare_cleanup_dispatch',side_effect=claim):self.assertIsNone(self.submit(view))
        self.assertEqual(self.posts,[])
        self.assertEqual(self.ledger.load(self.f.family)['native_stop']['stop']['filled_quantity'],2)

    def test_post_response_ledger_failure_stays_unknown_and_never_retries(self):
        self.stage();view=self.f.prepare();original=self.ledger._persist
        def persist(db,record,kind,at):
            if kind=='broker_observation':raise OSError('PRIVATE_DISK')
            return original(db,record,kind,at)
        with patch.object(self.ledger,'_persist',side_effect=persist):
            with self.assertRaises(BrokerError):self.submit(view)
        self.assertEqual(self.ledger.load(self.intent.intent_id)['state'],'submission_unknown')
        self.assertIsNone(self.submit(view));self.assertEqual(len(self.posts),1)

    def test_partial_sell_get_reconciliation_retains_allocation_until_observed_cancellation(self):
        self.stage();self.submit();self.n.now+=timedelta(seconds=1)
        self.row.update(status='partially_filled',filled_qty='2',filled_avg_price='150',updated_at=self.n.now.isoformat())
        self.n.visible=[deepcopy(self.row)]
        self.n.positions[0].update(qty='3',qty_available='0',market_value='450')
        self.n.account.update(cash='9550',equity='10000')
        self.n.fills.append(self.n.fill('00000000-0000-4000-8000-000000000081','sell','2','150',self.row['id'],11,'2','3'))
        view=self.f.prepare()
        self.assertEqual(view.remaining_quantity,3);self.assertEqual(view.available_quantity,0)
        self.assertEqual(self.ledger.load(self.intent.intent_id)['filled_quantity'],2)
        self.n.now+=timedelta(seconds=1)
        self.row.update(status='canceled',updated_at=self.n.now.isoformat())
        self.n.visible=[];self.n.positions[0]['qty_available']='3'
        next_view=self.f.prepare()
        self.assertEqual(next_view.remaining_quantity,3);self.assertEqual(next_view.available_quantity,3)
        self.assertEqual(self.f.book.snapshot().receipt.result.profit_usd,0)
        self.assertEqual(self.f.book.replay(self.f.book.snapshot().receipt.sequence),self.f.book.snapshot().receipt.result)

    def test_concurrent_gateway_calls_share_one_claim_and_send_only_once(self):
        self.stage();view=self.f.prepare();original=self.f.m.prepare_cleanup_dispatch
        barrier=threading.Barrier(2)
        def claim(*args,**kwargs):
            barrier.wait(timeout=5);return original(*args,**kwargs)
        with patch.object(self.f.m,'prepare_cleanup_dispatch',side_effect=claim):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results=list(pool.map(lambda _:self.submit(view),range(2)))
        self.assertEqual(sum(r is not None for r in results),1)
        self.assertEqual(len(self.posts),1)
        self.assertEqual(self.ledger.load(self.intent.intent_id)['state'],'open')

    def test_control_writer_cannot_commit_during_the_started_post(self):
        self.stage();view=self.f.prepare();started=threading.Event();done=threading.Event();errors=[]
        def writer():
            started.set()
            try:self.control_store.update(expected_revision=1,pause_entries=True,halt_orders=True,
                reason='operator_halt',now=self.n.now)
            except Exception as error:errors.append(type(error).__name__)
            finally:done.set()
        thread=threading.Thread(target=writer)
        def hook():
            thread.start();self.assertTrue(started.wait(2));self.assertFalse(done.wait(.1))
        self.post_hook=hook
        try:result=self.submit(view)
        finally:
            if thread.ident is not None:thread.join(5)
        self.assertEqual(result['state'],'open');self.assertTrue(done.is_set());self.assertEqual(errors,[])
        self.assertTrue(self.control_store.read(now=self.n.now).halt_orders)


if __name__=='__main__':unittest.main()
