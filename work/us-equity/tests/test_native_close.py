from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import importlib,importlib.util,unittest
import hashlib,json,os,sqlite3
from contextlib import closing
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

import test_native_risk_view
from paperbot_us.broker import BrokerError,HttpReply
from paperbot_us.controls import ControlJournal
from paperbot_us.decision_journal import DecisionJournal,DecisionError
from paperbot_us.execution_policy import ExecutionPolicy,PolicySnapshot
from paperbot_us.orders import LedgerError,_base,_hash,_json
from paperbot_us.risk import Quote,RiskLimits
from paperbot_us.risk_view import RiskViewError


class NativeCloseTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.native_close'),
            'explicit stop-to-close preparation required')
        self.m=importlib.import_module('paperbot_us.native_close')
        self.f=test_native_risk_view.NativeRiskPreparationTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.n=self.f.n;self.ledger=self.f.ledger;self.family=self.n.f.intent.intent_id
        self.limits=RiskLimits();self.controls=ControlJournal(self.n.f.f.path.with_name('close-controls.sqlite'))
        self.control=self.controls.initialize(now=self.n.now)
        self.policy=PolicySnapshot(ExecutionPolicy(1,10000,True,'1'*64,'2'*64,
            _hash(_json(_base(self.limits,RiskLimits))),5,True),'3'*64)
        self.journal=DecisionJournal(self.n.f.f.path.with_name('close-decisions.sqlite'),clock=lambda:self.n.now)
        self.journal.initialize();self.deletes=[];self.settle=True;self.failure=None;self.after_delete=None;self.reply_status=204
        self.old_transport=self.n.broker._transport;self.n.broker._transport=self.transport
        self.gateway=self.m.NativeCloseGateway(self.n.broker)

    def transport(self,method,url,headers,body):
        if method=='DELETE':
            self.assertEqual(urlsplit(url).path,'/v2/orders/'+self.n.f.child_id)
            self.assertIsNone(body);self.deletes.append(urlsplit(url).path)
            close=self.ledger.load(self.family)['native_stop']['close']
            self.assertEqual(close['outcome'],'claimed')
            if self.after_delete:self.after_delete()
            elif self.settle:
                self.n.f.child.update(status='canceled',updated_at=self.n.now.isoformat())
                self.n.visible=[];self.n.positions[0]['qty_available']=self.n.positions[0]['qty']
            if self.failure:raise self.failure
            return HttpReply(self.reply_status)
        self.assertEqual(method,'GET','close preparation cannot send a replacement sell')
        return self.old_transport(method,url,headers,body)

    def prepare(self,quotes=None):
        view=self.f.prepare(quotes=quotes or (Quote('AAPL','120','120.1',self.n.now,self.n.now),))
        return self.m.prepare_close(ledger=self.ledger,prepared=view,family_id=self.family,
            calendar=self.f.calendar,now=self.n.now,reason='account_risk')

    def record(self,plan=None):
        import paperbot_us.decision_journal as d
        self.plan=plan or self.prepare()
        self.assertTrue(hasattr(d,'capture_close_decision'),'close preparation needs its own decision scope')
        frame=d.capture_close_decision(ledger=self.ledger,prepared=self.plan,source=None,policy=self.policy,
            controls=self.control,limits=self.limits,calendar=self.f.calendar,now=self.n.now)
        self.receipt=self.journal.append(frame,ledger=self.ledger)
        return self.plan

    def request(self):
        return self.gateway.request(self.ledger,self.plan,decision=self.receipt,journal=self.journal,
            controls=self.controls,control_snapshot=self.control,limits=self.limits,calendar=self.f.calendar,
            book=self.f.book,baselines=self.f.baselines,sources=getattr(self,'sources',None))

    def test_preview_transfers_only_target_stop_while_actual_shares_remain_reserved(self):
        plan=self.prepare()
        self.assertEqual(plan.original.portfolio.stop_reservations[0].remaining_quantity,5)
        self.assertEqual(plan.preview.portfolio.stop_reservations,())
        self.assertEqual(plan.public_summary()['shares_available_now'],0)
        self.assertEqual(plan.public_summary()['preview_close_quantity'],5)
        self.assertFalse(plan.public_summary()['execution_enabled'])

    def test_account_risk_close_has_explicit_original_stop_evidence_and_replayable_prefix(self):
        self.record();row=self.receipt.read()
        self.assertEqual(row['inputs']['basis'],'native_risk_close')
        self.assertEqual(row['outcome']['action'],'sell')
        self.assertEqual(row['inputs']['close']['original_portfolio']['stop_reservations'][0]['remaining_quantity'],5)
        self.assertEqual(self.journal.replay(self.receipt.sequence,ledger=self.ledger),self.receipt)
        with self.assertRaises(DecisionError):self.journal.replay(self.receipt.sequence)

    def test_non_breached_risk_cannot_remove_protection(self):
        plan=self.prepare((Quote('AAPL','150','150.1',self.n.now,self.n.now),));self.record(plan)
        self.assertIn('close_risk_trigger_not_met',self.receipt.read()['outcome']['vetoes'])
        with self.assertRaises(BrokerError):self.request()
        self.assertEqual(self.deletes,[]);self.assertEqual(self.n.f.child['status'],'new')

    def test_ack_then_actual_terminal_child_requires_new_account_cleanup_not_a_sell(self):
        self.record();result=self.request()
        self.assertEqual(result.status,'cleanup_required');self.assertEqual(result.remaining_quantity,5)
        self.assertEqual(len(self.deletes),1)
        self.assertIn('reconcile_account',result.required_actions)
        self.assertFalse(result.public_summary()['ready_for_execution'])

    def test_ack_with_active_child_keeps_stop_reservation_and_never_repeats_delete(self):
        self.settle=False;self.record();result=self.request()
        self.assertEqual(result.status,'awaiting_stop_terminal')
        self.gateway=self.m.NativeCloseGateway(self.n.broker);again=self.request()
        self.assertEqual(again.status,'awaiting_stop_terminal');self.assertEqual(len(self.deletes),1)
        with self.assertRaises(RiskViewError):self.f.prepare()

    def test_confirmed_rejected_cancel_with_active_child_is_failed_close_not_pending_sale(self):
        self.settle=False;self.reply_status=422;self.record();result=self.request()
        self.assertEqual(result.status,'close_rejected_protected')
        self.assertEqual(result.remaining_quantity,5);self.assertEqual(self.n.f.child['status'],'new')

    def test_partial_stop_fill_retains_realized_loss_and_previews_only_remaining_shares(self):
        self.n.f.child.update(status='partially_filled',filled_qty='2',filled_avg_price='130',
            updated_at=(self.n.f.now+timedelta(seconds=4)).isoformat())
        self.n.visible=[deepcopy(self.n.f.child)]
        self.n.positions[0].update(qty='3',qty_available='0',market_value='450')
        self.n.account.update(cash='9510',equity='9960')
        self.n.fills.append(self.n.fill('00000000-0000-4000-8000-000000000082','sell','2','130',self.n.f.child_id,4,'2','3'))
        self.record();self.assertEqual(self.plan.intent.quantity,3)
        result=self.request();self.assertEqual(result.remaining_quantity,3)
        self.assertEqual(self.f.book.snapshot().receipt.result.realized_usd,-40)

    def test_get_only_restart_recovery_does_not_require_the_model_decision_archive(self):
        self.settle=False;self.record();self.request()
        self.assertTrue(hasattr(self.gateway,'recover'),'read-only recovery must survive missing model archive')
        with patch.object(self.journal,'replay',side_effect=OSError('PRIVATE_ARCHIVE')):
            result=self.gateway.recover(self.ledger,self.family)
        self.assertEqual(result.status,'awaiting_stop_terminal');self.assertEqual(len(self.deletes),1)

    def test_result_validator_cannot_mark_an_http_ack_as_known_unsent(self):
        self.record();ticket=self.m.claim_close(self.ledger,self.plan,self.receipt,self.control,now=self.n.now)
        with self.assertRaises(LedgerError):self.m.finish_close(self.ledger,ticket,outcome='not_sent',
            observed_at=self.n.now,http_status=204,vetoes=('close_deadline',))
        self.assertIsNotNone(self.ledger.load(self.family)['native_stop']['stop_cancel_requested_at'])

    def test_legacy_unresolved_stop_cancel_also_vetoes_ordinary_risk_preparation(self):
        self.f.prepare();self.ledger.prepare_stop_cancellation(self.family,at=self.n.now)
        with self.assertRaises(RiskViewError):self.f.prepare()

    def test_post_claim_fill_change_is_preserved_by_known_unsent_result(self):
        self.record();original=self.m.claim_close
        def changed(*args,**kwargs):
            result=original(*args,**kwargs);self.n.now+=timedelta(seconds=3)
            self.n.f.child.update(status='partially_filled',filled_qty='2',filled_avg_price='119',updated_at=self.n.now.isoformat())
            self.ledger.observe_native_group(self.family,row=self.n.f.parent,received_at=self.n.now)
            return result
        with patch.object(self.m,'claim_close',side_effect=changed):self.request()
        self.assertEqual(self.deletes,[])
        self.assertEqual(self.ledger.load(self.family)['native_stop']['stop']['filled_quantity'],2)

    def test_unknown_delete_cannot_become_permission_for_another_sell(self):
        self.settle=False;self.failure=OSError('PRIVATE_TRANSPORT');self.record();result=self.request()
        self.assertEqual(result.status,'awaiting_stop_terminal');self.assertEqual(result.remaining_quantity,5)
        self.assertNotIn('PRIVATE',str(result.public_summary()))
        self.failure=None;self.request();self.assertEqual(len(self.deletes),1)

    def test_current_global_halt_preserves_active_stop_and_clears_only_known_unsent_claim(self):
        self.record();self.controls.update(expected_revision=1,pause_entries=True,halt_orders=True,
            reason='operator_halt',now=self.n.now)
        result=self.request();self.assertEqual(self.deletes,[])
        record=self.ledger.load(self.family)
        self.assertEqual(record['native_stop']['close']['outcome'],'not_sent')
        self.assertIsNone(record['native_stop']['stop_cancel_requested_at'])
        self.assertEqual(self.n.f.child['status'],'new')

    def test_closed_during_cancel_is_observed_without_inventing_a_second_sell(self):
        def fill():
            self.n.f.child.update(status='filled',filled_qty='5',filled_avg_price='119',
                filled_at=self.n.now.isoformat(),updated_at=self.n.now.isoformat())
            self.n.visible=[];self.n.positions=[]
        self.after_delete=fill;self.record();result=self.request()
        self.assertEqual(result.status,'group_closed');self.assertEqual(result.remaining_quantity,0)
        self.assertIn('reconcile_account',result.required_actions);self.assertEqual(len(self.deletes),1)

    def test_latest_missing_lookup_never_uses_cached_protection_or_terminal_proof(self):
        self.record();self.n.missing=True
        result=self.request()
        self.assertEqual(result.status,'unconfirmed_group');self.assertIsNone(result.remaining_quantity)
        self.assertIn('reconcile_group',result.required_actions)

    def test_latest_missing_child_cannot_report_cached_remaining_shares_as_current(self):
        self.record();self.n.f.parent['legs']=[]
        result=self.request()
        self.assertEqual(result.status,'unconfirmed_group');self.assertIsNone(result.remaining_quantity)

    def test_delay_after_claim_records_known_unsent_without_removing_stop(self):
        self.record();original=self.m.claim_close
        def delayed(*args,**kwargs):
            value=original(*args,**kwargs);self.n.now+=timedelta(seconds=31);return value
        with patch.object(self.m,'claim_close',side_effect=delayed):self.request()
        self.assertEqual(self.deletes,[])
        self.assertEqual(self.ledger.load(self.family)['native_stop']['close']['outcome'],'not_sent')

    def test_clock_cannot_claim_before_the_committed_authorizing_decision(self):
        plan=self.prepare();self.n.now+=timedelta(seconds=1);self.record(plan)
        self.n.now-=timedelta(seconds=.5)
        with self.assertRaises(BrokerError):self.request()
        self.assertEqual(self.deletes,[])
        self.assertIsNone(self.ledger.load(self.family)['native_stop']['stop_cancel_requested_at'])

    def test_clock_rollback_after_claim_is_durable_known_unsent(self):
        plan=self.prepare();self.n.now+=timedelta(seconds=1);self.record(plan)
        original=self.m.claim_close
        def rollback(*args,**kwargs):
            value=original(*args,**kwargs);self.n.now-=timedelta(seconds=.5);return value
        with patch.object(self.m,'claim_close',side_effect=rollback):self.request()
        self.assertEqual(self.deletes,[])
        close=self.ledger.load(self.family)['native_stop']['close']
        self.assertEqual(close['outcome'],'not_sent');self.assertTrue(close['clock_regression'])

    def test_expired_recovery_deadline_does_not_repeat_cancel_or_free_uncertain_shares(self):
        self.settle=False;self.record();self.request();self.n.now+=timedelta(seconds=31)
        result=self.request()
        self.assertEqual(result.status,'recovery_overdue');self.assertEqual(len(self.deletes),1)
        self.assertIn('reconcile_group',result.required_actions)
        self.assertFalse(result.public_summary()['ready_for_execution'])

    def test_close_preparation_refuses_a_session_too_near_its_end(self):
        self.n.now=self.n.now.replace(hour=19,minute=59,second=20)
        with self.assertRaises(RiskViewError):self.prepare()
        self.assertEqual(self.deletes,[])

    def test_quantitative_close_requires_and_replays_actual_quant_and_model_sell_records(self):
        import test_shadow,test_replay_bridge
        from paperbot_us.data import Bar
        from paperbot_us.model_runner import assistance_once
        from paperbot_us.replay_bridge import PinnedRelease,ReplayBridge
        from paperbot_us.decision_journal import ReplaySources,capture_close_decision,strategy_digest
        shadow=test_shadow.ShadowTests();shadow.setUp();self.addCleanup(shadow.doCleanups)
        def bars(symbol):
            start=shadow.calendar.sessions[shadow.now.date()].opens
            count=int((shadow.now-start).total_seconds()//900)
            return [Bar(symbol,start+timedelta(minutes=15*i),shadow.now,150,155,145,
                150+i*.1,200 if i==count-1 else 100) for i in range(count)]
        shadow.bars=bars;shadow.run_once()
        class Client:
            def assess(self,request):return dict(action='sell',reason='退出条件满足，先准备平仓。',
                model='deepseek-flash',fingerprint='fp_test',prompt_tokens=200,completion_tokens=30,total_tokens=230)
        model=shadow.root/'sell-model.sqlite'
        for step in range(3):
            at=shadow.now+timedelta(seconds=61*step)
            assistance_once(client=Client(),source_archive=shadow.root/'decisions.sqlite',archive=model,
                publish=shadow.root/'sell-model.json',clock=lambda at=at:at)
        with closing(sqlite3.connect(model)) as db:
            reviews=[json.loads(r[0]) for r in db.execute('SELECT payload FROM observations')]
        with closing(sqlite3.connect(shadow.root/'decisions.sqlite')) as db:
            quant=[json.loads(r[0]) for r in db.execute('SELECT payload FROM observations')]
        identities={r['identity'] for r in quant if r['kind']=='decision' and r['symbol']=='AAPL'}
        review=next(r for r in reviews if r['kind']=='model_review' and r['source_identity'] in identities)
        release=shadow.root/'pinned';package=release/'paperbot_us';package.mkdir(parents=True);files={}
        live=Path(self.m.__file__).parent
        for name in test_replay_bridge.CORE:
            raw=(live/(name+'.py')).read_bytes();(package/(name+'.py')).write_bytes(raw)
            files['paperbot_us/'+name+'.py']=hashlib.sha256(raw).hexdigest()
        bridge=ReplayBridge(PinnedRelease(release,files=files,package_sha256='a'*64),
            timezone_root=os.environ.get('PYTHONTZPATH') or None)
        source=bridge.review(model,review['identity'],source_archive=shadow.root/'decisions.sqlite')
        self.sources=ReplaySources(bridge,shadow.root/'decisions.sqlite',model)
        self.assertTrue(source.read()['quant']['record']['exit_candidate'])
        self.n.now=shadow.now+timedelta(seconds=122);price=bars('AAPL')[-1].close
        self.n.positions[0].update(current_price=str(price),market_value=str(price*5))
        self.n.account['equity']=str(9250+price*5)
        view=self.f.prepare(quotes=(Quote('AAPL',str(price),str(round(price+.01,2)),self.n.now,self.n.now),))
        self.plan=self.m.prepare_close(ledger=self.ledger,prepared=view,family_id=self.family,
            calendar=self.f.calendar,now=self.n.now,reason='quantitative')
        policy=replace(self.policy,policy=replace(self.policy.policy,strategy_sha256=strategy_digest(source)))
        frame=capture_close_decision(ledger=self.ledger,prepared=self.plan,source=source,policy=policy,
            controls=self.control,limits=self.limits,calendar=self.f.calendar,now=self.n.now)
        self.receipt=self.journal.append(frame,sources=self.sources,ledger=self.ledger)
        self.assertEqual(self.receipt.read()['outcome']['action'],'sell')
        self.assertEqual(self.request().status,'cleanup_required');self.assertEqual(len(self.deletes),1)


if __name__=='__main__':unittest.main()
