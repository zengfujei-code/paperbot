from dataclasses import replace
from datetime import timedelta
import hashlib,importlib,json,unittest
from unittest.mock import patch
from types import SimpleNamespace
from copy import deepcopy
from urllib.parse import urlsplit
from paperbot_us.broker import HttpReply

import test_native_admission
from paperbot_us.orders import _json,_hash,_base
from paperbot_us.risk import RiskLimits


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.m=importlib.import_module('paperbot_us.pilot')
        self.f=test_native_admission.NativeAdmissionTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.root=self.f.d.root;self.now=self.f.n.now
        self.path=self.root/'pilot.json'
        self.value=dict(schema=1,kind='experimental_paper_pilot',mode='paper',capital_usd=10000,
            approved_at=self.now.isoformat(),expires_at=(self.now+timedelta(days=30)).isoformat(),
            strategy_sha256=self.f.policy.policy.strategy_sha256,
            risk_limits_sha256=self.f.policy.policy.risk_limits_sha256,
            formal_strategy_accepted=False,live_authorized=False)
        self.save()

    def save(self):
        self.path.write_text(_json(self.value),encoding='utf-8')
        self.digest=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.policy=replace(self.f.policy,policy=replace(self.f.policy.policy,acceptance_sha256=self.digest))
        self.approval=self.m.PilotAuthorization(self.path,self.digest)

    def test_explicit_pilot_is_separate_from_formal_strategy_acceptance(self):
        result=self.approval.verify(policy=self.policy,limits=self.f.limits,now=self.now)
        self.assertEqual(result['mode'],'paper');self.assertFalse(result['formal_strategy_accepted'])

    def test_expired_modified_live_or_different_risk_pilot_is_refused(self):
        for key,value in (('mode','live'),('formal_strategy_accepted',True),('capital_usd',40000),
                          ('risk_limits_sha256','f'*64)):
            with self.subTest(key=key):
                old=dict(self.value);self.value[key]=value;self.save()
                with self.assertRaises(self.m.PilotError):self.approval.verify(policy=self.policy,limits=self.f.limits,now=self.now)
                self.value=old
        self.save()
        with self.assertRaises(self.m.PilotError):self.approval.verify(policy=self.policy,limits=self.f.limits,now=self.now+timedelta(days=31))
        self.path.write_text('{}')
        with self.assertRaises(self.m.PilotError):self.approval.verify(policy=self.policy,limits=self.f.limits,now=self.now)

    def test_abandoned_never_sent_entry_releases_its_own_reservation_only(self):
        row=self.f.admit();self.now+=timedelta(seconds=21)
        retired=self.m.retire_unsent_entry(self.f.ledger,row['intent_id'],now=self.now)
        self.assertEqual(retired['state'],'expired');self.assertIsNone(retired['submission_claimed_at'])
        self.assertIsNotNone(retired['native_stop']);self.assertEqual(len(self.f.ledger.all_orders()),2)

    def test_unknown_submissions_are_never_retired_or_resent(self):
        self.f.admit()
        from paperbot_us.native_admission import claim_entry
        cycle=self.f.n.reconciler.refresh()
        view=self.f.f.m.prepare_native_risk_view(book=self.f.f.book,ledger=self.f.ledger,
            baselines=self.f.f.baselines,cycle=cycle,quotes=self.f.quotes,symbol=self.f.d.symbol,
            operator_state=self.f.view.state,calendar=self.f.f.calendar,now=self.now)
        claim_entry(self.f.ledger,self.f.intent.intent_id,sources=replace(self.f.inputs,prepared=view,cycle=cycle),now=self.now)
        self.now+=timedelta(seconds=40)
        self.assertIsNone(self.m.retire_unsent_entry(self.f.ledger,self.f.intent.intent_id,now=self.now))
        self.assertEqual(self.f.ledger.load(self.f.intent.intent_id)['state'],'submission_unknown')

    def test_pilot_entry_keeps_broker_protective_stop_good_until_cancelled(self):
        row=self.f.m.admit_entry(self.f.ledger,self.f.intent,decision=self.f.receipt,
            sources=self.f.inputs,now=self.now,time_in_force='gtc')
        self.assertEqual(row['native_stop']['plan']['time_in_force'],'gtc')
        self.assertEqual(row['state'],'reserved')

    def test_runtime_uses_real_native_admission_and_one_shot_gateway(self):
        p=self.f
        coordinator=self.m.PilotCoordinator(broker=p.n.broker,ledger=p.ledger,book=p.f.book,
            baselines=p.f.baselines,controls=p.d.controls,decisions=p.d.store,
            repair_decisions=None,reconciler=p.n.reconciler,policy=self.policy,authorization=self.approval,
            continuity=lambda:None,sources=p.d.sources,calendar=p.f.calendar,limits=p.limits,
            snapshots=lambda symbols:{},clock=lambda:self.now)
        # A missing provider quote must stop before reservation/network, while
        # the account observation still completes and explains the veto.
        with patch.object(p.d.sources.bridge,'latest_review',return_value=p.source):
            result=coordinator.tick()
        self.assertEqual(result['status'],'inputs_unavailable')
        self.assertFalse(result['formal_strategy_accepted']);self.assertEqual(len(p.ledger.all_orders()),1)

    def test_continuity_failure_prevents_all_broker_requests(self):
        p=self.f;calls=[]
        def broken():raise OSError('PRIVATE_SEED')
        coordinator=self.m.PilotCoordinator(broker=p.n.broker,ledger=p.ledger,book=p.f.book,
            baselines=p.f.baselines,controls=p.d.controls,decisions=p.d.store,repair_decisions=None,
            reconciler=p.n.reconciler,policy=self.policy,authorization=self.approval,continuity=broken,
            sources=p.d.sources,calendar=p.f.calendar,limits=p.limits,snapshots=lambda symbols:{},clock=lambda:self.now)
        with patch.object(p.n.broker,'_request',side_effect=lambda *a,**k:calls.append(a)):
            result=coordinator.tick()
        self.assertEqual(calls,[]);self.assertEqual(result['status'],'continuity_unavailable')
        self.assertNotIn('PRIVATE',_json(result))

    def test_coordinator_sends_one_protected_paper_order_after_durable_decision(self):
        p=self.f;posts=[];original=p.n.broker._transport
        def transport(method,url,headers,body):
            if method!='POST':return original(method,url,headers,body)
            posts.append(body);self.assertEqual(urlsplit(url).netloc,'paper-api.alpaca.markets')
            rows=p.ledger.all_orders();owned=next(r for r in rows if r['client_order_id']==body['client_order_id'])
            self.assertEqual(owned['state'],'submission_unknown')
            self.assertGreater(p.d.store.verify()['decisions'],1)
            parent,child=deepcopy(p.n.f.parent),deepcopy(p.n.f.child)
            common=dict(symbol=body['symbol'],qty=body['qty'],filled_qty='0',filled_avg_price=None,
                filled_at=None,created_at=self.now.isoformat(),updated_at=self.now.isoformat(),time_in_force=body['time_in_force'])
            parent.update(**common,id='00000000-0000-4000-8000-000000000061',status='new',
                limit_price=body['limit_price'],client_order_id=body['client_order_id'])
            child.update(**common,id='00000000-0000-4000-8000-000000000062',status='held',
                stop_price=body['stop_loss']['stop_price'],group_id=parent['id'])
            parent['legs']=[child];return HttpReply(201,parent)
        p.n.broker._transport=transport
        def quotes(symbols):
            return {s:SimpleNamespace(bid=150 if s=='AAPL' else 101.84,
                ask=150.1 if s=='AAPL' else 101.85,price=150.05 if s=='AAPL' else 101.85,
                quote_at=self.now,trade_at=self.now,received_at=self.now) for s in symbols}
        c=self.m.PilotCoordinator(broker=p.n.broker,ledger=p.ledger,book=p.f.book,
            baselines=p.f.baselines,controls=p.d.controls,decisions=p.d.store,repair_decisions=None,
            reconciler=p.n.reconciler,policy=self.policy,authorization=self.approval,continuity=lambda:None,
            sources=p.d.sources,calendar=p.f.calendar,limits=p.limits,snapshots=quotes,clock=lambda:self.now)
        with patch.object(p.d.sources.bridge,'latest_review',return_value=p.source):result=c.tick()
        self.assertEqual(result['status'],'entry_submitted',result)
        self.assertEqual(len(posts),1);self.assertEqual(posts[0]['order_class'],'oto')
        self.assertIn('stop_loss',posts[0]);self.assertFalse(result['live_authorized'])
        report=c._report('observing_signal',cycle=p.cycle)
        # A changed ledger invalidates the old cycle; use a fresh fixture's
        # proven current cycle to check displayed account arithmetic separately.

    def test_reconciled_projection_shows_actual_held_shares_and_book_profit(self):
        p=self.f
        c=self.m.PilotCoordinator(broker=p.n.broker,ledger=p.ledger,book=p.f.book,
            baselines=p.f.baselines,controls=p.d.controls,decisions=p.d.store,repair_decisions=None,
            reconciler=p.n.reconciler,policy=self.policy,authorization=self.approval,continuity=lambda:None,
            sources=p.d.sources,calendar=p.f.calendar,limits=p.limits,snapshots=lambda s:{},clock=lambda:self.now)
        result=c._report('observing_signal',cycle=p.cycle)
        self.assertTrue(result['accounting_ok']);self.assertEqual(result['cash_usd'],9250)
        self.assertEqual(result['holdings'][0]['symbol'],'AAPL');self.assertEqual(result['holdings'][0]['quantity'],5)
        self.assertEqual(result['open_orders_count'],1);self.assertEqual(result['profit_usd'],0)

    def coordinator(self,snapshots=lambda symbols:{}):
        p=self.f
        return self.m.PilotCoordinator(broker=p.n.broker,ledger=p.ledger,book=p.f.book,
            baselines=p.f.baselines,controls=p.d.controls,decisions=p.d.store,repair_decisions=None,
            reconciler=p.n.reconciler,policy=self.policy,authorization=self.approval,continuity=lambda:None,
            sources=p.d.sources,calendar=p.f.calendar,limits=p.limits,snapshots=snapshots,clock=lambda:self.now)

    def test_expired_pilot_still_recovers_existing_orders_without_new_exposure(self):
        c=self.coordinator();self.now+=timedelta(days=31)
        with patch('paperbot_us.pilot.NativeEntryRecovery.cancel_entry') as recovery:
            result=c.tick()
        self.assertTrue(recovery.called);self.assertFalse(result['execution_enabled'])

    def test_account_loss_exit_is_independent_of_missing_or_other_symbol_model(self):
        p=self.f;p.n.positions[0].update(current_price='128',market_value='640')
        p.n.account.update(equity='9890')
        def quotes(symbols):
            return {s:SimpleNamespace(bid=128,ask=128.01,quote_at=self.now,trade_at=self.now,
                received_at=self.now,price=128) for s in symbols}
        c=self.coordinator(snapshots=quotes)
        with patch.object(p.d.sources.bridge,'latest_review',side_effect=OSError('model missing')) as model:
            with patch('paperbot_us.pilot.NativeCloseGateway.request') as close:
                result=c._tick_locked()
        self.assertEqual(result['status'],'close_requested');self.assertTrue(close.called)
        self.assertFalse(model.called)

    def test_protection_sender_not_sent_and_unknown_are_distinct_from_acknowledged(self):
        self.assertEqual(self.m.repair_status(SimpleNamespace(status='not_sent',request_sent=False)),
            'protection_repair_not_sent')
        self.assertEqual(self.m.repair_status(SimpleNamespace(status='submission_unknown',request_sent=True)),
            'protection_repair_unknown')


if __name__=='__main__':unittest.main()
