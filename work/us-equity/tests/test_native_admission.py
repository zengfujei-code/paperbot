from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import importlib,importlib.util,unittest
from unittest.mock import patch
from urllib.parse import urlsplit

import test_native_risk_view,test_decision_journal
from paperbot_us.orders import LedgerError,_hash,_json
from paperbot_us.broker import HttpReply,BrokerError
from paperbot_us.native_execution import NativePaperGateway
from paperbot_us.risk import OrderIntent,Quote,RiskLimits,RiskState


class NativeAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.native_admission'),
            'group-aware admission must prove the actual account and every owned reservation')
        self.m=importlib.import_module('paperbot_us.native_admission')
        self.f=test_native_risk_view.NativeRiskPreparationTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.d=test_decision_journal.DecisionJournalTests();self.d.setUp();self.addCleanup(self.d.doCleanups)
        self.n=self.f.n;self.n.now=self.d.now;self.ledger=self.f.ledger
        self.source,self.policy=self.d.buy_proof();self.limits=RiskLimits()
        self.intent=replace(self.d.intent,intent_id='e'*64)
        self.prepare()

    def prepare(self):
        self.cycle=self.n.reconciler.refresh()
        self.quotes=(Quote('AAPL','150','150.1',self.n.now,self.n.now),
            Quote(self.d.symbol,'101.84','101.85',self.n.now,self.n.now))
        self.view=self.f.m.prepare_native_risk_view(book=self.f.book,ledger=self.ledger,
            baselines=self.f.baselines,cycle=self.cycle,quotes=self.quotes,symbol=self.d.symbol,
            operator_state=RiskState(pause_entries=False),calendar=self.f.calendar,now=self.n.now)
        self.receipt=self.d.append(self.d.frame(source=self.source,policy=self.policy,prepared=self.view,
            intent=self.intent,now=self.n.now,authorization_sha256='c'*64))
        self.assertEqual(self.receipt.read()['outcome']['action'],'buy',self.receipt.read()['outcome']['vetoes'])
        self.inputs=self.m.NativeEntrySources(self.view,self.cycle,self.f.book,self.f.baselines,
            self.d.store,self.d.sources,self.d.control,self.limits,self.f.calendar)

    def admit(self,**changes):
        return self.m.admit_entry(self.ledger,self.intent,decision=self.receipt,
            sources=changes.pop('sources',self.inputs),now=self.n.now,**changes)

    def test_other_protected_family_remains_reserved_and_new_entry_has_attached_plan(self):
        row=self.admit();self.assertEqual(row['state'],'reserved')
        self.assertEqual(row['native_stop']['plan']['symbol'],self.d.symbol)
        self.assertEqual(row['native_entry']['decision_digest'],self.receipt.digest)
        self.assertEqual(len(row['context']['portfolio']['stop_reservations']),1)
        self.assertEqual(row['context']['portfolio']['cash_usd'],'9250')

    def test_generic_admission_keeps_native_history_veto(self):
        row=self.ledger.admit(self.intent,portfolio=self.view.portfolio,quote=self.view.quote,
            state=self.view.state,limits=self.limits,calendar=self.f.calendar,now=self.n.now)
        self.assertEqual(row['state'],'rejected_by_risk')
        self.assertIn('native_group_coordinator_required',row['context']['reconciliation_findings'])

    def test_omitting_real_stop_or_inventing_cash_never_admits(self):
        for change in (dict(stop_reservations=()),dict(cash_usd='10000',equity_usd='10750')):
            forged=replace(self.view,portfolio=replace(self.view.portfolio,**change))
            with self.assertRaises(LedgerError):self.admit(sources=replace(self.inputs,prepared=forged))
        self.assertEqual(len(self.ledger.all_orders()),1)

    def test_changed_ledger_vetoes_stale_preparation(self):
        self.ledger.admit(replace(self.intent,intent_id='f'*64),portfolio=self.view.portfolio,
            quote=self.view.quote,state=self.view.state,limits=self.limits,calendar=self.f.calendar,now=self.n.now)
        with self.assertRaises(LedgerError):self.admit()

    def test_changed_account_or_stale_quotes_cannot_create_a_reservation(self):
        self.n.now+=timedelta(seconds=31)
        with self.assertRaises(LedgerError):self.admit()
        self.n.reconciler.refresh()
        with self.assertRaises(LedgerError):self.admit()

    def test_changed_limits_and_missing_decision_archive_fail_before_admission(self):
        with self.assertRaises(LedgerError):self.admit(sources=replace(self.inputs,
            limits=replace(self.limits,max_order_notional_usd='999')))
        with patch.object(self.d.store,'replay',side_effect=OSError('PRIVATE_ARCHIVE')):
            with self.assertRaises(LedgerError) as caught:self.admit()
        self.assertNotIn('PRIVATE',str(caught.exception))

    def test_same_committed_entry_is_idempotent_without_changing_its_stop(self):
        first=self.admit();self.assertEqual(self.admit(),first)
        self.intent=replace(self.intent,quantity=4)
        with self.assertRaises(LedgerError):self.admit()

    def test_second_reserved_buy_is_charged_against_symbol_cap(self):
        self.admit();self.intent=replace(self.intent,intent_id='f'*64);self.prepare()
        row=self.admit();self.assertEqual(row['state'],'rejected_by_risk')
        self.assertIn('symbol_exposure_limit',row['risk']['violations'])

    def test_global_halt_or_entry_pause_cannot_be_cleared_by_prepared_state(self):
        for field in ('pause_entries','halt_orders'):
            control=replace(self.d.control,**{field:True})
            with self.assertRaises(LedgerError):self.admit(sources=replace(self.inputs,control_snapshot=control))

    def test_generic_submission_cannot_reuse_grouped_reservation(self):
        row=self.admit()
        with self.assertRaises(LedgerError):self.ledger.prepare_native_submission(row['intent_id'],
            portfolio=self.view.portfolio,quote=self.view.quote,state=self.view.state,
            limits=self.limits,calendar=self.f.calendar,now=self.n.now)


class GroupedEntryDispatchTests(unittest.TestCase):
    prepare=NativeAdmissionTests.prepare
    admit=NativeAdmissionTests.admit

    def setUp(self):
        NativeAdmissionTests.setUp(self);self.admit();self.posts=[];self.failure=False
        cycle=self.n.reconciler.refresh()
        view=self.f.m.prepare_native_risk_view(book=self.f.book,ledger=self.ledger,baselines=self.f.baselines,
            cycle=cycle,quotes=self.quotes,symbol=self.d.symbol,operator_state=RiskState(pause_entries=False),
            calendar=self.f.calendar,now=self.n.now)
        self.inputs=replace(self.inputs,prepared=view,cycle=cycle)
        original=self.n.broker._transport
        def transport(method,url,headers,body):
            if method=='POST':
                self.assertEqual(urlsplit(url).netloc,'paper-api.alpaca.markets');self.posts.append(deepcopy(body))
                self.assertEqual(self.ledger.load(self.intent.intent_id)['state'],'submission_unknown')
                if self.failure:raise OSError('PRIVATE_TRANSPORT')
                parent=deepcopy(self.n.f.parent);child=deepcopy(self.n.f.child)
                common=dict(symbol=body['symbol'],qty=body['qty'],filled_qty='0',filled_avg_price=None,
                    filled_at=None,created_at=self.n.now.isoformat(),updated_at=self.n.now.isoformat())
                parent.update(**common,id='00000000-0000-4000-8000-000000000061',status='new',
                    limit_price=body['limit_price'],client_order_id=body['client_order_id'])
                child.update(**common,id='00000000-0000-4000-8000-000000000062',status='held',
                    stop_price=body['stop_loss']['stop_price'],group_id=parent['id'])
                parent['legs']=[child];return HttpReply(201,parent)
            return original(method,url,headers,body)
        self.n.broker._transport=transport;self.gateway=NativePaperGateway(self.n.broker)

    def submit(self):
        return self.gateway.submit(self.ledger,self.intent.intent_id,controls=self.d.controls,
            control_snapshot=self.d.control,market_price_usd='101.85',market_at=self.n.now,
            entry_sources=self.inputs)

    def test_one_grouped_post_retains_other_stop_and_recovers_both_new_legs(self):
        row=self.submit();self.assertEqual(row['state'],'open');self.assertEqual(len(self.posts),1)
        self.assertEqual(row['native_stop']['stop']['broker_order_id'],'00000000-0000-4000-8000-000000000062')
        self.assertIsNone(self.submit());self.assertEqual(len(self.posts),1)

    def test_unknown_grouped_post_never_retries(self):
        self.failure=True
        with self.assertRaises(BrokerError):self.submit()
        self.assertEqual(self.ledger.load(self.intent.intent_id)['state'],'submission_unknown')
        self.assertFalse(self.f.m._known_unsent(self.ledger.load(self.intent.intent_id),self.n.now))
        self.assertIsNone(self.submit());self.assertEqual(len(self.posts),1)

    def test_changed_current_control_prevents_grouped_post(self):
        self.d.controls.update(expected_revision=2,pause_entries=False,halt_orders=False,
            reason='operator_resume',now=self.n.now)
        self.assertIsNone(self.submit());self.assertEqual(self.posts,[])

    def test_delayed_group_claim_cannot_use_stale_account_or_expired_intent(self):
        original=self.m.claim_entry
        def delayed(*a,**k):
            value=original(*a,**k);self.n.now+=timedelta(seconds=31);return value
        with patch.object(self.m,'claim_entry',side_effect=delayed):self.assertIsNone(self.submit())
        self.assertEqual(self.posts,[])

    def test_changed_account_book_after_claim_blocks_the_network_request(self):
        original=self.m.claim_entry
        def changed(*a,**k):
            value=original(*a,**k);self.n.now+=timedelta(seconds=1);self.n.reconciler.refresh();return value
        with patch.object(self.m,'claim_entry',side_effect=changed):self.assertIsNone(self.submit())
        self.assertEqual(self.posts,[])

    def test_final_source_validation_delay_rechecks_actual_dispatch_clock(self):
        original=self.m.validate_dispatch
        def delayed(*a,**k):
            result=original(*a,**k);self.n.now+=timedelta(seconds=21);return result
        with patch.object(self.m,'validate_dispatch',side_effect=delayed):self.assertIsNone(self.submit())
        self.assertEqual(self.posts,[])

    def test_frozen_wall_clock_cannot_bypass_monotonic_dispatch_deadline(self):
        import inspect
        self.assertIn('monotonic',inspect.signature(NativePaperGateway).parameters)
        elapsed=[0];self.gateway=NativePaperGateway(self.n.broker,monotonic=lambda:elapsed[0])
        original=self.m.validate_dispatch
        def delayed(*a,**k):
            result=original(*a,**k);elapsed[0]=26;return result
        with patch.object(self.m,'validate_dispatch',side_effect=delayed):self.assertIsNone(self.submit())
        self.assertEqual(self.posts,[])

    def test_lost_decision_journal_after_claim_blocks_the_post(self):
        original=self.m.claim_entry
        def missing(*a,**k):
            result=original(*a,**k);self.d.store.path.unlink();return result
        with patch.object(self.m,'claim_entry',side_effect=missing):self.assertTrue(self.submit() is None)
        self.assertEqual(self.posts,[])

    def test_lost_original_model_evidence_after_claim_blocks_the_post(self):
        from pathlib import Path
        original=self.m.claim_entry
        def missing(*a,**k):
            result=original(*a,**k);Path(self.d.sources.model_archive).unlink();return result
        with patch.object(self.m,'claim_entry',side_effect=missing):self.assertTrue(self.submit() is None)
        self.assertEqual(self.posts,[])

    def test_provably_unsent_entry_does_not_poison_the_next_verified_portfolio(self):
        self.d.controls.update(expected_revision=2,pause_entries=False,halt_orders=False,
            reason='operator_resume',now=self.n.now)
        self.assertIsNone(self.submit());self.assertEqual(self.posts,[])
        row=self.ledger.load(self.intent.intent_id)
        self.assertEqual(row['state'],'rejected_by_risk');self.assertIsNotNone(row['submission_claimed_at'])
        cycle=self.n.reconciler.refresh();self.assertTrue(cycle.ready_for_risk,cycle.findings)
        view=self.f.m.prepare_native_risk_view(book=self.f.book,ledger=self.ledger,baselines=self.f.baselines,
            cycle=cycle,quotes=self.quotes,symbol=self.d.symbol,operator_state=RiskState(pause_entries=False),
            calendar=self.f.calendar,now=self.n.now)
        self.assertEqual(len(view.portfolio.stop_reservations),1)
        self.assertIsNone(self.submit());self.assertEqual(self.posts,[])


if __name__=='__main__':unittest.main()
