from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta
from dataclasses import replace
from decimal import Decimal
import importlib
import importlib.util
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

import test_native_orders
from paperbot_us.broker import PaperBroker,HttpReply,BrokerError
from paperbot_us.controls import ControlJournal


class NativeExecutionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.native_execution'),
                             'protected native entry gateway required')
        self.m=importlib.import_module('paperbot_us.native_execution')
        self.f=test_native_orders.NativeOrderLedgerTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.f.attach();self.now=self.f.now;self.calls=[]
        self.store=ControlJournal(self.f.f.path.with_name('control.sqlite'))
        first=self.store.initialize(now=self.now)
        self.snapshot=self.store.update(expected_revision=first.revision,pause_entries=False,halt_orders=False,
            reason='operator_resume',now=self.now)
        self.response=HttpReply(201,deepcopy(self.f.parent));self.fail_persist=False
        def transport(method,url,headers,body):
            record=self.f.ledger.load(self.f.intent.intent_id)
            self.assertEqual(record['state'],'submission_unknown')
            self.assertIsNotNone(record['submission_claimed_at'])
            self.assertIsNotNone(record['native_stop'])
            self.calls.append((method,url,body))
            if isinstance(self.response,Exception):raise self.response
            self.now+=timedelta(seconds=2)
            return self.response
        self.broker=PaperBroker('PK'+'T'*20,'T'*32,transport=transport,clock=lambda:self.now)
        self.gateway=self.m.NativePaperGateway(self.broker)

    def submit(self,**changes):
        args=dict(controls=self.store,control_snapshot=self.snapshot,market_price_usd='150',market_at=self.f.now,
                  **self.f.f.args)
        args.update(changes)
        return self.gateway.submit(self.f.ledger,self.f.intent.intent_id,**args)

    def test_one_post_uses_attached_oto_stop_and_atomically_records_both_legs(self):
        record=self.submit()
        self.assertEqual(record['state'],'open')
        self.assertEqual(record['native_stop']['stop']['broker_order_id'],self.f.child_id)
        self.assertEqual(len(self.calls),1)
        method,url,body=self.calls[0]
        self.assertEqual((method,urlsplit(url).netloc,urlsplit(url).path),('POST','paper-api.alpaca.markets','/v2/orders'))
        self.assertEqual(body['order_class'],'oto')
        self.assertEqual(body['stop_loss'],{'stop_price':'145.00'})
        self.assertFalse(body['extended_hours'])
        self.assertIsNone(self.submit())
        self.assertEqual(len(self.calls),1)

    def test_unknown_transport_or_invalid_response_retains_claim_without_post_retry(self):
        self.response=OSError('PRIVATE_PROVIDER_BODY')
        with self.assertRaises(BrokerError) as caught:self.submit()
        self.assertNotIn('PRIVATE',str(caught.exception))
        self.assertEqual(self.f.ledger.load(self.f.intent.intent_id)['state'],'submission_unknown')
        self.assertIsNone(self.submit());self.assertEqual(len(self.calls),1)

    def test_wrong_response_cannot_overwrite_plan_or_resolve_unknown_outcome(self):
        self.response=HttpReply(201,{**deepcopy(self.f.parent),'qty':'6'})
        with self.assertRaises(BrokerError):self.submit()
        self.assertIsNone(self.submit());self.assertEqual(len(self.calls),1)
        self.assertEqual(self.f.ledger.load(self.f.intent.intent_id)['state'],'submission_unknown')

    def test_invalid_market_stop_relationship_is_a_known_unsent_durable_veto(self):
        self.assertIsNone(self.submit(market_price_usd='144'))
        record=self.f.ledger.load(self.f.intent.intent_id)
        self.assertEqual(record['state'],'rejected_by_risk')
        self.assertIn('native_protection_invalid',record['risk']['violations'])
        self.assertEqual(self.calls,[])

    def test_changed_control_revision_rejects_unsent_claim_even_when_still_resumed(self):
        self.store.update(expected_revision=self.snapshot.revision,pause_entries=False,halt_orders=False,
                          reason='operator_resume',now=self.now)
        self.assertIsNone(self.submit())
        record=self.f.ledger.load(self.f.intent.intent_id)
        self.assertEqual(record['state'],'rejected_by_risk')
        self.assertIn('operator_control_changed',record['risk']['violations'])
        self.assertEqual(self.calls,[])

    def test_delayed_control_acquisition_rechecks_expiry_and_records_no_post(self):
        lease=self.store.dispatch_lease
        @contextmanager
        def delayed(*args,**kwargs):
            with lease(*args,**kwargs) as current:
                self.now+=timedelta(seconds=21);yield current
        with patch.object(self.store,'dispatch_lease',delayed):self.assertIsNone(self.submit())
        self.assertEqual(self.calls,[])
        self.assertEqual(self.f.ledger.load(self.f.intent.intent_id)['state'],'rejected_by_risk')

    def test_clock_rollback_after_claim_uses_audit_floor_and_never_sends(self):
        claim=self.f.ledger.prepare_native_submission
        def rollback(*args,**kwargs):
            ticket=claim(*args,**kwargs);self.now-=timedelta(seconds=1);return ticket
        with patch.object(self.f.ledger,'prepare_native_submission',rollback):self.assertIsNone(self.submit())
        record=self.f.ledger.load(self.f.intent.intent_id)
        self.assertEqual(record['state'],'rejected_by_risk')
        self.assertIn('submission_clock_regression',record['risk']['violations'])
        self.assertEqual(self.calls,[])

    def test_claim_persistence_failure_causes_no_network_and_post_success_persistence_failure_is_unknown(self):
        persist=self.f.ledger._persist
        with patch.object(self.f.ledger,'_persist',side_effect=OSError('PRIVATE_DISK')):
            with self.assertRaises(BrokerError):self.submit()
        self.assertEqual(self.calls,[])
        def fail_observation(db,record,kind,at):
            if kind=='native_group_observation':raise OSError('PRIVATE_DISK')
            return persist(db,record,kind,at)
        with patch.object(self.f.ledger,'_persist',fail_observation):
            with self.assertRaises(BrokerError):self.submit()
        self.assertEqual(self.f.ledger.load(self.f.intent.intent_id)['state'],'submission_unknown')
        self.assertIsNone(self.submit());self.assertEqual(len(self.calls),1)

    def test_no_simple_order_or_caller_gateway_override_can_strip_native_stop(self):
        with self.assertRaises(BrokerError):self.submit(gateway='simple_limit')
        self.assertEqual(self.calls,[])
        self.assertEqual(self.f.ledger.load(self.f.intent.intent_id)['state'],'reserved')

    def test_concurrent_invocations_claim_one_submission(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as workers:
            results=list(workers.map(lambda _:self.submit(),range(2)))
        self.assertEqual(len([r for r in results if r is not None]),1)
        self.assertEqual(len(self.calls),1)

    def test_control_writer_cannot_commit_during_native_http_dispatch(self):
        entered=threading.Event();committed=threading.Event();errors=[];writers=[]
        def update():
            try:
                entered.set()
                self.store.update(expected_revision=self.snapshot.revision,pause_entries=True,halt_orders=True,
                                  reason='operator_halt',now=self.now)
                committed.set()
            except Exception as error:errors.append(type(error).__name__)
        original=self.broker._transport
        def held(method,url,headers,body):
            writer=threading.Thread(target=update);writers.append(writer);writer.start()
            self.assertTrue(entered.wait(2));self.assertFalse(committed.wait(.05))
            return original(method,url,headers,body)
        self.broker._transport=held;self.submit()
        for writer in writers:writer.join(2)
        self.assertEqual(errors,[]);self.assertTrue(committed.is_set())

    def test_valid_exponential_decimal_cash_roundtrips_into_final_risk_and_can_submit(self):
        portfolio=replace(self.f.f.portfolio,**{name:Decimal('1E+4') for name in
            ('cash_usd','equity_usd','day_start_equity_usd','high_water_equity_usd')})
        record=self.submit(portfolio=portfolio)
        self.assertEqual(record['state'],'open')
        self.assertEqual(len(self.calls),1)

    def test_final_risk_recheck_exception_is_known_unsent_and_durably_rejected(self):
        with patch.object(self.m,'recheck_ticket',side_effect=ValueError('PRIVATE_RECHECK')):
            self.assertIsNone(self.submit())
        record=self.f.ledger.load(self.f.intent.intent_id)
        self.assertEqual(record['state'],'rejected_by_risk')
        self.assertIn('risk_recheck_unavailable',record['risk']['violations'])
        self.assertNotIn('PRIVATE',str(record))
        self.assertEqual(self.calls,[])


if __name__=='__main__':unittest.main()
