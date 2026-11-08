from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import json
import sqlite3
import unittest
from unittest.mock import patch

import test_native_reconciler as fixtures
from paperbot_us.broker_time_transport import CanonicalBrokerTimes
from paperbot_us.broker import BrokerError,HttpReply
from paperbot_us.data import utc_time
from paperbot_us.monitor import ObservationArchive
from paperbot_us.native_broker import NativePaperReader


class BrokerTimestampPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.NativeReconcilerTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        cycle=self.f.reconciler.refresh();self.assertTrue(cycle.ready_for_risk,cycle.findings)
        self.original=self.f.f.ledger.all_orders()[0]
        self.journal=self.f.f.f.path.with_name('broker-times.sqlite')
        self.adapter=CanonicalBrokerTimes(ledger=self.f.f.ledger,journal=self.journal,
            transport=self.f.transport,clock=lambda:self.f.now)
        self.f.broker._transport=self.adapter

    def shifted(self,value,microseconds):
        return (utc_time(value)+timedelta(microseconds=microseconds)).isoformat()

    def change_provider(self,offset):
        for role,row in (('parent',self.f.f.parent),('stop',self.f.f.child)):
            anchor=self.original['native_stop'][role]
            for field in ('created_at','updated_at','filled_at'):
                if anchor[field] is not None:row[field]=self.shifted(anchor[field],offset)
        self.f.visible=[deepcopy(self.f.f.child)];self.f.now+=timedelta(seconds=1)

    def audit(self):
        with sqlite3.connect(self.journal) as db:
            return [json.loads(row[0]) for row in db.execute('SELECT payload FROM observations ORDER BY seq')]

    def test_real_one_microsecond_roundtrip_preserves_core_anchors_and_raw_times(self):
        self.change_provider(1);cycle=self.f.reconciler.refresh()
        self.assertTrue(cycle.ready_for_risk,cycle.findings)
        saved=self.f.f.ledger.all_orders()[0]
        for role in ('parent','stop'):
            self.assertEqual(saved['native_stop'][role]['created_at'],self.original['native_stop'][role]['created_at'])
            self.assertEqual(saved['native_stop'][role]['updated_at'],self.original['native_stop'][role]['updated_at'])
        events=self.audit();self.assertTrue(events)
        self.assertTrue(any(change['raw']['created_at']==self.f.f.child['created_at'] for event in events for change in event['changes']))
        self.assertTrue(ObservationArchive(self.journal).verify()['integrity_verified'])
        self.assertTrue(self.f.f.ledger.verify()['integrity_verified'])
        self.assertTrue(all(method=='GET' for method,_ in self.f.calls))

    def test_alternation_cannot_walk_first_durable_anchor(self):
        for offset in (1,0,1,0):
            self.change_provider(offset);cycle=self.f.reconciler.refresh()
            self.assertTrue(cycle.ready_for_risk,cycle.findings)
            self.assertEqual(self.f.f.ledger.all_orders()[0]['native_stop']['parent']['created_at'],self.original['native_stop']['parent']['created_at'])
        self.change_provider(2);self.assertFalse(self.f.reconciler.refresh().ready_for_risk)

    def test_larger_completion_time_change_remains_vetoed(self):
        self.change_provider(0);self.f.f.parent['filled_at']=self.shifted(self.f.f.parent['filled_at'],2)
        self.assertFalse(self.f.reconciler.refresh().ready_for_risk)

    def test_changed_financial_event_cannot_move_backwards_one_microsecond(self):
        self.change_provider(1)
        child=self.f.f.child;child.update(status='partially_filled',filled_qty='1',filled_avg_price='140')
        child['updated_at']=self.shifted(self.original['native_stop']['stop']['updated_at'],-1)
        self.f.visible=[deepcopy(child)]
        self.assertFalse(self.f.reconciler.refresh().ready_for_risk)
        self.assertEqual(self.f.f.ledger.all_orders()[0]['native_stop']['stop']['filled_quantity'],0)

    def test_static_identity_and_price_mismatch_never_normalized_away(self):
        self.change_provider(1);self.f.visible[0]['stop_price']='144.99'
        self.assertIn('non_atomic_order_snapshot',self.f.reconciler.refresh().findings)

    def test_non_get_has_no_adapter_disk_io_and_payload_is_unchanged(self):
        calls=[]
        adapter=CanonicalBrokerTimes(ledger=self.f.f.ledger,journal=self.journal,
            transport=lambda *args:calls.append(args) or HttpReply(204),clock=lambda:self.f.now)
        with patch.object(self.f.f.ledger,'all_orders',side_effect=AssertionError('no_post_disk_io')):
            body={'synthetic':'exact'};headers={'synthetic':'exact'}
            self.assertEqual(adapter('POST','https://paper-api.alpaca.markets/v2/orders',headers,body).status,204)
        self.assertIs(calls[0][2],headers);self.assertIs(calls[0][3],body)
        self.assertFalse(self.journal.exists())

    def test_audit_commit_failure_refuses_normalized_input_without_secret_error(self):
        self.change_provider(1)
        with patch('paperbot_us.broker_time_transport.ObservationArchive.append',side_effect=RuntimeError('SECRET_MUST_NOT_APPEAR')):
            with self.assertRaises(BrokerError) as caught:
                self.adapter('GET','https://paper-api.alpaca.markets/v2/orders?status=open',{},None)
        self.assertNotIn('SECRET_MUST_NOT_APPEAR',str(caught.exception))

    def test_market_and_receipt_parser_policy_is_unchanged(self):
        a=utc_time('2026-10-27T18:49:05.761555999Z');b=utc_time('2026-10-27T18:49:05.761556Z')
        self.assertEqual(b-a,timedelta(microseconds=1))

    def test_restart_rejects_changed_event_reusing_prior_raw_provider_time(self):
        self.change_provider(1);self.assertTrue(self.f.reconciler.refresh().ready_for_risk)
        self.f.broker._transport=CanonicalBrokerTimes(ledger=self.f.f.ledger,journal=self.journal,
            transport=self.f.transport,clock=lambda:self.f.now)
        self.f.f.child['status']='canceled'
        with self.assertRaises(BrokerError):
            NativePaperReader(self.f.broker).recover(self.f.f.ledger,self.f.f.intent.intent_id)
        self.assertEqual(self.f.f.ledger.all_orders()[0]['native_stop']['stop']['status'],'new')


if __name__=='__main__':unittest.main()
