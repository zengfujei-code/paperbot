from copy import deepcopy
from datetime import timedelta
import importlib
import importlib.util
from urllib.parse import urlsplit,parse_qs
import unittest

import test_native_orders
from paperbot_us.broker import PaperBroker,HttpReply,BrokerError


class NativePaperReaderTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.native_broker'),
                             'group-aware paper GET adapter required')
        self.m=importlib.import_module('paperbot_us.native_broker')
        self.f=test_native_orders.NativeOrderLedgerTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.now=self.f.now+timedelta(seconds=2);self.calls=[];self.responses=[]
        def transport(method,url,headers,body):
            self.calls.append((method,url,body))
            response=self.responses.pop(0)
            if isinstance(response,Exception): raise response
            return response
        broker=PaperBroker('PK'+'T'*20,'T'*32,transport=transport,clock=lambda:self.now)
        self.reader=self.m.NativePaperReader(broker)

    def replies(self,parent=None):
        parent=deepcopy(parent or self.f.parent)
        first={**parent,'legs':None}
        self.responses=[HttpReply(200,first),HttpReply(200,parent)]

    def test_lookup_resolves_client_identity_then_reads_nested_parent_on_fixed_paper_host(self):
        self.replies();group=self.reader.lookup(self.f.plan)
        self.assertEqual(group.parent.broker_order_id,self.f.parent_id)
        self.assertEqual(group.stop.broker_order_id,self.f.child_id)
        self.assertEqual([c[0] for c in self.calls],['GET','GET'])
        urls=[urlsplit(c[1]) for c in self.calls]
        self.assertTrue(all(u.netloc=='paper-api.alpaca.markets' for u in urls))
        self.assertEqual(urls[0].path,'/v2/orders:by_client_order_id')
        self.assertEqual(parse_qs(urls[0].query),dict(client_order_id=['pb3-'+'a'*40]))
        self.assertEqual(urls[1].path,'/v2/orders/'+self.f.parent_id)
        self.assertEqual(parse_qs(urls[1].query),dict(nested=['true']))

    def test_absent_client_lookup_is_unknown_without_a_retry_or_write(self):
        self.f.claim();before=self.f.ledger.load(self.f.intent.intent_id)
        self.responses=[HttpReply(404)]
        self.assertIsNone(self.reader.recover(self.f.ledger,self.f.intent.intent_id))
        self.assertEqual(self.f.ledger.load(self.f.intent.intent_id),before)
        self.assertEqual([c[0] for c in self.calls],['GET'])

    def test_recovery_commits_owned_group_and_restart_keeps_native_child(self):
        self.f.claim();self.replies()
        record=self.reader.recover(self.f.ledger,self.f.intent.intent_id)
        self.assertEqual(record['state'],'open')
        self.assertEqual(self.f.f.Ledger(self.f.f.path).load(self.f.intent.intent_id),record)
        self.assertEqual(record['native_stop']['stop']['broker_order_id'],self.f.child_id)

    def test_mismatch_nested_disappearance_and_unknown_transport_preserve_unknown_claim(self):
        self.f.claim();before=self.f.ledger.all_orders()
        foreign={**deepcopy(self.f.parent),'client_order_id':'foreign'}
        for replies in ([HttpReply(200,foreign)],
                        [HttpReply(200,{**self.f.parent,'legs':[]}),HttpReply(404)],
                        [OSError('PRIVATE_PROVIDER_BODY')]):
            with self.subTest(case=len(replies)):
                self.responses=list(replies)
                with self.assertRaises(BrokerError) as caught: self.reader.recover(self.f.ledger,self.f.intent.intent_id)
                self.assertNotIn('PRIVATE',str(caught.exception))
                self.assertEqual(self.f.ledger.all_orders(),before)

    def test_two_reads_cannot_regress_parent_fills_or_change_its_identity(self):
        full=deepcopy(self.f.parent)
        full.update(status='filled',filled_qty='5',filled_avg_price='150',
                    filled_at=(self.f.now+timedelta(seconds=1)).isoformat())
        for second in (deepcopy(self.f.parent),{**full,'id':'00000000-0000-4000-8000-000000000099'}):
            with self.subTest(parent=second['status']):
                self.responses=[HttpReply(200,{**full,'legs':[]}),HttpReply(200,second)]
                with self.assertRaises(BrokerError): self.reader.lookup(self.f.plan)

    def test_flat_open_refs_cover_native_stops_without_claiming_they_are_limit_sells(self):
        child={**self.f.child,'status':'new'}
        self.responses=[HttpReply(200,[{**self.f.parent,'legs':None},child])]
        refs=self.reader.open_orders()
        self.assertEqual(len(refs),2)
        stop=next(o for o in refs if o.side=='sell')
        self.assertEqual((stop.order_type,stop.limit_price_usd,stop.stop_price_usd),('stop',None,145))
        self.assertNotIn(self.f.child_id,repr(stop))
        self.assertNotIn('private-provider-child',repr(stop))
        self.assertEqual(parse_qs(urlsplit(self.calls[0][1]).query),dict(status=['open'],limit=['500'],nested=['false']))

    def test_open_refs_refuse_truncation_duplicates_nested_or_unsupported_orders(self):
        row={**self.f.parent,'legs':None}
        for rows in ([row]*500,[row,row],[self.f.parent],[{**self.f.child,'type':'stop_limit','limit_price':'144'}]):
            with self.subTest(rows=len(rows)):
                self.responses=[HttpReply(200,rows)]
                with self.assertRaises(BrokerError): self.reader.open_orders()

    def test_lookup_deadline_exhaustion_cannot_return_a_healthy_group(self):
        ticks=iter((0,26))
        self.reader=self.m.NativePaperReader(self.reader.broker,monotonic=lambda:next(ticks))
        self.replies()
        with self.assertRaises(BrokerError): self.reader.lookup(self.f.plan)
        self.assertEqual(len(self.calls),1)

    def test_first_observed_child_identity_or_fills_cannot_disappear_in_the_nested_read(self):
        self.f.full_parent()
        first=deepcopy(self.f.parent)
        first['legs'][0].update(status='partially_filled',filled_qty='2',filled_avg_price='130')
        for second in (deepcopy(self.f.parent),{**deepcopy(self.f.parent),'legs':[]},
                       {**deepcopy(self.f.parent),'legs':[{**first['legs'][0],
                           'id':'00000000-0000-4000-8000-000000000099'}]}):
            with self.subTest(legs=len(second['legs'])):
                self.responses=[HttpReply(200,first),HttpReply(200,second)]
                with self.assertRaises(BrokerError): self.reader.lookup(self.f.plan)


if __name__=='__main__': unittest.main()
