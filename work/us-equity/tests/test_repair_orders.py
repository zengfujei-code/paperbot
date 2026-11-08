from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
import importlib,importlib.util,unittest
from unittest.mock import patch
from urllib.parse import urlsplit,parse_qs

import test_repair_protection
import test_native_cleanup
from paperbot_us.broker import HttpReply,BrokerError
from paperbot_us.orders import LedgerError,_hash,_json
from paperbot_us.risk import Quote,RiskState
from paperbot_us.repair_protection import prepare_repair


class RepairOrderTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.repair_orders'),
            'durable independent repair ownership/accounting required')
        self.m=importlib.import_module('paperbot_us.repair_orders')
        self.r=test_repair_protection.RepairProtectionTests();self.r.setUp();self.addCleanup(self.r.doCleanups)
        self.r.r.cancel();self.f=self.r.f;self.n=self.r.n;self.ledger=self.r.r.ledger
        self.plan=self.r.prepare();self.family=self.f.family
        self.token=None;self.row=None;self.missing=False;self.gets=[]
        original=self.n.broker._transport
        def transport(method,url,headers,body):
            self.assertEqual(method,'GET','owned repair recovery cannot submit/cancel')
            parsed=urlsplit(url)
            if (self.token is not None and parsed.path=='/v2/orders:by_client_order_id'
                    and parse_qs(parsed.query).get('client_order_id')==[self.token.client_order_id]):
                self.gets.append(parsed.path)
                return HttpReply(404) if self.missing else HttpReply(200,deepcopy(self.row))
            return original(method,url,headers,body)
        self.n.broker._transport=transport
        self.reader=self.m.RepairPaperReader(self.n.broker)

    def reserve(self,**changes):
        return self.m.reserve_repair(**{**dict(ledger=self.ledger,plan=self.plan,book=self.f.book,
            baselines=self.f.f.baselines,calendar=self.f.f.calendar,limits=self.f.limits,now=self.n.now),**changes})

    def claim(self):
        return self.m.claim_repair(self.ledger,self.family,self.token.repair_id,
            expected_sha256=self.token.orders_sha256,now=self.n.now)

    def stage(self):
        self.token=self.reserve();self.claim();self.n.now+=timedelta(seconds=1)
        self.row={**self.r.r.e.row,'id':'00000000-0000-4000-8000-000000000095',
            'client_order_id':self.token.client_order_id,'type':'stop','limit_price':None,'stop_price':'145',
            'time_in_force':'gtc','order_class':'simple','status':'new','filled_qty':'0','filled_avg_price':None,
            'created_at':self.n.now.isoformat(),'updated_at':self.n.now.isoformat(),'filled_at':None,'legs':None}
        self.n.visible=[deepcopy(self.row)];self.n.positions[0]['qty_available']='0'
        return self.recover()

    def recover(self):return self.reader.recover(self.ledger,self.family,self.token.repair_id)

    def fill(self,qty,status='partially_filled'):
        prior=int(self.row['filled_qty']);delta=qty-prior;self.n.now+=timedelta(seconds=1)
        self.row.update(status=status,filled_qty=str(qty),filled_avg_price='140',updated_at=self.n.now.isoformat(),
            filled_at=self.n.now.isoformat() if status=='filled' else None)
        self.n.visible=[] if status in ('filled','canceled') else [deepcopy(self.row)]
        left=5-qty
        self.n.positions=[] if not left else [{**self.n.positions[0],'qty':str(left),
            'qty_available':str(left) if status=='canceled' else '0','market_value':str(150*left)}]
        self.n.account.update(cash=str(9250+140*qty),equity=str(10000-10*qty))
        if delta:self.n.fills.append(self.n.fill('00000000-0000-4000-8000-00000000009'+str(qty),
            'sell',str(delta),'140',self.row['id'],int((self.n.now-self.n.f.now).total_seconds()),str(qty),str(5-qty)))

    def test_reservation_owns_all_available_shares_without_claiming_protection(self):
        original=self.ledger.load(self.family)['native_stop']['stop']['broker_order_id']
        self.token=self.reserve()
        group=self.f.m.project_groups(self.ledger.all_orders())[self.family]
        self.assertEqual(group.remaining_quantity,5);self.assertEqual(group.available_quantity,0)
        self.assertFalse(self.token.public_summary()['protection_active'])
        self.assertEqual(self.ledger.load(self.family)['native_stop']['stop']['broker_order_id'],original)
        with self.assertRaises(LedgerError):self.reserve()

    def test_claim_is_one_shot_and_an_unknown_get_cannot_release_shares(self):
        self.token=self.reserve();self.claim();self.missing=True
        self.assertIsNone(self.recover())
        self.assertIsNone(self.m.claim_repair(self.ledger,self.family,self.token.repair_id,
            expected_sha256=_hash(_json(self.ledger.all_orders())),now=self.n.now))
        self.assertEqual(self.f.m.project_groups(self.ledger.all_orders())[self.family].available_quantity,0)
        self.assertFalse(self.n.reconciler.refresh().ready_for_risk)

    def test_changed_sources_cannot_admit_or_claim_a_repair(self):
        self.n.now+=timedelta(seconds=1);self.n.reconciler.refresh()
        with self.assertRaises(LedgerError):self.reserve()
        self.plan=self.r.prepare();self.token=self.reserve();self.n.now+=timedelta(seconds=1)
        self.n.reconciler.refresh()
        with self.assertRaises(LedgerError):self.claim()

    def test_expired_or_failed_claim_never_becomes_an_order(self):
        self.token=self.reserve();before=self.ledger.all_orders()
        with patch.object(self.ledger,'_persist',side_effect=OSError('PRIVATE_DISK')):
            with self.assertRaises(LedgerError):self.claim()
        self.assertEqual(self.ledger.all_orders(),before);self.assertEqual(self.gets,[])
        self.n.now+=timedelta(seconds=21)
        with self.assertRaises(LedgerError):self.claim()

    def test_only_an_unchanged_never_claimed_reservation_can_be_retired(self):
        self.token=self.reserve();self.n.now+=timedelta(seconds=21)
        self.assertTrue(hasattr(self.m,'retire_repair'),'known-unsent repair allocations need explicit retirement')
        self.m.retire_repair(self.ledger,self.family,self.token.repair_id,expected_sha256=self.token.orders_sha256,
            reason='expired',now=self.n.now)
        group=self.f.m.project_groups(self.ledger.all_orders())[self.family]
        self.assertEqual(group.available_quantity,5)
        self.r.r.gateway.recover(self.ledger,self.r.r.identity)
        self.plan=self.r.prepare();self.token=self.reserve()
        self.assertEqual(self.f.m.project_groups(self.ledger.all_orders())[self.family].available_quantity,0)

    def test_claimed_unknown_repair_cannot_be_retired_by_expiry(self):
        self.token=self.reserve();self.claim();self.n.now+=timedelta(seconds=21)
        self.assertTrue(hasattr(self.m,'retire_repair'),'unknown repair reservations must survive expiry')
        result=self.m.retire_repair(self.ledger,self.family,self.token.repair_id,
            expected_sha256=_hash(_json(self.ledger.all_orders())),reason='expired',now=self.n.now)
        self.assertIsNone(result)
        self.assertEqual(self.f.m.project_groups(self.ledger.all_orders())[self.family].available_quantity,0)

    def test_a_malformed_typed_reference_cannot_assert_an_active_partial_new_order(self):
        self.stage();self.n.now+=timedelta(seconds=1)
        from paperbot_us.native_broker import _reference
        ref=_reference({**self.row,'updated_at':self.n.now.isoformat()},self.n.now)
        with patch('paperbot_us.native_broker._reference',return_value=replace(ref,filled_quantity=2,average_fill_price_usd=Decimal('140'))):
            with self.assertRaises(BrokerError):self.recover()

    def test_observed_owned_stop_reconciles_and_reserves_the_actual_remaining_position(self):
        self.stage();cycle=self.n.reconciler.refresh()
        self.assertTrue(cycle.ready_for_risk,cycle.findings)
        self.assertTrue(cycle.book_receipt.result.reconciled)
        view=self.f.f.prepare(cycle=cycle)
        self.assertEqual(view.portfolio.stop_reservations[0].remaining_quantity,5)
        self.assertEqual(cycle.native_states[0]['status'],'repaired_protected')

    def test_partial_and_full_gap_fills_reconcile_without_rewriting_original_child(self):
        self.stage();original=self.ledger.load(self.family)['native_stop']['stop']['broker_order_id']
        self.fill(2);first=self.n.reconciler.refresh()
        self.assertTrue(first.ready_for_risk,first.findings)
        self.assertEqual(first.book_receipt.result.profit_usd,Decimal('-20'))
        group=self.f.m.project_groups(self.ledger.all_orders())[self.family]
        self.assertEqual(group.remaining_quantity,3);self.assertEqual(group.reserved_quantity,3)
        self.fill(5,'filled');last=self.n.reconciler.refresh()
        self.assertTrue(last.ready_for_risk,last.findings);self.assertEqual(last.book_receipt.result.profit_usd,Decimal('-50'))
        self.assertTrue(self.f.m.project_groups(self.ledger.all_orders())[self.family].settled)
        self.assertEqual(self.ledger.load(self.family)['native_stop']['stop']['broker_order_id'],original)
        self.assertEqual(self.f.book.replay(first.book_receipt.sequence).profit_usd,Decimal('-20'))

    def test_new_unknown_lookup_invalidates_cached_active_protection(self):
        self.stage();self.assertTrue(self.n.reconciler.refresh().ready_for_risk)
        self.n.now+=timedelta(seconds=1);self.missing=True;self.assertIsNone(self.recover())
        cycle=self.n.reconciler.refresh();self.assertFalse(cycle.ready_for_risk)
        self.assertEqual(self.f.m.project_groups(self.ledger.all_orders())[self.family].available_quantity,0)

    def test_overlapping_old_success_cannot_overwrite_a_newer_unknown_lookup(self):
        self.stage();original=self.n.broker._transport;entered=False
        def transport(method,url,headers,body):
            nonlocal entered
            parsed=urlsplit(url)
            if (not entered and parsed.path=='/v2/orders:by_client_order_id'
                    and parse_qs(parsed.query).get('client_order_id')==[self.token.client_order_id]):
                entered=True;response=original(method,url,headers,body)
                self.n.now+=timedelta(seconds=1);self.missing=True
                self.assertIsNone(self.recover());self.missing=False
                return response
            return original(method,url,headers,body)
        self.n.broker._transport=transport
        cycle=self.n.reconciler.refresh()
        self.assertFalse(cycle.ready_for_risk)
        row=self.m.repair_rows(self.ledger.load(self.family))[0]
        self.assertEqual(row['lookup_state'],'unknown')

    def test_old_terminal_repair_does_not_stale_a_fresh_active_replacement(self):
        self.stage();self.fill(2,'canceled');self.n.reconciler.refresh()
        self.plan=self.r.prepare();self.token=self.reserve();self.claim();self.n.now+=timedelta(seconds=1)
        self.row={**self.row,'id':'00000000-0000-4000-8000-000000000096',
            'client_order_id':self.token.client_order_id,'qty':'3','status':'new','filled_qty':'0',
            'filled_avg_price':None,'created_at':self.n.now.isoformat(),'updated_at':self.n.now.isoformat(),'filled_at':None}
        self.n.visible=[deepcopy(self.row)];self.n.positions[0]['qty_available']='0'
        self.recover();self.assertTrue(self.n.reconciler.refresh().ready_for_risk)
        self.n.now+=timedelta(seconds=31)
        cycle=self.n.reconciler.refresh();self.assertTrue(cycle.ready_for_risk,cycle.findings)
        self.assertEqual(cycle.native_states[0]['status'],'repaired_protected')
        self.assertEqual(cycle.book_receipt.result.profit_usd,Decimal('-20'))

    def test_wrong_client_or_reused_original_child_id_cannot_become_a_repair(self):
        self.stage();before=self.ledger.all_orders();self.n.now+=timedelta(seconds=1)
        self.row['id']=self.ledger.load(self.family)['native_stop']['stop']['broker_order_id']
        with self.assertRaises(BrokerError):self.recover()
        records=self.ledger.all_orders()
        self.assertEqual(records[0]['native_stop']['stop'],before[0]['native_stop']['stop'])
        self.assertFalse(self.n.reconciler.refresh().ready_for_risk)

    def test_late_terminal_repair_fill_is_queried_and_reconciled(self):
        self.stage();self.fill(2,'canceled');first=self.n.reconciler.refresh()
        self.assertTrue(first.book_receipt.result.reconciled)
        self.fill(3,'canceled');last=self.n.reconciler.refresh()
        self.assertTrue(last.book_receipt.result.reconciled,last.findings)
        group=self.f.m.project_groups(self.ledger.all_orders())[self.family]
        self.assertEqual(group.remaining_quantity,2);self.assertEqual(group.available_quantity,2)
        self.assertEqual(last.book_receipt.result.profit_usd,Decimal('-30'))

    def test_terminal_partial_buy_keeps_real_parent_quantity_and_needs_no_fake_completion(self):
        f=test_native_cleanup.NativeCleanupTests();f.setUp();self.addCleanup(f.doCleanups);f.partial_parent()
        cycle=f.n.reconciler.refresh()
        plan=prepare_repair(book=f.book,ledger=f.ledger,baselines=f.f.baselines,cycle=cycle,
            quotes=(Quote('AAPL','150','150.1',f.n.now,f.n.now),),family_id=f.family,
            operator_state=RiskState(pause_entries=True),calendar=f.f.calendar,limits=f.limits,now=f.n.now)
        token=self.m.reserve_repair(ledger=f.ledger,plan=plan,book=f.book,baselines=f.f.baselines,
            calendar=f.f.calendar,limits=f.limits,now=f.n.now)
        self.m.claim_repair(f.ledger,f.family,token.repair_id,expected_sha256=token.orders_sha256,now=f.n.now)
        f.n.now+=timedelta(seconds=1)
        row={**f.n.f.parent,'id':'00000000-0000-4000-8000-000000000099','client_order_id':token.client_order_id,
            'side':'sell','type':'stop','limit_price':None,'stop_price':'145','qty':'2','filled_qty':'2',
            'filled_avg_price':'140','status':'filled','time_in_force':'gtc','order_class':'simple','legs':None,
            'created_at':f.n.now.isoformat(),'updated_at':f.n.now.isoformat(),'filled_at':f.n.now.isoformat()}
        from paperbot_us.native_broker import _reference
        self.m.observe_repair(f.ledger,f.family,token.repair_id,_reference(row,f.n.now))
        f.n.positions=[];f.n.account.update(cash='9980',equity='9980')
        f.n.fills.append(f.n.fill('00000000-0000-4000-8000-000000000098','sell','2','140',row['id'],
            int((f.n.now-f.n.f.now).total_seconds()),'2','0'))
        result=f.n.reconciler.refresh();self.assertTrue(result.ready_for_risk,result.findings)
        self.assertEqual(result.book_receipt.result.profit_usd,Decimal('-20'))
        parent=f.ledger.load(f.family)['native_stop']['parent']
        self.assertEqual(parent['quantity'],5);self.assertEqual(parent['filled_quantity'],2)
        self.assertEqual(parent['status'],'canceled');self.assertIsNone(parent['filled_at'])


if __name__=='__main__':unittest.main()
