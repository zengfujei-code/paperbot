from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
import importlib
import importlib.util
import unittest
from unittest.mock import patch

import test_native_risk_view
import test_native_orders
from paperbot_us.orders import OrderLedger, LedgerError, recheck_ticket, _hash, _json
from paperbot_us.risk import OrderIntent, Quote, RiskLimits, RiskState
from paperbot_us.risk_baselines import BaselineError
from paperbot_us.risk_view import RiskViewError


class NativeCleanupTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.native_cleanup'),
            'accounting-only group-owned cleanup required')
        self.m = importlib.import_module('paperbot_us.native_cleanup')
        self.f = test_native_risk_view.NativeRiskPreparationTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.n = self.f.n; self.ledger = self.f.ledger; self.book = self.f.book
        self.family = self.n.f.intent.intent_id
        self.n.f.child.update(status='canceled', updated_at=(self.n.f.now + timedelta(seconds=3)).isoformat())
        self.n.visible = []; self.n.positions[0]['qty_available'] = '5'
        self.limits = RiskLimits(); self.intent = self.sell('c', 5)

    def sell(self, letter, quantity):
        return OrderIntent(letter * 64, 'AAPL', 'sell', quantity, '150', None, self.n.now + timedelta(seconds=20))

    def prepare(self, **changes):
        cycle = changes.pop('cycle', None) or self.n.reconciler.refresh()
        return self.m.prepare_cleanup_risk_view(**{**dict(book=self.book, ledger=self.ledger,
            baselines=self.f.baselines, cycle=cycle, quotes=(Quote('AAPL','150','150.1',self.n.now,self.n.now),),
            family_id=self.family, operator_state=RiskState(pause_entries=False),
            calendar=self.f.calendar, now=self.n.now), **changes})

    def admit(self, intent=None, **changes):
        prepared=changes.pop('prepared',None)
        if prepared is None: prepared=self.prepare()
        return self.m.admit_cleanup(self.ledger, intent or self.intent, **{**dict(prepared=prepared,
            limits=self.limits, calendar=self.f.calendar, now=self.n.now), **changes})

    def claim(self, identity=None):
        return self.m.prepare_cleanup_submission(self.ledger, identity or self.intent.intent_id,
            prepared=self.prepare(), limits=self.limits, calendar=self.f.calendar, now=self.n.now)

    def partial_parent(self):
        self.n.f.parent.update(status='canceled', filled_qty='2', filled_avg_price='150', filled_at=None,
            updated_at=(self.n.f.now+timedelta(seconds=3)).isoformat())
        self.n.positions[0].update(qty='2',qty_available='2',market_value='300')
        self.n.fills[0].update(qty='2',cum_qty='2',leaves_qty='3')
        self.n.account.update(cash='9700',equity='10000')

    def test_accounting_only_preparation_keeps_protection_warning_and_pauses_entries(self):
        cycle = self.n.reconciler.refresh(); self.assertFalse(cycle.ready_for_risk)
        view = self.prepare(cycle=cycle)
        self.assertIn('native_protection_unprotected_position', view.risk.findings)
        self.assertTrue(view.risk.state.pause_entries)
        self.assertFalse(cycle.ready_for_risk)
        self.assertEqual(view.remaining_quantity,5)
        self.assertFalse(view.public_summary()['execution_enabled'])
        baseline = self.f.baselines.replay()
        self.assertFalse(baseline.entries_ready)
        self.assertIn('native_cleanup_only', baseline.findings)
        with self.assertRaises(BaselineError):
            self.f.baselines.observe(book=self.book,cycle=cycle,calendar=self.f.calendar,now=self.n.now)

    def test_active_missing_or_pending_child_never_authorizes_cleanup(self):
        for status in ('new','held','pending_cancel','pending_new'):
            with self.subTest(status=status):
                case=NativeCleanupTests();case.setUp();self.addCleanup(case.doCleanups)
                case.n.f.child['status']=status
                case.n.visible=[] if status=='held' else [deepcopy(case.n.f.child)]
                case.n.positions[0]['qty_available']='5' if status=='held' else '0'
                with self.assertRaises(RiskViewError): case.prepare()

    def test_terminal_partial_parent_can_allocate_only_its_known_filled_shares(self):
        self.partial_parent()
        too_large=self.admit(self.sell('d',3))
        self.assertEqual(too_large['state'],'rejected_by_risk')
        self.assertIn('native_cleanup_family_capacity',too_large['risk']['violations'])
        good=self.admit(self.sell('e',2)); self.assertEqual(good['state'],'reserved')
        self.assertEqual(good['native_cleanup']['family_intent_id'],self.family)

    def test_live_partial_parent_cannot_sell_while_remaining_buy_may_still_fill(self):
        self.partial_parent(); self.n.f.parent['status']='partially_filled'
        self.n.visible=[{**self.n.f.parent,'legs':None}]
        with self.assertRaises(RiskViewError): self.prepare()

    def test_allocation_is_durable_idempotent_and_concurrent_family_capacity_is_retained(self):
        first=self.admit(self.sell('c',3)); self.assertEqual(first['state'],'reserved')
        self.assertEqual(self.admit(self.sell('c',3)),first)
        second=self.admit(self.sell('d',3)); self.assertEqual(second['state'],'rejected_by_risk')
        restarted=OrderLedger(self.ledger.path)
        projection=self.m.project_groups(restarted.all_orders())[self.family]
        self.assertEqual(projection.remaining_quantity,5)
        self.assertEqual(projection.reserved_quantity,3)
        self.assertEqual(projection.available_quantity,2)

    def test_changed_ledger_and_stale_prepared_view_cannot_allocate(self):
        view=self.prepare(); self.admit(self.sell('d',1))
        with self.assertRaises(LedgerError): self.admit(prepared=view)
        view=self.prepare(); self.n.now+=timedelta(seconds=31)
        with self.assertRaises(LedgerError): self.m.admit_cleanup(self.ledger,self.intent,
            prepared=view,limits=self.limits,calendar=self.f.calendar,now=self.n.now)

    def test_buy_or_other_symbol_cannot_use_reducing_allocation(self):
        for intent in (self.n.f.intent,replace(self.intent,symbol='MSFT')):
            with self.subTest(side=intent.side,symbol=intent.symbol), self.assertRaises(LedgerError):
                self.admit(intent)

    def test_generic_gateway_cannot_claim_group_cleanup_and_dedicated_claim_is_one_shot(self):
        self.admit(); view=self.prepare()
        args=dict(portfolio=view.risk.portfolio,quote=view.risk.quote,state=view.risk.state,
            limits=self.limits,calendar=self.f.calendar,now=self.n.now)
        with self.assertRaises(LedgerError): self.ledger.prepare_submission(self.intent.intent_id,**args)
        ticket=self.claim(); self.assertEqual(ticket['state'],'submission_unknown')
        self.assertTrue(recheck_ticket(ticket,now=self.n.now).allowed)
        self.assertIsNone(self.m.prepare_cleanup_submission(self.ledger,self.intent.intent_id,
            prepared=view,limits=self.limits,calendar=self.f.calendar,now=self.n.now))
        with self.assertRaises(LedgerError): self.ledger.prepare_cancellation(self.intent.intent_id,at=self.n.now)

    def test_unknown_claim_reserves_family_capacity_and_cannot_be_resubmitted_as_new_intent(self):
        self.admit(); self.claim()
        # No broker response is supplied. The allocation remains unknown and
        # direct projection must retain all five shares across a restart.
        state=self.m.project_groups(OrderLedger(self.ledger.path).all_orders())[self.family]
        self.assertEqual(state.reserved_quantity,5)
        self.assertEqual(state.available_quantity,0)

    def test_expired_never_claimed_allocation_can_retire_without_fresh_network_evidence(self):
        record=self.admit();calls=len(self.n.calls);self.n.now+=timedelta(seconds=31)
        self.assertTrue(hasattr(self.m,'retire_cleanup_reservation'),'known-unsent cleanup needs a durable release')
        retired=self.m.retire_cleanup_reservation(self.ledger,self.intent.intent_id,
            expected_sha256=_hash(_json(record)),reason='expired',now=self.n.now)
        self.assertEqual(retired['state'],'expired')
        self.assertFalse(retired['risk']['allowed'])
        self.assertEqual(self.m.project_groups(self.ledger.all_orders())[self.family].available_quantity,5)
        self.assertEqual(len(self.n.calls),calls)

    def test_unknown_claim_cannot_retire_even_after_its_intent_expired(self):
        self.admit();record=self.claim();self.n.now+=timedelta(seconds=31)
        self.assertTrue(hasattr(self.m,'retire_cleanup_reservation'),'unknown outcomes must stay reserved')
        retired=self.m.retire_cleanup_reservation(self.ledger,self.intent.intent_id,
            expected_sha256=_hash(_json(record)),reason='expired',now=self.n.now)
        self.assertIsNone(retired)
        self.assertEqual(self.m.project_groups(self.ledger.all_orders())[self.family].reserved_quantity,5)

    def test_replaced_child_cannot_authorize_exit_without_successor_ownership(self):
        self.n.f.child['status']='replaced'
        with self.assertRaises(RiskViewError): self.prepare()

    def test_old_preparation_version_cannot_authorize_a_new_cleanup_allocation(self):
        view=self.prepare();old=replace(view,risk=replace(view.risk,code_sha256='0'*64))
        with self.assertRaises(LedgerError):self.admit(prepared=old)

    def test_two_families_reconcile_into_owned_cleanup_and_other_live_stop_reservations(self):
        self.prepare()
        second=test_native_orders.NativeOrderLedgerTests();second.setUp();self.addCleanup(second.doCleanups)
        second.intent=replace(second.intent,intent_id='b'*64);second.f.intent=second.intent
        second.plan=type(second.plan).from_intent(second.intent)
        second.parent_id='00000000-0000-4000-8000-000000000070'
        second.child_id='00000000-0000-4000-8000-000000000071'
        second.parent.update(id=second.parent_id,client_order_id='pb3-'+'b'*40)
        second.child['id']=second.child_id
        second.claim();second.full_parent();row=second.observe(3)
        # Import two synthetic observed families for reconciliation stress. This
        # is not evidence that the legacy admission gate accepts two families.
        with self.ledger._db(True) as db:
            self.ledger._persist(db,row,'synthetic_observed_family_fixture',self.n.now)
        old_transport=self.n.broker._transport
        from urllib.parse import urlsplit,parse_qs
        from paperbot_us.broker import HttpReply
        def transport(method,url,headers,body):
            self.assertEqual(method,'GET')
            parsed=urlsplit(url)
            if (parsed.path=='/v2/orders:by_client_order_id'
                    and parse_qs(parsed.query).get('client_order_id')==['pb3-'+'b'*40]):
                return HttpReply(200,{**second.parent,'legs':None})
            if parsed.path=='/v2/orders/'+second.parent_id:return HttpReply(200,deepcopy(second.parent))
            return old_transport(method,url,headers,body)
        self.n.broker._transport=transport
        self.n.visible=[deepcopy(second.child)]
        self.n.positions[0].update(qty='10',qty_available='5',market_value='1500')
        self.n.account.update(cash='8500',equity='10000')
        self.n.fills.append(self.n.fill('00000000-0000-4000-8000-000000000072','buy','5','150',second.parent_id,1,'5','0'))
        view=self.prepare()
        self.assertEqual(view.risk.portfolio.positions[0].quantity,10)
        self.assertEqual(view.risk.portfolio.stop_reservations[0].remaining_quantity,5)
        record=self.admit(prepared=view)
        self.assertEqual(record['state'],'reserved')
        self.assertEqual(record['native_cleanup']['family_intent_id'],self.family)
        self.assertEqual(record['risk']['gross_exposure_after_usd'],'1501.0')

    def test_loss_pause_does_not_block_independent_reducing_risk_but_global_halt_does(self):
        view=self.prepare(operator_state=RiskState(pause_entries=True,halt_orders=True))
        record=self.admit(prepared=view)
        self.assertEqual(record['state'],'rejected_by_risk')
        self.assertIn('all_orders_halted',record['risk']['violations'])

    def test_unexplained_account_or_available_share_difference_cannot_prepare(self):
        cycle=self.n.reconciler.refresh()
        for bad in (replace(cycle,findings=(*cycle.findings,'unrecognized_broker_order')),
                    replace(cycle,positions_available=tuple(replace(p,available_quantity=4) for p in cycle.positions_available))):
            with self.assertRaises(RiskViewError): self.prepare(cycle=bad)

    def test_durable_admission_failure_leaves_no_half_allocated_family(self):
        view=self.prepare(); before=self.ledger.all_orders()
        with patch.object(self.ledger,'_persist',side_effect=OSError('PRIVATE_DISK')):
            with self.assertRaises(LedgerError) as caught:self.admit(prepared=view)
        self.assertNotIn('PRIVATE',str(caught.exception));self.assertEqual(self.ledger.all_orders(),before)

    def test_filled_cleanup_closes_group_only_after_real_activity_and_cash_reconciliation(self):
        self.admit();self.claim()
        created=self.n.now+timedelta(seconds=1);filled=created+timedelta(seconds=1);self.n.now=filled
        cleanup_id='00000000-0000-4000-8000-000000000060'
        self.ledger.observe(self.intent.intent_id,status='filled',filled_quantity=5,average_fill_price_usd='150',
            broker_order_id=cleanup_id,broker_created_at=created,broker_event_at=filled,received_at=filled)
        # The discrepancy lookup must confirm this cleanup, not reuse the
        # fixture's unrelated OTO parent response as terminal evidence.
        from urllib.parse import urlsplit,parse_qs
        from paperbot_us.broker import HttpReply
        original=self.n.broker._transport
        row={**self.n.f.parent,'id':cleanup_id,'client_order_id':'pb3-'+self.intent.intent_id[:40],
            'side':'sell','order_class':'simple','limit_price':'150','status':'filled','filled_qty':'5','filled_avg_price':'150',
            'created_at':created.isoformat(),'updated_at':filled.isoformat(),'filled_at':filled.isoformat(),'legs':None}
        def transport(method,url,headers,body):
            parsed=urlsplit(url)
            if (parsed.path=='/v2/orders:by_client_order_id'
                    and parse_qs(parsed.query).get('client_order_id')==[row['client_order_id']]):
                self.assertEqual(method,'GET');return HttpReply(200,deepcopy(row))
            return original(method,url,headers,body)
        self.n.broker._transport=transport
        self.n.positions=[];self.n.account.update(cash='10000',equity='10000')
        before=self.n.reconciler.refresh()
        self.assertFalse(before.ready_for_risk,'a ledger fill alone cannot stand in for missing activities')
        self.n.fills.append(self.n.fill('00000000-0000-4000-8000-000000000061','sell','5','150',cleanup_id,12,'5','0'))
        self.n.now+=timedelta(seconds=1)
        final=self.n.reconciler.refresh();self.assertTrue(final.ready_for_risk,final.findings)
        self.assertEqual(final.book_receipt.result.profit_usd,0)
        self.assertEqual(self.book.replay(final.book_receipt.sequence),final.book_receipt.result)
        self.assertEqual(self.n.f.child['status'],'canceled','cleanup fills never turn into stop fills')
        self.assertEqual(self.m.project_groups(self.ledger.all_orders())[self.family].remaining_quantity,0)
        view=self.f.prepare(cycle=final)
        self.assertEqual(view.portfolio.stop_reservations,())


if __name__=='__main__':unittest.main()
