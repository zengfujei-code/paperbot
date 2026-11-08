from dataclasses import replace
from datetime import timedelta
import importlib,importlib.util,unittest

import test_cleanup_recovery
from paperbot_us.risk import Quote,RiskState


class RepairProtectionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.repair_protection'),
            'standalone protection repair must have an explicit safe contract')
        self.m=importlib.import_module('paperbot_us.repair_protection')
        self.r=test_cleanup_recovery.CleanupRecoveryTests();self.r.setUp();self.addCleanup(self.r.doCleanups)
        self.n=self.r.n;self.f=self.r.e.f

    def prepare(self,**changes):
        cycle=changes.pop('cycle',None) or self.n.reconciler.refresh()
        return self.m.prepare_repair(**{**dict(book=self.f.book,ledger=self.r.ledger,baselines=self.f.f.baselines,
            cycle=cycle,quotes=(Quote('AAPL','150','150.1',self.n.now,self.n.now),),family_id=self.f.family,
            operator_state=RiskState(pause_entries=True),calendar=self.f.f.calendar,limits=self.r.e.f.limits,now=self.n.now),**changes})

    def body(self,plan,**changes):
        return plan.request_body(**{**dict(book=self.f.book,ledger=self.r.ledger,baselines=self.f.f.baselines,
            calendar=self.f.f.calendar,limits=self.r.e.f.limits,now=self.n.now),**changes})

    def test_repair_body_is_a_real_stop_contract_and_never_claims_active_protection(self):
        self.r.cancel();plan=self.prepare();body=self.body(plan)
        self.assertEqual(body['type'],'stop');self.assertEqual(body['side'],'sell')
        self.assertEqual(body['qty'],'5');self.assertEqual(body['stop_price'],'145.00')
        self.assertEqual(body['time_in_force'],'gtc');self.assertNotIn('limit_price',body)
        self.assertFalse(body['extended_hours'])
        self.assertFalse(plan.public_summary()['protection_active'])
        self.assertFalse(plan.public_summary()['ready_for_execution'])

    def test_gap_below_old_trigger_uses_a_legal_current_trigger_without_profit_claim(self):
        self.r.cancel()
        plan=self.prepare(quotes=(Quote('AAPL','120','120.1',self.n.now,self.n.now),))
        self.assertEqual(self.body(plan)['stop_price'],'119.99')
        self.assertEqual(plan.public_summary()['quantity'],5)

    def test_pending_cleanup_or_unknown_cancellation_cannot_be_repaired_over(self):
        self.r.settle=False;self.r.cancel()
        with self.assertRaises(self.m.RepairError):self.prepare()

    def test_partial_sell_cancel_repairs_only_remaining_owned_shares(self):
        self.r.partial();self.r.cancel();plan=self.prepare()
        self.assertEqual(plan.quantity,3);self.assertEqual(self.body(plan)['qty'],'3')

    def test_global_halt_forbids_a_new_repair_order(self):
        self.r.cancel()
        with self.assertRaises(self.m.RepairError):self.prepare(operator_state=RiskState(pause_entries=True,halt_orders=True))

    def test_expired_plan_or_changed_account_sources_cannot_emit_a_request_body(self):
        self.r.cancel();plan=self.prepare();self.n.now+=timedelta(seconds=1)
        self.n.reconciler.refresh()
        with self.assertRaises(self.m.RepairError):self.body(plan)
        plan=self.prepare();self.n.now+=timedelta(seconds=21)
        with self.assertRaises(self.m.RepairError):self.body(plan)

    def test_invalid_tiny_price_never_becomes_a_zero_or_above_market_stop(self):
        self.r.cancel()
        with self.assertRaises(self.m.RepairError):self.prepare(
            quotes=(Quote('AAPL','0.0098','0.0098',self.n.now,self.n.now),))

    def test_repair_request_cannot_relax_the_recorded_limits(self):
        self.r.cancel();plan=self.prepare()
        with self.assertRaises(self.m.RepairError):self.body(plan,
            limits=replace(self.r.e.f.limits,max_spread_bps='100'))

    def test_stale_terminal_cleanup_requires_a_new_exact_lookup_before_repair(self):
        self.r.cancel();self.n.now+=timedelta(seconds=45)
        with self.assertRaises(self.m.RepairError):self.prepare()
        result=self.r.gateway.recover(self.r.ledger,self.r.identity)
        self.assertEqual(result.status,'cleanup_terminal')
        self.assertEqual(self.body(self.prepare())['qty'],'5')

    def test_newer_unknown_lookup_invalidates_even_fresh_terminal_handoff(self):
        self.r.cancel();plan=self.prepare();self.n.now+=timedelta(seconds=1)
        self.r.missing=True
        self.assertEqual(self.r.gateway.recover(self.r.ledger,self.r.identity).status,'cleanup_unknown')
        with self.assertRaises(self.m.RepairError):self.body(plan)
        cycle=self.n.reconciler.refresh()
        with self.assertRaises(self.m.RepairError):self.prepare(cycle=cycle)
        self.r.missing=False
        self.r.gateway.recover(self.r.ledger,self.r.identity)
        self.assertEqual(self.body(self.prepare())['qty'],'5')


if __name__=='__main__':unittest.main()
