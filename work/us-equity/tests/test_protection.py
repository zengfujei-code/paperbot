from dataclasses import replace
from datetime import datetime,timedelta,timezone
import importlib
import importlib.util
import unittest
from paperbot_us.data import Calendar,Session
from paperbot_us.risk import OrderIntent,Quote


class ProtectionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.protection'),'native stop contract and group states required')
        self.m=importlib.import_module('paperbot_us.protection')
        self.now=datetime(2026,10,23,15,tzinfo=timezone.utc)
        self.intent=OrderIntent('a'*64,'AAPL','buy',5,'100','98',self.now+timedelta(seconds=20))
        self.plan=self.m.NativeStopPlan.from_intent(self.intent)
        self.quote=Quote('AAPL','100','100.01',self.now,self.now)
        self.calendar=Calendar([Session.from_alpaca(dict(date='2026-10-23',open='09:30',close='16:00'))],
            start='2026-10-23',end='2026-10-23')
        self.group_id='00000000-0000-4000-8000-000000000010'
        common=dict(asset_class='us_equity',symbol='AAPL',qty='5',filled_qty='0',filled_avg_price=None,
            order_class='oto',time_in_force='day',extended_hours=False,
            created_at=(self.now-timedelta(seconds=3)).isoformat(),updated_at=(self.now-timedelta(seconds=1)).isoformat(),
            group_id=self.group_id)
        self.child={**common,'id':'00000000-0000-4000-8000-000000000002','client_order_id':'provider-child',
            'side':'sell','type':'stop','limit_price':None,'stop_price':'98','status':'held','group_role':'stop_loss'}
        self.parent={**common,'id':'00000000-0000-4000-8000-000000000001','client_order_id':'pb3-'+'a'*40,
            'side':'buy','type':'limit','limit_price':'100','stop_price':None,'status':'new','group_role':'entry','legs':[self.child]}

    def assess(self):
        group=self.m.parse_oto_group(self.parent,plan=self.plan,received_at=self.now)
        return self.m.assess_protection(group,calendar=self.calendar,now=self.now)

    def full_parent(self):
        self.parent.update(filled_qty='5',filled_avg_price='100',status='filled',
                           filled_at=(self.now-timedelta(seconds=2)).isoformat())

    def test_request_is_limit_entry_with_attached_native_sell_stop_and_no_network(self):
        body=self.plan.request_body(quote=self.quote,market_price_usd='100',market_at=self.now,
                                    calendar=self.calendar,now=self.now)
        self.assertEqual(body,dict(symbol='AAPL',side='buy',qty='5',type='limit',limit_price='100.00',
            time_in_force='day',extended_hours=False,order_class='oto',client_order_id='pb3-'+'a'*40,
            stop_loss=dict(stop_price='98.00')))
        for quote,price in ((replace(self.quote,as_of=self.now-timedelta(seconds=31)),'100'),(self.quote,'98')):
            with self.assertRaises(self.m.ProtectionError):
                self.plan.request_body(quote=quote,market_price_usd=price,market_at=self.now,calendar=self.calendar,now=self.now)

    def test_partial_entry_is_unprotected_and_requests_reconciliation_not_an_extra_sell(self):
        self.parent.update(filled_qty='2',filled_avg_price='100',status='partially_filled')
        state=self.assess()
        self.assertEqual(state.status,'unprotected_partial_entry')
        self.assertTrue(state.pause_entries)
        self.assertFalse(state.native_stop_active)
        self.assertEqual(state.remaining_quantity,2)
        self.assertIn('cancel_unfilled_entry',state.required_actions)
        self.assertNotIn('submit_sell',state.required_actions)

    def test_full_entry_requires_observed_active_stop_and_gap_fill_below_trigger_is_valid(self):
        self.full_parent()
        self.assertEqual(self.assess().status,'protection_pending')
        self.child['status']='new'
        state=self.assess();self.assertTrue(state.native_stop_active);self.assertFalse(state.pause_entries)
        self.child.update(status='partially_filled',filled_qty='2',filled_avg_price='90')
        state=self.assess();self.assertEqual(state.remaining_quantity,3);self.assertTrue(state.native_stop_active)
        self.child.update(status='filled',filled_qty='5',filled_avg_price='90',
                          filled_at=(self.now-timedelta(seconds=1)).isoformat())
        state=self.assess();self.assertEqual(state.status,'closed');self.assertEqual(state.remaining_quantity,0)

    def test_missing_failed_canceled_or_expired_protection_never_looks_active(self):
        self.full_parent()
        for status in ('rejected','canceled','expired','pending_cancel'):
            self.child['status']=status;state=self.assess()
            self.assertTrue(state.pause_entries);self.assertFalse(state.native_stop_active)
        self.parent['legs']=[]
        self.assertTrue(self.assess().pause_entries)

    def test_group_identity_quantity_side_and_stop_price_mismatches_are_refused(self):
        for changes in (dict(group_id='00000000-0000-4000-8000-000000000099'),dict(group_role='take_profit'),
                        dict(qty='6'),dict(side='buy'),dict(stop_price='97'),dict(type='stop_limit',limit_price='97')):
            with self.subTest(changes=changes):
                original=dict(self.child);self.child.update(changes)
                with self.assertRaises(self.m.ProtectionError): self.assess()
                self.child.clear();self.child.update(original)
        self.parent['client_order_id']='foreign-parent'
        with self.assertRaises(self.m.ProtectionError): self.assess()

    def test_old_evidence_and_closed_session_do_not_claim_effective_protection(self):
        self.full_parent();self.child['status']='new'
        group=self.m.parse_oto_group(self.parent,plan=self.plan,received_at=self.now)
        with self.assertRaises(self.m.ProtectionError):
            self.m.assess_protection(group,calendar=self.calendar,now=self.now+timedelta(seconds=31))
        self.now=self.now.replace(hour=21)
        state=self.assess();self.assertTrue(state.pause_entries);self.assertEqual(state.status,'out_of_session_exposure')
        self.assertFalse(state.native_stop_active)

    def test_old_child_and_unproven_activation_timeline_do_not_claim_protection(self):
        self.full_parent();self.child['status']='new'
        self.child.update(created_at=(self.now-timedelta(days=1)).isoformat(),
                          updated_at=(self.now-timedelta(days=1)).isoformat())
        with self.assertRaises(self.m.ProtectionError): self.assess()
        self.child.update(created_at=self.parent['created_at'],updated_at=self.parent['created_at'])
        self.assertTrue(self.assess().pause_entries)
        self.assertFalse(self.assess().native_stop_active)
        self.child['updated_at']=(self.now-timedelta(seconds=1)).isoformat()
        self.parent['filled_at']=None
        self.assertTrue(self.assess().pause_entries)

    def test_zero_or_partial_entry_with_an_active_stop_remains_an_unresolved_group(self):
        for parent_status in ('new','canceled','expired','rejected'):
            for child_status in ('new','accepted','pending_cancel'):
                with self.subTest(parent=parent_status,child=child_status):
                    self.parent['status']=parent_status;self.child['status']=child_status
                    state=self.assess();self.assertTrue(state.pause_entries)
                    self.assertFalse(state.native_stop_active)
                    self.assertIn('reconcile_group',state.required_actions)
        self.parent.update(status='partially_filled',filled_qty='2',filled_avg_price='100')
        self.child['status']='new'
        self.assertTrue(self.assess().pause_entries)

    def test_expired_or_out_of_session_pending_entry_is_not_normal_waiting(self):
        self.now+=timedelta(seconds=21)
        state=self.assess();self.assertTrue(state.pause_entries)
        self.assertIn('cancel_unfilled_entry',state.required_actions)
        self.now=self.now.replace(hour=21)
        state=self.assess();self.assertTrue(state.pause_entries)
        self.assertIn('reconcile_group',state.required_actions)

    def test_preopen_prices_are_not_used_to_build_a_regular_session_protected_entry(self):
        opened=self.now.replace(hour=13,minute=30,second=5)
        preopen=opened-timedelta(seconds=10)
        quote=replace(self.quote,as_of=preopen,received_at=opened)
        plan=self.m.NativeStopPlan.from_intent(replace(self.intent,expires_at=opened+timedelta(seconds=20)))
        with self.assertRaises(self.m.ProtectionError):
            plan.request_body(quote=quote,market_price_usd='100',market_at=preopen,calendar=self.calendar,now=opened)


if __name__=='__main__': unittest.main()
