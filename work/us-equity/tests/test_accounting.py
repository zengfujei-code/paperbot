from dataclasses import replace
from datetime import datetime,timedelta,timezone
from decimal import Decimal
import importlib
import unittest


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.m=importlib.import_module('paperbot_us.accounting')
        self.now=datetime(2026,10,23,15,tzinfo=timezone.utc);self.start=self.now-timedelta(hours=1)
        self.buy_id='00000000-0000-4000-8000-000000000001';self.sell_id='00000000-0000-4000-8000-000000000002'
        self.anchor=self.m.AccountAnchor(self.start,'10000','a'*64,'1'*64)
        self.buy=self.order(self.buy_id,'buy',5,'101',5,'500')
        self.sell=self.order(self.sell_id,'sell',2,'109',2,'220')
        self.fills=[self.fill('b'*64,self.buy_id,'buy',5,'100',5,0,30),
                    self.fill('c'*64,self.sell_id,'sell',2,'110',2,0,40)]
        self.fee=self.m.Activity('d'*64,'FEE',net_usd='-0.02',source_day='2026-10-23')

    def order(self,identity,side,quantity,limit,filled,notional):
        return self.m.OrderEvidence('e'*64 if side=='buy' else 'f'*64,identity,'AAPL',side,quantity,limit,
            filled,notional,self.start+timedelta(minutes=1),self.now-timedelta(seconds=5))

    def fill(self,identity,order,side,qty,price,cumulative,leaves,minute):
        return self.m.Activity(identity,'FILL',symbol='AAPL',side=side,quantity=qty,price_usd=price,
            cumulative_quantity=cumulative,leaves_quantity=leaves,broker_order_id=order,
            occurred_at=self.start+timedelta(minutes=minute))

    def inputs(self,**changes):
        positions=(self.m.AccountMark('AAPL',3,'110','330',self.now-timedelta(seconds=1)),)
        args=dict(anchor=self.anchor,window=self.m.ActivityWindow(self.start,self.now-timedelta(seconds=3),
            self.now-timedelta(seconds=2),tuple(self.fills+[self.fee])),orders=(self.buy,self.sell),
            account=self.m.AccountSnapshot(self.now,'9719.98','10049.98',False,positions,self.anchor.account_sha256),now=self.now)
        return {**args,**changes}

    def test_buy_partial_exit_and_fee_match_an_independent_cash_and_profit_oracle(self):
        result=self.m.reconcile(**self.inputs())
        self.assertTrue(result.reconciled,result.violations)
        self.assertEqual(result.expected_cash_usd,Decimal('9719.98'))
        self.assertEqual(result.realized_usd,Decimal(20));self.assertEqual(result.unrealized_usd,Decimal(30))
        self.assertEqual(result.fees_usd,Decimal('.02'));self.assertEqual(result.profit_usd,Decimal('49.98'))
        self.assertEqual(result.inventory[0].quantity,3);self.assertEqual(result.inventory[0].average_cost_usd,Decimal(100))

    def test_empty_window_is_zero_profit_only_with_a_verified_empty_10000_anchor(self):
        args=self.inputs(orders=(),account=self.m.AccountSnapshot(self.now,'10000','10000',False,(),self.anchor.account_sha256))
        args['window']=replace(args['window'],activities=())
        self.assertTrue(self.m.reconcile(**args).reconciled)
        for change in (dict(cash_usd='10001',equity_usd='10001'),dict(blocked=True)):
            with self.subTest(change=change):
                result=self.m.reconcile(**{**args,'account':replace(args['account'],**change)})
                self.assertFalse(result.reconciled);self.assertIsNone(result.profit_usd)

    def test_missing_foreign_limit_breaking_or_wrong_cumulative_fills_block_reconciliation(self):
        args=self.inputs()
        cases=[tuple(self.fills[1:]+[self.fee]),tuple([replace(self.fills[0],broker_order_id='00000000-0000-4000-8000-000000000003'),self.fills[1],self.fee]),
            tuple([replace(self.fills[0],price_usd='102'),self.fills[1],self.fee]),
            tuple([replace(self.fills[0],cumulative_quantity=4),self.fills[1],self.fee])]
        for activities in cases:
            with self.subTest(activities=activities):
                result=self.m.reconcile(**{**args,'window':replace(args['window'],activities=activities)})
                self.assertFalse(result.reconciled);self.assertIsNone(result.profit_usd)

    def test_cash_positions_equity_marks_and_stale_account_are_independent_checks(self):
        args=self.inputs();account=args['account'];mark=account.positions[0]
        for changed in (replace(account,cash_usd='9720'),replace(account,equity_usd='10050'),
            replace(account,positions=(replace(mark,quantity=2),)),replace(account,positions=(replace(mark,market_value_usd='329'),)),
            replace(account,received_at=self.now-timedelta(seconds=31)),
            replace(account,positions=(replace(mark,received_at=self.now-timedelta(seconds=31)),))):
            with self.subTest(changed=changed):
                result=self.m.reconcile(**{**args,'account':changed});self.assertFalse(result.reconciled);self.assertIsNone(result.profit_usd)

    def test_unsupported_cash_activity_and_preanchor_fill_never_become_profit(self):
        args=self.inputs()
        for extra in (self.m.Activity('1'*64,'UNSUPPORTED',provider_kind='CSD'),
                      replace(self.fee,identity='3'*64,net_usd='0.02'),replace(self.fills[0],identity='2'*64,occurred_at=self.start-timedelta(seconds=1))):
            with self.subTest(extra=extra):
                result=self.m.reconcile(**{**args,'window':replace(args['window'],activities=tuple(self.fills+[self.fee,extra]))})
                self.assertFalse(result.reconciled);self.assertIsNone(result.profit_usd)

    def test_partial_fills_use_individual_prices_and_cumulative_economics(self):
        first=self.fill('1'*64,self.buy_id,'buy',2,'99',2,3,20)
        second=self.fill('2'*64,self.buy_id,'buy',3,'100.66666666',5,0,30)
        args=self.inputs();notional=Decimal(198)+Decimal('100.66666666')*3
        orders=(replace(self.buy,filled_notional_usd=notional),self.sell)
        result=self.m.reconcile(**{**args,'orders':orders,'window':replace(args['window'],activities=(first,second,self.fills[1],self.fee))})
        self.assertTrue(result.reconciled,result.violations)

    def test_duplicate_events_future_times_bad_types_and_incomplete_windows_are_rejected(self):
        args=self.inputs()
        with self.assertRaises(ValueError): replace(args['window'],activities=(self.fills[0],self.fills[0]))
        for changed in (replace(args['window'],complete=False),replace(args['window'],until=self.now+timedelta(seconds=1),received_at=self.now+timedelta(seconds=2))):
            self.assertFalse(self.m.reconcile(**{**args,'window':changed}).reconciled)
        for changes in (dict(quantity=True),dict(price_usd='NaN'),dict(quantity=1.5),dict(symbol='<script>')):
            with self.subTest(changes=changes),self.assertRaises(ValueError): replace(self.fills[0],**changes)

    def test_future_fee_source_time_cannot_hide_inside_the_current_date(self):
        args=self.inputs(orders=(),account=self.m.AccountSnapshot(self.now,'9999','9999',False,(),self.anchor.account_sha256))
        fee=self.m.Activity('9'*64,'FEE',net_usd='-1',source_day=self.now.date().isoformat(),source_time=self.now+timedelta(hours=1))
        args['window']=replace(args['window'],activities=(fee,))
        result=self.m.reconcile(**args)
        self.assertFalse(result.reconciled);self.assertIsNone(result.profit_usd)

    def test_timestamped_fee_before_or_at_anchor_is_not_experiment_profit(self):
        args=self.inputs(orders=(),account=self.m.AccountSnapshot(self.now,'9999','9999',False,(),self.anchor.account_sha256))
        for at in (self.start,self.start-timedelta(seconds=1)):
            fee=self.m.Activity('8'*64,'FEE',net_usd='-1',source_day=self.start.date().isoformat(),source_time=at)
            args['window']=replace(args['window'],activities=(fee,))
            result=self.m.reconcile(**args)
            self.assertFalse(result.reconciled);self.assertIsNone(result.profit_usd)

    def test_matching_money_from_another_paper_account_is_not_reconciled(self):
        args=self.inputs();self.assertTrue(hasattr(args['account'],'account_sha256'),'private accounting binding required')
        changed=replace(args['account'],account_sha256='9'*64)
        value=self.m.reconcile(**{**args,'account':changed})
        self.assertFalse(value.reconciled);self.assertIn('account_identity_mismatch',value.violations)

    def stop_inputs(self):
        self.assertIn('order_type',self.m.OrderEvidence.__dataclass_fields__,
                      'native stop-market accounting must be distinct from limit accounting')
        parent=replace(self.buy,filled_at=self.start+timedelta(minutes=30))
        stop=replace(self.sell,quantity=5,limit_price_usd=None,filled_quantity=5,filled_notional_usd='450',
            order_type='stop',stop_price_usd='98',parent_order_id=self.buy_id,
            filled_at=self.start+timedelta(minutes=40))
        args=self.inputs(orders=(parent,stop),account=self.m.AccountSnapshot(self.now,'9950','9950',False,(),self.anchor.account_sha256))
        args['window']=replace(args['window'],activities=(self.fills[0],self.fill('c'*64,self.sell_id,'sell',5,'90',5,0,40)))
        return args

    def test_native_stop_gap_fill_uses_actual_price_and_realized_loss(self):
        args=self.stop_inputs();result=self.m.reconcile(**args)
        self.assertTrue(result.reconciled,result.violations)
        self.assertEqual(result.expected_cash_usd,Decimal('9950'))
        self.assertEqual(result.realized_usd,Decimal('-50'))
        self.assertEqual(result.profit_usd,Decimal('-50'))
        self.assertEqual(result.inventory,())
        book_order=args['orders'][1]
        self.assertNotIn(self.sell_id,repr(book_order))
        self.assertNotIn(self.buy_id,repr(book_order))

    def test_stop_fill_needs_an_owned_matching_fully_filled_parent_and_completion_time(self):
        args=self.stop_inputs();parent,child=args['orders']
        cases=((child,), (replace(parent,symbol='MSFT'),child),
               (replace(parent,filled_at=None),child),
               (replace(parent,filled_quantity=4,filled_notional_usd='400',filled_at=None),child),
               (parent,replace(child,quantity=6,filled_at=None)))
        for orders in cases:
            with self.subTest(orders=orders):
                result=self.m.reconcile(**{**args,'orders':orders})
                self.assertFalse(result.reconciled);self.assertIsNone(result.profit_usd)
                self.assertIn('stop_parent_mismatch',result.violations)
        # The exit cannot precede complete entry activation, even if cash agrees.
        bad=replace(args['window'],activities=(args['window'].activities[0],
            replace(args['window'].activities[1],occurred_at=self.start+timedelta(minutes=29))))
        result=self.m.reconcile(**{**args,'window':bad})
        self.assertIn('stop_fill_before_entry_completion',result.violations)

    def test_simple_sell_limit_cannot_be_relabelled_as_an_unowned_stop(self):
        self.stop_inputs()
        for changes in (dict(order_type='market'),dict(order_type='stop',limit_price_usd=None),
                        dict(order_type='stop',side='buy',limit_price_usd=None,stop_price_usd='98',parent_order_id=self.buy_id),
                        dict(order_type='limit',parent_order_id=self.buy_id),dict(order_type='limit',stop_price_usd='98')):
            with self.subTest(changes=changes),self.assertRaises(ValueError): replace(self.sell,**changes)
        args=self.inputs();fill=replace(self.fills[1],price_usd='90')
        result=self.m.reconcile(**{**args,'window':replace(args['window'],activities=(self.fills[0],fill,self.fee))})
        self.assertIn('individual_fill_limit_violation',result.violations)

    def test_equal_time_parent_and_child_fills_follow_parent_dependency_not_uuid_order(self):
        args=self.stop_inputs();parent,stop=args['orders'];first,last=args['window'].activities
        parent=replace(parent,broker_order_id=self.sell_id)
        stop=replace(stop,broker_order_id=self.buy_id,parent_order_id=self.sell_id,filled_at=parent.filled_at)
        first=replace(first,broker_order_id=self.sell_id)
        last=replace(last,broker_order_id=self.buy_id,occurred_at=first.occurred_at)
        result=self.m.reconcile(**{**args,'orders':(parent,stop),'window':replace(args['window'],activities=(last,first))})
        self.assertTrue(result.reconciled,result.violations)

    def test_repair_cannot_precede_the_reported_complete_entry_even_when_money_matches(self):
        args=self.stop_inputs();parent,stop=args['orders']
        parent=replace(parent,filled_at=self.start+timedelta(minutes=35))
        stop=replace(stop,created_at=self.start+timedelta(minutes=36),protection_kind='repair',
            parent_terminal_at=self.start+timedelta(minutes=31),allocation_sha256='7'*64)
        result=self.m.reconcile(**{**args,'orders':(parent,stop)})
        self.assertFalse(result.reconciled);self.assertIn('repair_parent_mismatch',result.violations)

    def test_terminal_partial_entry_allocation_cannot_precede_its_actual_buy_fill(self):
        args=self.stop_inputs();parent,stop=args['orders']
        parent=replace(parent,quantity=10,filled_at=None)
        stop=replace(stop,created_at=self.start+timedelta(minutes=36),protection_kind='repair',
            parent_terminal_at=self.start+timedelta(minutes=29),allocation_sha256='7'*64)
        first,last=args['window'].activities
        window=replace(args['window'],activities=(replace(first,leaves_quantity=5),last))
        result=self.m.reconcile(**{**args,'orders':(parent,stop),'window':window})
        self.assertFalse(result.reconciled);self.assertIn('repair_parent_mismatch',result.violations)


if __name__=='__main__': unittest.main()
