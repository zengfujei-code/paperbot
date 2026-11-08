from copy import deepcopy
from datetime import timedelta
import unittest
from urllib.parse import urlsplit,parse_qs
from unittest.mock import patch

import test_native_close
from paperbot_us.broker import HttpReply
from paperbot_us.decision_journal import capture_close_decision
from paperbot_us.native_close import prepare_close
from paperbot_us.compensation import CloseCompensator,CompensationError
from paperbot_us.repair_decisions import RepairDecisionJournal
from paperbot_us.risk import Quote
from paperbot_us.runner import _exclusive


class CompensationTests(unittest.TestCase):
    def setUp(self):
        self.c=test_native_close.NativeCloseTests();self.c.setUp();self.addCleanup(self.c.doCleanups)
        self.n=self.c.n;self.rows={};self.posts=[];self.cancels=[];self.missing=set();self.failure_kind=None
        self.allow_repair_close=False
        original=self.n.broker._transport
        def transport(method,url,headers,body):
            parsed=urlsplit(url)
            if method=='POST':
                self.assertEqual(parsed.netloc,'paper-api.alpaca.markets');self.posts.append(deepcopy(body))
                row=dict(**body,id='00000000-0000-4000-8000-'+format(500+len(self.posts),'012d'),
                    asset_class='us_equity',status='new',filled_qty='0',filled_avg_price=None,filled_at=None,
                    created_at=self.n.now.isoformat(),updated_at=self.n.now.isoformat(),legs=None)
                row.setdefault('limit_price',None);row.setdefault('stop_price',None)
                self.rows[row['client_order_id']]=row;self.n.visible=[deepcopy(row)]
                if self.n.positions:self.n.positions[0]['qty_available']='0'
                if self.failure_kind==body['type']:raise OSError('PRIVATE_PROVIDER')
                return HttpReply(201,deepcopy(row))
            if parsed.path=='/v2/orders:by_client_order_id':
                client=parse_qs(parsed.query).get('client_order_id',[None])[0]
                if client in self.missing:return HttpReply(404)
                if client in self.rows:return HttpReply(200,deepcopy(self.rows[client]))
            if method=='DELETE':
                matches=[r for r in self.rows.values() if parsed.path=='/v2/orders/'+r['id']]
                if matches:
                    row=matches[0]
                    self.assertTrue(row['type']=='limit' or self.allow_repair_close,
                        'compensation must preserve active repair stops')
                    self.cancels.append(row['id']);self.n.now+=timedelta(seconds=1)
                    row.update(status='canceled',updated_at=self.n.now.isoformat());self.n.visible=[]
                    if self.n.positions:self.n.positions[0]['qty_available']=self.n.positions[0]['qty']
                    return HttpReply(204)
            return original(method,url,headers,body)
        self.n.broker._transport=transport
        self.repairs=RepairDecisionJournal(self.n.f.f.path.with_name('compensation-decisions.sqlite'),clock=lambda:self.n.now)
        self.worker=CloseCompensator(broker=self.n.broker,ledger=self.c.ledger,reconciler=self.n.reconciler,
            controls=self.c.controls,decisions=self.c.journal,repair_decisions=self.repairs,
            policy=self.c.policy,limits=self.c.limits,calendar=self.c.f.calendar)

    def start(self):self.c.record();return self.c.request()

    def advance(self,quotes=True):
        provider=(lambda:(Quote('AAPL','150','150.1',self.n.now,self.n.now),)) if quotes else None
        return self.worker.advance(self.c.family,baselines=self.c.f.baselines,quotes=provider)

    def limit(self):return next(r for r in reversed(tuple(self.rows.values())) if r['type']=='limit')

    def fill_cleanup(self,quantity):
        row=self.limit();prior=int(row['filled_qty']);self.n.now+=timedelta(seconds=1)
        row.update(status='filled' if quantity==5 else 'partially_filled',filled_qty=str(quantity),
            filled_avg_price='150',updated_at=self.n.now.isoformat(),
            filled_at=self.n.now.isoformat() if quantity==5 else None)
        left=5-quantity;self.n.visible=[] if not left else [deepcopy(row)]
        self.n.positions=[] if not left else [{**self.n.positions[0],'qty':str(left),'qty_available':'0','market_value':str(150*left)}]
        self.n.account.update(cash=str(9250+150*quantity),equity='10000')
        self.n.fills.append(self.n.fill('00000000-0000-4000-8000-00000000008'+str(quantity),'sell',
            str(quantity-prior),'150',row['id'],int((self.n.now-self.n.f.now).total_seconds()),str(quantity),str(left)))

    def test_observed_stop_terminal_starts_one_owned_cleanup(self):
        self.assertEqual(self.start().status,'cleanup_required');result=self.advance()
        self.assertEqual(result.status,'cleanup_dispatched');self.assertEqual(len(self.posts),1)
        self.assertEqual(self.posts[0]['type'],'limit');self.advance();self.assertEqual(len(self.posts),1)
        self.assertEqual(len(self.c.deletes),1)

    def test_full_cleanup_fill_and_cash_proof_finish_the_task_without_another_sell(self):
        self.start();self.advance();self.fill_cleanup(5);result=self.advance()
        self.assertEqual(result.status,'closed');self.assertEqual(result.remaining_quantity,0)
        self.assertEqual(self.c.f.book.snapshot().receipt.result.profit_usd,0)
        self.assertEqual(self.advance().status,'previously_resolved');self.assertEqual(len(self.posts),1)

    def test_expired_partial_cleanup_is_canceled_before_protecting_only_remaining_shares(self):
        self.start();self.advance();self.fill_cleanup(2);self.n.now+=timedelta(seconds=21)
        result=self.advance();self.assertEqual(result.status,'repair_dispatched')
        self.assertEqual([r['type'] for r in self.posts],['limit','stop']);self.assertEqual(self.posts[-1]['qty'],'3')
        self.assertEqual(len(self.cancels),1);self.assertEqual(self.advance().status,'restored')
        self.advance();self.assertEqual(len(self.posts),2)

    def test_unknown_cleanup_post_preserves_the_one_shot_claim_then_get_recovers(self):
        self.start();self.failure_kind='limit';result=self.advance()
        self.assertEqual(result.status,'cleanup_submission_unknown')
        self.failure_kind=None;self.advance();self.assertEqual(len(self.posts),1)
        self.fill_cleanup(5);self.assertEqual(self.advance().status,'closed')

    def test_unknown_cleanup_lookup_never_allows_a_replacement_stop(self):
        self.start();self.advance();self.missing.add(self.limit()['client_order_id'])
        self.n.now+=timedelta(seconds=31);result=self.advance()
        self.assertEqual(result.status,'cleanup_unknown_or_working');self.assertIsNone(result.remaining_quantity)
        self.assertEqual(len(self.posts),1);self.assertEqual(self.cancels,[])

    def test_unknown_repair_post_does_not_create_another_repair_after_restart(self):
        self.start();self.advance();self.n.now+=timedelta(seconds=21);self.failure_kind='stop'
        self.assertEqual(self.advance().status,'repair_dispatched');self.failure_kind=None
        repair=next(r for r in self.rows.values() if r['type']=='stop');self.missing.add(repair['client_order_id'])
        self.n.now+=timedelta(seconds=21);self.advance();self.assertEqual(len(self.posts),2)
        self.missing.clear();self.assertEqual(self.advance().status,'restored');self.assertEqual(len(self.posts),2)

    def test_current_halt_after_original_cancel_never_creates_a_new_order(self):
        self.start();self.c.controls.update(expected_revision=1,pause_entries=True,halt_orders=True,
            reason='operator_halt',now=self.n.now)
        self.assertEqual(self.advance().status,'unprotected_global_halt');self.assertEqual(self.posts,[])

    def test_missing_quotes_after_deadline_is_reported_and_never_guesses_a_price(self):
        self.start();self.n.now+=timedelta(seconds=31)
        result=self.advance(quotes=False)
        self.assertEqual(result.status,'repair_inputs_unavailable');self.assertEqual(self.posts,[])

    def test_missing_quotes_before_deadline_records_inputs_unavailable_without_a_sell(self):
        self.start();result=self.advance(quotes=False)
        self.assertEqual(result.status,'cleanup_inputs_unavailable');self.assertEqual(self.posts,[])

    def test_terminal_cleanup_is_refreshed_before_later_repair_confirmation(self):
        self.start();self.advance();self.fill_cleanup(2);self.n.now+=timedelta(seconds=21)
        self.assertEqual(self.advance().status,'repair_dispatched');self.n.now+=timedelta(seconds=31)
        self.assertEqual(self.advance().status,'restored');self.assertEqual(len(self.posts),2)

    def test_interrupted_terminal_cleanup_handoff_recovers_after_quote_outage(self):
        self.start();self.advance();self.n.now+=timedelta(seconds=21)
        self.assertEqual(self.advance(quotes=False).status,'repair_inputs_unavailable')
        self.assertEqual(len(self.cancels),1);self.n.now+=timedelta(seconds=31)
        self.assertEqual(self.advance().status,'repair_dispatched')
        self.assertEqual(self.advance().status,'restored');self.assertEqual(len(self.posts),2)

    def test_missing_terminal_cleanup_lookup_prevents_cached_handoff(self):
        self.start();self.advance();self.n.now+=timedelta(seconds=21)
        self.assertEqual(self.advance(quotes=False).status,'repair_inputs_unavailable')
        self.n.now+=timedelta(seconds=31);self.missing.add(self.limit()['client_order_id'])
        self.assertEqual(self.advance().status,'cleanup_unknown_or_working')
        self.assertEqual(len(self.posts),1);self.assertEqual(len(self.cancels),1)

    def test_known_unsent_stop_cancel_finishes_protected_without_a_replacement(self):
        self.c.record();original=self.c.m.claim_close
        def delayed(*args,**kwargs):
            value=original(*args,**kwargs);self.n.now+=timedelta(seconds=31);return value
        with patch.object(self.c.m,'claim_close',side_effect=delayed):self.c.request()
        self.assertEqual(self.advance().status,'close_not_sent_protected')
        self.assertEqual(self.c.deletes,[]);self.assertEqual(self.posts,[])

    def test_restored_repair_can_be_closed_through_its_own_owned_lifecycle(self):
        self.start();self.advance();self.n.now+=timedelta(seconds=21)
        self.assertEqual(self.advance().status,'repair_dispatched')
        self.assertEqual(self.advance().status,'restored')
        original_close=deepcopy(self.c.ledger.load(self.c.family)['native_stop']['close'])
        view=self.c.f.prepare(quotes=(Quote('AAPL','120','120.1',self.n.now,self.n.now),))
        plan=prepare_close(ledger=self.c.ledger,prepared=view,family_id=self.c.family,
            calendar=self.c.f.calendar,now=self.n.now,reason='account_risk')
        self.assertIsNotNone(plan.target_repair_id)
        current=self.c.controls.read(now=self.n.now)
        frame=capture_close_decision(ledger=self.c.ledger,prepared=plan,source=None,policy=self.c.policy,
            controls=current,limits=self.c.limits,calendar=self.c.f.calendar,now=self.n.now)
        receipt=self.c.journal.append(frame,ledger=self.c.ledger);self.allow_repair_close=True
        self.assertEqual(self.c.gateway.request(self.c.ledger,plan,decision=receipt,journal=self.c.journal,
            controls=self.c.controls,control_snapshot=current,limits=self.c.limits,calendar=self.c.f.calendar,
            book=self.c.f.book,baselines=self.c.f.baselines).status,'cleanup_required')
        result=self.worker.advance(self.c.family,baselines=self.c.f.baselines,target_repair_id=plan.target_repair_id,
            quotes=lambda:(Quote('AAPL','120','120.1',self.n.now,self.n.now),))
        self.assertEqual(result.status,'cleanup_dispatched');self.fill_cleanup(5)
        result=self.worker.advance(self.c.family,baselines=self.c.f.baselines,target_repair_id=plan.target_repair_id)
        self.assertEqual(result.status,'closed');self.assertEqual(result.remaining_quantity,0)
        self.assertEqual(self.c.ledger.load(self.c.family)['native_stop']['close'],original_close)
        self.assertEqual([p['type'] for p in self.posts],['limit','stop','limit'])

    def test_unknown_original_cancel_escalates_without_retry_or_alternative_sell(self):
        self.c.settle=False;self.c.failure=OSError('PRIVATE_PROVIDER');self.start()
        self.c.failure=None;self.advance();self.n.now+=timedelta(seconds=31)
        result=self.advance();self.assertEqual(result.status,'escalated_unknown_stop')
        self.assertEqual(len(self.c.deletes),1);self.assertEqual(self.posts,[])

    def test_second_execution_owner_cannot_enter_while_global_lock_is_held(self):
        self.start()
        with _exclusive(self.worker.lock_path):
            with self.assertRaises(CompensationError):self.advance()
        self.assertEqual(self.posts,[])


if __name__=='__main__':unittest.main()
