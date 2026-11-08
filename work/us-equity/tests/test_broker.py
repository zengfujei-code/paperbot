from datetime import datetime,timedelta,timezone
from decimal import Decimal
import importlib
import hashlib
import io
import threading
import unittest
import urllib.error
from urllib.parse import urlsplit,parse_qs
from unittest.mock import patch
from paperbot_us.risk import OrderIntent


class PaperBrokerTests(unittest.TestCase):
    def setUp(self):
        self.m=importlib.import_module('paperbot_us.broker')
        self.now=datetime(2026,10,23,17,tzinfo=timezone.utc)
        self.calls=[];self.responses=[]
        self.client_id='pb3-'+'a'*40
        self.order_id='00000000-0000-4000-8000-000000000001'
        self.intent=OrderIntent('a'*64,'AAPL','buy',5,'150.10','145',self.now+timedelta(seconds=20))

    def client(self):
        def transport(method,url,headers,body):
            self.calls.append((method,url,headers,body))
            result=self.responses.pop(0)
            if isinstance(result,Exception): raise result
            return result
        return self.m.PaperBroker('PK'+'T'*20,'T'*32,transport=transport,clock=lambda:self.now)

    def reply(self,status=200,payload=None): return self.m.HttpReply(status,payload)

    def order(self,**changes):
        return dict(id=self.order_id,client_order_id=self.client_id,asset_class='us_equity',
            symbol='AAPL',side='buy',qty='5',filled_qty='0',filled_avg_price=None,
            type='limit',order_class='simple',time_in_force='day',extended_hours=False,
            limit_price='150.1',status='new',created_at=self.now.isoformat(),
            updated_at=self.now.isoformat(),**changes)

    def test_paper_credentials_required_and_repr_has_no_secrets(self):
        for key in ('AK'+'T'*20,'','PK\nBAD'):
            with self.subTest(key=key),self.assertRaises(self.m.BrokerError): self.m.PaperBroker(key,'T'*32)
        self.assertEqual(repr(self.client()),'<PaperBroker paper-only>')

    def test_account_discards_identifiers_and_ignores_margin_buying_power(self):
        self.responses=[self.reply(payload=dict(id='00000000-0000-4000-8000-000000000050',account_number='PRIVATE_FIXTURE',
            currency='USD',status='ACTIVE',cash='10000',equity='10000',buying_power='40000',
            trading_blocked=False,account_blocked=False))]
        result=self.client().account()
        self.assertEqual((result.cash_usd,result.equity_usd),(Decimal(10000),Decimal(10000)))
        self.assertNotIn('PRIVATE',repr(result));self.assertFalse(result.blocked)
        self.assertEqual(urlsplit(self.calls[0][1]).netloc,'paper-api.alpaca.markets')

    def test_user_suspension_is_a_block_and_malformed_flag_fails_closed(self):
        base=dict(id='00000000-0000-4000-8000-000000000050',currency='USD',status='ACTIVE',cash='10000',equity='10000',trading_blocked=False,account_blocked=False)
        self.responses=[self.reply(payload={**base,'trade_suspended_by_user':True})]
        self.assertTrue(self.client().account().blocked)
        self.responses=[self.reply(payload={**base,'trade_suspended_by_user':'false'})]
        with self.assertRaises(self.m.BrokerError): self.client().account()

    def test_private_account_binding_discards_raw_id_and_rejects_missing_identity(self):
        identity='00000000-0000-4000-8000-000000000050'
        base=dict(id=identity,currency='USD',status='ACTIVE',cash='10000',equity='10000',trading_blocked=False,account_blocked=False)
        self.responses=[self.reply(payload=base)];value=self.client().account()
        self.assertTrue(hasattr(value,'account_sha256'),'private account binding required')
        expected=hashlib.sha256(b'paperbot-paper-account-v1\0'+identity.encode()).hexdigest()
        self.assertEqual(value.account_sha256,expected);self.assertNotIn(identity,repr(value));self.assertNotIn(expected,repr(value))
        for identity in (None,'bad'):
            self.responses=[self.reply(payload={**base,'id':identity})]
            with self.assertRaises(self.m.BrokerError): self.client().account()

    def test_positions_require_long_whole_us_shares(self):
        self.responses=[self.reply(payload=[dict(asset_class='us_equity',symbol='AAPL',side='long',qty='5',qty_available='4')])]
        self.assertEqual(self.client().positions()[0].quantity,5)
        for change in ({'qty':'1.5'},{'side':'short'},{'qty_available':'6'},{'asset_class':'crypto'}):
            row=dict(asset_class='us_equity',symbol='AAPL',side='long',qty='5',qty_available='5');row.update(change)
            self.responses=[self.reply(payload=[row])]
            with self.subTest(change=change),self.assertRaises(self.m.BrokerError): self.client().positions()

    def test_normalized_order_never_carries_unrecognized_metadata(self):
        row=self.order();row['untrusted']='PRIVATE_FIXTURE'
        self.responses=[self.reply(payload=row)]
        result=self.client().lookup(self.client_id)
        self.assertEqual(result.status,'open');self.assertEqual(result.filled_quantity,0)
        self.assertNotIn('PRIVATE',repr(result));self.assertTrue(result.owned)
        self.assertEqual(parse_qs(urlsplit(self.calls[0][1]).query),dict(client_order_id=[self.client_id]))

    def test_lookup_miss_is_only_a_read_result_and_non404_errors_are_sanitized(self):
        self.responses=[self.reply(404,{'message':'PRIVATE_FIXTURE'})]
        self.assertIsNone(self.client().lookup(self.client_id))
        self.responses=[self.reply(401,{'message':'PRIVATE_FIXTURE'})]
        with self.assertRaises(self.m.BrokerError) as caught: self.client().lookup(self.client_id)
        self.assertNotIn('PRIVATE',str(caught.exception))
        self.assertEqual([c[0] for c in self.calls],['GET','GET'])

    def test_open_book_detects_truncation_unknown_orders_and_duplicates(self):
        unknown=self.order();unknown['client_order_id']='manual-fixture'
        self.responses=[self.reply(payload=[unknown])]
        self.assertFalse(self.client().open_orders()[0].owned)
        for rows in ([self.order()]*2,[self.order()]*500):
            self.responses=[self.reply(payload=rows)]
            with self.assertRaises(self.m.BrokerError): self.client().open_orders()

    def test_submit_is_exact_regular_session_limit_and_response_matches_intent(self):
        self.responses=[self.reply(payload=self.order())]
        result=self.client()._submit_limit(self.intent)
        self.assertEqual(result.quantity,5)
        method,url,headers,body=self.calls[0]
        self.assertEqual((method,url),('POST','https://paper-api.alpaca.markets/v2/orders'))
        self.assertEqual(body,dict(symbol='AAPL',qty='5',side='buy',type='limit',time_in_force='day',
            limit_price='150.10',client_order_id=self.client_id,extended_hours=False,order_class='simple'))
        self.assertNotIn('stop_loss',body)
        changed=self.order();changed['qty']='6'
        self.responses=[self.reply(payload=changed)]
        with self.assertRaisesRegex(self.m.BrokerError,'submission_outcome_unknown'): self.client()._submit_limit(self.intent)

    def test_submission_timeout_or_http_error_never_retries(self):
        for value in (OSError('PRIVATE_FIXTURE'),self.reply(429,{'message':'PRIVATE_FIXTURE'}),
                      self.reply(302,{'location':'https://api.alpaca.markets'})):
            self.calls=[];self.responses=[value]
            with self.subTest(value=type(value)),self.assertRaisesRegex(self.m.BrokerError,'submission_outcome_unknown'):
                self.client()._submit_limit(self.intent)
            self.assertEqual(len(self.calls),1)

    def test_invalid_ticks_expiration_and_nonexact_intents_do_not_make_requests(self):
        from dataclasses import replace
        for value in (replace(self.intent,limit_price_usd=Decimal('150.101')),
                      replace(self.intent,expires_at=self.now),{'symbol':'AAPL'}):
            with self.subTest(value=type(value)),self.assertRaises(self.m.BrokerError): self.client()._submit_limit(value)
        self.assertEqual(self.calls,[])

    def test_cancel_ack_does_not_mean_terminal_and_failure_stays_unknown(self):
        self.responses=[self.reply(204)]
        result=self.client()._cancel(self.order_id)
        self.assertEqual(result,'cancel_requested')
        self.responses=[self.reply(422,{'message':'PRIVATE_FIXTURE'})]
        with self.assertRaisesRegex(self.m.BrokerError,'cancel_outcome_unknown'): self.client()._cancel(self.order_id)
        self.assertEqual([c[0] for c in self.calls],['DELETE','DELETE'])

    def test_unsupported_pending_states_do_not_release_orders(self):
        for status in ('pending_cancel','pending_replace','suspended','done_for_day'):
            row=self.order();row['status']=status
            self.responses=[self.reply(payload=row)]
            with self.subTest(status=status),self.assertRaises(self.m.BrokerError): self.client().lookup(self.client_id)

    def test_default_transport_refuses_redirects_and_bounds_response(self):
        class Response:
            status=200;headers={}
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read1(self,n): return b'x'*n
        with patch('paperbot_us.broker.urllib.request.build_opener') as build:
            build.return_value.open.return_value=Response()
            with self.assertRaises(self.m.BrokerError): self.m.PaperBroker._http('GET','https://paper-api.alpaca.markets/v2/account',{},None)
            self.assertTrue(any(isinstance(arg,self.m._NoRedirect) for arg in build.call_args.args))
            self.assertTrue(any(isinstance(arg,self.m.urllib.request.ProxyHandler) and arg.proxies=={} for arg in build.call_args.args))

    def activity(self,index=1,**changes):
        row=dict(id=f'20261023170000000::00000000-0000-4000-8000-{index:012d}',activity_type='FILL',
            order_id=self.order_id,symbol='AAPL',side='buy',type='fill',qty='1',cum_qty='1',leaves_qty='0',
            price='150',transaction_time=(self.now-timedelta(seconds=1)).isoformat(),account_id='PRIVATE_ACCOUNT')
        return {**row,**changes}

    def test_complete_activity_window_is_get_only_allowlisted_and_has_explicit_coverage(self):
        self.responses=[self.reply(payload=[self.activity()])]
        result=self.client().activities(after=self.now-timedelta(hours=1),until=self.now-timedelta(milliseconds=1))
        self.assertTrue(result.complete);self.assertEqual(result.activities[0].quantity,1)
        self.assertNotIn('PRIVATE',repr(result));self.assertEqual(self.calls[0][0],'GET')
        query=parse_qs(urlsplit(self.calls[0][1]).query)
        self.assertEqual(query['page_size'],['100']);self.assertEqual(query['direction'],['asc'])
        self.assertNotIn('activity_types',query)

    def test_activity_pagination_exhaustion_and_duplicate_rows_are_not_complete_history(self):
        self.responses=[self.reply(payload=[self.activity(i) for i in range(1,101)])]
        with self.assertRaises(self.m.BrokerError): self.client().activities(after=self.now-timedelta(hours=1),max_pages=1)
        self.responses=[self.reply(payload=[self.activity(),self.activity()])]
        with self.assertRaises(self.m.BrokerError): self.client().activities(after=self.now-timedelta(hours=1))
        self.responses=[self.reply(payload=[self.activity(i) for i in range(1,101)]),self.reply(payload=[])]
        result=self.client().activities(after=self.now-timedelta(hours=1))
        self.assertEqual(len(result.activities),100)
        self.assertIn('page_token',parse_qs(urlsplit(self.calls[-1][1]).query))

    def test_fees_and_unknown_cash_activity_are_retained_as_typed_evidence(self):
        fee=self.activity(activity_type='FEE',date='2026-10-23',net_amount='-0.02')
        unknown=self.activity(2,activity_type='CSD',net_amount='100')
        self.responses=[self.reply(payload=[fee,unknown])]
        result=self.client().activities(after=self.now-timedelta(hours=1))
        self.assertEqual([a.kind for a in result.activities],['FEE','UNSUPPORTED'])
        self.assertEqual(result.activities[0].net_usd,Decimal('-.02'))
        self.assertEqual(result.activities[1].provider_kind,'CSD')

    def test_activity_future_source_bad_identity_and_nonfinite_values_are_sanitized(self):
        for change in (dict(id='PRIVATE_BAD_ID'),dict(price='NaN'),dict(transaction_time=(self.now+timedelta(seconds=1)).isoformat())):
            self.responses=[self.reply(payload=[self.activity(**change)])]
            with self.subTest(change=change),self.assertRaises(self.m.BrokerError) as caught:
                self.client().activities(after=self.now-timedelta(hours=1))
            self.assertNotIn('PRIVATE',str(caught.exception))

    def test_account_marks_are_broker_accounting_values_not_executable_quotes(self):
        row=dict(asset_class='us_equity',symbol='AAPL',side='long',qty='2',current_price='150',market_value='300',id='PRIVATE_ACCOUNT')
        self.responses=[self.reply(payload=[row])]
        result=self.client().account_marks()
        self.assertEqual(result[0].market_value_usd,Decimal(300));self.assertNotIn('PRIVATE',repr(result))

    def test_fee_timestamp_is_checked_before_reducing_it_to_a_date(self):
        row=self.activity(activity_type='FEE',date=(self.now+timedelta(hours=1)).isoformat(),net_amount='-1')
        self.responses=[self.reply(payload=[row])]
        with self.assertRaises(self.m.BrokerError): self.client().activities(after=self.now-timedelta(hours=1))
        row['date']=(self.now-timedelta(seconds=1)).isoformat();self.responses=[self.reply(payload=[row])]
        event=self.client().activities(after=self.now-timedelta(hours=1)).activities[0]
        self.assertEqual(event.source_time,self.now-timedelta(seconds=1))

    def ledger(self):
        from pathlib import Path
        import tempfile
        from paperbot_us.orders import OrderLedger
        from paperbot_us.data import Calendar,Session
        from paperbot_us.risk import Portfolio,Quote,RiskState,RiskLimits
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        ledger=OrderLedger(Path(temp.name)/'orders.sqlite');ledger.initialize()
        calendar=Calendar([Session.from_alpaca({'date':'2026-10-23','open':'09:30','close':'16:00'})],start='2026-10-23',end='2026-10-23')
        args=dict(portfolio=Portfolio(self.now,'10000','10000','10000','10000'),
            quote=Quote('AAPL','150','150.1',self.now,self.now),state=RiskState(pause_entries=False),
            limits=RiskLimits(),calendar=calendar,now=self.now)
        ledger.admit(self.intent,**args)
        from paperbot_us.controls import ControlJournal
        controls=ControlJournal(Path(temp.name)/'operator-controls.sqlite');controls.initialize(now=self.now)
        snapshot=controls.update(expected_revision=1,pause_entries=False,halt_orders=False,reason='operator_resume',now=self.now)
        args.update(controls=controls,control_snapshot=snapshot)
        return ledger,args

    def test_public_submit_claims_once_then_records_normalized_broker_result(self):
        ledger,args=self.ledger();self.responses=[self.reply(payload=self.order())]
        client=self.client()
        self.assertEqual(client.submit(ledger,self.intent.intent_id,**args)['state'],'open')
        self.assertIsNone(client.submit(ledger,self.intent.intent_id,**args))
        self.assertEqual(len(self.calls),1)
        self.assertEqual(ledger.verify()['events'],3)

    def test_operator_halt_after_durable_claim_vetoes_before_the_http_request(self):
        from paperbot_us.controls import ControlJournal
        ledger,args=self.ledger()
        controls=ControlJournal(ledger.path.with_name('test-controls.sqlite'));controls.initialize(now=self.now)
        snapshot=controls.update(expected_revision=1,pause_entries=False,halt_orders=False,
            reason='operator_resume',now=self.now)
        client=self.client();original=ledger._persist
        def persist(*values):
            value=original(*values)
            if values[2]=='submission_claim':
                controls.update(expected_revision=2,pause_entries=True,halt_orders=True,reason='operator_halt',now=self.now)
            return value
        with patch.object(ledger,'_persist',side_effect=persist):
            self.assertIsNone(client.submit(ledger,self.intent.intent_id,
                **{**args,'controls':controls,'control_snapshot':snapshot}))
        self.assertEqual(self.calls,[])
        saved=ledger.load(self.intent.intent_id)
        self.assertEqual(saved['state'],'rejected_by_risk')
        self.assertIn('operator_control_changed',saved['risk']['violations'])

    def test_slow_operator_read_cannot_leave_a_previously_fresh_risk_check_usable(self):
        from paperbot_us.controls import ControlJournal
        ledger,args=self.ledger()
        controls=ControlJournal(ledger.path.with_name('test-controls.sqlite'));controls.initialize(now=self.now)
        snapshot=controls.update(expected_revision=1,pause_entries=False,halt_orders=False,
            reason='operator_resume',now=self.now)
        client=self.client();original=controls._read
        def slow(*values,**kwargs):
            result=original(*values,**kwargs);self.now+=timedelta(seconds=31);return result
        with patch.object(controls,'_read',side_effect=slow):
            self.assertIsNone(client.submit(ledger,self.intent.intent_id,
                **{**args,'controls':controls,'control_snapshot':snapshot}))
        self.assertEqual(self.calls,[])
        self.assertIn('stale_quote',ledger.load(self.intent.intent_id)['risk']['violations'])

    def test_missing_controls_and_a_paused_snapshot_are_independent_final_vetoes(self):
        from paperbot_us.controls import ControlJournal
        for missing in (True,False):
            with self.subTest(missing=missing):
                ledger,args=self.ledger()
                controls=ControlJournal(ledger.path.with_name('test-controls.sqlite'));snapshot=controls.initialize(now=self.now)
                if missing: controls.path.unlink()
                self.assertIsNone(self.client().submit(ledger,self.intent.intent_id,
                    **{**args,'controls':controls,'control_snapshot':snapshot}))
                self.assertEqual(self.calls,[])
                self.assertEqual(ledger.load(self.intent.intent_id)['state'],'rejected_by_risk')

    def test_controls_are_mandatory_and_an_unexpected_guard_failure_is_a_durable_veto(self):
        ledger,args=self.ledger()
        bare={k:v for k,v in args.items() if k not in ('controls','control_snapshot')}
        with self.assertRaises(self.m.BrokerError):
            self.client().submit(ledger,self.intent.intent_id,**bare)
        self.assertEqual(ledger.load(self.intent.intent_id)['state'],'reserved')
        with patch.object(args['controls'],'_read',side_effect=OSError('PRIVATE_CONTROL_ERROR')):
            self.assertIsNone(self.client().submit(ledger,self.intent.intent_id,**args))
        self.assertEqual(self.calls,[])
        self.assertEqual(ledger.load(self.intent.intent_id)['state'],'rejected_by_risk')

    def test_clock_regression_before_send_is_a_durable_veto_with_the_actual_observed_time(self):
        ledger,args=self.ledger();client=self.client();original=args['controls']._read
        claim_at=self.now
        def regressed(*values,**kwargs):
            result=original(*values,**kwargs);self.now-=timedelta(seconds=1);return result
        with patch.object(args['controls'],'_read',side_effect=regressed):
            self.assertIsNone(client.submit(ledger,self.intent.intent_id,**args))
        self.assertEqual(self.calls,[])
        saved=ledger.load(self.intent.intent_id)
        self.assertEqual(saved['state'],'rejected_by_risk')
        self.assertIn('submission_clock_regression',saved['risk']['violations'])
        self.assertEqual(saved['updated_at'],claim_at.isoformat(timespec='microseconds'))
        self.assertEqual(saved['context']['observed_at'],self.now.isoformat(timespec='microseconds'))
        self.assertEqual(saved['context']['veto_recorded_at'],claim_at.isoformat(timespec='microseconds'))

    def test_dispatch_and_operator_commit_have_a_shared_serialization_boundary(self):
        ledger,args=self.ledger();controls=args['controls'];client=self.client()
        self.responses=[self.reply(payload=self.order())]
        entered,committed=threading.Event(),threading.Event();errors=[];writers=[]
        append=controls._append;transport=client._transport
        def write(*values):
            result=append(*values);entered.set();return result
        def halt():
            try:
                controls.update(expected_revision=2,pause_entries=True,halt_orders=True,reason='operator_halt',now=self.now)
                committed.set()
            except Exception as error: errors.append(type(error).__name__)
        def dispatch(*values):
            writer=threading.Thread(target=halt);writers.append(writer);writer.start()
            self.assertTrue(entered.wait(2))
            self.assertFalse(committed.wait(.05),'a new halt must not commit before a dispatch using the old snapshot')
            return transport(*values)
        try:
            with patch.object(controls,'_append',side_effect=write),patch.object(client,'_transport',side_effect=dispatch):
                result=client.submit(ledger,self.intent.intent_id,**args)
            self.assertEqual(result['state'],'open')
        finally:
            for writer in writers: writer.join(2)
        self.assertEqual(errors,[])
        self.assertTrue(committed.is_set())
        self.assertTrue(controls.read(now=self.now).halt_orders)
        self.assertEqual(len(self.calls),1)

    def test_public_submit_rechecks_at_current_clock_and_never_posts_stale_data(self):
        ledger,args=self.ledger();client=self.client()
        self.now+=timedelta(seconds=31)
        self.assertIsNone(client.submit(ledger,self.intent.intent_id,**args))
        self.assertEqual(self.calls,[])

    def test_public_submit_unknown_result_survives_retry_and_restart(self):
        ledger,args=self.ledger();self.responses=[OSError('PRIVATE_FIXTURE')];client=self.client()
        with self.assertRaises(self.m.BrokerError): client.submit(ledger,self.intent.intent_id,**args)
        self.assertEqual(ledger.load(self.intent.intent_id)['state'],'submission_unknown')
        self.assertEqual(len(ledger.active()),1)
        self.assertIsNone(client.submit(ledger,self.intent.intent_id,**args));self.assertEqual(len(self.calls),1)

    def test_public_cancel_is_durable_and_never_releases_on_ack_or_failure(self):
        for reply in (204,422):
            with self.subTest(reply=reply):
                ledger,args=self.ledger();self.responses=[self.reply(payload=self.order()),self.reply(reply)]
                client=self.client();client.submit(ledger,self.intent.intent_id,**args)
                if reply==204: self.assertEqual(client.cancel(ledger,self.intent.intent_id),'cancel_requested')
                else:
                    with self.assertRaises(self.m.BrokerError): client.cancel(ledger,self.intent.intent_id)
                record=ledger.load(self.intent.intent_id)
                self.assertIsNotNone(record['cancel_requested_at']);self.assertEqual(record['state'],'open')
                self.assertEqual(len(ledger.active()),1)
                self.assertIsNone(client.cancel(ledger,self.intent.intent_id))

    def test_persistence_latency_cannot_cross_close_or_quote_expiry_then_post(self):
        from dataclasses import replace
        from paperbot_us.risk import OrderIntent
        for delay,near_close in ((31,False),(2,True)):
            with self.subTest(delay=delay):
                self.now=datetime(2026,10,23,19,59,59,tzinfo=timezone.utc) if near_close else datetime(2026,10,23,17,tzinfo=timezone.utc)
                self.intent=replace(self.intent,expires_at=self.now+timedelta(seconds=60))
                ledger,args=self.ledger();client=self.client();self.responses=[self.reply(payload=self.order())]
                original=ledger._persist
                def persist(*values):
                    result=original(*values)
                    if values[2]=='submission_claim': self.now+=timedelta(seconds=delay)
                    return result
                with patch.object(ledger,'_persist',side_effect=persist):
                    self.assertIsNone(client.submit(ledger,self.intent.intent_id,**args))
                record=ledger.load(self.intent.intent_id)
                self.assertEqual(record['state'],'rejected_by_risk')
                self.assertIn('market_closed' if near_close else 'stale_quote',record['risk']['violations'])
                self.assertEqual(self.calls,[])


if __name__=='__main__': unittest.main()
