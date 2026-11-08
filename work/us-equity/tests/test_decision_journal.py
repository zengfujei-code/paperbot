from contextlib import closing
from dataclasses import replace
from datetime import datetime,timedelta,timezone
from decimal import Decimal
import hashlib,importlib,importlib.util,json,sqlite3,unittest
from unittest.mock import patch

import test_replay_bridge
from paperbot_us.controls import ControlJournal
from paperbot_us.execution_policy import ExecutionPolicy,PolicySnapshot
from paperbot_us.orders import OrderLedger,_json,_base
from paperbot_us.risk import Portfolio,Position,Quote,RiskState,RiskLimits,OrderIntent
from paperbot_us.risk_view import PreparedRiskView,_version as view_version


class DecisionJournalTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.decision_journal'),
                             'complete decision trace journal required')
        self.m=importlib.import_module('paperbot_us.decision_journal')
        self.f=test_replay_bridge.ReplayBridgeTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.root=self.f.root;self.now=self.f.f.now
        self.source=self.f.bridge.review(self.root/'model.sqlite',self.f.review_id,source_archive=self.root/'decisions.sqlite')
        quant=self.source.read()['quant']['record'];self.symbol=quant['symbol']
        self.calendar=self.f.f.fixture.calendar
        self.sources=self.m.ReplaySources(self.f.bridge,self.root/'decisions.sqlite',self.root/'model.sqlite')
        self.limits=RiskLimits();self.quote=Quote(self.symbol,'101.84','101.85',self.now,self.now)
        quotes=(self.quote,)
        self.view=PreparedRiskView(Portfolio(self.now,'10000','10000','10000','10000',valuation_quotes=quotes),
            self.quote,RiskState(pause_entries=False),(),'1'*64,'2'*64,hashlib.sha256(b'[]').hexdigest(),
            hashlib.sha256(_json([_base(self.quote,Quote)]).encode()).hexdigest(),view_version())
        self.controls=ControlJournal(self.root/'controls.sqlite');self.controls.initialize(now=self.now)
        self.control=self.controls.update(expected_revision=1,pause_entries=False,halt_orders=False,reason='operator_resume',now=self.now)
        limits_sha=hashlib.sha256(_json(_base(self.limits,RiskLimits)).encode()).hexdigest()
        self.policy=PolicySnapshot(ExecutionPolicy(1,10000,False,None,None,limits_sha,5,True),'3'*64)
        self.intent=OrderIntent('a'*64,self.symbol,'buy',5,'101.85','99',self.now+timedelta(seconds=20))
        self.path=self.root/'execution-decisions.sqlite';self.store=self.m.DecisionJournal(self.path,clock=lambda:self.now)
        self.store.initialize()

    def frame(self,**changes):
        return self.m.capture_decision(**{**dict(source=self.source,prepared=self.view,intent=self.intent,
            policy=self.policy,controls=self.control,limits=self.limits,calendar=self.calendar,now=self.now,
            requested_action='buy',authorization_sha256=None,basis='quant_model'),**changes})

    def append(self,frame=None): return self.store.append(frame or self.frame(),sources=self.sources)

    def test_complete_veto_captures_actual_inputs_risk_and_original_source_before_receipt(self):
        receipt=self.append();record=receipt.read()
        self.assertEqual(record['outcome']['action'],'wait')
        self.assertIn('execution_disabled',record['outcome']['vetoes'])
        self.assertIn('strategy_acceptance_unavailable',record['outcome']['vetoes'])
        self.assertEqual(record['inputs']['source']['model']['record']['receipt']['action'],'hold')
        self.assertTrue(record['outcome']['risk']['allowed'])
        self.assertEqual(record['inputs']['risk']['portfolio']['cash_usd'],'10000')
        self.assertEqual(record['inputs']['controls']['digest'],self.control.digest)
        self.assertEqual(record['inputs']['prepared']['book_digest'],'1'*64)
        restarted=self.m.DecisionJournal(self.path,clock=lambda:self.now)
        self.assertEqual(restarted.replay(receipt.sequence,sources=self.sources).read(),record)
        self.assertFalse(receipt.public_summary()['execution_enabled'])
        self.assertEqual(receipt.public_summary()['orders_submitted'],0)

    def test_same_complete_frame_is_idempotent_and_source_prefix_may_grow(self):
        frame=self.frame();first=self.append(frame)
        self.assertEqual(self.append(frame),first)
        from paperbot_us.monitor import ObservationArchive
        ObservationArchive(self.root/'decisions.sqlite').append(dict(kind='cycle',identity='b'*64,
            generated_at=self.now.timestamp(),session_day='2026-10-23',status='analyzing'))
        self.assertEqual(self.store.replay(first.sequence,sources=self.sources),first)
        self.assertEqual(self.store.verify()['decisions'],1)

    def test_missing_inputs_and_operator_pause_record_wait_without_authority(self):
        frame=self.frame(source=None,prepared=None,policy=None,controls=None)
        receipt=self.store.append(frame)
        self.assertEqual(receipt.read()['outcome']['action'],'wait')
        self.assertIn('risk_unavailable',receipt.read()['outcome']['vetoes'])
        paused=self.controls.update(expected_revision=2,pause_entries=True,halt_orders=False,reason='operator_pause',now=self.now)
        result=self.append(self.frame(controls=paused)).read()
        self.assertIn('entries_paused',result['outcome']['risk']['violations'])

    def test_old_quotes_or_historical_model_cannot_be_made_current_by_replay(self):
        stale=replace(self.view,quote=replace(self.quote,as_of=self.now-timedelta(seconds=31)))
        receipt=self.append(self.frame(prepared=stale))
        self.assertIn('stale_quote',receipt.read()['outcome']['risk']['violations'])
        self.now+=timedelta(seconds=1021)
        record=self.append(self.frame(now=self.now)).read()
        self.assertIn('model_input_stale',record['outcome']['vetoes'])

    def test_disk_failure_rolls_back_before_returning_any_committed_decision(self):
        with patch.object(self.store,'_persist',side_effect=OSError('PRIVATE_DISK')):
            with self.assertRaises(self.m.DecisionError) as caught:self.append()
        self.assertNotIn('PRIVATE',str(caught.exception))
        self.assertEqual(self.store.verify()['events'],0)
        self.assertEqual(self.append().sequence,1)

    def test_corruption_and_current_code_mismatch_are_not_accepted_as_replay(self):
        first=self.append()
        with patch.object(self.m,'_version',return_value='f'*64):
            with self.assertRaises(self.m.DecisionError):self.store.replay(first.sequence,sources=self.sources)
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("UPDATE events SET digest='broken' WHERE sequence=1");db.commit()
        with self.assertRaises(self.m.DecisionError):self.store.verify()

    def test_clock_rollback_is_a_durable_veto_with_separate_observed_and_audit_time(self):
        first=self.append();self.now-=timedelta(seconds=1)
        second=self.append(self.frame(now=self.now))
        record=second.read()
        self.assertEqual(record['recorded_at'],first.read()['recorded_at'])
        self.assertLess(record['inputs']['observed_at'],record['recorded_at'])
        self.assertIn('decision_clock_regression',record['outcome']['vetoes'])

    def test_order_progress_is_linked_without_mutating_or_releasing_order_state(self):
        decision=self.append()
        ledger=OrderLedger(self.root/'orders.sqlite');ledger.initialize()
        args=dict(portfolio=self.view.portfolio,quote=self.quote,state=self.view.state,limits=self.limits,
            calendar=self.calendar,now=self.now)
        ledger.admit(self.intent,**args);ledger.prepare_submission(self.intent.intent_id,**args)
        before=ledger.all_orders()
        linked=self.store.observe_order(decision.sequence,ledger=ledger)
        self.assertEqual(ledger.all_orders(),before)
        record=linked.read()
        self.assertEqual(record['kind'],'order_observation')
        self.assertEqual(record['order']['state'],'submission_unknown')
        self.assertTrue(record['unexpected_after_veto'])
        self.assertIn('pause_entries',record['required_actions'])
        self.assertEqual(self.store.verify()['events'],2)

    def test_account_performance_uses_the_same_book_anchor_and_preserves_the_source_receipt(self):
        self.assertTrue(hasattr(self.store,'observe_account'),'subsequent account performance must link to the decision')
        from paperbot_us.accountbook import AccountBook
        from paperbot_us.accounting import AccountAnchor,ActivityWindow,AccountSnapshot
        anchor=AccountAnchor(self.now-timedelta(seconds=1),'10000','a'*64,'b'*64)
        book=AccountBook(self.root/'book.sqlite');book.initialize(anchor)
        first=book.refresh(window=ActivityWindow(anchor.as_of,self.now,self.now,()),orders=(),
            account=AccountSnapshot(self.now,'10000','10000',False,(),anchor.account_sha256),now=self.now)
        self.view=replace(self.view,book_digest=first.digest)
        decision=self.append()
        self.now+=timedelta(seconds=1)
        second=book.refresh(window=ActivityWindow(anchor.as_of,self.now,self.now,()),orders=(),
            account=AccountSnapshot(self.now,'10000','10000',False,(),anchor.account_sha256),now=self.now)
        before=book.verify()
        linked=self.store.observe_account(decision.sequence,book=book)
        self.assertEqual(book.verify(),before)
        record=linked.read()
        self.assertEqual(record['kind'],'account_observation')
        self.assertEqual(record['source_book_digest'],first.digest)
        self.assertEqual(record['book_digest'],second.digest)
        self.assertTrue(record['account']['accounting_ok'])
        self.assertEqual(record['account']['profit_usd'],'0')
        self.assertEqual(self.store.observe_account(decision.sequence,book=book),linked)
        foreign=AccountBook(self.root/'foreign.sqlite')
        foreign.initialize(replace(anchor,account_sha256='c'*64))
        foreign.refresh(window=ActivityWindow(anchor.as_of,self.now,self.now,()),orders=(),
            account=AccountSnapshot(self.now,'10000','10000',False,(),'c'*64),now=self.now)
        with self.assertRaises(self.m.DecisionError):self.store.observe_account(decision.sequence,book=foreign)

    def test_missing_calendar_coverage_is_recorded_as_a_veto_instead_of_losing_the_decision(self):
        self.now+=timedelta(days=1)
        receipt=self.append(self.frame(now=self.now))
        self.assertEqual(receipt.read()['outcome']['action'],'wait')
        self.assertIn('calendar_unavailable',receipt.read()['outcome']['vetoes'])

    def buy_proof(self):
        class Client:
            def assess(self,request):
                return dict(action='buy',reason='量价满足候选条件，模拟验证。',model='deepseek-flash',fingerprint='fp_test',
                    prompt_tokens=200,completion_tokens=30,total_tokens=230)
        archive=self.root/'buy-model.sqlite'
        self.f.f.m.assistance_once(client=Client(),source_archive=self.root/'decisions.sqlite',archive=archive,
            publish=self.root/'buy-model.json',clock=lambda:self.now)
        proof=self.f.bridge.latest_review(archive,source_archive=self.root/'decisions.sqlite')
        self.sources=self.m.ReplaySources(self.f.bridge,self.root/'decisions.sqlite',archive)
        active=replace(self.policy.policy,execution_enabled=True,strategy_sha256=self.m.strategy_digest(proof),acceptance_sha256='c'*64)
        return proof,replace(self.policy,policy=active)

    def test_recorded_candidate_replays_but_audit_receipt_grants_no_order_authority(self):
        source,policy=self.buy_proof()
        receipt=self.append(self.frame(source=source,policy=policy,authorization_sha256='c'*64))
        self.assertEqual(receipt.read()['outcome']['action'],'buy')
        self.assertEqual(receipt.read()['outcome']['vetoes'],[])
        self.assertTrue(receipt.read()['outcome']['authorization_is_reference_only'])
        self.assertFalse(receipt.public_summary()['execution_enabled'])
        self.assertEqual(receipt.public_summary()['orders_submitted'],0)
        self.assertEqual(self.store.replay(receipt.sequence,sources=self.sources),receipt)

    def test_quote_binding_mismatch_vetoes_an_otherwise_valid_candidate(self):
        source,policy=self.buy_proof()
        changed=replace(self.view,quotes_digest='f'*64)
        receipt=self.append(self.frame(source=source,policy=policy,prepared=changed,authorization_sha256='c'*64))
        self.assertEqual(receipt.read()['outcome']['action'],'wait')
        self.assertIn('quote_binding_mismatch',receipt.read()['outcome']['vetoes'])

    def test_reducing_exit_survives_entry_pause_without_model_but_not_global_halt(self):
        held=replace(self.view.portfolio,cash_usd='9490.75',positions=(Position(self.symbol,5,'101.85',self.now),))
        view=replace(self.view,portfolio=held)
        intent=replace(self.intent,side='sell',limit_price_usd='101.84',stop_price_usd=None)
        active=replace(self.policy,policy=replace(self.policy.policy,execution_enabled=True,strategy_sha256='b'*64,acceptance_sha256='c'*64))
        paused=self.controls.update(expected_revision=2,pause_entries=True,halt_orders=False,reason='operator_pause',now=self.now)
        args=dict(source=None,prepared=view,intent=intent,policy=active,controls=paused,requested_action='sell',basis='risk_reduction')
        exit_receipt=self.append(self.frame(**args))
        self.assertEqual(exit_receipt.read()['outcome']['action'],'sell')
        halted=self.controls.update(expected_revision=3,pause_entries=True,halt_orders=True,reason='operator_halt',now=self.now)
        blocked=self.append(self.frame(**{**args,'controls':halted}))
        self.assertEqual(blocked.read()['outcome']['action'],'hold')
        self.assertIn('all_orders_halted',blocked.read()['outcome']['risk']['violations'])

    def test_source_head_fork_cannot_be_disguised_as_a_longer_valid_chain(self):
        from paperbot_us.monitor import ObservationArchive
        archive=ObservationArchive(self.root/'decisions.sqlite')
        archive.append(dict(kind='cycle',identity='c'*64,generated_at=self.now.timestamp(),
            session_day='2026-10-23',status='analyzing'))
        self.source=self.f.bridge.review(self.root/'model.sqlite',self.f.review_id,source_archive=archive.path)
        receipt=self.append()
        with closing(sqlite3.connect(archive.path)) as db:
            sequence,payload,previous=db.execute('SELECT seq,payload,previous_digest FROM observations ORDER BY seq DESC LIMIT 1').fetchone()
            row=json.loads(payload);row['status']='market_closed';payload=_json(row)
            digest=hashlib.sha256((previous+'\0'+payload).encode()).hexdigest()
            db.execute('UPDATE observations SET payload=?,digest=? WHERE seq=?',(payload,digest,sequence))
            db.execute('UPDATE metadata SET head_digest=?',(digest,));db.commit()
        archive.append(dict(kind='cycle',identity='d'*64,generated_at=self.now.timestamp(),
            session_day='2026-10-23',status='analyzing'))
        with self.assertRaises(self.m.DecisionError):self.store.replay(receipt.sequence,sources=self.sources)

    def test_unknown_record_fields_are_rejected_even_when_the_hash_chain_matches(self):
        self.append()
        with closing(sqlite3.connect(self.path)) as db:
            row=db.execute('SELECT payload,previous_digest FROM events WHERE sequence=1').fetchone()
            record=json.loads(row[0]);record['secret']='PRIVATE_SYNTHETIC_MARKER';payload=_json(record)
            digest=hashlib.sha256((row[1]+'\0'+payload).encode()).hexdigest()
            db.execute('UPDATE events SET payload=?,digest=? WHERE sequence=1',(payload,digest));db.commit()
        with self.assertRaises(self.m.DecisionError) as caught:self.store.verify()
        self.assertNotIn('PRIVATE',str(caught.exception))
        with self.assertRaises(self.m.DecisionError):self.store.replay(1,sources=self.sources)


if __name__=='__main__':unittest.main()
