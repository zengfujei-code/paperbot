from contextlib import closing
from datetime import timedelta
import importlib
import importlib.util
import json
import sqlite3
import unittest
from unittest.mock import patch
import test_shadow


class ModelRunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.model_runner'),'durable model coordinator required')
        self.m=importlib.import_module('paperbot_us.model_runner')
        self.fixture=test_shadow.ShadowTests('test_repeated_bar_is_deduplicated_and_preserves_first_decision_time')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.fixture.run_once();self.root=self.fixture.root;self.now=self.fixture.now;owner=self
        class Client:
            calls=[]
            def assess(self,request):
                self.calls.append(request)
                return dict(action='hold',reason='趋势偏强，等待量价确认。',model='deepseek-flash',fingerprint='fp_test',
                    prompt_tokens=200,completion_tokens=30,total_tokens=230)
        self.client=Client()

    def run_once(self,historical=False):
        return self.m.assistance_once(client=self.client,source_archive=self.root/'decisions.sqlite',
            archive=self.root/'model.sqlite',publish=self.root/'model.json',clock=lambda:self.now,historical=historical)

    def records(self):
        with closing(sqlite3.connect(self.root/'model.sqlite')) as db:
            return [json.loads(r[0]) for r in db.execute('SELECT payload FROM observations')]

    def test_actual_assessment_is_durable_replayable_and_cannot_enable_orders(self):
        result=self.run_once();self.assertEqual(result['status'],'reviewed');self.assertEqual(len(self.client.calls),1)
        self.assertFalse(result['execution_enabled']);self.assertEqual(result['decisions'][0]['final_action'],'hold')
        self.assertIn('execution_not_enabled',result['decisions'][0]['vetoes'])
        records=self.records();self.assertEqual([r['kind'] for r in records],['model_attempt','model_review'])
        source=self.m.source_decision(self.root/'decisions.sqlite',records[1]['source_identity'])
        self.assertEqual(self.m.replay_review(records[1],source)['reason'],result['decisions'][0]['reason'])
        records[1]['request']['max_tokens']=500
        with self.assertRaises(ValueError): self.m.replay_review(records[1],source)

    def test_minimum_interval_and_restart_deduplicate_billed_attempts(self):
        first=self.run_once();self.now+=timedelta(seconds=10);second=self.run_once()
        self.assertEqual(len(self.client.calls),1);self.assertEqual(second['status'],'rate_limited')
        self.now+=timedelta(seconds=51);self.run_once()
        self.assertEqual(len(self.client.calls),2)
        self.assertNotEqual(self.client.calls[0]['messages'][1]['content'],self.client.calls[1]['messages'][1]['content'])
        attempts=[r['identity'] for r in self.records() if r['kind']=='model_attempt']
        self.assertEqual(len(attempts),len(set(attempts)))
        self.assertEqual(first['decisions'][0]['reviewed_at'],second['decisions'][0]['reviewed_at'])
        self.assertEqual(first['audit_sha256'],second['audit_sha256'])

    def test_audit_failure_before_network_blocks_billing_and_clears_projection(self):
        with patch.object(self.m.ObservationArchive,'append',side_effect=OSError('PRIVATE_DISK')):
            result=self.run_once()
        self.assertEqual(self.client.calls,[]);self.assertEqual(result['status'],'unavailable')
        self.assertEqual(result['decisions'],[]);self.assertNotIn('PRIVATE',json.dumps(result))

    def test_unknown_or_invalid_response_consumes_attempt_and_never_retries_same_bar(self):
        def fail(request): self.client.calls.append(request);raise OSError('PRIVATE_ERROR')
        self.client.assess=fail
        result=self.run_once();self.assertEqual(result['status'],'model_unavailable')
        self.assertEqual(result['decisions'],[]);self.assertEqual(len(self.client.calls),1)
        self.run_once();self.assertEqual(len(self.client.calls),1)
        self.assertEqual(self.records()[-1]['kind'],'model_failure')
        self.assertNotIn('PRIVATE',json.dumps(self.records()))

    def test_market_close_does_not_call_model_unless_historical_explicit(self):
        self.now=self.now.replace(hour=21)
        result=self.run_once();self.assertEqual(result['status'],'market_closed');self.assertEqual(self.client.calls,[])
        result=self.run_once(historical=True);self.assertEqual(result['status'],'reviewed')
        self.assertEqual(result['decisions'][0]['context'],'historical_review')
        self.assertFalse(result['decisions'][0]['signal_usable'])
        self.now+=timedelta(days=2);self.assertEqual(self.run_once(historical=True)['status'],'input_unavailable')

    def test_stale_source_cannot_be_relabelled_live_by_fresh_heartbeat(self):
        self.now+=timedelta(seconds=1021)
        result=self.run_once();self.assertEqual(result['status'],'input_unavailable');self.assertEqual(self.client.calls,[])

    def test_previous_session_cannot_claim_current_session_is_closed(self):
        self.now+=timedelta(days=3)
        result=self.run_once();self.assertEqual(result['status'],'input_unavailable');self.assertEqual(self.client.calls,[])

    def test_daily_budget_counts_committed_attempts_even_without_response(self):
        journal=self.m.ObservationArchive(self.root/'model.sqlite')
        for n in range(60):
            journal.append(dict(kind='model_attempt',identity=f'{n:064x}',generated_at=self.now.timestamp()-7200+n))
        result=self.run_once();self.assertEqual(result['status'],'budget_exhausted');self.assertEqual(self.client.calls,[])
        self.assertEqual(result['attempts_today'],60)

    def test_slow_commit_cannot_shorten_actual_call_cooldown(self):
        original=self.m.ObservationArchive.append
        def delayed(journal,record):
            value=original(journal,record)
            if record['kind']=='model_attempt': self.now+=timedelta(seconds=20)
            return value
        with patch.object(self.m.ObservationArchive,'append',delayed): first=self.run_once()
        self.assertEqual(first['status'],'reviewed');self.now+=timedelta(seconds=41)
        self.assertEqual(self.run_once()['status'],'rate_limited');self.assertEqual(len(self.client.calls),1)

    def test_committed_attempt_crossing_close_or_input_expiry_is_not_sent(self):
        original=self.m.ObservationArchive.append
        def delayed(journal,record):
            value=original(journal,record)
            if record['kind']=='model_attempt': self.now=self.now.replace(hour=20)
            return value
        with patch.object(self.m.ObservationArchive,'append',delayed): result=self.run_once()
        self.assertEqual(self.client.calls,[]);self.assertEqual(result['status'],'input_unavailable')
        self.assertEqual(result['attempts_today'],1)

    def test_response_expiry_is_disclosed_and_never_creates_an_executable_signal(self):
        original=self.client.assess
        def delayed(request):
            value=original(request);self.now+=timedelta(seconds=1021);return value
        self.client.assess=delayed;result=self.run_once()
        self.assertEqual(result['status'],'reviewed');row=result['decisions'][0]
        self.assertFalse(row['input_current_at_review']);self.assertFalse(row['signal_usable'])

    def test_model_cannot_promote_non_candidate_and_corrupt_source_is_rejected(self):
        real=self.client.assess
        self.client.assess=lambda request:{**real(request),'action':'sell'}
        result=self.run_once();row=result['decisions'][0]
        self.assertEqual(row['final_action'],'hold')
        # Which direction is a candidate is taken only from replayed quant inputs.
        data=json.loads(self.client.calls[0]['messages'][1]['content'])
        if not data['exit_candidate']: self.assertIn('model_outside_candidate',row['vetoes'])
        with closing(sqlite3.connect(self.root/'decisions.sqlite')) as db:
            db.execute("UPDATE observations SET digest='bad' WHERE seq=1");db.commit()
        self.assertEqual(self.run_once()['status'],'unavailable')


if __name__=='__main__': unittest.main()
