"""Final test report contracts/storage; arithmetic alone grants no authority."""
from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[3]),str(Path(__file__).resolve().parents[1])]
from test_common_scores import plan,fits,frame
from test_training_data import Tick,TZ
from paperbot_r2 import test_protocol as protocol,evaluation_book,validation_scores
from paperbot_r2.contracts import aware,digest
from paperbot_r2.model_training import FAMILIES
try:
    from paperbot_r2 import test_scores as scores
except ImportError:
    scores=None

FINAL=aware('2026-12-11T21:02:01+00:00')


class TestScoreContractTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(scores,'test score journal not implemented')
    def report(self,*,empty=False,infinite=False):
        p=plan();probabilities={f:(0. if infinite else .9 if f=='prevalence' else .1) for f in FAMILIES}
        values=() if empty else tuple(frame(s,probabilities=probabilities,label=1 if infinite else 0) for s in protocol.planned_slots(p))
        return protocol.evaluate_test_frames(p,values,as_of=FINAL,fit_statuses=fits(),selected_family='prevalence')

    def test_final_report_keeps_validation_selection_even_when_test_prefers_another(self):
        with patch('sqlite3.connect',side_effect=AssertionError('arithmetic only')):
            report=self.report();scores.validate_report(report,'prevalence')
        self.assertEqual(report['selected_family'],'prevalence');self.assertIsNone(report['proposed_family'])
        self.assertGreater(report['macro']['prevalence']['brier'],report['macro']['ewma']['brier'])
        self.assertEqual(sum(g['counts']['planned'] for g in report['groups']),240)
        self.assertEqual(len(report['training_fits']),15);self.assertEqual(report['test_sessions'],5)

    def test_missing_counts_and_failed_coverage_remain_a_valid_inconclusive_report(self):
        report=self.report(empty=True);scores.validate_report(report,'prevalence')
        self.assertFalse(report['engineering_gate_passed'])
        self.assertEqual(sum(g['counts']['unissued'] for g in report['groups']),240)
        self.assertIsNone(report['macro']['prevalence']['brier'])

    def test_infinite_log_loss_keeps_declared_json_tag(self):
        report=self.report(infinite=True);scores.validate_report(report,'prevalence')
        self.assertEqual(report['macro']['prevalence']['log_loss'],{'loss':'positive_infinity'})
        self.assertNotIn('Infinity',json.dumps(report,allow_nan=False))

    def test_changed_selection_phase_flags_grid_fit_count_or_math_source_rejected(self):
        original=self.report();bads=[]
        for name,value in (('selected_family','ewma'),('phase','validation'),('complete',False),('math_only',False),
            ('orders',True),('promotion',True),('test_admission',True),('proposed_family','ewma'),
            ('source_sha256','0'*64),('evidence_class','trading_accepted')):
            bad=deepcopy(original);bad[name]=value;bads.append(bad)
        bad=deepcopy(original);bad['groups'].pop();bads.append(bad)
        bad=deepcopy(original);bad['training_fits'].pop();bads.append(bad)
        bad=deepcopy(original);bad['groups'][0]['counts']['planned']=15;bads.append(bad)
        for report in bads:
            with self.assertRaises(scores.TestScoreError):scores.validate_report(report,'prevalence')

    def test_freeze_load_and_initialization_phase_gates_precede_sql_and_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);book=scores.TestScoreJournal(root/'scores.sqlite')
            for at in ('2026-11-12T17:00:00+00:00','2026-12-07T21:02:01+00:00','2026-12-11T21:02:00+00:00'):
                with patch('sqlite3.connect',side_effect=AssertionError('phase before SQL')) as connect:
                    for method in (book.freeze_once,book.load_frozen):
                        with self.assertRaises(scores.TestScoreError):method(timezone_file=TZ,clock=lambda:aware(at))
                    connect.assert_not_called()
            with patch('sqlite3.connect',side_effect=AssertionError('initialization before SQL')) as connect:
                with self.assertRaises(scores.TestScoreError):
                    book.initialize(None,timezone_file=TZ,clock=lambda:aware('2026-11-24T14:30:00+00:00'))
                connect.assert_not_called()
            self.assertEqual(list(root.iterdir()),[])


class TestScoreStorageTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(scores,'test score journal not implemented')
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        class StorageOnly(scores.TestScoreJournal):
            def source_sha256(self):return hashlib.sha256(b'explicit final test storage fixture; no phase authority').hexdigest()
        self.book=StorageOnly(self.root/'scores.sqlite')
        policy=validation_scores.make_policy(protocol.source_sha256())
        self.binding=dict(policy=asdict(policy),registration_identity='a'*64)
        self.clock=Tick(aware('2026-11-24T13:00:00+00:00'))
        self.book.create(self.binding,clock=self.clock,check=lambda:None);self.clock=Tick(FINAL+timedelta(seconds=1))
        p=plan();report=protocol.evaluate_test_frames(p,tuple(frame(s) for s in protocol.planned_slots(p)),
            as_of=FINAL,fit_statuses=fits(),selected_family='prevalence')
        # Deliberately non-authoritative storage projection; no original books.
        self.body=dict(report=report,basis=dict(source_basis=dict(policy_seal=policy.seal,registration_identity='a'*64),
            selection_reference=dict(selected_family='prevalence',selection_head_digest='b'*64,
                selection_commit_observed_at='2026-12-04T21:03:00+00:00',validation_report_sha256='c'*64),
            protocol_sha256=protocol.source_sha256()))

    def append(self,kind,body):
        with scores._exclusive(self.book.path):return self.book.append(self.book.snapshot(),kind,body,clock=self.clock,check=lambda:None)

    def test_interrupted_evaluation_commit_cannot_replace_report_and_only_ack_recovers(self):
        commit=self.book._commit
        def interrupted(db,kind):
            commit(db,kind)
            if kind=='evaluated':raise RuntimeError('synthetic interrupt after durable comparison')
        with patch.object(self.book,'_commit',side_effect=interrupted):
            with self.assertRaises(evaluation_book.EvaluationBookError):self.append('evaluated',self.body)
        first=self.book.snapshot();self.assertEqual(first['head_seq'],3)
        with self.assertRaises(evaluation_book.EvaluationBookError):self.append('evaluated',self.body)
        with patch.object(protocol,'evaluate_test_frames',side_effect=AssertionError('never reevaluate first comparison')):
            state=self.append('confirmed',dict(evaluated_digest=first['domain']['evaluated_digest']))
        self.assertEqual(state['head_seq'],4);self.assertEqual(state['domain']['report'],self.body['report'])
        self.assertEqual(state['domain']['report']['selected_family'],'prevalence')
        with self.assertRaises(scores.TestScoreError):self.book.load_frozen(timezone_file=TZ,clock=self.clock)

    def test_confirmation_cannot_precede_evaluation_or_repeat(self):
        with self.assertRaises(evaluation_book.EvaluationBookError):self.append('confirmed',dict(evaluated_digest='f'*64))
        state=self.append('evaluated',self.body)
        body=dict(evaluated_digest=state['domain']['evaluated_digest'])
        self.append('confirmed',body)
        with self.assertRaises(evaluation_book.EvaluationBookError):self.append('confirmed',body)

    def test_copied_report_book_cannot_become_another_first_history(self):
        self.append('evaluated',self.body);other=self.root/'copy.sqlite';shutil.copy2(self.book.path,other)
        with self.assertRaises(evaluation_book.EvaluationBookError):type(self.book)(other).snapshot()


if __name__=='__main__':unittest.main()
