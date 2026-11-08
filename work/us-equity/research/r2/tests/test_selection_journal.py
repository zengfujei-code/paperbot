"""Selection arithmetic/reducer/gates; actual-source admission follows full protocol."""
from copy import deepcopy
from datetime import timedelta
import hashlib
from pathlib import Path
import shutil
import socket
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[3]),str(Path(__file__).resolve().parents[1])]
from test_common_scores import plan,fits,frame
from paperbot_r2 import common_scores as scores,test_protocol as protocol,validation_scores
from paperbot_r2 import evaluation_book
from paperbot_r2.contracts import aware,digest
from paperbot_r2.model_training import FAMILIES
try:
    from paperbot_r2 import selection_journal as selection
except ImportError:
    selection=None

TZ=Path(__file__).resolve().parents[5]/'outputs/zoneinfo/America/New_York'
AFTER=aware('2026-12-04T21:02:01+00:00')


class SelectionContractTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(selection,'one-time selection journal not implemented')
        self.plan=plan();self.slots=scores.planned_slots(self.plan)
        for target in ('sqlite3.connect','socket.socket'):
            guard=patch(target,side_effect=AssertionError('contract tests forbid SQL/network'))
            guard.start();self.addCleanup(guard.stop)
    def report(self,values=None):
        probabilities=values or {f:(.2 if f=='garch' else .8) for f in FAMILIES}
        return scores.evaluate_frames(self.plan,tuple(frame(s,probabilities=probabilities) for s in self.slots),
                                      as_of=AFTER,fit_statuses=fits())
    def test_predeclared_minimum_macro_and_original_fit_trial_count(self):
        report=self.report();decision=selection.project_report(report)
        self.assertEqual(decision['selected_family'],'garch')
        self.assertTrue(decision['engineering_gate_passed']);self.assertEqual(decision['report_sha256'],digest(report))
        self.assertEqual(len(decision['training_fits']),15);self.assertEqual(len(decision['coverage']),24)
        self.assertEqual(decision['reason'],'minimum_macro_brier')
    def test_exact_tie_uses_frozen_baseline_order(self):
        decision=selection.project_report(self.report({f:.2 for f in FAMILIES}))
        self.assertEqual(decision['selected_family'],'prevalence')
    def test_failed_coverage_is_terminal_no_admission_choice(self):
        report=scores.evaluate_frames(self.plan,(),as_of=AFTER,fit_statuses=fits())
        decision=selection.project_report(report)
        self.assertFalse(decision['engineering_gate_passed']);self.assertIsNone(decision['selected_family'])
        self.assertEqual(decision['reason'],'engineering_gate_failed')
    def test_altered_proposal_incomplete_report_and_changed_source_rejected(self):
        original=self.report()
        for key,value in (('proposed_family','ewma'),('complete',False),('math_only',False),
                          ('phase','test'),('source_sha256','0'*64),('orders',True),('test_admission',True)):
            bad=deepcopy(original);bad[key]=value
            with self.assertRaises(selection.SelectionError):selection.project_report(bad)
    def test_omitted_group_fit_or_invalid_macro_rejected(self):
        original=self.report()
        bads=[]
        bad=deepcopy(original);bad['groups'].pop();bads.append(bad)
        bad=deepcopy(original);bad['training_fits'].pop();bads.append(bad)
        bad=deepcopy(original);bad['macro']['garch']['brier']=True;bads.append(bad)
        bad=deepcopy(original);bad['macro']['garch']['brier']=None;bads.append(bad)
        bad=deepcopy(original);bad['groups'][0]['counts']['planned']=15;bads.append(bad)
        for bad in bads:
            with self.assertRaises(selection.SelectionError):selection.project_report(bad)
    def test_reducer_one_decision_and_separate_commit_observation(self):
        book=selection.SelectionJournal(Path('not-created.sqlite'));domain=book.domain_initial()
        decision=selection.project_report(self.report())
        body=dict(decision=decision,score_book=dict(path=str(Path('score.sqlite').absolute()),device=1,inode=2,head_seq=4,head_digest='a'*64),
                  score_commit_observed_at=AFTER+timedelta(seconds=1),score_available_at=AFTER+timedelta(seconds=2))
        at=AFTER+timedelta(seconds=3)
        book.domain_apply(domain,'decided',body,at,'b'*64,{})
        self.assertIsNone(domain['confirmed_at'])
        with self.assertRaises(selection.SelectionError):book.domain_apply(domain,'decided',body,at,'c'*64,{})
        with self.assertRaises(selection.SelectionError):book.domain_apply(domain,'confirmed',dict(decision_digest='b'*64),at,'c'*64,{})
        book.domain_apply(domain,'confirmed',dict(decision_digest='b'*64),at+timedelta(seconds=1),'c'*64,{})
        self.assertEqual(domain['decision']['selected_family'],'garch')
        with self.assertRaises(selection.SelectionError):
            book.domain_apply(domain,'confirmed',dict(decision_digest='b'*64),at+timedelta(seconds=2),'d'*64,{})
    def test_decision_or_confirmation_after_test_open_rejected(self):
        book=selection.SelectionJournal(Path('not-created.sqlite'));domain=book.domain_initial()
        decision=selection.project_report(self.report())
        body=dict(decision=decision,score_book=dict(path=str(Path('score.sqlite').absolute()),device=1,inode=2,head_seq=4,head_digest='a'*64),
                  score_commit_observed_at=AFTER+timedelta(seconds=1),score_available_at=AFTER+timedelta(seconds=2))
        with self.assertRaises(selection.SelectionError):book.domain_apply(domain,'decided',body,protocol.TEST_OPEN,'b'*64,{})
        book.domain_apply(domain,'decided',body,AFTER+timedelta(seconds=3),'b'*64,{})
        with self.assertRaises(selection.SelectionError):
            book.domain_apply(domain,'confirmed',dict(decision_digest='b'*64),protocol.TEST_OPEN,'c'*64,{})
    def test_phase_rejection_before_sql_or_lock_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'selection.sqlite';book=selection.SelectionJournal(path)
            for at in (aware('2026-11-12T16:00:00+00:00'),AFTER-timedelta(seconds=1),protocol.TEST_OPEN):
                with self.assertRaises(selection.SelectionError):book.select_once(timezone_file=TZ,clock=lambda:at)
            with self.assertRaises(selection.SelectionError):book.load_current(timezone_file=TZ,clock=lambda:AFTER-timedelta(seconds=1))
            with self.assertRaises(selection.SelectionError):
                book.initialize(validation_scores.ValidationScoreJournal(Path(temp)/'scores.sqlite'),timezone_file=TZ,
                                clock=lambda:aware('2026-11-24T14:30:00+00:00'))
            self.assertEqual(list(Path(temp).iterdir()),[])
    def test_incomplete_protocol_cannot_initialize_source_books(self):
        if all(Path(protocol.__file__).with_name(n).exists() for n in protocol.PROTOCOL_FILES):
            return  # Full-protocol original-source integration owns this case later.
        with tempfile.TemporaryDirectory() as temp:
            book=selection.SelectionJournal(Path(temp)/'selection.sqlite')
            with self.assertRaises(selection.SelectionError):
                book.initialize(validation_scores.ValidationScoreJournal(Path(temp)/'scores.sqlite'),timezone_file=TZ,
                                clock=lambda:aware('2026-11-24T13:00:00+00:00'))
            self.assertEqual(list(Path(temp).iterdir()),[])


class SelectionStorageTests(unittest.TestCase):
    """Real domain/storage only; the synthetic source grants no admission."""
    def setUp(self):
        self.assertIsNotNone(selection)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        class StorageOnly(selection.SelectionJournal):
            def source_sha256(self):return hashlib.sha256(b'explicit synthetic storage fixture; no admission').hexdigest()
        self.book=StorageOnly(self.root/'selection.sqlite')
        self.at=aware('2026-11-24T13:00:00+00:00')
        def tick():
            self.at+=timedelta(microseconds=1);return self.at
        self.clock=tick
        self.book.create(dict(fixture='storage_only_no_authority'),clock=self.clock,check=lambda:None)
        p=plan();report=scores.evaluate_frames(p,tuple(frame(s) for s in scores.planned_slots(p)),as_of=AFTER,fit_statuses=fits())
        self.body=dict(decision=selection.project_report(report),
            score_book=dict(path=str((self.root/'not-a-source-score.sqlite').absolute()),device=1,inode=2,head_seq=4,head_digest='a'*64),
            score_commit_observed_at=AFTER+timedelta(seconds=1),score_available_at=AFTER+timedelta(seconds=2))
        self.at=AFTER+timedelta(seconds=3)
    def append(self,kind,body,clock=None):
        with selection._exclusive(self.book.path):
            return self.book.append(self.book.snapshot(),kind,body,clock=clock or self.clock,check=lambda:None)
    def test_committed_decision_survives_interruption_and_cannot_be_replaced(self):
        original=self.book._commit
        def interrupted(db,kind):
            original(db,kind)
            if kind=='decided':raise RuntimeError('synthetic interruption after actual COMMIT')
        with patch.object(self.book,'_commit',side_effect=interrupted):
            with self.assertRaises(evaluation_book.EvaluationBookError):self.append('decided',self.body)
        state=self.book.snapshot();self.assertEqual(state['head_seq'],3)
        self.assertIsNone(state['domain']['confirmed_at'])
        with self.assertRaises(evaluation_book.EvaluationBookError):self.append('decided',self.body)
        confirmed=self.append('confirmed',dict(decision_digest=state['domain']['decision_digest']))
        self.assertEqual(confirmed['head_seq'],4)
        self.assertGreater(confirmed['domain']['confirmed_at'],confirmed['domain']['decided_at'])
        # An actual storage confirmation alone cannot manufacture source authority.
        with self.assertRaises(selection.SelectionError):self.book.load_current(timezone_file=TZ,clock=self.clock)
    def test_result_commit_crossing_open_remains_unconfirmed(self):
        self.at=protocol.TEST_OPEN-timedelta(seconds=1);original=self.book._commit
        def crossed(db,kind):
            original(db,kind)
            if kind=='decided':self.at=protocol.TEST_OPEN
        clock=protocol.SelectionClock(TZ,self.clock)
        with patch.object(self.book,'_commit',side_effect=crossed):
            with self.assertRaises(evaluation_book.EvaluationBookError):self.append('decided',self.body,clock)
        state=self.book.snapshot();self.assertEqual(state['head_seq'],3)
        with self.assertRaises(evaluation_book.EvaluationBookError):
            self.append('confirmed',dict(decision_digest=state['domain']['decision_digest']),clock)
        self.assertEqual(self.book.snapshot()['head_seq'],3)
        self.assertIsNone(self.book.snapshot()['domain']['confirmed_at'])
    def test_copied_selection_is_not_another_first_history(self):
        other=self.root/'copied.sqlite';shutil.copy2(self.book.path,other)
        with self.assertRaises(evaluation_book.EvaluationBookError):type(self.book)(other).snapshot()


if __name__=='__main__':unittest.main()
