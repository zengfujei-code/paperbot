"""Real captured synthetic input; reducer/storage are not phase admission."""
from copy import deepcopy
from datetime import timedelta
import hashlib
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch
import zoneinfo

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[3]),str(Path(__file__).resolve().parents[1])]
from issue_fixture import IssueWorld,Tick,TZ
from test_training_data import ROOT
from paperbot_r2 import forecast_inputs as inputs,candidate_forecasts as maths,evaluation_book,training_codec as codec
from paperbot_r2.contracts import aware
try:
    from paperbot_r2 import test_issue_journal as test_issuer
except ImportError:
    test_issuer=None


class TestIssuePhaseTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(test_issuer,'test issue journal not implemented')
    def test_bad_dates_deadlines_and_initialization_precede_sql_and_locks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);book=test_issuer.TestIssueJournal(root/'test.sqlite')
            for key,at in (('AAPL|2026-11-24T16:45:00+00:00','2026-11-24T16:45:01+00:00'),
                ('AAPL|2026-11-12T16:45:00+00:00','2026-11-12T16:45:01+00:00'),
                ('AAPL|2026-12-07T16:45:00+00:00','2026-12-07T16:47:01+00:00')):
                with patch('sqlite3.connect',side_effect=AssertionError('phase before SQL')):
                    with self.assertRaises(test_issuer.TestIssueError):
                        book.issue_once(key,selection_journal=None,timezone_file=TZ,clock=lambda:aware(at))
            with patch('sqlite3.connect',side_effect=AssertionError('initialization before SQL')):
                with self.assertRaises(test_issuer.TestIssueError):
                    book.initialize(None,timezone_file=TZ,clock=lambda:aware('2026-11-24T14:30:00+00:00'))
            self.assertEqual(list(root.iterdir()),[])


class TestIssueDomainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if test_issuer is None:return
        cls.old_tz=zoneinfo.TZPATH;zoneinfo.reset_tzpath([str(ROOT/'outputs/zoneinfo')]);zoneinfo.ZoneInfo.clear_cache()
        cls.addClassCleanup(zoneinfo.reset_tzpath,cls.old_tz);cls.addClassCleanup(zoneinfo.ZoneInfo.clear_cache)
        cls.temp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        with patch.dict(os.environ,{},clear=True),patch.object(socket,'socket',side_effect=AssertionError('no network')):
            cls.w=IssueWorld(Path(cls.temp.name));cls.key=cls.w.receive_until('2026-12-07',9)
            cls.captured=inputs.capture_issue_input(cls.w.training_receipt,cls.w.registration,cls.w.source_pairs(),cls.key,
                timezone_file=TZ,clock=Tick(cls.w.at))
            cls.panel=maths.compute_candidates(cls.w.training_receipt,cls.captured,clock=Tick(cls.w.at+timedelta(seconds=1)))
        cls.binding=dict(validation_issue_binding=dict(identity=cls.captured.registration.identity,
            plan_json=cls.captured.registration.plan_json,training=dict(fits_sha256=cls.captured.fits_sha256)))
        cls.reference=dict(selected_family='prevalence',selection_head_digest='a'*64,
            selection_commit_observed_at='2026-12-04T21:03:00+00:00',validation_report_sha256='b'*64)
        cls.reserved=dict(key=cls.key,input=inputs.encode_input(cls.captured),candidates=list(maths.model_training.FAMILIES),admission=cls.reference)
        cls.reserve_at=cls.captured.available_at+timedelta(microseconds=1)
    def setUp(self):
        self.assertIsNotNone(test_issuer,'test issue journal not implemented')
        self.book=test_issuer.TestIssueJournal(Path('domain-only-no-file.sqlite'));self.domain=self.book.domain_initial()
    def apply(self,kind,body,at,sha='c'*64):self.book.domain_apply(self.domain,kind,body,at,sha,self.binding)
    def reserve(self):self.apply('reserved',deepcopy(self.reserved),self.reserve_at)
    def compute(self):
        self.reserve();self.apply('computed',dict(key=self.key,panel=codec.encode(self.panel),reservation_digest='c'*64),self.panel.completed_at,'d'*64)
    def test_actual_captured_input_and_five_slots_then_separate_issue_confirmation(self):
        self.compute();at=self.panel.completed_at+timedelta(microseconds=1)
        self.apply('issued',dict(key=self.key,result_digest='d'*64),at,'e'*64)
        self.apply('confirmed',dict(key=self.key,issue_digest='e'*64),at+timedelta(microseconds=1),'f'*64)
        frame=self.domain['frames'][self.key]
        self.assertEqual(frame['consumed_slots'],5);self.assertEqual(frame['phase'],'confirmed')
        self.assertEqual(frame['input'].seal,self.captured.seal)
        self.assertEqual(frame['admission']['selected_family'],'prevalence')
        self.assertEqual(frame['panel'].rows[-1].status,'failed')
    def test_lost_reservation_abandons_once_without_new_computation(self):
        self.reserve();self.apply('abandoned',dict(key=self.key,reservation_digest='c'*64),self.reserve_at+timedelta(microseconds=1))
        self.assertEqual(self.domain['frames'][self.key]['consumed_slots'],5)
        self.assertEqual(self.domain['frames'][self.key]['phase'],'abandoned')
        with self.assertRaises(test_issuer.TestIssueError):
            self.apply('computed',dict(key=self.key,panel=codec.encode(self.panel),reservation_digest='c'*64),self.panel.completed_at)
        with self.assertRaises(test_issuer.TestIssueError):self.reserve()
    def test_confirmation_must_observe_prior_issue_before_deadline(self):
        self.compute();at=self.panel.completed_at+timedelta(microseconds=1)
        self.apply('issued',dict(key=self.key,result_digest='d'*64),at,'e'*64)
        for when in (at,self.captured.bundle.vector.event_at+timedelta(seconds=121)):
            with self.assertRaises(test_issuer.TestIssueError):self.apply('confirmed',dict(key=self.key,issue_digest='e'*64),when)
    def test_changed_candidate_set_selection_reference_and_key_rejected(self):
        variants=[]
        bad=deepcopy(self.reserved);bad['candidates'].pop();variants.append(bad)
        bad=deepcopy(self.reserved);bad['admission']['selection_commit_observed_at']='2026-12-07T14:30:00+00:00';variants.append(bad)
        bad=deepcopy(self.reserved);bad['admission']['selected_family']='new_method';variants.append(bad)
        bad=deepcopy(self.reserved);bad['key']='MSFT|2026-12-07T16:45:00+00:00';variants.append(bad)
        for body in variants:
            with self.assertRaises(test_issuer.TestIssueError):self.apply('reserved',body,self.reserve_at)
    def test_real_storage_recovers_committed_reservation_without_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            class StorageOnly(test_issuer.TestIssueJournal):
                def source_sha256(self):return hashlib.sha256(b'explicit issue storage fixture; no phase authority').hexdigest()
            book=StorageOnly(Path(temp)/'test.sqlite');clock=Tick(aware('2026-11-24T13:00:00+00:00'))
            book.create(self.binding,clock=clock,check=lambda:None);clock=Tick(self.reserve_at)
            commit=book._commit
            def interrupted(db,kind):
                commit(db,kind)
                if kind=='reserved':raise RuntimeError('synthetic interrupted COMMIT')
            with book_lock(book),patch.object(book,'_commit',side_effect=interrupted):
                with self.assertRaises(evaluation_book.EvaluationBookError):
                    book.append(book.snapshot(),'reserved',self.reserved,clock=clock,check=lambda:None)
            state=book.snapshot();self.assertEqual(state['head_seq'],3)
            self.assertEqual(state['domain']['frames'][self.key]['consumed_slots'],5)
            with book_lock(book):
                state=book.append(state,'abandoned',dict(key=self.key,reservation_digest=state['domain']['frames'][self.key]['reservation_digest']),clock=clock,check=lambda:None)
            self.assertEqual(state['domain']['frames'][self.key]['phase'],'abandoned')
            with self.assertRaises(test_issuer.TestIssueError):book.load_issued(self.key,timezone_file=TZ,clock=clock)
            other=Path(temp)/'copy.sqlite';shutil.copy2(book.path,other)
            with self.assertRaises(evaluation_book.EvaluationBookError):StorageOnly(other).snapshot()


def book_lock(book):return test_issuer._exclusive(book.path)


if __name__=='__main__':unittest.main()
