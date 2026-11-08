"""Temporary private-storage tests, no market or phase permission."""
from contextlib import closing
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).resolve().parents[3]),str(Path(__file__).resolve().parents[1])]
from paperbot_r2 import evaluation_book as books,common_registration
from paperbot_r2.contracts import aware


class Tick:
    def __init__(self,value):self.value=value
    def __call__(self):self.value+=timedelta(microseconds=100);return self.value


class ExampleBook(books.EvaluationBook):
    def source_sha256(self):return hashlib.sha256(Path(books.__file__).read_bytes()).hexdigest()
    def domain_initial(self):return dict(values=[])
    def domain_apply(self,domain,kind,body,at,sha,binding):
        books.need(kind=='value' and set(body)=={'value'} and type(body['value']) is int and 0<=body['value']<=10)
        domain['values'].append(body['value'])


class EvaluationBookTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.directory=Path(temp.name);self.book=ExampleBook(self.directory/'book.sqlite')
        self.clock=Tick(aware('2026-11-24T05:01:00+00:00'))
    def create(self):return self.book.create(dict(scope='synthetic-storage-only'),clock=self.clock,check=lambda:None)

    def test_initialized_marker_observes_committed_header_then_source_bound_replay(self):
        state=self.create();self.assertTrue(state['initialized'])
        self.assertEqual(state['head_seq'],2);self.assertGreater(state['initialized_at'],state['created_at'])
        state=self.book.append(state,'value',dict(value=2),clock=self.clock,check=lambda:None)
        self.assertEqual(state['domain']['values'],[2]);self.assertEqual(state['head_seq'],3)
        self.assertEqual(self.create()['head_seq'],3)
        with patch.object(self.book,'source_sha256',return_value='0'*64):
            with self.assertRaises(books.EvaluationBookError):self.book.snapshot()

    def test_ambiguous_header_commit_recovers_only_initialization_without_duplicate_header(self):
        commit=self.book._commit
        def crash(db,kind):
            commit(db,kind)
            if kind=='created':raise RuntimeError('committed header, interrupted caller')
        with patch.object(self.book,'_commit',side_effect=crash):
            with self.assertRaises(books.EvaluationBookError):self.create()
        self.assertFalse(self.book.snapshot()['initialized']);self.assertEqual(self.book.snapshot()['head_seq'],1)
        with self.assertRaises(books.EvaluationBookError):
            self.book.append(self.book.snapshot(),'value',dict(value=1),clock=self.clock,check=lambda:None)
        self.assertTrue(self.create()['initialized']);self.assertEqual(self.book.snapshot()['head_seq'],2)

    def test_header_commit_crossing_first_open_cannot_become_initialized(self):
        commit=self.book._commit
        def late(db,kind):
            commit(db,kind)
            if kind=='created':self.clock.value=common_registration.FIRST_OPEN
        with patch.object(self.book,'_commit',side_effect=late):
            with self.assertRaises(books.EvaluationBookError):self.create()
        self.assertFalse(self.book.snapshot()['initialized'])
        with self.assertRaises(books.EvaluationBookError):self.create()

    def test_copied_changed_payload_future_timestamp_and_budget_reject(self):
        self.create();copy=self.directory/'copy.sqlite';shutil.copy2(self.book.path,copy)
        with self.assertRaises(books.EvaluationBookError):ExampleBook(copy).snapshot()
        with self.assertRaises(books.EvaluationBookError):
            self.book.snapshot(clock=Tick(aware('2026-11-24T04:00:00+00:00')))
        state=self.book.snapshot()
        with patch.object(books,'MAX_EVENTS',3):
            state=self.book.append(state,'value',dict(value=1),clock=self.clock,check=lambda:None)
            with self.assertRaises(books.EvaluationBookError):self.book.append(state,'value',dict(value=2),clock=self.clock,check=lambda:None)
        with closing(sqlite3.connect(self.book.path,isolation_level=None)) as db:
            value=json.loads(db.execute('SELECT payload FROM events WHERE seq=3').fetchone()[0]);value['body']['value']=5
            db.execute('UPDATE events SET payload=? WHERE seq=3',(json.dumps(value),))
        with self.assertRaises(books.EvaluationBookError):self.book.snapshot()

    def test_append_stamp_is_sampled_after_transaction_validation_callback(self):
        state=self.create();changed=[]
        def check():
            if not changed:self.clock.value+=timedelta(seconds=10);changed.append(self.clock.value)
        result=self.book.append(state,'value',dict(value=1),clock=self.clock,check=check)
        self.assertGreater(result['last_at'],changed[0])
        with self.assertRaises(books.EvaluationBookError):
            self.book.append(state,'value',dict(value=2),clock=self.clock,check=lambda:None)

    def test_failing_check_rolls_back_and_no_uninitialized_domain_append(self):
        state=self.create();before=self.book.path.read_bytes()
        def invalid():raise RuntimeError('changed source')
        with self.assertRaises(books.EvaluationBookError):self.book.append(state,'value',dict(value=1),clock=self.clock,check=invalid)
        self.assertEqual(self.book.path.read_bytes(),before)
