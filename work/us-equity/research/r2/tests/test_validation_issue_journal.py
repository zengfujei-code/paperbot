"""Synthetic future publication lifecycle, never a market prediction/test."""
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import os
import shutil
import socket
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import zoneinfo

sys.path.insert(0,str(Path(__file__).resolve().parent))
from issue_fixture import IssueWorld,Tick,TZ
from test_training_data import ROOT
from test_canonical_history import HANDOFF
from paperbot_r2 import validation_issue_journal as journal
from paperbot_r2 import candidate_forecasts as maths,forecast_inputs,common_registration as common
from paperbot_us.data import Calendar


class ValidationIssueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_tz=zoneinfo.TZPATH
        zoneinfo.reset_tzpath([str(ROOT/'outputs/zoneinfo')]);zoneinfo.ZoneInfo.clear_cache()

    @classmethod
    def tearDownClass(cls):
        zoneinfo.reset_tzpath(cls.old_tz);zoneinfo.ZoneInfo.clear_cache()

    def setUp(self):
        env=patch.dict(os.environ,{},clear=True);env.start();self.addCleanup(env.stop)
        net=patch.object(socket,'create_connection',side_effect=AssertionError('no network'))
        net.start();self.addCleanup(net.stop)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.directory=Path(temp.name)
        self.w=IssueWorld(self.directory,issue_source_sha256=journal.source_sha256())
        self.book=journal.ValidationIssueJournal(self.directory/('r2-validation-'+self.w.identity+'.sqlite'))
        self.initialize()
        self.key=self.w.receive_until('2026-11-24',9);self.clock=Tick(self.w.at)

    def initialize(self):
        self.book.initialize(self.w.registry,self.w.identity,self.w.training,self.w.source_pairs(),
                             timezone_file=TZ,clock=Tick(HANDOFF+timedelta(minutes=1)))

    def issue(self,clock=None):
        return self.book.issue_once(self.key,registry=self.w.registry,training_journal=self.w.training,
            pairs=self.w.source_pairs(),timezone_file=TZ,clock=clock or self.clock)

    def frame(self):return self.book.snapshot()['frames'][self.key]

    def test_normal_five_slots_exact_input_replay_actual_commit_and_duplicate(self):
        seen=[];original=maths.compute_candidates
        def compute(*args,**kwargs):
            self.assertEqual(self.frame()['phase'],'reserved');seen.append(True)
            return original(*args,**kwargs)
        with patch.object(maths,'compute_candidates',side_effect=compute):self.issue()
        state=self.book.snapshot();frame=self.frame()
        self.assertEqual(state['head_seq'],5);self.assertEqual(frame['phase'],'confirmed')
        self.assertEqual(frame['consumed_slots'],5);self.assertEqual(len(frame['panel'].rows),5)
        self.assertEqual(len(seen),1)
        with patch.object(maths,'compute_candidates',side_effect=AssertionError('no retry')):
            self.issue()
        receipt=self.book.load_issued(self.key,clock=Tick(self.clock.value+timedelta(hours=2)))
        self.assertEqual(receipt.panel,frame['panel']);self.assertEqual(receipt.issue_input,frame['input'])
        self.assertGreater(receipt.commit_observed_at,receipt.issued_at)
        self.assertGreater(receipt.available_at,receipt.commit_observed_at)
        self.assertTrue(receipt.validation_only);self.assertFalse(receipt.test_admission)

    def test_reserved_crash_consumes_five_and_never_recomputes(self):
        with patch.object(maths,'compute_candidates',side_effect=RuntimeError('synthetic interruption')):
            with self.assertRaises(journal.ValidationIssueError):self.issue()
        self.assertEqual(self.frame()['phase'],'reserved')
        with (patch.object(forecast_inputs,'capture_issue_input',side_effect=AssertionError('no recapture')),
              patch.object(maths,'compute_candidates',side_effect=AssertionError('no retry'))):
            self.issue()
        self.assertEqual(self.frame()['phase'],'abandoned');self.assertEqual(self.frame()['consumed_slots'],5)
        with self.assertRaises(journal.ValidationIssueError):self.book.load_issued(self.key,clock=self.clock)

    def test_uncertain_reservation_commit_does_not_attempt_math_or_reuse_slot(self):
        commit=self.book._commit
        def interrupted(db,kind):
            commit(db,kind)
            if kind=='reserved':raise RuntimeError('unknown commit outcome')
        with (patch.object(self.book,'_commit',side_effect=interrupted),
              patch.object(maths,'compute_candidates',side_effect=AssertionError('not admitted')) as compute):
            with self.assertRaises(journal.ValidationIssueError):self.issue()
            compute.assert_not_called()
        self.assertEqual(self.frame()['phase'],'reserved')
        self.issue();self.assertEqual(self.frame()['phase'],'abandoned')

    def test_computed_recovery_only_acknowledges_persistence(self):
        append=self.book._append
        def stop(state,kind,*args,**kwargs):
            if kind=='issued':raise RuntimeError('interrupted after result commit')
            return append(state,kind,*args,**kwargs)
        with patch.object(self.book,'_append',side_effect=stop):
            with self.assertRaises(journal.ValidationIssueError):self.issue()
        self.assertEqual(self.frame()['phase'],'computed')
        with (patch.object(forecast_inputs,'capture_issue_input',side_effect=AssertionError('no reread')),
             patch.object(maths,'compute_candidates',side_effect=AssertionError('no retry')),
              patch.object(self.w.f.raw,'_transaction',side_effect=AssertionError('no price replay'))):
            self.issue()
        self.assertEqual(self.frame()['phase'],'confirmed')

    def test_issue_commit_observed_only_after_replay_then_confirmation(self):
        append=self.book._append
        def stop(state,kind,*args,**kwargs):
            if kind=='confirmed':raise RuntimeError('interrupted after durable issue')
            return append(state,kind,*args,**kwargs)
        with patch.object(self.book,'_append',side_effect=stop):
            with self.assertRaises(journal.ValidationIssueError):self.issue()
        self.assertEqual(self.frame()['phase'],'issued')
        with self.assertRaises(journal.ValidationIssueError):self.book.load_issued(self.key,clock=self.clock)
        with patch.object(maths,'compute_candidates',side_effect=AssertionError('no retry')):self.issue()
        self.assertEqual(self.frame()['phase'],'confirmed')

    def test_commit_crossing_deadline_leaves_unconfirmed_unusable_evidence(self):
        commit=self.book._commit
        def cross(db,kind):
            commit(db,kind)
            if kind=='issued':self.clock.value=forecast_inputs.aware(self.key.split('|')[1])+timedelta(seconds=121)
        with patch.object(self.book,'_commit',side_effect=cross):
            with self.assertRaises(journal.ValidationIssueError):self.issue()
        self.assertEqual(self.frame()['phase'],'issued')
        with self.assertRaises(journal.ValidationIssueError):self.book.load_issued(self.key,clock=self.clock)
        with patch.object(self.book,'snapshot',side_effect=AssertionError('deadline before journal read')):
            with self.assertRaises(journal.ValidationIssueError):self.issue()

    def test_canonical_exposure_during_math_vetoes_publication(self):
        compute=maths.compute_candidates
        def expose(*args,**kwargs):
            result=compute(*args,**kwargs)
            common.import_exposure(self.w.registry,domain='us_equity',days=['2026-11-24'],
                source_sha256='e'*64,timezone_file=TZ,clock=self.clock)
            return result
        with patch.object(maths,'compute_candidates',side_effect=expose):
            with self.assertRaises(journal.ValidationIssueError):self.issue()
        self.assertNotIn(self.frame()['phase'],('issued','confirmed'))

    def test_invalidation_after_issue_commit_cannot_gain_confirmation(self):
        commit=self.book._commit
        def invalidate(db,kind):
            commit(db,kind)
            if kind=='issued':
                common.import_exposure(self.w.registry,domain='us_equity',days=['2026-11-24'],
                    source_sha256='e'*64,timezone_file=TZ,clock=self.clock)
        with patch.object(self.book,'_commit',side_effect=invalidate):
            with self.assertRaises(journal.ValidationIssueError):self.issue()
        self.assertEqual(self.frame()['phase'],'issued')
        with self.assertRaises(journal.ValidationIssueError):self.book.load_issued(self.key,clock=self.clock)

    def test_later_valid_source_descendant_loads_without_reading_new_prices(self):
        self.issue();before=self.frame()['panel']
        self.w.receive_until('2026-11-24',10,derive=False)
        with patch.object(forecast_inputs.receipt_book.ReceiptBook,'_transaction',side_effect=AssertionError('no future price replay')):
            receipt=self.book.load_issued(self.key,clock=Tick(self.w.at))
        self.assertEqual(receipt.panel,before)

    def test_current_metadata_guard_does_not_redecode_models_for_each_commit(self):
        original=journal._current;calls=[]
        def checked(*args,**kwargs):calls.append(True);return original(*args,**kwargs)
        with patch.object(journal,'_current',side_effect=checked):self.issue()
        self.assertEqual(len(calls),1)

    def test_calendar_change_during_math_vetoes_and_later_load_is_rechecked(self):
        self.issue();session=self.w.session('2026-11-24')
        changed=replace(session,closes=session.closes-timedelta(minutes=15))
        self.w.f.raw.record_calendar(Calendar([changed],start='2026-11-24',end='2026-11-24'),
                                    day='2026-11-24',received_at=self.clock())
        with self.assertRaises(journal.ValidationIssueError):self.book.load_issued(self.key,clock=self.clock)

    def test_test_phase_and_reserved_clock_refuse_before_journal_read(self):
        for day in ('2026-11-12','2026-12-07'):
            self.key='AAPL|'+day+'T16:45:00+00:00'
            with patch.object(self.book,'snapshot',side_effect=AssertionError('phase before SQL')):
                with self.assertRaises(journal.ValidationIssueError):
                    self.issue(clock=Tick(forecast_inputs.aware(self.key.split('|')[1])+timedelta(seconds=3)))

    def test_copied_journal_and_successor_source_are_rejected(self):
        clone=self.directory/'copy.sqlite';shutil.copy2(self.book.path,clone)
        with self.assertRaises(journal.ValidationIssueError):journal.ValidationIssueJournal(clone).snapshot()
        original=self.w.next_raw.path;other=self.directory/'other-raw.sqlite';shutil.copy2(original,other)
        self.w.next_raw.path=other
        with self.assertRaises(journal.ValidationIssueError):self.issue()
        self.assertNotIn(self.key,self.book.snapshot()['frames'])

    def test_wrong_code_hash_or_late_initialization_never_creates_journal(self):
        location=self.directory/'wrong-code';location.mkdir()
        world=IssueWorld(location,issue_source_sha256='0'*64)
        other=journal.ValidationIssueJournal(location/('r2-validation-'+world.identity+'.sqlite'))
        with self.assertRaises(journal.ValidationIssueError):
            other.initialize(world.registry,world.identity,world.training,world.source_pairs(),
                             timezone_file=TZ,clock=Tick(HANDOFF+timedelta(minutes=2)))
        self.assertFalse(other.path.exists())

        with patch.object(common,'load_registration',side_effect=AssertionError('late gate before registry')) as read:
            with self.assertRaises(journal.ValidationIssueError):
                self.book.initialize(self.w.registry,self.w.identity,self.w.training,self.w.source_pairs(),
                                     timezone_file=TZ,clock=self.clock)
            read.assert_not_called()

    def test_one_registration_cannot_initialize_a_second_alternate_journal(self):
        # Use a fresh pre-open world: the main fixture already has later bars,
        # which would otherwise reject this attempt for an unrelated time gate.
        location=self.directory/'pre-open';location.mkdir()
        world=IssueWorld(location,issue_source_sha256=journal.source_sha256())
        other=journal.ValidationIssueJournal(location/'alternate.sqlite')
        with self.assertRaises(journal.ValidationIssueError):
            other.initialize(world.registry,world.identity,world.training,world.source_pairs(),
                             timezone_file=TZ,clock=Tick(HANDOFF+timedelta(minutes=2)))
        self.assertFalse(other.path.exists())

    def test_future_journal_payload_is_not_read_before_timestamp_preflight(self):
        self.issue()
        with patch.object(self.book,'snapshot',side_effect=AssertionError('no future payload')) as read:
            with self.assertRaises(journal.ValidationIssueError):
                self.book.load_issued(self.key,clock=Tick(HANDOFF))
            read.assert_not_called()

    def test_later_feature_quarantine_vetoes_loaded_publication(self):
        self.issue();features=self.w.f.features;context=features.snapshot()['raw_context']
        features._append('quarantined',dict(key=self.key,raw_context=context,calendar=None),self.clock())
        with self.assertRaises(journal.ValidationIssueError):self.book.load_issued(self.key,clock=self.clock)

    def test_corrupted_quarantine_payload_cannot_hide_its_veto(self):
        self.issue();features=self.w.f.features;context=features.snapshot()['raw_context']
        features._append('quarantined',dict(key=self.key,raw_context=context,calendar=None),self.clock())
        with closing(sqlite3.connect(features.path,isolation_level=None)) as db:
            seq,payload=db.execute('SELECT seq,payload FROM observations ORDER BY seq DESC LIMIT 1').fetchone()
            value=json.loads(payload);value['kind']='quarantineX'
            db.execute('UPDATE observations SET payload=? WHERE seq=?',(json.dumps(value),seq))
        with self.assertRaises(Exception):features.snapshot()
        with self.assertRaises(journal.ValidationIssueError):self.book.load_issued(self.key,clock=self.clock)

    def test_corrupted_raw_descendant_payload_cannot_be_loaded(self):
        self.issue();self.w.receive_until('2026-11-24',10,derive=False)
        with closing(sqlite3.connect(self.w.f.raw.path,isolation_level=None)) as db:
            seq,payload=db.execute("SELECT seq,payload FROM events WHERE json_extract(payload,'$.kind')='receipt' ORDER BY seq DESC LIMIT 1").fetchone()
            value=json.loads(payload);value['body']['bar']['close']+=.01
            db.execute('UPDATE events SET payload=? WHERE seq=?',(json.dumps(value),seq))
        with self.assertRaises(Exception):self.w.f.raw.snapshot()
        with self.assertRaises(journal.ValidationIssueError):self.book.load_issued(self.key,clock=Tick(self.w.at))

    def test_corrupted_raw_metadata_checksum_cannot_be_loaded(self):
        self.issue()
        with closing(sqlite3.connect(self.w.f.raw.path,isolation_level=None)) as db:
            db.execute('UPDATE metadata SET anchor_digest=?',('0'*64,))
        with self.assertRaises(Exception):self.w.f.raw.snapshot()
        with self.assertRaises(journal.ValidationIssueError):self.book.load_issued(self.key,clock=self.clock)

    def test_authority_watch_covers_the_initial_full_replay(self):
        original=journal._current;changed=[]
        def corrupt(*args,**kwargs):
            result=original(*args,**kwargs)
            if not changed:
                with closing(sqlite3.connect(self.w.training.path,isolation_level=None)) as db:
                    payload=db.execute('SELECT payload FROM events WHERE seq=1').fetchone()[0]
                    value=json.loads(payload);value['body']['scope']='corrupted-scope'
                    db.execute('UPDATE events SET payload=? WHERE seq=1',(json.dumps(value),))
                changed.append(True)
            return result
        with patch.object(journal,'_current',side_effect=corrupt):
            with self.assertRaises(journal.ValidationIssueError):self.issue()
        self.assertEqual(changed,[True]);self.assertNotIn(self.key,self.book.snapshot()['frames'])

    def test_publication_watch_covers_input_capture_without_an_epoch_gap(self):
        original=forecast_inputs.capture_issue_input
        def corrupt(*args,**kwargs):
            result=original(*args,**kwargs)
            with closing(sqlite3.connect(self.w.f.features.path,isolation_level=None)) as db:
                seq,payload=db.execute('SELECT seq,payload FROM observations ORDER BY seq DESC LIMIT 1').fetchone()
                value=json.loads(payload);value['observing_started_at']=123.
                db.execute('UPDATE observations SET payload=? WHERE seq=?',(json.dumps(value),seq))
            return result
        with patch.object(forecast_inputs,'capture_issue_input',side_effect=corrupt):
            with self.assertRaises(journal.ValidationIssueError):self.issue()
        self.assertNotIn(self.key,self.book.snapshot()['frames'])

    def test_final_loading_check_detects_replacement_after_sources_were_checked(self):
        self.issue();raw=self.w.f.raw.path;copy=self.directory/'replacement.sqlite';shutil.copy2(raw,copy)
        original=journal._current;identity=forecast_inputs._identity;calls=[];mode=[]
        def current(*args,**kwargs):
            result=original(*args,**kwargs);calls.append(True)
            if len(calls)==2:
                try:os.replace(copy,raw);mode.append('native')
                except PermissionError:mode.append('simulated-sharing-restriction')
            return result
        def changed_identity(path):
            value=identity(path)
            # Windows SQLite may deny file deletion while a watcher is open.
            # Exercise the same documented OS identity change there; Linux
            # must additionally execute the actual pathname replacement.
            if mode==['simulated-sharing-restriction'] and path==raw:
                return value[0],value[1]+1,*value[2:]
            return value
        with (patch.object(journal,'_current',side_effect=current),
              patch.object(forecast_inputs,'_identity',side_effect=changed_identity)):
            with self.assertRaises(journal.ValidationIssueError):self.book.load_issued(self.key,clock=self.clock)
        self.assertEqual(len(calls),2)
        if os.name!='nt':self.assertEqual(mode,['native'])
