"""Source-bound first report fixtures; no real market/selection/order claim."""
from contextlib import closing
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import zoneinfo

sys.path.insert(0,str(Path(__file__).resolve().parent))
from comparison_fixture import ComparisonWorld
from test_training_data import ROOT,TZ,Tick
from test_canonical_history import HANDOFF
from paperbot_r2 import common_outcome,common_scores,receipt_book,validation_settlements as settlements
from paperbot_r2 import validation_scores as scores,common_registration
from paperbot_r2.validation_scores import ValidationScoreJournal
from paperbot_r2.contracts import aware

FINAL=aware('2026-12-04T21:02:01+00:00')


class ValidationScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_tz=zoneinfo.TZPATH
        zoneinfo.reset_tzpath([str(ROOT/'outputs/zoneinfo')]);zoneinfo.ZoneInfo.clear_cache()
    @classmethod
    def tearDownClass(cls):zoneinfo.reset_tzpath(cls.old_tz);zoneinfo.ZoneInfo.clear_cache()
    def setUp(self):
        env=patch.dict(os.environ,{},clear=True);env.start();self.addCleanup(env.stop)
        net=patch.object(socket,'create_connection',side_effect=AssertionError('no network'))
        net.start();self.addCleanup(net.stop)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.directory=Path(temp.name);self.c=ComparisonWorld(self.directory)
        self.book=ValidationScoreJournal(self.directory/('r2-validation-scores-'+self.c.w.identity+'.sqlite'))
        self.book.initialize(self.c.settlements,timezone_file=TZ,clock=Tick(HANDOFF+timedelta(minutes=3)))
        self.c.issue();self.other=self.c.key.replace('AAPL|','MSFT|')
        self.c.issues.issue_once(self.other,registry=self.c.w.registry,training_journal=self.c.w.training,
            pairs=self.c.w.source_pairs(),timezone_file=TZ,clock=self.c.clock)
        self.c.outcome();self.c.settle();self.clock=Tick(FINAL)
    def freeze(self):return self.book.freeze_once(timezone_file=TZ,clock=self.clock)
    def load(self):return self.book.load_frozen(timezone_file=TZ,clock=self.clock)

    def test_first_report_common_denominators_failures_and_no_admission(self):
        state=self.freeze();report=state['domain']['report']
        self.assertFalse(report['engineering_gate_passed']);self.assertIsNone(report['proposed_family'])
        self.assertEqual(sum(g['counts']['planned'] for g in report['groups']),348)
        self.assertEqual(sum(g['counts']['issued'] for g in report['groups']),2)
        self.assertEqual(sum(g['counts']['settled'] for g in report['groups']),1)
        self.assertEqual(sum(g['counts']['common'] for g in report['groups']),0)
        self.assertEqual(report['trial_units'],15);self.assertEqual(state['head_seq'],4)
        receipt=self.load()
        self.assertEqual(json.loads(receipt.report_json),report)
        self.assertTrue(receipt.requires_selection_admission);self.assertFalse(receipt.test_admission);self.assertFalse(receipt.orders)
        self.assertGreater(receipt.available_at,receipt.commit_observed_at)

    def test_early_and_test_open_freeze_gate_precedes_sql(self):
        for at in (FINAL-timedelta(seconds=1),aware('2026-11-12T18:00:00+00:00'),settlements.TEST_OPEN):
            with patch('sqlite3.connect',side_effect=AssertionError('no premature/post-test read')) as connect:
                with self.assertRaises(scores.ValidationScoreError):self.book.freeze_once(timezone_file=TZ,clock=lambda:at)
                connect.assert_not_called()

    def test_bulk_scoring_and_later_loading_never_decode_new_raw_prices(self):
        with (patch.object(receipt_book.ReceiptBook,'_transaction',side_effect=AssertionError('opaque source hashing only')),
              patch.object(common_outcome,'observe_common_outcome',side_effect=AssertionError('do not reobserve'))):
            self.freeze();first=self.load()
            self.clock=Tick(settlements.TEST_OPEN+timedelta(days=1));later=self.load()
        self.assertEqual(later.report_json,first.report_json)

    def test_ambiguous_report_commit_only_recovers_confirmation(self):
        commit=self.book._commit
        def interrupt(db,kind):
            commit(db,kind)
            if kind=='evaluated':raise RuntimeError('first report durable')
        with patch.object(self.book,'_commit',side_effect=interrupt):
            with self.assertRaises(scores.ValidationScoreError):self.freeze()
        first=self.book.snapshot()['domain']['report']
        with patch.object(common_scores,'evaluate_frames',side_effect=AssertionError('no second calculation')):
            self.freeze();receipt=self.load()
        self.assertEqual(json.loads(receipt.report_json),first)

    def test_report_commit_crossing_test_open_never_confirms_afterward(self):
        commit=self.book._commit
        def cross(db,kind):
            commit(db,kind)
            if kind=='evaluated':self.clock.value=settlements.TEST_OPEN
        with patch.object(self.book,'_commit',side_effect=cross):
            with self.assertRaises(scores.ValidationScoreError):self.freeze()
        self.assertIsNone(self.book.snapshot()['domain']['confirmed_at'])
        with self.assertRaises(scores.ValidationScoreError):self.load()

    def test_corrupt_source_and_later_quarantine_cannot_become_current_permission(self):
        self.freeze();features=self.c.w.f.features
        features._append('quarantined',dict(key=self.c.key,raw_context=features.snapshot()['raw_context'],calendar=None),self.clock())
        with self.assertRaises(scores.ValidationScoreError):self.load()
        with closing(sqlite3.connect(features.path,isolation_level=None)) as db:
            seq,payload=db.execute('SELECT seq,payload FROM observations ORDER BY seq DESC LIMIT 1').fetchone()
            raw=json.loads(payload);raw['kind']='quarantineX';db.execute('UPDATE observations SET payload=? WHERE seq=?',(json.dumps(raw),seq))
        with self.assertRaises(scores.ValidationScoreError):self.load()

    def test_canonical_exposure_after_freeze_invalidates_report_receipt(self):
        self.freeze()
        common_registration.import_exposure(self.c.w.registry,domain='us_equity',days=['2026-12-07'],
            source_sha256='e'*64,timezone_file=TZ,clock=self.clock)
        with self.assertRaises(scores.ValidationScoreError):self.load()

    def test_later_late_settlement_does_not_recompute_or_replace_frozen_report(self):
        self.freeze();first=self.load();self.c.key=self.other;self.c.clock=Tick(self.clock.value+timedelta(seconds=1))
        late=self.c.settle();self.assertTrue(late['effective_late'])
        with patch.object(common_scores,'evaluate_frames',side_effect=AssertionError('frozen means frozen')):
            self.clock=Tick(self.c.clock.value+timedelta(seconds=1));later=self.load()
        self.assertEqual(later.report_json,first.report_json)

    def test_final_source_identity_is_checked_after_basis_verification(self):
        self.freeze();raw=self.c.w.f.raw.path;copy=self.directory/'replacement.sqlite';shutil.copy2(raw,copy)
        original=scores.verify_basis;identity=settlements.issuer.inputs._identity;mode=[]
        def replaced(*args,**kwargs):
            result=original(*args,**kwargs)
            try:os.replace(copy,raw);mode.append('native')
            except PermissionError:mode.append('simulated-sharing-restriction')
            return result
        def changed(path):
            value=identity(path)
            if mode==['simulated-sharing-restriction'] and path==raw:return value[0],value[1]+1,*value[2:]
            return value
        with (patch.object(scores,'verify_basis',side_effect=replaced),
              patch.object(settlements.issuer.inputs,'_identity',side_effect=changed)):
            with self.assertRaises(scores.ValidationScoreError):self.load()
        self.assertEqual(len(mode),1)
        if os.name!='nt':self.assertEqual(mode,['native'])

    def test_copied_score_book_and_alternative_path_are_not_a_second_history(self):
        self.freeze();copy=self.directory/'score-copy.sqlite';shutil.copy2(self.book.path,copy)
        with self.assertRaises(Exception):ValidationScoreJournal(copy).snapshot()

    def test_original_training_outcome_corruption_vetoes_current_score(self):
        original=self.c.w.f.outcomes
        with closing(sqlite3.connect(original.path,isolation_level=None)) as db:
            db.execute('UPDATE observations SET payload=payload||? WHERE seq=(SELECT MAX(seq) FROM observations)',(' ',))
        with self.assertRaises(Exception):original.snapshot()
        with self.assertRaises(scores.ValidationScoreError):self.freeze()

    def test_original_training_outcome_consumed_prefix_cannot_roll_back(self):
        original=self.c.w.f.outcomes;before=original.snapshot()
        with closing(sqlite3.connect(original.path,isolation_level=None)) as db:
            previous=db.execute('SELECT previous_digest FROM observations WHERE seq=?',(before['head_seq'],)).fetchone()[0]
            db.execute('BEGIN IMMEDIATE')
            db.execute('DELETE FROM observations WHERE seq=?',(before['head_seq'],))
            db.execute('UPDATE metadata SET head_seq=?,head_digest=?',(before['head_seq']-1,previous));db.commit()
        self.assertEqual(original.snapshot()['head_seq'],before['head_seq']-1)
        with self.assertRaises(scores.ValidationScoreError):self.freeze()

    def test_original_training_outcome_same_bytes_replacement_is_rejected(self):
        original=self.c.w.f.outcomes;copy=self.directory/'outcome-copy.sqlite'
        before=original.path.stat().st_ino;shutil.copy2(original.path,copy);os.replace(copy,original.path)
        self.assertNotEqual(before,original.path.stat().st_ino)
        original.snapshot()  # The source's own schema/hash chain remains valid.
        with self.assertRaises(scores.ValidationScoreError):self.freeze()

    def test_original_training_outcome_wal_commit_is_watched_during_scoring(self):
        original=self.c.w.f.outcomes;evaluate=common_scores.evaluate_frames
        with closing(sqlite3.connect(original.path,isolation_level=None)) as keep:
            self.assertEqual(keep.execute('PRAGMA journal_mode=WAL').fetchone()[0].lower(),'wal')
            changed=[]
            def concurrent(*args,**kwargs):
                report=evaluate(*args,**kwargs);stat=original.path.stat();before=(stat.st_size,stat.st_mtime_ns)
                keep.execute('BEGIN IMMEDIATE');keep.execute('UPDATE metadata SET started_at=started_at+1')
                keep.execute('UPDATE metadata SET started_at=started_at-1');keep.commit()
                stat=original.path.stat();changed.append(before==(stat.st_size,stat.st_mtime_ns))
                original.snapshot()  # Net values and chain are still valid.
                return report
            with patch.object(common_scores,'evaluate_frames',side_effect=concurrent):
                with self.assertRaises(scores.ValidationScoreError):self.freeze()
            self.assertEqual(changed,[True])
        self.assertIsNone(self.book.snapshot()['domain']['report'])

    def test_original_training_outcome_replacement_after_basis_check_is_rejected(self):
        self.freeze();original=self.c.w.f.outcomes.path;copy=self.directory/'late-outcome-copy.sqlite';shutil.copy2(original,copy)
        verify=scores.verify_basis;identity=settlements.issuer.inputs._identity;mode=[]
        def replaced(*args,**kwargs):
            result=verify(*args,**kwargs)
            try:os.replace(copy,original);mode.append('native')
            except PermissionError:mode.append('simulated-sharing-restriction')
            return result
        def changed(path):
            value=identity(path)
            if mode==['simulated-sharing-restriction'] and path==original:return value[0],value[1]+1,*value[2:]
            return value
        with (patch.object(scores,'verify_basis',side_effect=replaced),patch.object(settlements.issuer.inputs,'_identity',side_effect=changed)):
            with self.assertRaises(scores.ValidationScoreError):self.load()
        self.assertEqual(len(mode),1)
        if os.name!='nt':self.assertEqual(mode,['native'])
