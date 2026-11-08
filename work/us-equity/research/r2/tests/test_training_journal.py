"""Durable training lifecycle tests; all upstream evidence is synthetic."""
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import os
import json
from pathlib import Path
import shutil
import socket
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import zoneinfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_training_data import Fixture, Tick, ROOT, TZ
from paperbot_r2 import training_journal as journal, model_training, training_data
from paperbot_r2.contracts import digest


class TrainingJournalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_tz = zoneinfo.TZPATH
        zoneinfo.reset_tzpath([str(ROOT / 'outputs/zoneinfo')]); zoneinfo.ZoneInfo.clear_cache()

    @classmethod
    def tearDownClass(cls):
        zoneinfo.reset_tzpath(cls.old_tz); zoneinfo.ZoneInfo.clear_cache()

    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True); env.start(); self.addCleanup(env.stop)
        network = patch.object(socket, 'create_connection', side_effect=AssertionError('no network'))
        network.start(); self.addCleanup(network.stop)
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name); self.f = Fixture(self.directory)
        self.book = journal.TrainingJournal(self.directory / 'training.sqlite')
        self.book.initialize(timezone_file=TZ, clock=Tick(self.f.before))
        self.clock = Tick(self.f.at + timedelta(seconds=1))

    def mature(self):
        self.f.finish(); self.clock = Tick(self.f.at + timedelta(seconds=1))

    def train(self, clock=None):
        return self.book.run_once(self.f.raw, self.f.features, self.f.outcomes,
                                  timezone_file=TZ, clock=clock or self.clock)

    def test_unready_does_not_reserve_or_fit(self):
        with patch.object(model_training, 'fit_training_snapshot', side_effect=AssertionError('must not fit')) as fit:
            state = self.train()
            fit.assert_not_called()
        self.assertEqual(state['phase'], 'created')
        self.assertEqual(state['head_seq'], 1)
        self.assertEqual(state['reserved_trial_units'], 0)
        with self.assertRaises(journal.TrainingJournalError):
            self.book.load_acknowledged(clock=self.clock)

    def test_reservation_is_durable_before_fit_then_result_ack_after_commit(self):
        self.mature()
        paths = (self.f.raw.path, self.f.features.path, self.f.outcomes.path)
        before = [sha256(p.read_bytes()).hexdigest() for p in paths]
        original_fit = model_training.fit_training_snapshot
        seen = []
        def fit(snapshot, **kwargs):
            state = self.book.snapshot()
            seen.append((state['phase'], state['head_seq'], state['reserved_trial_units'], state['training'].seal))
            return original_fit(snapshot, **kwargs)
        with patch.object(model_training, 'fit_training_snapshot', fit):
            state = self.train()
        self.assertEqual(seen, [('reserved', 2, 15, state['training'].seal)])
        self.assertEqual(state['phase'], 'acknowledged')
        self.assertEqual(state['head_seq'], 5)
        self.assertEqual(len(state['fits'].attempts), 15)
        self.assertEqual(state['reserved_trial_units'], 15)
        self.assertGreater(state['acknowledged_at'], state['result_at'])
        self.assertGreater(state['commit_observed_at'], state['acknowledged_at'])
        self.assertGreaterEqual(state['result_at'], state['fits'].available_at)
        self.assertGreater(min(a.started_at for a in state['fits'].attempts), state['reserved_at'])
        receipt = self.book.load_acknowledged(clock=self.clock)
        self.assertEqual(receipt.fits, state['fits'])
        self.assertEqual(receipt.acknowledged_at, state['acknowledged_at'])
        self.assertGreaterEqual(receipt.loaded_at, receipt.acknowledged_at)
        self.assertTrue(receipt.canonical_registration_required)
        self.assertEqual(before, [sha256(p.read_bytes()).hexdigest() for p in paths])

    def test_acknowledged_is_idempotent_without_upstream_read_or_refit(self):
        self.mature(); first = self.train()
        with patch.object(training_data, 'read_training_snapshot', side_effect=AssertionError('no upstream read')), \
             patch.object(model_training, 'fit_training_snapshot', side_effect=AssertionError('no refit')):
            second = self.train()
        self.assertEqual(second, first)

    def test_interrupted_fit_consumes_all_slots_and_recovery_never_refits(self):
        self.mature()
        with patch.object(model_training, 'fit_training_snapshot', side_effect=RuntimeError('sensitive error')):
            with self.assertRaises(journal.TrainingJournalError) as caught:
                self.train()
        self.assertNotIn('sensitive', str(caught.exception))
        self.assertEqual(self.book.snapshot()['phase'], 'reserved')
        with patch.object(model_training, 'fit_training_snapshot', side_effect=AssertionError('never retry')) as fit, \
             patch.object(training_data, 'read_training_snapshot', side_effect=AssertionError('no source reread')):
            state = self.train(); fit.assert_not_called()
        self.assertEqual(state['phase'], 'abandoned')
        self.assertEqual(state['reserved_trial_units'], 15)
        self.assertEqual(state['head_seq'], 3)
        self.assertIsNone(state['fits'])
        self.assertEqual(self.train(), state)

    def test_ambiguous_reserved_commit_never_starts_or_repeats_fit(self):
        self.mature(); append = self.book._append
        def fail_after_commit(state, kind, body, at, **kwargs):
            result = append(state, kind, body, at, **kwargs)
            if kind == 'reserved':
                raise OSError('synthetic post-commit failure')
            return result
        with patch.object(self.book, '_append', fail_after_commit), \
             patch.object(model_training, 'fit_training_snapshot', side_effect=AssertionError('fit after ambiguous commit')) as fit:
            with self.assertRaises(journal.TrainingJournalError):
                self.train()
            fit.assert_not_called()
        self.assertEqual(self.book.snapshot()['phase'], 'reserved')
        self.assertEqual(self.train()['phase'], 'abandoned')

    def test_ack_failure_keeps_result_unusable_and_recovers_without_refit(self):
        self.mature(); append = self.book._append
        def fail_ack(state, kind, body, at, **kwargs):
            if kind == 'acknowledged':
                raise OSError('synthetic acknowledgment failure')
            return append(state, kind, body, at, **kwargs)
        with patch.object(self.book, '_append', fail_ack):
            with self.assertRaises(journal.TrainingJournalError):
                self.train()
        pending = self.book.snapshot()
        self.assertEqual(pending['phase'], 'result_pending_ack')
        with self.assertRaises(journal.TrainingJournalError):
            self.book.load_acknowledged(clock=self.clock)
        with patch.object(model_training, 'fit_training_snapshot', side_effect=AssertionError('no retry')):
            result = self.train()
        self.assertEqual(result['fits'], pending['fits'])
        self.assertEqual(result['result_digest'], pending['result_digest'])
        self.assertGreater(result['acknowledged_at'], pending['result_at'])

    def test_result_write_failure_recovers_as_abandoned_not_retrained(self):
        self.mature(); append = self.book._append
        def fail_result(state, kind, body, at, **kwargs):
            if kind == 'result':
                raise OSError('synthetic result write failure')
            return append(state, kind, body, at, **kwargs)
        with patch.object(self.book, '_append', fail_result):
            with self.assertRaises(journal.TrainingJournalError):
                self.train()
        self.assertEqual(self.book.snapshot()['phase'], 'reserved')
        self.assertEqual(self.train()['phase'], 'abandoned')

    def test_corruption_copied_book_source_change_and_foreign_wal_are_rejected(self):
        copied = self.directory / 'copied.sqlite'; shutil.copyfile(self.book.path, copied)
        with self.assertRaises(journal.TrainingJournalError):
            journal.TrainingJournal(copied).snapshot()
        with patch.object(journal, 'source_sha256', return_value='0' * 64):
            with self.assertRaises(journal.TrainingJournalError):
                self.book.snapshot()
        with closing(sqlite3.connect(self.book.path, isolation_level=None)) as db:
            db.execute('PRAGMA journal_mode=WAL')
            with self.assertRaises(journal.TrainingJournalError):
                self.book.snapshot()
            db.execute('PRAGMA journal_mode=DELETE')
            db.execute('UPDATE events SET digest=? WHERE seq=1', ('0' * 64,))
        with self.assertRaises(journal.TrainingJournalError):
            self.book.snapshot()

    def test_cutoff_precedes_reads_and_loader_rejects_early_clock(self):
        at = datetime(2026, 11, 10, 4, 55, tzinfo=timezone.utc)
        with patch('sqlite3.connect', side_effect=AssertionError('no database read')) as connect:
            with self.assertRaises(journal.TrainingJournalError):
                self.train(clock=lambda: at)
            connect.assert_not_called()
        self.mature(); state = self.train()
        with self.assertRaises(journal.TrainingJournalError):
            self.book.load_acknowledged(clock=lambda: state['result_at'])

    def test_initialize_never_overwrites_an_existing_book(self):
        before = self.book.path.read_bytes()
        with self.assertRaises(journal.TrainingJournalError):
            self.book.initialize(timezone_file=TZ, clock=self.clock)
        self.assertEqual(before, self.book.path.read_bytes())

    def test_fit_result_must_match_reserved_input_rows_not_only_snapshot_header(self):
        self.mature(); original = model_training.fit_training_snapshot
        def altered(snapshot, **kwargs):
            fits = original(snapshot, **kwargs)
            first = next(a for a in fits.attempts if a.status == 'failed')
            fields = {k: v for k, v in asdict(first).items() if k != 'seal'}
            fields['input_rows_sha256'] = '0' * 64
            changed = model_training.FitAttempt(**fields, seal=digest(fields))
            attempts = tuple(changed if a is first else a for a in fits.attempts)
            fields = dict(snapshot_sha256=fits.snapshot_sha256, attempts=attempts,
                          available_at=fits.available_at, source_sha256=fits.source_sha256)
            return model_training.TrainingFits(**fields, seal=digest({**fields,
                'attempts': [asdict(a) for a in attempts]}))
        with patch.object(model_training, 'fit_training_snapshot', altered):
            with self.assertRaises(journal.TrainingJournalError):
                self.train()
        self.assertEqual(self.book.snapshot()['phase'], 'reserved')

    def test_clock_regression_after_fit_cannot_commit_or_publish_result(self):
        self.mature(); original = model_training.fit_training_snapshot
        def regress(snapshot, **kwargs):
            result = original(snapshot, **kwargs)
            self.clock.value -= timedelta(seconds=1)
            return result
        with patch.object(model_training, 'fit_training_snapshot', regress):
            with self.assertRaises(journal.TrainingJournalError):
                self.train()
        self.assertEqual(self.book.snapshot()['phase'], 'reserved')

    def test_corrupt_old_event_is_rejected_even_with_unchanged_last_head(self):
        self.mature(); state = self.train()
        with closing(sqlite3.connect(self.book.path, isolation_level=None)) as db:
            head = db.execute('SELECT head_digest FROM metadata').fetchone()[0]
            payload = db.execute('SELECT payload FROM events WHERE seq=1').fetchone()[0]
            db.execute('UPDATE events SET payload=? WHERE seq=1', (payload + ' ',))
            self.assertEqual(db.execute('SELECT head_digest FROM metadata').fetchone()[0], head)
        self.assertEqual(head, state['head_digest'])
        with self.assertRaises(journal.TrainingJournalError):
            self.book.load_acknowledged(clock=self.clock)

    def test_ack_write_crossing_cutoff_cannot_publish_usable_result(self):
        self.mature(); original = self.book._write
        cutoff = datetime(2026, 11, 10, 4, 56, tzinfo=timezone.utc)
        def delay(db, state, meta, kind, body, at):
            if kind == 'acknowledged':
                self.clock.value = cutoff
            return original(db, state, meta, kind, body, at)
        with patch.object(self.book, '_write', delay):
            with self.assertRaises(journal.TrainingJournalError):
                self.train()
        self.assertEqual(self.book.snapshot()['phase'], 'result_pending_ack')
        with self.assertRaises(journal.TrainingJournalError):
            self.book.load_acknowledged(clock=self.clock)

    def test_ack_commit_crossing_cutoff_stays_unconfirmed_and_unusable(self):
        self.mature(); clock = self.clock
        cutoff = datetime(2026, 11, 10, 4, 56, tzinfo=timezone.utc)
        target = str(self.book.path)
        class LateCommit(sqlite3.Connection):
            def commit(connection):
                value = connection.execute('PRAGMA database_list').fetchone()[2]
                is_target = Path(value) == Path(target)
                kind = None
                if is_target:
                    value = connection.execute('SELECT payload FROM events ORDER BY seq DESC LIMIT 1').fetchone()
                    kind = json.loads(value[0])['kind'] if value else None
                super().commit()
                if kind == 'acknowledged':
                    clock.value = cutoff
        original = sqlite3.connect
        def connect(*args, **kwargs):
            return original(*args, **kwargs, factory=LateCommit)
        with patch('sqlite3.connect', connect):
            with self.assertRaises(journal.TrainingJournalError):
                self.train()
        self.assertEqual(self.book.snapshot()['phase'], 'ack_pending_confirmation')
        with self.assertRaises(journal.TrainingJournalError):
            self.book.load_acknowledged(clock=self.clock)

    def test_confirmation_recovery_keeps_original_ack_and_never_refits(self):
        self.mature(); append = self.book._append
        def fail_confirmation(state, kind, body, at, **kwargs):
            if kind == 'confirmed':
                raise OSError('synthetic confirmation interruption')
            return append(state, kind, body, at, **kwargs)
        with patch.object(self.book, '_append', fail_confirmation):
            with self.assertRaises(journal.TrainingJournalError):
                self.train()
        pending = self.book.snapshot()
        self.assertEqual(pending['phase'], 'ack_pending_confirmation')
        self.assertEqual(pending['head_seq'], 4)
        with self.assertRaises(journal.TrainingJournalError):
            self.book.load_acknowledged(clock=self.clock)
        with patch.object(model_training, 'fit_training_snapshot', side_effect=AssertionError('no refit')), \
             patch.object(training_data, 'read_training_snapshot', side_effect=AssertionError('no upstream read')):
            state = self.train()
        self.assertEqual(state['phase'], 'acknowledged')
        self.assertEqual(state['head_seq'], 5)
        self.assertEqual(state['ack_digest'], pending['ack_digest'])
        self.assertEqual(state['acknowledged_at'], pending['acknowledged_at'])
        self.assertEqual(state['fits'], pending['fits'])

    def test_ack_clock_regression_inside_transaction_rolls_back_ack(self):
        self.mature(); original = self.book._write
        def regress(db, state, meta, kind, body, at):
            if kind == 'acknowledged':
                self.clock.value -= timedelta(seconds=1)
            return original(db, state, meta, kind, body, at)
        with patch.object(self.book, '_write', regress):
            with self.assertRaises(journal.TrainingJournalError):
                self.train()
        self.assertEqual(self.book.snapshot()['phase'], 'result_pending_ack')


if __name__ == '__main__':
    unittest.main()
