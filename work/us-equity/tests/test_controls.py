from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import importlib
import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.controls'), 'durable operator controls required')
        self.m = importlib.import_module('paperbot_us.controls')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'controls.sqlite'
        self.store = self.m.ControlJournal(self.path)
        self.now = datetime(2026, 10, 24, 6, tzinfo=timezone.utc)
        self.store.initialize(now=self.now)

    def change(self, **changes):
        return self.store.update(expected_revision=changes.get('expected_revision', 1),
                                 pause_entries=changes.get('pause_entries', False),
                                 halt_orders=changes.get('halt_orders', False),
                                 reason=changes.get('reason', 'operator_resume'), now=changes.get('now', self.now))

    def test_initial_pause_is_durable_and_never_auto_expires_into_resume(self):
        initial = self.store.read(now=self.now)
        self.assertTrue(initial.pause_entries)
        self.assertFalse(initial.halt_orders)
        self.assertEqual(initial.revision, 1)
        later = self.m.ControlJournal(self.path).read(now=self.now + timedelta(days=31))
        self.assertEqual(initial, later)
        self.assertFalse(initial.public_summary()['execution_enabled'])

    def test_revision_compare_and_swap_preserves_every_operator_change(self):
        initial = self.store.read(now=self.now)
        resumed = self.change()
        self.assertFalse(resumed.pause_entries)
        self.assertFalse(self.store.is_current(initial, now=self.now))
        self.assertTrue(self.store.is_current(resumed, now=self.now))
        with self.assertRaises(self.m.ControlError):
            self.change(expected_revision=1, pause_entries=True, reason='operator_pause')
        self.assertEqual(self.store.read(now=self.now), resumed)
        self.assertEqual(self.store.verify(now=self.now)['revisions'], 2)

    def test_halt_release_preserves_pause_and_requires_a_separate_explicit_resume(self):
        halted = self.change(pause_entries=True, halt_orders=True, reason='operator_halt')
        self.assertTrue(halted.halt_orders)
        released = self.change(expected_revision=2, pause_entries=True, halt_orders=False,
                               reason='operator_release_halt')
        self.assertTrue(released.pause_entries)
        self.assertFalse(released.halt_orders)
        resumed = self.change(expected_revision=3)
        self.assertFalse(resumed.pause_entries)

    def test_invalid_states_reasons_and_boolean_revision_cannot_change_controls(self):
        cases = [dict(pause_entries='false'), dict(expected_revision=True),
                 dict(pause_entries=False, halt_orders=True, reason='operator_halt'),
                 dict(pause_entries=True, reason='operator_resume'),
                 dict(reason='PRIVATE_API_KEY=DO_NOT_STORE')]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(self.m.ControlError):
                self.change(**args)
        self.assertEqual(self.store.verify(now=self.now)['revisions'], 1)

    def test_missing_corrupt_or_clock_regressed_control_state_never_becomes_normal_resume(self):
        missing = self.m.ControlJournal(self.path.with_name('missing.sqlite'))
        with self.assertRaises(self.m.ControlError):
            missing.read(now=self.now)
        self.assertFalse(missing.path.exists())
        with self.assertRaises(self.m.ControlError):
            self.store.read(now=self.now - timedelta(seconds=1))
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("UPDATE events SET payload='{}' WHERE revision=1")
            db.commit()
        with self.assertRaises(self.m.ControlError):
            self.store.read(now=self.now)

    def test_failed_commit_preserves_previous_control_and_sanitizes_failure(self):
        before = self.store.read(now=self.now)
        with patch.object(self.store, '_append', side_effect=OSError('PRIVATE_DISK_ERROR')):
            with self.assertRaises(self.m.ControlError) as caught:
                self.change()
        self.assertNotIn('PRIVATE', str(caught.exception))
        self.assertEqual(self.store.read(now=self.now), before)

    def test_existing_state_cannot_be_reset_and_only_exact_current_snapshot_is_current(self):
        with self.assertRaises(self.m.ControlError):
            self.store.initialize(now=self.now)
        actual = self.store.read(now=self.now)
        self.assertFalse(self.store.is_current(replace(actual, pause_entries=False), now=self.now))
        self.assertFalse(self.store.is_current(replace(actual, digest='f' * 64), now=self.now))

    def test_operator_update_cannot_commit_inside_a_dispatch_read_lease(self):
        self.assertTrue(hasattr(self.store, 'dispatch_lease'), 'cross-process dispatch/control serialization required')
        entered, committed = threading.Event(), threading.Event()
        errors = []
        original = self.store._append
        def append(*args):
            result = original(*args)
            entered.set()
            return result
        def update():
            try:
                self.change()
                committed.set()
            except Exception as error:
                errors.append(type(error).__name__)
        snapshot = self.store.read(now=self.now)
        with patch.object(self.store, '_append', side_effect=append):
            with self.store.dispatch_lease(snapshot, now=self.now) as current:
                self.assertTrue(current)
                writer = threading.Thread(target=update)
                writer.start()
                self.assertTrue(entered.wait(2))
                self.assertFalse(committed.wait(.05), 'halt/resume must not commit between control check and dispatch')
            writer.join(2)
        self.assertFalse(writer.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(committed.is_set())
        with self.store.dispatch_lease(snapshot, now=self.now) as current:
            self.assertFalse(current)

    def test_wal_mode_is_rejected_instead_of_using_incompatible_dispatch_locking(self):
        snapshot = self.store.read(now=self.now)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute('PRAGMA journal_mode=WAL').fetchone()[0], 'wal')
        with self.assertRaises(self.m.ControlError):
            self.store.read(now=self.now)
        with self.assertRaises(self.m.ControlError):
            self.change()
        with self.assertRaises(self.m.ControlError):
            with self.store.dispatch_lease(snapshot, now=self.now):
                self.fail('a WAL control database cannot supply a rollback-journal dispatch lease')
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute('PRAGMA journal_mode').fetchone()[0], 'wal', 'do not silently change database mode')
            self.assertEqual(db.execute('SELECT COUNT(*) FROM events').fetchone()[0], 1)


if __name__ == '__main__':
    unittest.main()
