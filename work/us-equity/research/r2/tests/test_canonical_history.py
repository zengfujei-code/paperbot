"""Canonical continuity fixtures are synthetic and never touch a live registry."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[1])]
from paperbot_us import experiments as legacy
from paperbot_r2 import canonical_history as canonical
from test_training_data import Tick, TZ

HANDOFF = datetime(2026, 11, 24, 5, 1, tzinfo=timezone.utc)


def legacy_fixture(path, *, terminal=True, domain='us_equity'):
    registry = legacy.ExperimentRegistry(path); registry.initialize()
    registry.import_exposure(domain=domain, days=['2026-10-02'], source_sha256='a' * 64)
    plan = dict(schema=1, domain=domain, family='ewma_cusum', data_sha256='a' * 64,
        code_sha256='b' * 64, configuration_sha256='c' * 64,
        split=dict(train=['2026-10-01'], validation=['2026-10-26'], test=['2026-11-23'], context=[]),
        candidate_ids=['1' * 64], trial_budget=1, selection_stage='validation', objective='variance_mse_min')
    identity = registry.register(plan)
    registry.start_trial(identity, '1' * 64, 'train')
    if terminal:
        registry.finish(identity, report_sha256='d' * 64, completed=False)
    return registry, identity


class CanonicalHistoryTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True); env.start(); self.addCleanup(env.stop)
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.registry, self.identity = legacy_fixture(self.directory / 'shared.sqlite')
        self.store = canonical.CanonicalRegistry(self.registry.path)
        self.clock = Tick(HANDOFF)

    def inspect(self, clock=None):
        return canonical.inspect_legacy(self.registry, timezone_file=TZ, clock=clock or self.clock)

    def activate(self, history=None, clock=None):
        return self.store.continue_legacy(expected_history=history or self.inspect(),
                                          timezone_file=TZ, clock=clock or self.clock)

    def snapshot(self):
        return self.store.snapshot(timezone_file=TZ, clock=self.clock)

    def test_inspection_is_read_only_and_preserves_complete_state_and_unknown_trials(self):
        before = self.registry.path.read_bytes(); history = self.inspect()
        self.assertEqual(before, self.registry.path.read_bytes())
        self.assertEqual(history.path, str(self.registry.path))
        self.assertEqual(history.sequence, 5)
        self.assertTrue(history.all_reports_terminal)
        self.assertEqual(history.legacy_state()['experiments'][self.identity]['trials']['1' * 64 + ':train'],
                         {'status': 'started'})
        self.assertEqual(history.used_days()['us_equity'],
                         ('2026-10-01', '2026-10-02', '2026-10-26', '2026-11-23'))

    def test_transfer_preserves_legacy_prefix_and_read_api_but_freezes_legacy_writes(self):
        history = self.inspect(); old = self.registry.load(self.identity)
        with closing(sqlite3.connect(self.registry.path)) as db:
            prefix = db.execute('SELECT seq,payload,digest FROM events ORDER BY seq').fetchall()
        state = self.activate(history)
        self.assertEqual(state['legacy_state'], history.legacy_state())
        self.assertEqual(state['head_seq'], 1)
        self.assertEqual(self.registry.load(self.identity), old)
        with self.assertRaises(legacy.ExperimentError):
            self.registry.import_exposure(domain='us_equity', days=['2026-12-01'], source_sha256='f' * 64)
        with closing(sqlite3.connect(self.registry.path)) as db:
            self.assertEqual(db.execute('SELECT seq,payload,digest FROM events ORDER BY seq').fetchall(), prefix)
            self.assertEqual(len(db.execute("SELECT name FROM sqlite_schema WHERE type='trigger'").fetchall()), 6)
        self.assertEqual(self.activate(history), self.snapshot())

    def test_active_report_and_wrong_time_refuse_without_mutation(self):
        active, _ = legacy_fixture(self.directory / 'active.sqlite', terminal=False)
        history = canonical.inspect_legacy(active, timezone_file=TZ, clock=self.clock)
        self.assertFalse(history.all_reports_terminal)
        before = active.path.read_bytes()
        with self.assertRaises(canonical.CanonicalHistoryError):
            canonical.CanonicalRegistry(active.path).continue_legacy(expected_history=history,
                timezone_file=TZ, clock=self.clock)
        self.assertEqual(before, active.path.read_bytes())
        before = self.registry.path.read_bytes()
        with self.assertRaises(canonical.CanonicalHistoryError):
            self.activate(clock=lambda: datetime(2026, 10, 29, tzinfo=timezone.utc))
        self.assertEqual(before, self.registry.path.read_bytes())

    def test_r1_reserved_period_is_blocked_before_database_read(self):
        with patch('sqlite3.connect', side_effect=AssertionError('no read')) as connect:
            with self.assertRaises(canonical.CanonicalHistoryError):
                self.inspect(clock=lambda: datetime(2026, 11, 12, 12, tzinfo=timezone.utc))
            connect.assert_not_called()

    def test_original_anchor_allows_descendant_append_but_not_copied_book(self):
        history = self.inspect()
        self.registry.import_exposure(domain='synthetic', days=['2026-09-01'], source_sha256='e' * 64)
        copy = self.directory / 'copy.sqlite'; shutil.copyfile(self.registry.path, copy)
        with self.assertRaises(canonical.CanonicalHistoryError):
            canonical.CanonicalRegistry(copy).continue_legacy(expected_history=history,
                timezone_file=TZ, clock=self.clock)
        state = self.activate(history)
        self.assertEqual(len(state['legacy_state']['imports']), 2)

    def test_missing_freeze_trigger_fails_closed(self):
        self.activate()
        with closing(sqlite3.connect(self.registry.path, isolation_level=None)) as db:
            trigger = db.execute("SELECT name FROM sqlite_schema WHERE type='trigger' ORDER BY name LIMIT 1").fetchone()[0]
            db.execute('DROP TRIGGER ' + trigger)
        with self.assertRaises(canonical.CanonicalHistoryError):
            self.snapshot()

    def test_corrupted_legacy_prefix_fails_even_with_all_triggers_restored(self):
        self.activate()
        name = 'paperbot_v2_freeze_events_update'
        with closing(sqlite3.connect(self.registry.path, isolation_level=None)) as db:
            db.execute('DROP TRIGGER ' + name)
            db.execute('UPDATE events SET digest=? WHERE seq=2', ('0' * 64,))
            db.execute(canonical.FREEZE_SQL[name])
        with self.assertRaises(canonical.CanonicalHistoryError):
            self.snapshot()

    def test_changed_continuation_source_is_not_a_new_empty_authority(self):
        self.activate()
        with patch.object(canonical, 'source_sha256', return_value='0' * 64):
            with self.assertRaises(canonical.CanonicalHistoryError):
                self.snapshot()

    def test_transaction_failure_rolls_back_new_schema_and_keeps_old_reader_working(self):
        history = self.inspect(); original = self.store._write
        def fail(*args, **kwargs):
            raise OSError('synthetic handoff failure')
        with patch.object(self.store, '_write', fail):
            with self.assertRaises(canonical.CanonicalHistoryError):
                self.activate(history)
        self.assertEqual(self.registry.load(self.identity), history.legacy_state()['experiments'][self.identity])
        with closing(sqlite3.connect(self.registry.path)) as db:
            self.assertEqual(db.execute("SELECT name FROM sqlite_schema WHERE type='trigger'").fetchall(), [])
            self.assertEqual(set(r[0] for r in db.execute("SELECT name FROM sqlite_schema WHERE type='table'")), {'events','head'})

    def test_wal_commit_during_read_invalidates_history(self):
        with closing(sqlite3.connect(self.registry.path, isolation_level=None)) as keeper:
            keeper.execute('PRAGMA journal_mode=WAL'); keeper.execute('PRAGMA user_version=1')
            original = legacy.ExperimentRegistry._read; changed = []
            def mutate(registry, db):
                result = original(registry, db)
                if not changed:
                    changed.append(True)
                    self.registry.import_exposure(domain='synthetic', days=['2026-09-02'], source_sha256='e' * 64)
                return result
            with patch.object(legacy.ExperimentRegistry, '_read', mutate):
                with self.assertRaises(canonical.CanonicalHistoryError):
                    self.inspect()
            self.assertEqual(changed, [True])


if __name__ == '__main__':
    unittest.main()
