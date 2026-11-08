from datetime import datetime, timedelta
from dataclasses import replace
import os
import sys
from pathlib import Path
import unittest
import io
import json
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_common_outcome_journal as fixtures
from paperbot_us.data import Calendar
from paperbot_r2.feature_pipeline import derive_once
from paperbot_r2.common_outcome_runner import collect_once, main


class OutcomeRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.OutcomeJournalTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        fixtures.OutcomeJournalTests.tearDownClass.__func__(cls)

    def setUp(self):
        fixtures.OutcomeJournalTests.setUp(self)

    bar = fixtures.OutcomeJournalTests.bar
    receive = fixtures.OutcomeJournalTests.receive
    ready_feature = fixtures.OutcomeJournalTests.ready_feature

    def cycle(self, max_per_cycle=3):
        return collect_once(self.journal, feature_journal=self.features, raw_book=self.raw,
                            clock=fixtures.Tick(self.at), max_per_cycle=max_per_cycle)

    def test_matured_first_outcome_is_durable_and_repeat_is_idle(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 100.5)
        result = self.cycle()
        self.assertEqual(result['acknowledged_count'], 1)
        self.assertEqual(result['cycle_recorded'], 1)
        self.assertEqual(result['score_eligible_count'], 0)
        self.assertFalse(result['execution_enabled'])
        self.assertEqual(result['predictions_issued'], 0)
        head = self.journal.snapshot()['head_seq']
        self.assertEqual(self.cycle()['cycle_recorded'], 0)
        self.assertEqual(self.journal.snapshot()['head_seq'], head)
        self.assertIn(key, self.journal.snapshot()['entries'])

    def test_one_cycle_reuses_a_deeply_verified_unchanged_feature_source(self):
        self.ready_feature()
        self.receive(3,100.)
        self.receive(4,100.5)
        original=self.features.snapshot
        with patch.object(self.features,'snapshot',wraps=original) as source:
            result=self.cycle()
        self.assertEqual(result['cycle_recorded'],1)
        self.assertEqual(source.call_count,1)

    def test_missing_target_stays_in_unobserved_denominator(self):
        self.ready_feature()
        self.receive(3, 100.)
        self.at += timedelta(minutes=16)
        result = self.cycle()
        self.assertEqual(result['acknowledged_count'], 0)
        self.assertEqual(result['unobserved_matured_count'], 1)
        self.assertEqual(result['cycle_unavailable'], 1)
        self.assertEqual(result['status'], 'awaiting_receipts')

    def test_late_backlog_cannot_be_reported_as_on_time(self):
        self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 100.5)
        self.at += timedelta(minutes=5)
        result = self.cycle()
        self.assertEqual(result['acknowledged_count'], 1)
        self.assertEqual(result['late_count'], 1)

    def test_calendar_quarantine_remains_visible_and_unrecorded(self):
        self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 100.5)
        changed = replace(self.session, closes=self.session.closes - timedelta(minutes=15))
        self.at += timedelta(seconds=1)
        self.raw.record_calendar(Calendar([changed], start='2020-03-09', end='2020-03-09'),
                                 day='2020-03-09', received_at=self.at)
        self.at += timedelta(seconds=1)
        derive_once(self.features, raw_book=self.raw, clock=lambda: self.at)
        result = self.cycle()
        self.assertEqual(result['quarantined_count'], 1)
        self.assertEqual(result['acknowledged_count'], 0)

    def test_reserved_date_and_secret_names_refuse_before_any_input_read(self):
        with patch.object(self.features, 'snapshot', side_effect=AssertionError('feature read')) as f:
            with patch.object(self.raw, 'snapshot', side_effect=AssertionError('raw read')) as r:
                result = collect_once(self.journal, feature_journal=self.features, raw_book=self.raw,
                    clock=lambda: datetime.fromisoformat('2026-11-10T15:00:00+00:00'))
                self.assertEqual(result['status'], 'reserved_dates_paused')
                self.assertEqual((f.call_count, r.call_count), (0, 0))
                with patch.dict(os.environ, {'UNRELATED_API_KEY': 'not-a-real-secret'}):
                    result = self.cycle()
                self.assertEqual(result['status'], 'credential_environment_refused')
                self.assertEqual((f.call_count, r.call_count), (0, 0))

    def test_cycle_bounds_work_and_prioritizes_fresh_targets(self):
        # Four matured feature slots; the newest is still within its120s window.
        for offset in range(8):
            self.receive(offset, 100. + offset * .1)
            derive_once(self.features, raw_book=self.raw, clock=lambda: self.at)
        result = self.cycle(max_per_cycle=1)
        self.assertEqual(result['cycle_attempted'], 1)
        self.assertEqual(result['cycle_recorded'], 1)
        self.assertGreaterEqual(result['unobserved_matured_count'], 1)
        first = next(iter(self.journal.snapshot()['entries'].values()))
        self.assertFalse(first['effective_late'])

    def test_permanently_missing_old_gap_does_not_starve_later_ready_backlog(self):
        self.ready_feature()
        # Gap offset3 never arrives. Later contiguous windows are valid.
        for offset in range(4, 9):
            self.receive(offset, 100. + offset * .1)
            derive_once(self.features, raw_book=self.raw, clock=lambda: self.at)
        self.at += timedelta(minutes=5)
        result = self.cycle(max_per_cycle=1)
        self.assertEqual(result['cycle_recorded'], 1)
        self.assertGreaterEqual(result['cycle_unavailable'], 1)
        self.assertEqual(result['late_count'], 1)

    def test_pending_ack_is_recovered_by_worker_without_replacing_outcome(self):
        key = self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 100.5)
        original = fixtures.ObservationArchive.append

        def fail_ack(archive, report):
            if report['kind'] == 'acknowledged':
                raise OSError('simulated interrupted acknowledgment')
            return original(archive, report)

        with patch.object(fixtures.ObservationArchive, 'append', fail_ack):
            result = self.cycle()
        self.assertEqual(result['pending_ack_count'], 1)
        stored = self.journal.snapshot()['entries'][key]['outcome']
        self.at += timedelta(minutes=5)
        result = self.cycle()
        self.assertEqual(result['acknowledged_count'], 1)
        self.assertEqual(result['pending_ack_count'], 0)
        self.assertEqual(self.journal.snapshot()['entries'][key]['outcome'], stored)

    def test_clock_behind_committed_feature_fails_without_label(self):
        self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 100.5)
        derive_once(self.features, raw_book=self.raw, clock=lambda: self.at)
        self.at -= timedelta(minutes=10)
        result = self.cycle()
        self.assertEqual(result['status'], 'outcome_worker_unavailable')
        self.assertEqual(self.journal.snapshot()['entries'], {})

    def test_in_place_source_rollback_cannot_mix_disconnected_later_history(self):
        initial_feature = self.features.path.read_bytes()
        initial_raw = self.raw.path.read_bytes()
        self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 100.5)
        self.assertEqual(self.cycle()['acknowledged_count'], 1)
        head = self.journal.snapshot()['head_seq']
        self.features.path.write_bytes(initial_feature)
        self.raw.path.write_bytes(initial_raw)
        for offset in range(5, 10):
            self.receive(offset, 101. + offset * .1)
            derive_once(self.features, raw_book=self.raw, clock=lambda: self.at)
        result = self.cycle()
        self.assertEqual(result['status'], 'outcome_worker_unavailable')
        self.assertEqual(self.journal.snapshot()['head_seq'], head)

    def test_cli_explicit_initialize_refuses_second_initialization(self):
        target = self.directory / 'cli.sqlite'
        argv = ['worker', '--outcome-book', str(target), '--feature-book', str(self.features.path),
                '--raw-book', str(self.raw.path), '--publish', str(self.directory / 'status.json'),
                '--timezone-root', str(fixtures.ROOT / 'outputs/zoneinfo'),
                '--timezone-sha256', self.raw.snapshot()['plan'].timezone_sha256, '--initialize-only']
        output = io.StringIO()
        with patch.object(sys, 'argv', argv), redirect_stdout(output):
            self.assertEqual(main(), 0)
        self.assertEqual(json.loads(output.getvalue())['status'], 'initialized')
        before = target.read_bytes()
        output = io.StringIO()
        with patch.object(sys, 'argv', argv), redirect_stdout(output):
            self.assertEqual(main(), 1)
        self.assertEqual(target.read_bytes(), before)
        self.assertEqual(json.loads(output.getvalue())['status'], 'outcome_worker_unavailable')

    def test_corrupt_own_prefix_has_null_counts_and_no_reset(self):
        self.ready_feature()
        self.receive(3, 100.)
        self.receive(4, 100.5)
        self.cycle()
        from contextlib import closing
        import sqlite3
        with closing(sqlite3.connect(self.journal.path)) as db:
            db.execute('UPDATE observations SET digest=? WHERE seq=1', ('0' * 64,))
            db.commit()
        result = self.cycle()
        self.assertEqual(result['status'], 'outcome_worker_unavailable')
        self.assertIsNone(result['acknowledged_count'])


if __name__ == '__main__':
    unittest.main()
