"""Fixed training fits on synthetic evidence only, never a market claim."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch
import zoneinfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_training_data import Fixture, Tick, ROOT, SYMBOLS, TZ
from paperbot_r2 import model_training as trainer
from paperbot_r2 import garch, hmm, classifier, training_data
from paperbot_r2.contracts import ChallengerError
from paperbot_us import regime_baselines


class ModelTrainingTests(unittest.TestCase):
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
        self.f = Fixture(Path(temp.name))

    def ready(self):
        self.f.finish()
        self.snapshot = self.f.read()
        return self.snapshot

    def fit(self, snapshot=None, clock=None):
        return trainer.fit_training_snapshot(snapshot or self.snapshot,
            timezone_file=TZ, clock=clock or Tick(self.f.at + timedelta(seconds=1)))

    def test_unready_snapshot_invokes_no_numeric_fit(self):
        value = self.f.read()
        with patch.object(garch, 'fit_garch', side_effect=AssertionError('no fit')) as fit:
            with self.assertRaises(trainer.ModelTrainingError):
                self.fit(value)
            fit.assert_not_called()

    def test_fifteen_fixed_attempts_include_failures_and_actual_completion(self):
        self.ready()
        with patch.object(regime_baselines, 'fit_baseline', wraps=regime_baselines.fit_baseline) as ewma_fit, \
             patch.object(garch, 'fit_garch', wraps=garch.fit_garch) as garch_fit, \
             patch.object(hmm, 'fit_hmm', wraps=hmm.fit_hmm) as hmm_fit, \
             patch.object(classifier, 'fit_classifier', wraps=classifier.fit_classifier) as nb_fit:
            fits = self.fit()
        self.assertEqual(len(fits.attempts), 15)
        self.assertEqual(len(set((a.symbol, a.family) for a in fits.attempts)), 15)
        self.assertEqual(ewma_fit.call_count, 3)
        self.assertTrue(all(c.kwargs['decay'] == .94 for c in ewma_fit.call_args_list))
        self.assertEqual(garch_fit.call_count, 3)
        self.assertTrue(all((c.kwargs['alpha'], c.kwargs['beta']) == (.1, .85) for c in garch_fit.call_args_list))
        self.assertEqual(hmm_fit.call_count, 3)
        self.assertTrue(all((c.kwargs['max_iterations'], c.kwargs['tolerance']) == (80, 1e-7)
                            for c in hmm_fit.call_args_list))
        self.assertEqual(nb_fit.call_count, 3)
        self.assertTrue(all(c.kwargs['variance_floor'] == 1e-6 for c in nb_fit.call_args_list))
        for attempt in fits.attempts:
            self.assertEqual(attempt.snapshot_sha256, self.snapshot.seal)
            self.assertGreater(attempt.completed_at, attempt.started_at)
            self.assertGreaterEqual(attempt.started_at, self.snapshot.available_at)
            if attempt.family == 'gaussian_nb':
                self.assertEqual(attempt.status, 'failed')
                self.assertIsNone(attempt.payload)
        self.assertGreaterEqual(fits.available_at, max(a.completed_at for a in fits.attempts))

    def test_prevalence_and_ewma_use_training_only_independent_arithmetic(self):
        self.ready(); result = self.fit()
        attempts = {a.family: a for a in result.attempts if a.symbol == 'AAPL'}
        self.assertEqual(attempts['prevalence'].payload.model.probability, 2. / 3.)
        values = tuple(r.return_fraction for r in self.snapshot.symbols[0].returns)
        initial = math.fsum(r*r for r in values) / len(values)
        variance = initial
        for r in values:
            variance = .94 * variance + .06 * r*r
        payload = attempts['ewma'].payload
        self.assertAlmostEqual(payload.model.initial_variance, initial, places=15)
        self.assertAlmostEqual(payload.state.variance, variance, places=15)
        self.assertEqual(payload.state.last_ends_at, self.snapshot.symbols[0].returns[-1].ends_at)

    def test_bound_model_rejects_other_symbol_other_snapshot_and_early_use(self):
        self.ready(); result = self.fit()
        attempt = next(a for a in result.attempts if a.symbol == 'AAPL' and a.family == 'garch')
        payload = trainer.bound_model(attempt, symbol='AAPL', snapshot_sha256=self.snapshot.seal,
                                      as_of=attempt.completed_at)
        self.assertIs(payload, attempt.payload)
        for symbol, sha, at in (('MSFT', self.snapshot.seal, attempt.completed_at),
                                ('AAPL', '0' * 64, attempt.completed_at),
                                ('AAPL', self.snapshot.seal, attempt.started_at)):
            with self.assertRaises(trainer.ModelTrainingError):
                trainer.bound_model(attempt, symbol=symbol, snapshot_sha256=sha, as_of=at)
        with self.assertRaises(trainer.ModelTrainingError):
            replace(attempt, symbol='MSFT')

    def test_failed_hmm_is_single_attempt_without_replacement_and_error_is_safe(self):
        self.ready()
        with patch.object(hmm, 'fit_hmm', side_effect=RuntimeError('sensitive arbitrary message')) as fit:
            result = self.fit()
        self.assertEqual(fit.call_count, 3)
        failures = [a for a in result.attempts if a.family == 'hmm']
        self.assertEqual(len(failures), 3)
        self.assertTrue(all(a.status == 'failed' and a.failure_code == 'fixed_fit_failed' for a in failures))
        self.assertNotIn('sensitive', str(result))

    def test_backdated_changed_source_and_cutoff_fits_are_rejected(self):
        snapshot = self.ready()
        with self.assertRaises(trainer.ModelTrainingError):
            self.fit(clock=lambda: snapshot.available_at - timedelta(seconds=1))
        with patch.object(training_data, 'source_sha256', return_value='0' * 64):
            with self.assertRaises(trainer.ModelTrainingError):
                self.fit()
        with self.assertRaises(trainer.ModelTrainingError):
            self.fit(clock=lambda: datetime(2026, 11, 10, 4, 55, tzinfo=timezone.utc))

    def test_clock_regression_during_fit_cannot_publish_partial_success(self):
        self.ready()
        at = self.f.at + timedelta(seconds=1)
        times = iter((at, at + timedelta(seconds=1), at))
        with self.assertRaises(trainer.ModelTrainingError):
            self.fit(clock=lambda: next(times))


if __name__ == '__main__':
    unittest.main()
