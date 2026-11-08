from dataclasses import replace
from datetime import datetime, timedelta, timezone
import itertools
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paperbot_us.regime_baselines import ReturnObservation
from paperbot_r2.contracts import ChallengerError
from paperbot_r2.hmm import (GaussianParameters, filter_hmm, predictive_hmm, fit_hmm,
                            initial_hmm_state, observe_hmm, forecast_hmm)


class HmmTests(unittest.TestCase):
    def parameters(self):
        return GaussianParameters((.6, .4), ((.8, .2), (.3, .7)), (-.01, .02), (.0001, .0004))

    def rows(self):
        rng = random.Random(508)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        values = [rng.gauss(0., .002 if (i // 48) % 2 == 0 else .025) for i in range(192)]
        return tuple(ReturnObservation(start + timedelta(minutes=15 * i),
                                       start + timedelta(minutes=15 * i, seconds=3), r)
                     for i, r in enumerate(values))

    def fit(self, **kwargs):
        rows = self.rows()
        return fit_hmm(rows, as_of=rows[-1].available_at, **kwargs)

    def test_forward_filter_matches_brute_force_paths_at_every_prefix(self):
        params = self.parameters()
        values = (-.012, .018, .004, .03)
        posterior = None
        log_likelihood = 0.
        for count, value in enumerate(values, 1):
            posterior, density = filter_hmm(params, posterior, value, initial=count == 1)
            log_likelihood += density
            weights = [0., 0.]
            for path in itertools.product(range(2), repeat=count):
                weight = params.initial[path[0]]
                for i, state in enumerate(path):
                    if i:
                        weight *= params.transition[path[i - 1]][state]
                    variance = params.variances[state]
                    weight *= math.exp(-.5 * (values[i] - params.means[state]) ** 2 / variance) / math.sqrt(2 * math.pi * variance)
                weights[path[-1]] += weight
            total = sum(weights)
            self.assertAlmostEqual(log_likelihood, math.log(total), places=11)
            for state in range(2):
                self.assertAlmostEqual(math.exp(posterior[state]), weights[state] / total, places=12)

    def test_predictive_mixture_variance_includes_between_state_means(self):
        params = self.parameters()
        mean, variance, moment = predictive_hmm(params, (math.log(.75), math.log(.25)))
        weights = (.75 * .8 + .25 * .3, .75 * .2 + .25 * .7)
        expected_mean = sum(p * m for p, m in zip(weights, params.means))
        expected_moment = sum(p * (v + m * m) for p, v, m in zip(weights, params.variances, params.means))
        self.assertAlmostEqual(mean, expected_mean)
        self.assertAlmostEqual(moment, expected_moment)
        self.assertAlmostEqual(variance, expected_moment - expected_mean ** 2)

    def test_log_filter_does_not_permanently_zero_a_small_state(self):
        params = GaussianParameters((.5, .5), ((1., 0.), (0., 1.)), (-.1, .1), (1e-8, 1e-8))
        log_post, _ = filter_hmm(params, None, -.1, initial=True)
        self.assertLess(log_post[1], -100000)
        recovered, score = filter_hmm(params, log_post, .1)
        self.assertAlmostEqual(math.exp(recovered[0]), .5, places=10)
        self.assertTrue(math.isfinite(score))

    def test_stochastic_and_numeric_contracts(self):
        good = self.parameters()
        for changes in [dict(initial=(.2, .2)), dict(transition=((.2, .2), (.3, .7))),
                        dict(variances=(0., .1)), dict(means=(math.nan, .1)), dict(initial=(True, 0.))]:
            with self.subTest(changes=changes), self.assertRaises(ChallengerError):
                replace(good, **changes)
        for posterior in [(0., 0.), (math.nan, 0.), (math.inf, 0.)]:
            with self.assertRaises(ChallengerError):
                filter_hmm(good, posterior, .01)

    def test_fit_is_deterministic_converged_and_train_likelihood_monotone(self):
        one = self.fit()
        self.assertEqual(one, self.fit())
        self.assertTrue(one.converged)
        self.assertEqual(one.iterations, len(one.log_likelihood_trace) - 1)
        self.assertLessEqual(one.iterations, 80)
        self.assertLessEqual(one.parameters.variances[0], one.parameters.variances[1])
        self.assertTrue(all(b >= a - 1e-7 for a, b in zip(one.log_likelihood_trace, one.log_likelihood_trace[1:])))

    def test_nonconvergence_has_bounded_retained_diagnostics(self):
        with self.assertRaises(ChallengerError) as failure:
            self.fit(max_iterations=1, tolerance=1e-10)
        self.assertEqual(str(failure.exception), 'hmm_not_converged')
        self.assertEqual(failure.exception.diagnostics['iterations'], 1)
        self.assertEqual(len(failure.exception.diagnostics['log_likelihood_trace']), 2)

    def test_training_bounds_singularity_and_future_receipts(self):
        rows = self.rows()
        cases = [rows[:63], tuple(replace(r, return_fraction=.01) for r in rows),
                 tuple(reversed(rows)), rows[:-1] + (replace(rows[-1], available_at=rows[-1].available_at + timedelta(days=1)),)]
        for values in cases:
            with self.assertRaises(ChallengerError):
                fit_hmm(values, as_of=rows[-1].available_at)
        for args in [dict(max_iterations=81), dict(max_iterations=True), dict(tolerance=math.nan), dict(tolerance=0.)]:
            with self.assertRaises(ChallengerError):
                self.fit(**args)

    def test_singular_fitted_components_are_rejected_before_variance_floor(self):
        rows = tuple(replace(row, return_fraction=0. if i < 128 else .02)
                     for i, row in enumerate(self.rows()))
        with self.assertRaises(ChallengerError) as failure:
            fit_hmm(rows, as_of=rows[-1].available_at)
        self.assertEqual(str(failure.exception), 'singular_hmm_component_variance')

    def test_causal_state_fit_identity_delayed_updates_and_duplicates(self):
        model = self.fit()
        state = initial_hmm_state(model)
        ends = state.last_ends_at + timedelta(minutes=15)
        row = ReturnObservation(ends, ends + timedelta(hours=1), .02)
        before = forecast_hmm(model, state, issued_at=state.updated_at)
        with self.assertRaises(ChallengerError):
            observe_hmm(model, state, row, observed_at=ends)
        later = observe_hmm(model, state, row, observed_at=row.available_at)
        self.assertEqual(before, forecast_hmm(model, state, issued_at=state.updated_at))
        self.assertEqual(later.updated_at, row.available_at)
        with self.assertRaises(ChallengerError):
            observe_hmm(model, later, row, observed_at=row.available_at)
        with self.assertRaises(ChallengerError):
            forecast_hmm(model, later, issued_at=ends)
        with self.assertRaises(ChallengerError):
            forecast_hmm(model, replace(later, model_sha256='0' * 64), issued_at=row.available_at)
        with self.assertRaises(ChallengerError):
            replace(model, source_sha256='0' * 64)

    def test_state_permutation_preserves_filter_and_predictive_distribution(self):
        params = self.parameters()
        reverse = GaussianParameters(tuple(reversed(params.initial)),
            tuple(tuple(reversed(row)) for row in reversed(params.transition)),
            tuple(reversed(params.means)), tuple(reversed(params.variances)))
        one, ll1 = filter_hmm(params, None, .01, initial=True)
        two, ll2 = filter_hmm(reverse, None, .01, initial=True)
        self.assertAlmostEqual(ll1, ll2)
        self.assertAlmostEqual(one[0], two[1])
        for a, b in zip(predictive_hmm(params, one), predictive_hmm(reverse, two)):
            self.assertAlmostEqual(a, b)

    def test_exact_zero_path_has_explicit_hashable_log_representation(self):
        model = replace(self.fit(), final_log_posterior=(-math.inf, 0.))
        state = initial_hmm_state(model)
        forecast = forecast_hmm(model, state, issued_at=state.updated_at)
        self.assertTrue(math.isfinite(forecast.variance))
        self.assertEqual(state.log_posterior, (-math.inf, 0.))


if __name__ == '__main__':
    unittest.main()
