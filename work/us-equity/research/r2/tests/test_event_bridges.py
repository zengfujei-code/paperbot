from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperbot_us.regime_baselines import ReturnObservation, fit_baseline, initial_state
from paperbot_r2.contracts import digest, source_digest
from paperbot_r2.garch import fit_garch, initial_garch_state, observe_garch
from paperbot_r2.hmm import (GaussianParameters, HMMModel, HMMState,
                             initial_hmm_state, observe_hmm)
from paperbot_r2.event_bridges import (BridgeError, bridge_ewma, bridge_garch,
                                       bridge_hmm)


def normal_tail(mean, variance):
    scale = math.sqrt(2 * variance)
    return .5 * math.erfc((.005 - mean) / scale) + .5 * math.erfc((.005 + mean) / scale)


class EventBridgeTests(unittest.TestCase):
    def setUp(self):
        self.at = datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc)

    def rows(self, shift=0.):
        return tuple(ReturnObservation(self.at + timedelta(minutes=15 * i),
                                       self.at + timedelta(minutes=15 * i, seconds=1),
                                       shift + (.01 if i % 2 else -.01)) for i in range(64))

    def baseline(self, shift=0., decay=.94):
        rows = self.rows(shift)
        model = fit_baseline(rows, as_of=rows[-1].available_at, decay=decay,
                             cusum_k=.5, cusum_h=5.)
        return model, initial_state(model)

    def garch(self, alpha=.1, beta=.85, shift=.001):
        rows = self.rows(shift)
        model = fit_garch(rows, as_of=rows[-1].available_at, alpha=alpha, beta=beta)
        return model, initial_garch_state(model)

    def hmm(self, log_shift=0., row_shift=0.):
        parameters = GaussianParameters((.6, .4), ((.8 + row_shift, .2),
                                                     (.3, .7 + row_shift)),
                                        (-.01, .02), (.0001, .0004))
        model = HMMModel(parameters, (math.log(.75) + log_shift,
                                      math.log(.25) + log_shift), 64,
                         self.at, self.at + timedelta(seconds=1), '1' * 64,
                         source_digest('hmm'), 1, (0., 0.), 1e-7)
        state = initial_hmm_state(model)
        return model, state

    def test_ewma_uses_zero_mean_second_moment_for_both_unseen_steps(self):
        model, state = self.baseline(shift=.02)
        result = bridge_ewma(model, state)
        self.assertEqual(result.family, 'ewma')
        self.assertEqual(result.horizon_steps, 2)
        self.assertAlmostEqual(result.gap_variance, state.variance)
        self.assertAlmostEqual(result.target_variance, state.variance)
        self.assertAlmostEqual(result.probability, math.erfc(.005 / math.sqrt(2 * state.variance)))
        self.assertNotAlmostEqual(result.probability, normal_tail(model.training_mean, state.variance))
        self.assertEqual(result.state_sha256, digest(asdict(state)))
        self.assertEqual(result.target_weights, ())

    def test_garch_second_step_uses_expected_gap_shock_and_frozen_mean(self):
        model, state = self.garch()
        row = ReturnObservation(state.last_ends_at + timedelta(minutes=15),
                                state.last_ends_at + timedelta(minutes=15, seconds=1), .05)
        state = observe_garch(model, state, row, observed_at=row.available_at)
        result = bridge_garch(model, state)
        expected_variance = model.omega + (model.alpha + model.beta) * state.variance
        self.assertEqual(result.family, 'garch')
        self.assertAlmostEqual(result.gap_variance, state.variance)
        self.assertAlmostEqual(result.target_variance, expected_variance)
        self.assertAlmostEqual(result.probability, normal_tail(model.mean, expected_variance))
        self.assertNotAlmostEqual(result.target_variance, state.variance)
        self.assertEqual(result.target_weights, ())

    def test_hmm_two_transitions_and_asymmetric_component_tails(self):
        model, state = self.hmm()
        result = bridge_hmm(model, state)
        one = (.75 * .8 + .25 * .3, .75 * .2 + .25 * .7)
        two = (one[0] * .8 + one[1] * .3, one[0] * .2 + one[1] * .7)
        expected = sum(w * normal_tail(mean, var) for w, mean, var in
                       zip(two, model.parameters.means, model.parameters.variances))
        one_step = sum(w * normal_tail(mean, var) for w, mean, var in
                       zip(one, model.parameters.means, model.parameters.variances))
        self.assertEqual(result.family, 'hmm')
        self.assertAlmostEqual(result.target_weights[0], two[0])
        self.assertAlmostEqual(result.target_weights[1], two[1])
        self.assertAlmostEqual(result.probability, expected)
        self.assertNotAlmostEqual(result.probability, one_step)
        mean = sum(w * m for w, m in zip(two, model.parameters.means))
        expected_variance = sum(w * (v + (m - mean) ** 2) for w, v, m in
                                zip(two, model.parameters.variances, model.parameters.means))
        self.assertAlmostEqual(result.target_variance, expected_variance)

    def test_later_gap_observation_does_not_rewrite_an_earlier_result(self):
        model, state = self.hmm()
        original = bridge_hmm(model, state)
        row = ReturnObservation(state.last_ends_at + timedelta(minutes=15),
                                state.last_ends_at + timedelta(minutes=15, seconds=1), .08)
        later = observe_hmm(model, state, row, observed_at=row.available_at)
        updated = bridge_hmm(model, later)
        self.assertNotEqual(updated.seal, original.seal)
        self.assertNotEqual(updated.probability, original.probability)
        self.assertEqual(original, bridge_hmm(model, state))

    def test_hmm_accepts_models_inside_the_existing_normalization_tolerance(self):
        for changes in ({'log_shift': 1e-11}, {'row_shift': 9e-13}):
            with self.subTest(changes=changes):
                model, state = self.hmm(**changes)
                result = bridge_hmm(model, state)
                current = [math.exp(value) for value in state.log_posterior]
                current = [value / sum(current) for value in current]
                gap = [sum(current[i] * model.parameters.transition[i][j]
                           for i in range(2)) for j in range(2)]
                gap = [value / sum(gap) for value in gap]
                target = [sum(gap[i] * model.parameters.transition[i][j]
                              for i in range(2)) for j in range(2)]
                target = [value / sum(target) for value in target]
                expected = sum(w * normal_tail(mean, var) for w, mean, var in
                               zip(target, model.parameters.means, model.parameters.variances))
                self.assertAlmostEqual(sum(result.target_weights), 1., places=13)
                self.assertAlmostEqual(result.probability, expected, places=13)

    def test_extreme_represented_probabilities_keep_unclipped_log_values(self):
        model, _ = self.hmm()
        for mean, expected in ((0., 0.), (.1, 1.)):
            with self.subTest(mean=mean):
                parameters = replace(model.parameters, means=(mean, mean),
                                     variances=(1e-12, 1e-12))
                candidate = replace(model, parameters=parameters)
                result = bridge_hmm(candidate, initial_hmm_state(candidate))
                self.assertEqual(result.probability, expected)
                self.assertEqual(result.log_event_probability,
                                 -math.inf if expected == 0. else 0.)
                self.assertEqual(result.log_no_event_probability,
                                 -math.inf if expected == 1. else 0.)

    def test_wrong_or_nonfrozen_models_and_degenerate_variance_fail_closed(self):
        baseline, _ = self.baseline(decay=.9)
        with self.assertRaises(BridgeError):
            bridge_ewma(baseline, initial_state(baseline))
        zero, zero_state = self.baseline(shift=0.)
        zero = replace(zero, initial_variance=0., training_mean=0., training_sigma=0.)
        with self.assertRaises(BridgeError):
            bridge_ewma(zero, initial_state(zero))
        garch, state = self.garch()
        other, other_state = self.garch(alpha=.1, beta=.8)
        with self.assertRaises(BridgeError):
            bridge_garch(garch, other_state)
        with self.assertRaises(BridgeError):
            bridge_garch(other, other_state)
        hmm, hmm_state = self.hmm()
        with self.assertRaises(BridgeError):
            bridge_hmm(hmm, replace(hmm_state, model_sha256='0' * 64))

    def test_forged_seal_or_probability_fails_validation(self):
        model, state = self.garch()
        result = bridge_garch(model, state)
        with self.assertRaises(BridgeError):
            replace(result, seal='0' * 64)
        with self.assertRaises(BridgeError):
            replace(result, probability=.99)


if __name__ == '__main__':
    unittest.main()
