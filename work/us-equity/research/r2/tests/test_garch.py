from dataclasses import replace
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paperbot_us.regime_baselines import ReturnObservation
from paperbot_r2.contracts import ChallengerError, number
from paperbot_r2.garch import fit_garch, initial_garch_state, observe_garch, forecast_garch, normal_nll


class GarchTests(unittest.TestCase):
    def rows(self, count=64):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        return tuple(ReturnObservation(start + timedelta(minutes=15 * i),
                                       start + timedelta(minutes=15 * i, seconds=2),
                                       .01 if i % 2 else -.01) for i in range(count))

    def fit(self, **kwargs):
        rows = self.rows()
        return fit_garch(rows, as_of=rows[-1].available_at, alpha=.1, beta=.8, **kwargs)

    def next_row(self, rows, value=.02, delay=0):
        ends = rows[-1].ends_at + timedelta(minutes=15)
        return ReturnObservation(ends, ends + timedelta(seconds=delay), value)

    def test_independent_manual_recurrence_and_pre_outcome_score(self):
        model = self.fit()
        self.assertAlmostEqual(model.mean, 0.)
        self.assertAlmostEqual(model.omega, .00001)
        self.assertAlmostEqual(model.initial_variance, .0001)
        state = initial_garch_state(model)
        first = forecast_garch(model, state, issued_at=state.updated_at)
        self.assertAlmostEqual(first.variance, .0001)
        self.assertAlmostEqual(normal_nll(first, .02), .5 * (math.log(2 * math.pi * .0001) + 4))
        row = self.next_row(self.rows())
        state = observe_garch(model, state, row, observed_at=row.available_at)
        self.assertAlmostEqual(state.variance, .00013)
        row2 = ReturnObservation(row.ends_at + timedelta(minutes=15), row.ends_at + timedelta(minutes=15), -.03)
        state = observe_garch(model, state, row2, observed_at=row2.available_at)
        self.assertAlmostEqual(state.variance, .000204)
        self.assertEqual(state.observations, 2)

    def test_nonzero_mean_second_moment_is_not_centered_variance(self):
        rows = tuple(replace(r, return_fraction=r.return_fraction + .02) for r in self.rows())
        model = fit_garch(rows, as_of=rows[-1].available_at, alpha=.1, beta=.8)
        prediction = forecast_garch(model, initial_garch_state(model), issued_at=model.fitted_at)
        self.assertAlmostEqual(prediction.mean, .02)
        self.assertAlmostEqual(prediction.variance, .0001)
        self.assertAlmostEqual(prediction.second_moment, .0005)
        self.assertEqual(prediction.variance_unit, 'fraction_squared_per_15m')
        self.assertFalse(prediction.annualized)
        self.assertEqual(prediction.target_clock, 'next_observed_return_not_calendar_bound')

    def test_training_fit_matches_independent_population_variance_targeting(self):
        rows = tuple(replace(r, return_fraction=.001 * ((i % 7) - 2)) for i, r in enumerate(self.rows()))
        mean = sum(r.return_fraction for r in rows) / len(rows)
        variance = sum((r.return_fraction - mean) ** 2 for r in rows) / len(rows)
        expected = variance
        for row in rows:
            expected = .1 * variance + .2 * (row.return_fraction - mean) ** 2 + .7 * expected
        model = fit_garch(rows, as_of=rows[-1].available_at, alpha=.2, beta=.7)
        self.assertAlmostEqual(model.mean, mean)
        self.assertAlmostEqual(model.omega, .1 * variance)
        self.assertAlmostEqual(model.final_variance, expected)

    def test_deterministic_fit_and_model_bound_state(self):
        model = self.fit()
        self.assertEqual(model, self.fit())
        state = initial_garch_state(model)
        other = replace(model, alpha=.05, omega=(1 - .05 - .8) * model.initial_variance)
        with self.assertRaises(ChallengerError):
            forecast_garch(other, state, issued_at=state.updated_at)
        with self.assertRaises(ChallengerError):
            replace(model, source_sha256='0' * 64)

    def test_coefficients_and_finite_number_bounds(self):
        for alpha, beta in [(-.1, .8), (.2, -.1), (.2, .8), (0., .999), (True, .5), (math.nan, .5), (.1, math.inf)]:
            with self.subTest(alpha=alpha, beta=beta), self.assertRaises(ChallengerError):
                fit_garch(self.rows(), as_of=self.rows()[-1].available_at, alpha=alpha, beta=beta)

    def test_oversized_integer_uses_bounded_number_error(self):
        with self.assertRaises(ChallengerError) as failure:
            number(10 ** 1000)
        self.assertEqual(str(failure.exception), 'finite_bounded_number_required')

    def test_training_size_chronology_receipt_and_singularity(self):
        good = self.rows()
        cases = [good[:63], self.rows(2001), tuple(reversed(good)), good[:-1] + (good[-2],),
                 tuple(replace(r, return_fraction=0.) for r in good),
                 good[:-1] + (replace(good[-1], available_at=good[-1].available_at + timedelta(days=1)),)]
        for rows in cases:
            with self.subTest(count=len(rows)), self.assertRaises(ChallengerError):
                fit_garch(rows, as_of=good[-1].available_at, alpha=.1, beta=.8)

    def test_delayed_receipt_cannot_be_retroactively_used(self):
        model = self.fit()
        state = initial_garch_state(model)
        row = self.next_row(self.rows(), delay=86400)
        with self.assertRaises(ChallengerError):
            observe_garch(model, state, row, observed_at=row.ends_at)
        later = observe_garch(model, state, row, observed_at=row.available_at)
        self.assertEqual(later.updated_at, row.available_at)
        with self.assertRaises(ChallengerError):
            forecast_garch(model, later, issued_at=row.ends_at)

    def test_duplicate_overlapping_and_out_of_order_updates_are_rejected(self):
        model = self.fit()
        state = initial_garch_state(model)
        row = self.next_row(self.rows())
        state = observe_garch(model, state, row, observed_at=row.available_at)
        for old in (row, self.rows()[-1], replace(row, ends_at=row.ends_at + timedelta(seconds=1),
                                                available_at=row.available_at + timedelta(seconds=1))):
            with self.assertRaises(ChallengerError):
                observe_garch(model, state, old, observed_at=row.available_at + timedelta(hours=1))

    def test_earlier_receipt_change_changes_training_commitment(self):
        rows = self.rows()
        changed = tuple(replace(r, available_at=r.available_at + timedelta(seconds=1)) for r in rows)
        one = self.fit()
        two = fit_garch(changed, as_of=changed[-1].available_at, alpha=.1, beta=.8)
        self.assertNotEqual(one.training_sha256, two.training_sha256)
        self.assertEqual(one.final_variance, two.final_variance)

    def test_forecasts_are_prefix_invariant_and_do_not_mutate_fitted_parameters(self):
        model = self.fit()
        initial = initial_garch_state(model)
        old_forecast = forecast_garch(model, initial, issued_at=initial.updated_at)
        for value in (.02, -.2):
            row = self.next_row(self.rows(), value)
            later = observe_garch(model, initial, row, observed_at=row.available_at)
            self.assertNotEqual(later.variance, initial.variance)
            self.assertEqual(old_forecast, forecast_garch(model, initial, issued_at=initial.updated_at))
        self.assertEqual(model, self.fit())


if __name__ == '__main__':
    unittest.main()
