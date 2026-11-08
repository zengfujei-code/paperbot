from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paperbot_r2.contracts import ChallengerError, digest
from paperbot_us import data, regime_baselines
import paperbot_r2.contracts as drift_contracts
import paperbot_r2.drift as drift_module
from paperbot_r2.drift import DriftPoint, DriftState, fit_drift, start_drift, observe_drift


class DriftTests(unittest.TestCase):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def point(self, i, value, *, kind='feature', delay=0, name=None):
        end = self.base + timedelta(minutes=15 * (i + 1))
        start = end - timedelta(minutes=15)
        return DriftPoint(kind=kind, stream_id=sha256(('stream:' + kind).encode()).hexdigest(),
                          interval_seconds=900, value=value, event_at=end,
                          available_at=end + timedelta(seconds=delay),
                          source_id=sha256(str(name if name is not None else i).encode()).hexdigest(),
                          target_start=start if kind == 'forecast_error' else None,
                          prediction_at=start if kind == 'forecast_error' else None)

    def training(self, kind='feature'):
        return tuple(self.point(i, 1. if i % 2 else -1., kind=kind) for i in range(32))

    def model(self, kind='feature'):
        rows = self.training(kind)
        return fit_drift(rows, fitted_at=rows[-1].available_at, slack=.5, threshold=2., fault_z=8.)

    def test_manual_upper_lower_recurrence_and_sticky_alert(self):
        model = self.model()
        self.assertAlmostEqual(model.mean, 0.)
        self.assertAlmostEqual(model.sigma, 1.)
        state = start_drift(model)
        for i, value in [(32, 1.5), (33, 1.5)]:
            point = self.point(i, value)
            state = observe_drift(model, state, point, observed_at=point.available_at)
        self.assertAlmostEqual(state.upper, 2.)
        self.assertAlmostEqual(state.lower, 0.)
        self.assertTrue(state.alert)
        point = self.point(34, -1.5)
        state = observe_drift(model, state, point, observed_at=point.available_at)
        self.assertAlmostEqual(state.upper, 0.)
        self.assertAlmostEqual(state.lower, 1.)
        self.assertTrue(state.alert)

    def test_stable_control_and_negative_shift(self):
        model = self.model()
        state = start_drift(model)
        for i in range(32, 52):
            point = self.point(i, 1. if i % 2 else -1.)
            state = observe_drift(model, state, point, observed_at=point.available_at)
        self.assertFalse(state.alert)
        for i in (52, 53):
            point = self.point(i, -1.5)
            state = observe_drift(model, state, point, observed_at=point.available_at)
        self.assertTrue(state.alert)
        self.assertGreaterEqual(state.lower, 2.)

    def test_extreme_point_is_recorded_as_data_fault_without_driving_drift(self):
        model = self.model()
        state = start_drift(model)
        bad = self.point(32, 9.)
        later = observe_drift(model, state, bad, observed_at=bad.available_at)
        self.assertTrue(later.data_fault)
        self.assertEqual(later.fault_count, 1)
        self.assertEqual((later.upper, later.lower), (0., 0.))
        self.assertFalse(later.alert)
        with self.assertRaises(ChallengerError):
            observe_drift(model, later, bad, observed_at=bad.available_at)

    def test_delayed_forecast_outcome_and_prediction_chronology(self):
        model = self.model('forecast_error')
        state = start_drift(model)
        point = self.point(32, .2, kind='forecast_error', delay=30)
        with self.assertRaises(ChallengerError):
            observe_drift(model, state, point, observed_at=point.event_at)
        state = observe_drift(model, state, point, observed_at=point.available_at)
        self.assertEqual(state.last_observed_at, point.available_at)
        with self.assertRaises(ChallengerError):
            replace(point, prediction_at=point.target_start + timedelta(seconds=1))
        with self.assertRaises(ChallengerError):
            replace(point, available_at=point.event_at - timedelta(seconds=1))

    def test_duplicates_overlaps_and_cross_stream_are_rejected(self):
        model = self.model('forecast_error')
        state = start_drift(model)
        point = self.point(32, .2, kind='forecast_error')
        state = observe_drift(model, state, point, observed_at=point.available_at)
        invalid = [point, self.point(33, .2, kind='feature'),
                   self.point(33, .2, kind='forecast_error', name=32),
                   replace(self.point(33, .2, kind='forecast_error'),
                           event_at=point.event_at + timedelta(minutes=15, seconds=-1),
                           target_start=point.event_at - timedelta(seconds=1),
                           prediction_at=point.event_at - timedelta(seconds=2))]
        for item in invalid:
            with self.subTest(kind=item.kind, at=item.event_at), self.assertRaises(ChallengerError):
                observe_drift(model, state, item, observed_at=item.available_at)

    def test_bad_training_parameters_and_nonfinite_values_fail_closed(self):
        rows = self.training()
        for values in [rows[:31], rows + (rows[-1],), tuple(reversed(rows)),
                       tuple(replace(r, value=0.) for r in rows),
                       rows[:-1] + (replace(rows[-1], available_at=rows[-1].available_at + timedelta(days=1)),)]:
            with self.subTest(count=len(values)), self.assertRaises(ChallengerError):
                fit_drift(values, fitted_at=rows[-1].available_at, slack=.5, threshold=2., fault_z=8.)
        for kwargs in [dict(slack=-1, threshold=2, fault_z=8), dict(slack=.5, threshold=2, fault_z=2),
                       dict(slack=True, threshold=2, fault_z=8), dict(slack=.5, threshold=math.inf, fault_z=8)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ChallengerError):
                fit_drift(rows, fitted_at=rows[-1].available_at, **kwargs)
        for value in (math.nan, math.inf, 10 ** 1000, True):
            with self.subTest(value=str(value)[:20]), self.assertRaises(ChallengerError):
                self.point(32, value)

    def test_model_state_binding_tamper_and_prefix_invariance(self):
        model = self.model()
        state = start_drift(model)
        point = self.point(32, 1.5)
        prefix = observe_drift(model, state, point, observed_at=point.available_at)
        next_point = self.point(33, 1.5)
        observe_drift(model, prefix, next_point, observed_at=next_point.available_at)
        self.assertEqual(prefix, observe_drift(model, state, point, observed_at=point.available_at))
        with self.assertRaises(ChallengerError):
            replace(model, mean=5.)
        with self.assertRaises(ChallengerError):
            replace(prefix, upper=0.)
        other_rows = tuple(replace(p, value=p.value + .1) for p in self.training())
        other = fit_drift(other_rows, fitted_at=other_rows[-1].available_at, slack=.5, threshold=2., fault_z=8.)
        with self.assertRaises(ChallengerError):
            observe_drift(other, prefix, next_point, observed_at=next_point.available_at)

    def test_source_identity_includes_shared_time_contract_dependencies(self):
        paths = [Path(drift_module.__file__), Path(drift_contracts.__file__),
                 Path(data.__file__), Path(regime_baselines.__file__)]
        expected = digest({p.name: sha256(p.read_bytes()).hexdigest() for p in paths})
        self.assertEqual(self.model().source_sha256, expected)

    def test_monitor_never_reuses_events_or_predictions_from_before_fitting(self):
        for kind in ('feature', 'forecast_error'):
            rows = self.training(kind)
            fitted = rows[-1].available_at + timedelta(hours=1)
            model = fit_drift(rows, fitted_at=fitted, slack=.5, threshold=2., fault_z=8.)
            point = self.point(35, .1, kind=kind)
            with self.subTest(kind=kind), self.assertRaises(ChallengerError):
                observe_drift(model, start_drift(model), point, observed_at=point.available_at)

    def test_same_kind_different_stream_is_rejected_in_fit_and_monitor(self):
        rows = self.training()
        self.assertTrue(hasattr(rows[0], 'stream_id'), 'samples must identify their stream')
        other_stream = sha256(b'other-symbol-or-model').hexdigest()
        mixed = rows[:-1] + (replace(rows[-1], stream_id=other_stream),)
        with self.assertRaises(ChallengerError):
            fit_drift(mixed, fitted_at=rows[-1].available_at, slack=.5, threshold=2., fault_z=8.)
        point = replace(self.point(32, .1), stream_id=other_stream)
        model = self.model()
        with self.assertRaises(ChallengerError):
            observe_drift(model, start_drift(model), point, observed_at=point.available_at)

    def test_data_fault_latches_across_repeated_extremes_and_normal_input(self):
        model = self.model()
        state = start_drift(model)
        for i, value in [(32, 9.), (33, 9.), (34, 0.)]:
            point = self.point(i, value)
            state = observe_drift(model, state, point, observed_at=point.available_at)
        self.assertTrue(state.data_fault)
        self.assertEqual(state.fault_count, 2)
        self.assertFalse(state.alert)

    def test_restart_batch_uses_actual_receipt_order_without_backdating_alert(self):
        model = self.model()
        state = start_drift(model)
        first, second = self.point(32, 1.5), self.point(33, 1.5)
        processed = second.available_at + timedelta(minutes=30)
        state = observe_drift(model, state, first, observed_at=processed)
        state = observe_drift(model, state, second, observed_at=processed)
        self.assertTrue(state.alert)
        self.assertEqual(state.last_observed_at, processed)
        self.assertEqual(state.last_available_at, second.available_at)

    def test_capacity_exhaustion_is_explicit_and_does_not_reset_history(self):
        model = self.model()
        state = start_drift(model)
        extras = tuple(sha256(('extra-' + str(i)).encode()).hexdigest() for i in range(4096 - len(state.seen_ids)))
        payload = asdict(state)
        payload.pop('seal')
        payload.update(seen_ids=state.seen_ids + extras, steps=len(extras))
        full = DriftState(**payload, seal=digest(payload))
        point = self.point(32, .1)
        with self.assertRaises(ChallengerError) as failure:
            observe_drift(model, full, point, observed_at=point.available_at)
        self.assertEqual(str(failure.exception), 'drift_capacity_exhausted')
        self.assertEqual(len(full.seen_ids), 4096)

    def test_forecast_duration_and_frozen_interval_cannot_be_mixed(self):
        point = self.point(32, .1, kind='forecast_error')
        with self.assertRaises(ChallengerError):
            replace(point, target_start=point.event_at - timedelta(seconds=1),
                    prediction_at=point.event_at - timedelta(seconds=1))
        short = replace(point, interval_seconds=1,
                        target_start=point.event_at - timedelta(seconds=1),
                        prediction_at=point.event_at - timedelta(seconds=1))
        model = self.model('forecast_error')
        with self.assertRaises(ChallengerError):
            observe_drift(model, start_drift(model), short, observed_at=short.available_at)
        rows = self.training('forecast_error')
        last = replace(rows[-1], interval_seconds=1,
                       target_start=rows[-1].event_at - timedelta(seconds=1),
                       prediction_at=rows[-1].event_at - timedelta(seconds=1))
        with self.assertRaises(ChallengerError):
            fit_drift(rows[:-1] + (last,), fitted_at=rows[-1].available_at,
                      slack=.5, threshold=2., fault_z=8.)
        for interval in (True, 0, 86401, 1.5):
            with self.subTest(interval=interval), self.assertRaises(ChallengerError):
                replace(point, interval_seconds=interval)


if __name__ == '__main__':
    unittest.main()
