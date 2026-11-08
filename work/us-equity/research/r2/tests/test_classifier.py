from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import math
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[1])]
from paperbot_r2.contracts import ChallengerError, digest
from paperbot_r2.classifier import (FeatureVector, ClassifierExample, fit_classifier,
                                   classify_regime, classification_loss)


class ClassifierTests(unittest.TestCase):
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    stream = sha256(b'synthetic-declared-feature-stream').hexdigest()
    labels = sha256(b'externally-defined-risk-class-v1').hexdigest()

    def example(self, i, x, label):
        start = self.base + timedelta(seconds=900 * i)
        features = FeatureVector(stream_id=self.stream, feature_names=('trend',),
            values=(x,), event_at=start - timedelta(seconds=3),
            available_at=start - timedelta(seconds=2), source_id=digest(['feature', i]),
            interval_seconds=900)
        return ClassifierExample(features=features, label_schema_id=self.labels,
            target_start=start, target_end=start + timedelta(seconds=900),
            outcome_available_at=start + timedelta(seconds=901), label=label,
            outcome_id=digest(['outcome', i]))

    def training(self):
        return tuple(self.example(i, (-3., -1., 1., 3.)[i % 4], int(i % 4 >= 2)) for i in range(64))

    def model(self):
        rows = self.training()
        return fit_classifier(rows, as_of=rows[-1].outcome_available_at, variance_floor=1e-6)

    def forecast(self, model, x=1., *, i=65):
        row = self.example(i, x, 1)
        issued = row.target_start - timedelta(seconds=1)
        return classify_regime(model, row.features, issued_at=issued,
                               target_start=row.target_start, target_end=row.target_end), row

    def test_independent_manual_moments_posterior_and_tie(self):
        model = self.model()
        self.assertEqual(model.class_counts, (32, 32))
        self.assertAlmostEqual(model.center[0], 0.)
        self.assertAlmostEqual(model.scale[0], math.sqrt(5.))
        for actual, expected in zip(model.class_means, (-2. / math.sqrt(5.), 2. / math.sqrt(5.))):
            self.assertAlmostEqual(actual[0], expected)
        self.assertAlmostEqual(model.class_variances[0][0], .2)
        self.assertAlmostEqual(model.class_variances[1][0], .2)
        prediction, _ = self.forecast(model)
        self.assertAlmostEqual(prediction.probabilities[1], 1. / (1. + math.exp(-4.)), places=12)
        tie, _ = self.forecast(model, x=0.)
        self.assertEqual(tie.probabilities, (.5, .5))

    def test_training_only_scaler_and_prefix_replay_are_frozen(self):
        model = self.model()
        before = asdict(model)
        prediction, row = self.forecast(model, 0.)
        self.forecast(model, 10., i=66)
        again = classify_regime(model, row.features, issued_at=prediction.issued_at,
            target_start=row.target_start, target_end=row.target_end)
        self.assertEqual(again, prediction)
        self.assertEqual(asdict(model), before)

    def test_future_feature_fit_and_late_outcome_are_rejected(self):
        rows = self.training()
        with self.assertRaises(ChallengerError):
            fit_classifier(rows, as_of=rows[-1].target_end, variance_floor=1e-6)
        row = self.example(65, 1., 1)
        with self.assertRaises(ChallengerError):
            replace(row, features=replace(row.features, available_at=row.target_start + timedelta(seconds=1)))
        model = self.model()
        for issued in (model.fitted_at - timedelta(seconds=1),
                       row.features.available_at - timedelta(seconds=1),
                       row.target_start + timedelta(seconds=1)):
            with self.subTest(issued=issued), self.assertRaises(ChallengerError):
                classify_regime(model, row.features, issued_at=issued,
                    target_start=row.target_start, target_end=row.target_end)

    def test_training_overlap_duplicate_feature_and_outcome_ids_are_rejected(self):
        rows = self.training()
        invalid = [rows[:-1] + (replace(rows[-1], features=replace(rows[-1].features, source_id=rows[0].features.source_id)),),
                   rows[:-1] + (replace(rows[-1], outcome_id=rows[0].outcome_id),),
                   tuple(reversed(rows)), rows[:63],
                   rows[:-1] + (self.example(62, 3., 1),)]
        for values in invalid:
            with self.subTest(last=values[-1].target_start), self.assertRaises(ChallengerError):
                fit_classifier(values, as_of=rows[-1].outcome_available_at, variance_floor=1e-6)

    def test_schema_label_identity_and_sampling_interval_are_bound(self):
        rows = self.training()
        changed = [replace(rows[-1], label_schema_id=digest('other-label-policy')),
                   replace(rows[-1], features=replace(rows[-1].features, stream_id=digest('other-stream'))),
                   replace(rows[-1], features=replace(rows[-1].features, feature_names=('volume',)))]
        for value in changed:
            with self.assertRaises(ChallengerError):
                fit_classifier(rows[:-1] + (value,), as_of=rows[-1].outcome_available_at, variance_floor=1e-6)
        with self.assertRaises(ChallengerError):
            replace(rows[-1].features, interval_seconds=60)
        model = self.model()
        row = self.example(65, 1., 1)
        for features in (replace(row.features, stream_id=digest('other-stream')),
                         replace(row.features, feature_names=('volume',))):
            with self.assertRaises(ChallengerError):
                classify_regime(model, features, issued_at=row.target_start,
                    target_start=row.target_start, target_end=row.target_end)

    def test_stale_features_wrong_target_duration_and_training_reuse_are_rejected(self):
        model = self.model()
        row = self.example(65, 1., 1)
        with self.assertRaises(ChallengerError):
            replace(row, features=replace(row.features, event_at=row.target_start - timedelta(seconds=901)))
        with self.assertRaises(ChallengerError):
            replace(row, target_end=row.target_end + timedelta(seconds=1))
        for features in (replace(row.features, event_at=row.target_start - timedelta(seconds=901)),
                         replace(row.features, source_id=self.training()[0].features.source_id)):
            with self.assertRaises(ChallengerError):
                classify_regime(model, features, issued_at=row.target_start,
                    target_start=row.target_start, target_end=row.target_end)

    def test_imbalanced_prior_and_within_class_variance_floor_are_explicit(self):
        rows = tuple(self.example(i, -1. if i < 48 else 1., int(i >= 48)) for i in range(64))
        model = fit_classifier(rows, as_of=rows[-1].outcome_available_at, variance_floor=1e-6)
        self.assertEqual(model.class_counts, (48, 16))
        self.assertEqual(model.priors, (.75, .25))
        self.assertEqual(model.floored, ((True,), (True,)))
        self.assertEqual(model.class_variances, ((1e-6,), (1e-6,)))
        prediction, _ = self.forecast(model, 1.)
        self.assertEqual(prediction.probabilities[1], 1.)
        self.assertTrue(all(math.isfinite(p) and p <= 0 for p in prediction.log_probabilities))

    def test_invalid_numbers_sizes_classes_and_constant_columns_fail_explicitly(self):
        rows = self.training()
        invalid = [tuple(replace(row, label=0) for row in rows),
                   tuple(replace(row, features=replace(row.features, values=(0.,))) for row in rows)]
        for values in invalid:
            with self.assertRaises(ChallengerError):
                fit_classifier(values, as_of=rows[-1].outcome_available_at, variance_floor=1e-6)
        for floor in (True, 0., math.nan, math.inf, 2.):
            with self.subTest(floor=floor), self.assertRaises(ChallengerError):
                fit_classifier(rows, as_of=rows[-1].outcome_available_at, variance_floor=floor)
        for value in (True, math.nan, math.inf, 10 ** 1000):
            with self.assertRaises(ChallengerError):
                replace(rows[0].features, values=(value,))
        for names, values in [(('x', 'x'), (1., 2.)), (('x',), (1., 2.)),
                              ((), ()), (('x',) * 9, (1.,) * 9)]:
            with self.assertRaises(ChallengerError):
                replace(rows[0].features, feature_names=names, values=values)
        with self.assertRaises(ChallengerError):
            replace(rows[0], label=True)

    def test_out_of_training_support_is_rejected_without_clipping(self):
        rows = self.training()
        small = tuple(replace(row, features=replace(row.features, values=(row.features.values[0] * .001,))) for row in rows)
        model = fit_classifier(small, as_of=small[-1].outcome_available_at, variance_floor=1e-6)
        with self.assertRaises(ChallengerError):
            self.forecast(model, 1.)

    def test_matured_scores_bind_complete_feature_target_and_label_policy(self):
        model = self.model()
        prediction, row = self.forecast(model)
        score = classification_loss(prediction, row, observed_at=row.outcome_available_at)
        self.assertAlmostEqual(score['brier'], (prediction.probabilities[1] - 1.) ** 2)
        self.assertAlmostEqual(score['negative_log_score'], -math.log(prediction.probabilities[1]))
        with self.assertRaises(ChallengerError):
            classification_loss(prediction, row, observed_at=row.target_end)
        changed = [replace(row, label_schema_id=digest('other-policy')),
                   replace(row, features=replace(row.features, values=(2.,))),
                   replace(row, target_start=row.target_start + timedelta(seconds=1), target_end=row.target_end + timedelta(seconds=1))]
        for item in changed:
            with self.assertRaises(ChallengerError):
                classification_loss(prediction, item, observed_at=item.outcome_available_at)

    def test_two_feature_joint_log_likelihood_matches_independent_calculation(self):
        rows = tuple(replace(row, features=replace(row.features,
            feature_names=('trend', 'duplicate_information'), values=row.features.values * 2)) for row in self.training())
        model = fit_classifier(rows, as_of=rows[-1].outcome_available_at, variance_floor=1e-6)
        row = self.example(65, 1., 1)
        features = replace(row.features, feature_names=('trend', 'duplicate_information'), values=(1., 1.))
        prediction = classify_regime(model, features, issued_at=row.target_start,
            target_start=row.target_start, target_end=row.target_end)
        self.assertAlmostEqual(prediction.probabilities[1], 1. / (1. + math.exp(-8.)), places=12)

    def test_model_prediction_seals_and_dependency_hashes(self):
        model = self.model()
        prediction, _ = self.forecast(model)
        with self.assertRaises(ChallengerError):
            replace(model, priors=(.9, .1))
        with self.assertRaises(ChallengerError):
            replace(prediction, probabilities=(.5, .5))
        import paperbot_r2.classifier as module
        import paperbot_r2.contracts as contracts
        from paperbot_us import data, regime_baselines
        paths = [Path(m.__file__) for m in (module, contracts, data, regime_baselines)]
        self.assertEqual(model.source_sha256, digest({p.name:sha256(p.read_bytes()).hexdigest() for p in paths}))

    def test_restored_iso_artifact_timestamps_remain_usable_and_canonical(self):
        model = self.model()
        restored = replace(model, fitted_at=model.fitted_at.isoformat(),
            last_target_end=model.last_target_end.isoformat(), last_feature_event=model.last_feature_event.isoformat())
        prediction, row = self.forecast(restored)
        restored_prediction = replace(prediction, issued_at=prediction.issued_at.isoformat(),
            target_start=prediction.target_start.isoformat(), target_end=prediction.target_end.isoformat())
        self.assertEqual(restored, model)
        self.assertEqual(restored_prediction, prediction)
        self.assertEqual(classification_loss(restored_prediction, row, observed_at=row.outcome_available_at),
                         classification_loss(prediction, row, observed_at=row.outcome_available_at))

    def test_near_minimum_scale_and_maximum_class_imbalance(self):
        rows = self.training()
        tiny = tuple(replace(row, features=replace(row.features,
            values=(row.features.values[0] * 1e-12,))) for row in rows)
        model = fit_classifier(tiny, as_of=tiny[-1].outcome_available_at, variance_floor=1e-6)
        prediction, _ = self.forecast(model, 1e-12)
        self.assertAlmostEqual(prediction.probabilities[1], 1. / (1. + math.exp(-4.)), places=12)
        imbalanced = tuple(self.example(i, (-1. if i < 1984 else 1.) + (.1 if i % 2 else -.1),
                                       int(i >= 1984)) for i in range(2000))
        model = fit_classifier(imbalanced, as_of=imbalanced[-1].outcome_available_at, variance_floor=1e-6)
        self.assertEqual(model.priors, (.992, .008))
        self.assertEqual(model.class_counts, (1984, 16))
        prediction, _ = self.forecast(model, 1., i=2001)
        self.assertTrue(all(math.isfinite(value) for value in prediction.log_probabilities))


if __name__ == '__main__':
    unittest.main()
