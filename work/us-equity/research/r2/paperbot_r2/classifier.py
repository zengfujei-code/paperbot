"""Frozen, offline Gaussian naive Bayes; class probabilities are not profit odds.

The caller defines classes, feature units, stream identity and point-in-time
provenance. No dataset/provider/broker, feature selection or online updates.
"""
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import math
from pathlib import Path
import re

from paperbot_us import data, regime_baselines
from .contracts import aware, digest, is_sha, need, number


def _source():
    paths = (Path(__file__), Path(__file__).with_name('contracts.py'),
             Path(data.__file__), Path(regime_baselines.__file__))
    need(all(p.is_file() and not p.is_symlink() and p.stat().st_size <= 524288 for p in paths))
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def _seal(value):
    return digest({k: v for k, v in asdict(value).items() if k != 'seal'})


def _make(cls, **fields):
    return cls(**fields, seal=digest(fields))


def _names(names):
    need(type(names) is tuple and 1 <= len(names) <= 8
         and all(type(n) is str and re.fullmatch('[a-z][a-z0-9_]{0,39}', n) for n in names)
         and len(set(names)) == len(names), 'invalid_feature_schema')


def _vector(values, size, *, minimum=-100., maximum=100.):
    need(type(values) is tuple and len(values) == size, 'invalid_feature_dimensions')
    return tuple(number(v, minimum=minimum, maximum=maximum) for v in values)


@dataclass(frozen=True)
class FeatureVector:
    stream_id: str
    feature_names: tuple
    values: tuple
    event_at: datetime
    available_at: datetime
    source_id: str
    interval_seconds: int

    def __post_init__(self):
        need(is_sha(self.stream_id) and is_sha(self.source_id))
        need(type(self.interval_seconds) is int and self.interval_seconds == 900)
        _names(self.feature_names)
        object.__setattr__(self, 'values', _vector(self.values, len(self.feature_names)))
        for name in ('event_at', 'available_at'):
            object.__setattr__(self, name, aware(getattr(self, name)))
        need(self.available_at >= self.event_at, 'feature_unavailable_at_event')


@dataclass(frozen=True)
class ClassifierExample:
    features: FeatureVector
    label_schema_id: str
    target_start: datetime
    target_end: datetime
    outcome_available_at: datetime
    label: int
    outcome_id: str

    def __post_init__(self):
        need(type(self.features) is FeatureVector and is_sha(self.label_schema_id)
             and is_sha(self.outcome_id) and type(self.label) is int and self.label in (0, 1))
        for name in ('target_start', 'target_end', 'outcome_available_at'):
            object.__setattr__(self, name, aware(getattr(self, name)))
        need((self.target_end - self.target_start).total_seconds() == self.features.interval_seconds
             and self.outcome_available_at >= self.target_end, 'invalid_label_interval')
        need(self.features.available_at <= self.target_start
             and 0 <= (self.target_start - self.features.event_at).total_seconds() <= 900,
             'unavailable_or_stale_training_features')


@dataclass(frozen=True)
class RegimeClassifier:
    stream_id: str
    label_schema_id: str
    feature_names: tuple
    center: tuple
    scale: tuple
    class_means: tuple
    class_variances: tuple
    floored: tuple
    priors: tuple
    class_counts: tuple
    variance_floor: float
    fitted_at: datetime
    last_target_end: datetime
    last_feature_event: datetime
    reference_ids: tuple
    outcome_ids: tuple
    training_sha256: str
    source_sha256: str
    seal: str
    interval_seconds: int = 900

    def __post_init__(self):
        _names(self.feature_names)
        width = len(self.feature_names)
        need(is_sha(self.stream_id) and is_sha(self.label_schema_id))
        need(type(self.interval_seconds) is int and self.interval_seconds == 900)
        _vector(self.center, width)
        _vector(self.scale, width, minimum=1e-12)
        number(self.variance_floor, minimum=1e-8, maximum=1.)
        for field in (self.class_means, self.class_variances, self.floored):
            need(type(field) is tuple and len(field) == 2)
        for values in self.class_means:
            _vector(values, width, minimum=-20., maximum=20.)
        for values in self.class_variances:
            _vector(values, width, minimum=self.variance_floor, maximum=1e4)
        need(all(type(row) is tuple and len(row) == width
                 and all(type(v) is bool for v in row) for row in self.floored))
        need(type(self.class_counts) is tuple and len(self.class_counts) == 2
             and all(type(n) is int and 16 <= n <= 2000 for n in self.class_counts)
             and 64 <= sum(self.class_counts) <= 2000)
        _vector(self.priors, 2, minimum=0., maximum=1.)
        need(all(p == n / sum(self.class_counts) for p, n in zip(self.priors, self.class_counts)))
        for name in ('fitted_at', 'last_target_end', 'last_feature_event'):
            object.__setattr__(self, name, aware(getattr(self, name)))
        need(aware(self.fitted_at) >= aware(self.last_target_end) > aware(self.last_feature_event))
        for ids in (self.reference_ids, self.outcome_ids):
            need(type(ids) is tuple and len(ids) == sum(self.class_counts)
                 and len(set(ids)) == len(ids) and all(is_sha(x) for x in ids))
        need(is_sha(self.training_sha256) and self.source_sha256 == _source()
             and self.seal == _seal(self), 'invalid_classifier_model')


@dataclass(frozen=True)
class RegimePrediction:
    model_sha256: str
    source_sha256: str
    feature_sha256: str
    label_schema_id: str
    issued_at: datetime
    target_start: datetime
    target_end: datetime
    probabilities: tuple
    log_probabilities: tuple
    seal: str

    def __post_init__(self):
        need(all(is_sha(v) for v in (self.model_sha256, self.source_sha256,
                                    self.feature_sha256, self.label_schema_id)))
        for name in ('issued_at', 'target_start', 'target_end'):
            object.__setattr__(self, name, aware(getattr(self, name)))
        need(aware(self.issued_at) <= aware(self.target_start)
             and (aware(self.target_end) - aware(self.target_start)).total_seconds() == 900)
        _vector(self.probabilities, 2, minimum=0., maximum=1.)
        _vector(self.log_probabilities, 2, minimum=-1e16, maximum=0.)
        need(math.isclose(math.fsum(self.probabilities), 1., rel_tol=1e-12)
             and all(math.isclose(p, math.exp(lp), rel_tol=1e-12, abs_tol=1e-300)
                     for p, lp in zip(self.probabilities, self.log_probabilities))
             and self.source_sha256 == _source() and self.seal == _seal(self),
             'invalid_classifier_prediction')


def fit_classifier(rows, *, as_of, variance_floor):
    at = aware(as_of)
    floor = number(variance_floor, minimum=1e-8, maximum=1.)
    need(type(rows) in (list, tuple) and 64 <= len(rows) <= 2000)
    need(all(type(r) is ClassifierExample and r.outcome_available_at <= at for r in rows),
         'unavailable_training_label')
    first = rows[0]
    previous = None
    for row in rows:
        need(row.features.stream_id == first.features.stream_id
             and row.features.feature_names == first.features.feature_names
             and row.label_schema_id == first.label_schema_id, 'mixed_classifier_schema')
        if previous is not None:
            need(row.target_start >= previous.target_end
                 and row.features.event_at > previous.features.event_at, 'overlapping_training_targets')
        previous = row
    references = tuple(r.features.source_id for r in rows)
    outcomes = tuple(r.outcome_id for r in rows)
    need(len(set(references)) == len(rows) and len(set(outcomes)) == len(rows), 'duplicate_training_evidence')
    counts = tuple(sum(r.label == k for r in rows) for k in (0, 1))
    need(min(counts) >= 16, 'insufficient_class_evidence')
    width = len(first.features.values)
    center = tuple(math.fsum(r.features.values[j] for r in rows) / len(rows) for j in range(width))
    scale = tuple(math.sqrt(math.fsum((r.features.values[j] - center[j]) ** 2 for r in rows) / len(rows))
                  for j in range(width))
    need(min(scale) > 1e-12, 'constant_training_feature')
    standardized = tuple(tuple((v - m) / s for v, m, s in zip(r.features.values, center, scale)) for r in rows)
    need(all(abs(v) <= 20 for point in standardized for v in point), 'feature_outside_training_support')
    means, variances, floored = [], [], []
    for label in (0, 1):
        values = [point for point, row in zip(standardized, rows) if row.label == label]
        mean = tuple(math.fsum(point[j] for point in values) / len(values) for j in range(width))
        raw = tuple(math.fsum((point[j] - mean[j]) ** 2 for point in values) / len(values) for j in range(width))
        means.append(mean)
        variances.append(tuple(max(floor, v) for v in raw))
        floored.append(tuple(v < floor for v in raw))
    return _make(RegimeClassifier, stream_id=first.features.stream_id, label_schema_id=first.label_schema_id,
        feature_names=first.features.feature_names, center=center, scale=scale,
        class_means=tuple(means), class_variances=tuple(variances), floored=tuple(floored),
        priors=tuple(n / len(rows) for n in counts), class_counts=counts, variance_floor=floor,
        fitted_at=at, last_target_end=rows[-1].target_end, last_feature_event=rows[-1].features.event_at,
        reference_ids=references, outcome_ids=outcomes,
        training_sha256=digest([asdict(r) for r in rows]), source_sha256=_source(), interval_seconds=900)


def classify_regime(model, features, *, issued_at, target_start, target_end):
    need(type(model) is RegimeClassifier and type(features) is FeatureVector)
    need(model.source_sha256 == _source() and model.seal == _seal(model), 'classifier_binding_mismatch')
    issued, start, end = map(aware, (issued_at, target_start, target_end))
    need(features.stream_id == model.stream_id and features.feature_names == model.feature_names
         and features.source_id not in model.reference_ids and features.event_at > model.last_feature_event,
         'classifier_feature_binding_mismatch')
    need(issued >= model.fitted_at and features.available_at <= issued <= start
         and start >= model.last_target_end and (end - start).total_seconds() == model.interval_seconds
         and 0 <= (start - features.event_at).total_seconds() <= model.interval_seconds,
         'unavailable_or_stale_prediction_features')
    values = tuple((v - m) / s for v, m, s in zip(features.values, model.center, model.scale))
    need(all(abs(v) <= 20 for v in values), 'feature_outside_training_support')
    log_weights = tuple(math.log(prior) - .5 * math.fsum(
        math.log(2 * math.pi * variance) + (v - mean) ** 2 / variance
        for v, mean, variance in zip(values, means, variances))
        for prior, means, variances in zip(model.priors, model.class_means, model.class_variances))
    odds = log_weights[1] - log_weights[0]
    tail = math.log1p(math.exp(-abs(odds)))
    logs = (-max(0., odds) - tail, -max(0., -odds) - tail)
    return _make(RegimePrediction, model_sha256=model.seal, source_sha256=_source(),
        feature_sha256=digest(asdict(features)), label_schema_id=model.label_schema_id,
        issued_at=issued, target_start=start, target_end=end,
        probabilities=tuple(math.exp(v) for v in logs), log_probabilities=logs)


def classification_loss(prediction, example, *, observed_at):
    need(type(prediction) is RegimePrediction and type(example) is ClassifierExample)
    at = aware(observed_at)
    need(prediction.seal == _seal(prediction) and prediction.source_sha256 == _source()
         and prediction.feature_sha256 == digest(asdict(example.features))
         and prediction.label_schema_id == example.label_schema_id
         and prediction.target_start == example.target_start and prediction.target_end == example.target_end
         and at >= example.outcome_available_at, 'unavailable_or_mismatched_classifier_outcome')
    return dict(brier=(prediction.probabilities[1] - example.label) ** 2,
                negative_log_score=-prediction.log_probabilities[example.label],
                prediction_sha256=prediction.seal, outcome_sha256=digest(asdict(example)),
                observed_at=at)
