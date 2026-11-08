"""Offline, frozen-reference drift warnings; no trading or provider access.

Inputs are dimensionless declared features or already matured forecast errors.
Timestamps and IDs express caller-supplied provenance; they do not prove it.
"""
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import math
from pathlib import Path

from paperbot_us import data, regime_baselines
from .contracts import aware, digest, is_sha, need, number


def _source():
    paths = (Path(__file__), Path(__file__).with_name('contracts.py'),
             Path(data.__file__), Path(regime_baselines.__file__))
    need(all(p.is_file() and not p.is_symlink() and p.stat().st_size <= 524288 for p in paths))
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def _seal(value):
    return digest({k: v for k, v in asdict(value).items() if k != 'seal'})


@dataclass(frozen=True)
class DriftPoint:
    kind: str
    stream_id: str
    interval_seconds: int
    value: float
    event_at: datetime
    available_at: datetime
    source_id: str
    target_start: datetime | None = None
    prediction_at: datetime | None = None

    def __post_init__(self):
        need(self.kind in ('feature', 'forecast_error') and is_sha(self.source_id) and is_sha(self.stream_id))
        need(type(self.interval_seconds) is int and 1 <= self.interval_seconds <= 86400,
             'invalid_drift_interval')
        object.__setattr__(self, 'value', number(self.value))
        for name in ('event_at', 'available_at'):
            object.__setattr__(self, name, aware(getattr(self, name)))
        need(self.available_at >= self.event_at, 'point_not_yet_available')
        if self.kind == 'feature':
            need(self.target_start is None and self.prediction_at is None)
        else:
            object.__setattr__(self, 'target_start', aware(self.target_start))
            object.__setattr__(self, 'prediction_at', aware(self.prediction_at))
            need(self.prediction_at <= self.target_start < self.event_at,
                 'forecast_not_issued_before_target')
            need((self.event_at - self.target_start).total_seconds() == self.interval_seconds,
                 'forecast_interval_mismatch')


@dataclass(frozen=True)
class DriftModel:
    kind: str
    stream_id: str
    interval_seconds: int
    mean: float
    sigma: float
    slack: float
    threshold: float
    fault_z: float
    fitted_at: datetime
    last_event_at: datetime
    last_available_at: datetime
    reference_ids: tuple
    training_sha256: str
    source_sha256: str
    seal: str

    def __post_init__(self):
        need(self.kind in ('feature', 'forecast_error') and is_sha(self.stream_id))
        need(type(self.interval_seconds) is int and 1 <= self.interval_seconds <= 86400)
        for name in ('mean', 'sigma', 'slack', 'threshold', 'fault_z'):
            number(getattr(self, name), minimum=-100. if name == 'mean' else 0., maximum=1000.)
        need(self.sigma > 1e-12 and 0 <= self.slack <= 5 and 1 <= self.threshold <= 100
             and self.threshold < self.fault_z <= 1000)
        need(aware(self.fitted_at) >= aware(self.last_available_at) >= aware(self.last_event_at))
        need(type(self.reference_ids) is tuple and 32 <= len(self.reference_ids) <= 2000
             and len(set(self.reference_ids)) == len(self.reference_ids)
             and all(is_sha(x) for x in self.reference_ids))
        need(is_sha(self.training_sha256) and self.source_sha256 == _source()
             and is_sha(self.seal) and self.seal == _seal(self), 'invalid_drift_model')


@dataclass(frozen=True)
class DriftState:
    model_sha256: str
    last_event_at: datetime
    last_available_at: datetime
    last_observed_at: datetime
    upper: float
    lower: float
    alert: bool
    data_fault: bool
    fault_count: int
    steps: int
    seen_ids: tuple
    seal: str

    def __post_init__(self):
        need(is_sha(self.model_sha256) and aware(self.last_observed_at) >= aware(self.last_available_at)
             >= aware(self.last_event_at))
        for value in (self.upper, self.lower):
            number(value, minimum=0., maximum=1e7)
        need(type(self.alert) is bool and type(self.data_fault) is bool
             and type(self.fault_count) is int and type(self.steps) is int
             and 0 <= self.fault_count <= self.steps <= 4096)
        need(type(self.seen_ids) is tuple and len(self.seen_ids) <= 4096
             and len(set(self.seen_ids)) == len(self.seen_ids)
             and all(is_sha(x) for x in self.seen_ids))
        need(is_sha(self.seal) and self.seal == _seal(self), 'invalid_drift_state')


def _make(cls, **fields):
    fields['seal'] = digest(fields)
    return cls(**fields)


def fit_drift(points, *, fitted_at, slack, threshold, fault_z):
    at = aware(fitted_at)
    slack = number(slack, minimum=0., maximum=5.)
    threshold = number(threshold, minimum=1., maximum=100.)
    fault_z = number(fault_z, minimum=0., maximum=1000.)
    need(threshold < fault_z and type(points) in (list, tuple) and 32 <= len(points) <= 2000)
    previous = None
    ids = []
    for point in points:
        need(type(point) is DriftPoint and point.available_at <= at)
        if previous is not None:
            need(point.kind == previous.kind and point.stream_id == previous.stream_id
                 and point.interval_seconds == previous.interval_seconds
                 and point.event_at > previous.event_at
                 and point.available_at >= previous.available_at)
            if point.kind == 'forecast_error':
                need(point.target_start >= previous.event_at, 'overlapping_forecast_targets')
        ids.append(point.source_id)
        previous = point
    need(len(set(ids)) == len(ids), 'duplicate_source')
    mean = math.fsum(point.value for point in points) / len(points)
    variance = math.fsum((point.value - mean) ** 2 for point in points) / len(points)
    sigma = math.sqrt(variance)
    need(math.isfinite(sigma) and sigma > 1e-12, 'unidentifiable_reference')
    return _make(DriftModel, kind=points[0].kind, stream_id=points[0].stream_id,
                 interval_seconds=points[0].interval_seconds, mean=mean, sigma=sigma, slack=slack,
                 threshold=threshold, fault_z=fault_z, fitted_at=at,
                 last_event_at=points[-1].event_at, last_available_at=points[-1].available_at, reference_ids=tuple(ids),
                 training_sha256=digest([asdict(point) for point in points]), source_sha256=_source())


def start_drift(model):
    need(type(model) is DriftModel and model.source_sha256 == _source() and model.seal == _seal(model))
    return _make(DriftState, model_sha256=model.seal, last_event_at=model.last_event_at,
                 last_available_at=model.last_available_at,
                 last_observed_at=model.fitted_at, upper=0., lower=0., alert=False,
                 data_fault=False, fault_count=0, steps=0, seen_ids=model.reference_ids)


def observe_drift(model, state, point, *, observed_at):
    need(type(model) is DriftModel and type(state) is DriftState and type(point) is DriftPoint)
    need(model.source_sha256 == _source() and model.seal == _seal(model)
         and state.seal == _seal(state) and state.model_sha256 == model.seal,
         'drift_binding_mismatch')
    at = aware(observed_at)
    need(len(state.seen_ids) < 4096, 'drift_capacity_exhausted')
    need(point.kind == model.kind and point.stream_id == model.stream_id and point.event_at > state.last_event_at
         and point.interval_seconds == model.interval_seconds
         and point.available_at >= state.last_available_at and point.available_at <= at
         and at >= state.last_observed_at and point.source_id not in state.seen_ids
         and len(state.seen_ids) == len(model.reference_ids) + state.steps
         and state.seen_ids[:len(model.reference_ids)] == model.reference_ids,
         'unavailable_or_duplicate_drift_point')
    if point.kind == 'forecast_error':
        need(point.prediction_at >= model.fitted_at and point.target_start >= state.last_event_at,
             'forecast_outside_monitoring_period')
    else:
        need(point.event_at > model.fitted_at, 'feature_predates_monitor')
    z = (point.value - model.mean) / model.sigma
    need(math.isfinite(z))
    fault = abs(z) > model.fault_z
    upper = state.upper if fault else max(0., state.upper + z - model.slack)
    lower = state.lower if fault else max(0., state.lower - z - model.slack)
    return _make(DriftState, model_sha256=model.seal, last_event_at=point.event_at,
                 last_available_at=point.available_at,
                 last_observed_at=at, upper=upper, lower=lower,
                 alert=state.alert or upper >= model.threshold or lower >= model.threshold,
                 data_fault=state.data_fault or fault, fault_count=state.fault_count + int(fault), steps=state.steps + 1,
                 seen_ids=state.seen_ids + (point.source_id,))
