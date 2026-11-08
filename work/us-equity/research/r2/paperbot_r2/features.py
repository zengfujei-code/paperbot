"""Causal archived-bar feature/label adapter. No provider or order access.

Receipt truth and archive authentication remain upstream responsibilities.
Offline replay with these functions is not an actual historical live forecast.
"""
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from fractions import Fraction
import hashlib
import math
from pathlib import Path

from paperbot_us import data, regime_baselines
from . import classifier
from .classifier import FeatureVector, ClassifierExample
from .contracts import aware, digest, is_sha, need, number

FEATURE_NAMES = ('trend_fraction', 'volatility_fraction', 'iex_volume_ratio')
POLICIES = ('archived_receipts', 'synthetic_fixture')


def _source():
    paths = (Path(__file__), Path(classifier.__file__), Path(__file__).with_name('contracts.py'),
             Path(data.__file__), Path(regime_baselines.__file__))
    need(all(p.is_file() and not p.is_symlink() and p.stat().st_size <= 524288 for p in paths))
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def _seal(value):
    return digest({k: v for k, v in asdict(value).items() if k != 'seal'})


def _policy(value):
    need(value in POLICIES, 'retrospective_availability_not_receipt_proof')


def _identity(bar):
    return (bar.symbol, bar.source, bar.feed, bar.adjustment, bar.interval_seconds, bar.currency, bar.volume_unit)


@dataclass(frozen=True)
class FeatureBundle:
    vector: FeatureVector
    price_identity: tuple
    issued_at: datetime
    input_available_at: datetime
    target_start: datetime
    target_end: datetime
    session_day: str
    session_open: datetime
    session_close: datetime
    lookback_returns: int
    availability_policy: str
    record_sha256: str
    bar_sha256: tuple
    calendar_sha256: str
    source_sha256: str
    seal: str

    def __post_init__(self):
        need(type(self.vector) is FeatureVector and self.vector.feature_names == FEATURE_NAMES)
        _policy(self.availability_policy)
        need(type(self.lookback_returns) is int and 2 <= self.lookback_returns <= 20)
        need(type(self.price_identity) is tuple and len(self.price_identity) == 7
             and self.price_identity[1:] == ('alpaca', 'iex', 'raw', 900, 'USD', 'shares'))
        need(type(self.bar_sha256) is tuple and len(self.bar_sha256) == self.lookback_returns + 1
             and len(set(self.bar_sha256)) == len(self.bar_sha256) and all(is_sha(v) for v in self.bar_sha256))
        for name in ('issued_at', 'input_available_at', 'target_start', 'target_end', 'session_open', 'session_close'):
            object.__setattr__(self, name, aware(getattr(self, name)))
        need(self.vector.available_at == self.issued_at >= self.input_available_at >= self.vector.event_at
             and 0 <= (self.issued_at - self.vector.event_at).total_seconds() <= 120
             and self.target_start == self.vector.event_at + timedelta(seconds=900)
             and self.target_end == self.target_start + timedelta(seconds=900)
             and self.session_open < self.vector.event_at <= self.issued_at < self.target_start
             and self.target_end <= self.session_close)
        need(is_sha(self.record_sha256) and is_sha(self.calendar_sha256) and self.source_sha256 == _source()
             and self.seal == _seal(self), 'invalid_feature_bundle')


def build_features(bars, *, calendar, issued_at, lookback_returns, availability_policy, record_sha256):
    _policy(availability_policy)
    need(type(lookback_returns) is int and 2 <= lookback_returns <= 20)
    need(type(bars) in (list, tuple) and len(bars) == lookback_returns + 1
         and all(type(b) is data.Bar for b in bars) and type(calendar) is data.Calendar
         and is_sha(record_sha256), 'invalid_feature_window')
    at = aware(issued_at)
    need(bars[0].feed == 'iex' and all(b.volume > 0 for b in bars), 'missing_venue_volume')
    try:
        data.validate_series(bars, as_of=at, calendar=calendar, max_age_seconds=120)
        session = calendar.at(bars[-1].starts)
        need(calendar.at(at) == session and all(calendar.at(b.starts) == session for b in bars))
        target_start, target_end = bars[-1].ends + timedelta(seconds=900), bars[-1].ends + timedelta(seconds=1800)
        need(target_end <= session.closes, 'target_outside_regular_session')
    except (data.DataQualityError, OverflowError):
        need(False, 'invalid_or_unavailable_feature_window')
    returns = tuple(number(b.close / a.close - 1., minimum=-1.) for a, b in zip(bars, bars[1:]))
    mean = math.fsum(returns) / lookback_returns
    volatility = math.sqrt(math.fsum((r - mean) ** 2 for r in returns) / lookback_returns)
    # Normalize by a prior volume before averaging: at least one term is one,
    # including when every positive input is a subnormal float.
    volume_scale = max(b.volume for b in bars[:-1])
    scaled_mean = math.fsum((b.volume / volume_scale) / lookback_returns for b in bars[:-1])
    volume_ratio = number((bars[-1].volume / volume_scale) / scaled_mean - 1.)
    values = (bars[-1].close / bars[0].close - 1., volatility, volume_ratio)
    source = _source()
    calendar_hash = digest(dict(start=calendar.start.isoformat(), end=calendar.end.isoformat(),
        sessions=[dict(day=s.day.isoformat(), opens=s.opens, closes=s.closes)
                  for _, s in sorted(calendar.sessions.items())]))
    bar_hashes = tuple(digest(asdict(b)) for b in bars)
    stream = digest(dict(source_sha256=source, price_identity=_identity(bars[0]),
        feature_names=FEATURE_NAMES, lookback_returns=lookback_returns, interval_seconds=900,
        availability_policy=availability_policy, target_gap_seconds=900))
    vector = FeatureVector(stream_id=stream, feature_names=FEATURE_NAMES, values=values,
        event_at=bars[-1].ends, available_at=at, interval_seconds=900,
        source_id=digest(dict(stream=stream, bars=bar_hashes, record_sha256=record_sha256,
                             calendar_sha256=calendar_hash, issued_at=at)))
    fields = dict(vector=vector, price_identity=_identity(bars[0]), issued_at=at,
        input_available_at=max(b.received_at for b in bars), target_start=target_start, target_end=target_end,
        session_day=session.day.isoformat(), session_open=session.opens, session_close=session.closes,
        lookback_returns=lookback_returns, availability_policy=availability_policy,
        record_sha256=record_sha256, bar_sha256=bar_hashes, calendar_sha256=calendar_hash, source_sha256=source)
    return FeatureBundle(**fields, seal=digest({k: asdict(v) if k == 'vector' else v for k, v in fields.items()}))


def label_example(bundle, target_bar, *, record_sha256, threshold_fraction, observed_at, availability_policy):
    need(type(bundle) is FeatureBundle and type(target_bar) is data.Bar)
    _policy(availability_policy)
    need(bundle.source_sha256 == _source() and bundle.seal == _seal(bundle)
         and availability_policy == bundle.availability_policy and is_sha(record_sha256), 'feature_label_binding_mismatch')
    at = aware(observed_at)
    threshold = number(threshold_fraction, minimum=0., maximum=.2)
    need(threshold > 0 and _identity(target_bar) == bundle.price_identity
         and target_bar.starts == bundle.target_start and target_bar.ends == bundle.target_end
         and target_bar.volume > 0, 'invalid_regime_target')
    try:
        session = data.Session(date.fromisoformat(bundle.session_day), bundle.session_open, bundle.session_close)
        target_bar.validate(as_of=at, session=session)
    except (data.DataQualityError, ValueError):
        need(False, 'unavailable_regime_target')
    schema = digest(dict(definition='abs(close-open)>=threshold_fraction*open',
        threshold_fraction=threshold, price_identity=bundle.price_identity,
        interval_seconds=900, target_basis='venue_bar_open_to_close',
        arithmetic='exact_rationals_of_canonical_decimal_float_strings', source_sha256=_source()))
    # Preserve decimal threshold equality (100 -> 100.5 at .005) without a
    # tolerance, cancellation or underflow turning a flat price into class1.
    opened, closed, cutoff = (Fraction(str(v)) for v in (target_bar.open, target_bar.close, threshold))
    label = int(abs(closed - opened) >= cutoff * opened)
    return ClassifierExample(features=bundle.vector, label_schema_id=schema,
        target_start=bundle.target_start, target_end=bundle.target_end, outcome_available_at=at,
        label=label, outcome_id=digest(dict(bar=asdict(target_bar), record_sha256=record_sha256,
            bundle_sha256=bundle.seal, observed_at=at, label_schema_id=schema)))
