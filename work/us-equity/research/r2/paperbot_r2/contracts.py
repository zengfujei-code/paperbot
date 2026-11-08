"""Bounded numeric/causal contracts, independent of execution or credentials."""
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

from paperbot_us import data, regime_baselines
from paperbot_us.regime_baselines import ReturnObservation


class ChallengerError(ValueError):
    def __init__(self, code, *, diagnostics=None):
        super().__init__(code)
        self.diagnostics = diagnostics


def need(condition, code='invalid_challenger_evidence'):
    if not condition:
        raise ChallengerError(code)


def number(value, *, minimum=-100., maximum=100.):
    need(type(value) in (int, float) and minimum <= value <= maximum and math.isfinite(value),
         'finite_bounded_number_required')
    return float(value)


def aware(value):
    try:
        return data.utc_time(value)
    except Exception:
        raise ChallengerError('aware_time_required') from None


def json_value(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if type(value) is float:
        # The exact log of a zero probability is legitimate. Use a tagged JSON
        # value instead of nonstandard Infinity/NaN in sealed math artifacts.
        if value == -math.inf:
            return {'log_probability': 'negative_infinity'}
        need(math.isfinite(value), 'nonfinite_challenger_artifact')
        return value
    if type(value) in (tuple, list):
        return [json_value(v) for v in value]
    if type(value) is dict:
        need(all(type(k) is str for k in value))
        return {k: json_value(v) for k, v in value.items()}
    need(value is None or type(value) in (str, int, bool), 'unsupported_challenger_field')
    return value


def digest(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def is_sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def source_digest(module):
    need(module in ('garch', 'hmm'))
    paths = [Path(__file__), Path(__file__).with_name(module + '.py'),
             Path(__file__).with_name('__init__.py'), Path(data.__file__), Path(regime_baselines.__file__)]
    need(all(p.is_file() and not p.is_symlink() and p.stat().st_size <= 524288 for p in paths))
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def training_rows(rows, as_of):
    at = aware(as_of)
    need(type(rows) in (list, tuple) and 64 <= len(rows) <= 2000, 'bounded_training_rows_required')
    previous = None
    for row in rows:
        need(type(row) is ReturnObservation and row.available_at <= at
             and (previous is None or row.starts_at >= previous), 'unavailable_or_unordered_training')
        number(row.return_fraction, minimum=-.9999999999999999)
        previous = row.ends_at
    return tuple(rows), at


def row_update(row, *, last_ends_at, updated_at, observed_at):
    at = aware(observed_at)
    need(type(row) is ReturnObservation and row.available_at <= at and at >= updated_at
         and row.starts_at >= last_ends_at, 'unavailable_or_unordered_update')
    number(row.return_fraction, minimum=-.9999999999999999)
    return at


def rows_digest(rows):
    return digest([asdict(row) for row in rows])


@dataclass(frozen=True)
class RiskForecast:
    model_sha256: str
    issued_at: datetime
    last_observed_end: datetime
    mean: float
    variance: float
    second_moment: float
    variance_unit: str = 'fraction_squared_per_15m'
    annualized: bool = False
    target_clock: str = 'next_observed_return_not_calendar_bound'
    interval_seconds: int = 900

    def __post_init__(self):
        for name in ('issued_at', 'last_observed_end'):
            object.__setattr__(self, name, aware(getattr(self, name)))
        for name in ('mean', 'variance', 'second_moment'):
            object.__setattr__(self, name, number(getattr(self, name), maximum=1e6, minimum=-100. if name == 'mean' else 0.))
        need(is_sha(self.model_sha256) and self.issued_at >= self.last_observed_end and self.variance >= 1e-12
             and math.isclose(self.second_moment, self.variance + self.mean ** 2, rel_tol=1e-12, abs_tol=1e-24)
             and self.variance_unit == 'fraction_squared_per_15m' and self.annualized is False
             and self.target_clock == 'next_observed_return_not_calendar_bound'
             and type(self.interval_seconds) is int and self.interval_seconds == 900)


def normal_nll(forecast, value):
    need(type(forecast) is RiskForecast)
    value = number(value, minimum=-.9999999999999999)
    score = .5 * (math.log(2 * math.pi * forecast.variance) + (value - forecast.mean) ** 2 / forecast.variance)
    need(math.isfinite(score), 'nonfinite_challenger_score')
    return score
