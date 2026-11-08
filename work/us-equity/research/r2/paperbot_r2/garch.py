"""Variance-targeted GARCH(1,1), one declared coefficient pair per fit.

No hidden grid, optimization restart, annualization or execution authority.
Updates count observed returns, not elapsed wall time or overnight risk.
"""
from dataclasses import asdict, dataclass
from datetime import datetime
import math
import statistics

from .contracts import (RiskForecast, aware, digest, is_sha, need, normal_nll,
                        number, row_update, rows_digest, source_digest, training_rows)


@dataclass(frozen=True)
class GarchModel:
    mean: float
    alpha: float
    beta: float
    omega: float
    initial_variance: float
    final_variance: float
    training_count: int
    training_ends_at: datetime
    fitted_at: datetime
    training_sha256: str
    source_sha256: str
    formula_version: str = 'variance-targeted-garch11-v1'

    def __post_init__(self):
        for name in ('mean', 'alpha', 'beta', 'omega', 'initial_variance', 'final_variance'):
            object.__setattr__(self, name, number(getattr(self, name), minimum=-100. if name == 'mean' else 0., maximum=1e6))
        for name in ('training_ends_at', 'fitted_at'):
            object.__setattr__(self, name, aware(getattr(self, name)))
        need(self.alpha + self.beta < .999 and min(self.initial_variance, self.final_variance) >= 1e-12
             and self.omega > 0 and math.isclose(self.omega, (1 - self.alpha - self.beta) * self.initial_variance, rel_tol=1e-12)
             and type(self.training_count) is int and 64 <= self.training_count <= 2000
             and self.training_ends_at <= self.fitted_at and is_sha(self.training_sha256)
             and self.source_sha256 == source_digest('garch') and self.formula_version == 'variance-targeted-garch11-v1')


def fit_garch(rows, *, as_of, alpha, beta):
    rows, at = training_rows(rows, as_of)
    alpha = number(alpha, minimum=0., maximum=1.)
    beta = number(beta, minimum=0., maximum=1.)
    need(alpha + beta < .999, 'nonstationary_garch_parameters')
    values = tuple(row.return_fraction for row in rows)
    mean = statistics.fmean(values)
    variance = statistics.fmean((v - mean) ** 2 for v in values)
    need(variance >= 1e-12, 'singular_training_variance')
    omega = (1 - alpha - beta) * variance
    final = variance
    for value in values:
        final = omega + alpha * (value - mean) ** 2 + beta * final
        need(math.isfinite(final) and final >= 1e-12)
    return GarchModel(mean, alpha, beta, omega, variance, final, len(rows), rows[-1].ends_at,
                      at, rows_digest(rows), source_digest('garch'))


@dataclass(frozen=True)
class GarchState:
    model_sha256: str
    variance: float
    last_ends_at: datetime
    updated_at: datetime
    observations: int

    def __post_init__(self):
        object.__setattr__(self, 'variance', number(self.variance, minimum=1e-12, maximum=1e6))
        for name in ('last_ends_at', 'updated_at'):
            object.__setattr__(self, name, aware(getattr(self, name)))
        need(is_sha(self.model_sha256) and self.last_ends_at <= self.updated_at
             and type(self.observations) is int and 0 <= self.observations <= 100000)


def _identity(model):
    need(type(model) is GarchModel and model.source_sha256 == source_digest('garch'))
    return digest(asdict(model))


def _state(model, state):
    need(type(state) is GarchState and state.model_sha256 == _identity(model)
         and state.last_ends_at >= model.training_ends_at and state.updated_at >= model.fitted_at)


def initial_garch_state(model):
    return GarchState(_identity(model), model.final_variance, model.training_ends_at, model.fitted_at, 0)


def observe_garch(model, state, row, *, observed_at):
    _state(model, state)
    at = row_update(row, last_ends_at=state.last_ends_at, updated_at=state.updated_at, observed_at=observed_at)
    variance = model.omega + model.alpha * (row.return_fraction - model.mean) ** 2 + model.beta * state.variance
    return GarchState(state.model_sha256, variance, row.ends_at, at, state.observations + 1)


def forecast_garch(model, state, *, issued_at):
    _state(model, state)
    at = aware(issued_at)
    need(at >= state.updated_at, 'forecast_before_state_available')
    return RiskForecast(state.model_sha256, at, state.last_ends_at, model.mean,
                        state.variance, state.variance + model.mean ** 2)
