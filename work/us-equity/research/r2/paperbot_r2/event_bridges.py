"""Pure second-step risk-event probabilities; no symbol or issue authority.

These mathematical results are not prospective predictions. A later issuer
must bind a result to a symbol, active feature, actual issue time and durable
journal before it may be scored against a future common outcome.
"""

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

from paperbot_us import data, regime_baselines
from . import contracts, garch, hmm
from .contracts import digest, is_sha


class BridgeError(ValueError):
    pass


def _need(condition):
    if not condition:
        raise BridgeError('event_bridge_unavailable')


def _source():
    paths = (Path(__file__), Path(contracts.__file__), Path(garch.__file__),
             Path(hmm.__file__), Path(data.__file__), Path(regime_baselines.__file__))
    _need(all(p.is_file() and not p.is_symlink() and p.stat().st_size <= 524288 for p in paths))
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def _seal(result):
    return digest({k: v for k, v in asdict(result).items() if k != 'seal'})


def _logs(probability):
    return (math.log(probability) if probability else -math.inf,
            math.log1p(-probability) if probability < 1. else -math.inf)


def _normal_tail(mean, variance):
    _need(type(mean) is float and type(variance) is float
          and math.isfinite(mean) and math.isfinite(variance) and variance > 0)
    scale = math.sqrt(2 * variance)
    value = .5 * math.erfc((.005 - mean) / scale) + .5 * math.erfc((.005 + mean) / scale)
    _need(math.isfinite(value) and 0 <= value <= 1)
    return value


def _mixture_variance(weights, means, variances):
    mean = math.fsum(w * m for w, m in zip(weights, means))
    value = math.fsum(w * (v + (m - mean) ** 2)
                      for w, m, v in zip(weights, means, variances))
    _need(math.isfinite(value) and value > 0)
    return value


def _unit_weights(values):
    # The HMM contract permits tiny row/posterior normalization error.
    # Match its normalized filtering semantics after each transition.
    total = math.fsum(values)
    _need(math.isfinite(total) and math.isclose(total, 1., rel_tol=0., abs_tol=2e-10))
    result = tuple(float(value / total) for value in values)
    _need(all(math.isfinite(value) and 0 <= value <= 1 for value in result))
    return result


@dataclass(frozen=True)
class BridgeResult:
    family: str
    probability: float
    log_event_probability: float
    log_no_event_probability: float
    gap_variance: float
    target_variance: float
    target_weights: tuple
    model_sha256: str
    state_sha256: str
    source_sha256: str
    seal: str
    threshold_fraction: float = .005
    horizon_steps: int = 2
    formula_version: str = 'two-step-absolute-return-risk-v1'

    def __post_init__(self):
        _need(self.family in ('ewma', 'garch', 'hmm')
              and type(self.probability) is float and math.isfinite(self.probability)
              and 0 <= self.probability <= 1
              and type(self.gap_variance) is float and math.isfinite(self.gap_variance)
              and self.gap_variance > 0
              and type(self.target_variance) is float and math.isfinite(self.target_variance)
              and self.target_variance > 0
              and type(self.target_weights) is tuple
              and ((self.family == 'hmm' and len(self.target_weights) == 2
                    and all(type(v) is float and math.isfinite(v) and 0 <= v <= 1
                            for v in self.target_weights)
                    and math.isclose(math.fsum(self.target_weights), 1., rel_tol=0., abs_tol=1e-12))
                   or (self.family != 'hmm' and self.target_weights == ()))
              and all(is_sha(v) for v in (self.model_sha256, self.state_sha256,
                                           self.source_sha256, self.seal))
              and self.source_sha256 == _source()
              and self.threshold_fraction == .005
              and type(self.horizon_steps) is int and self.horizon_steps == 2
              and self.formula_version == 'two-step-absolute-return-risk-v1'
              and (self.log_event_probability, self.log_no_event_probability)
              == _logs(self.probability)
              and self.seal == _seal(self))


def _result(family, probability, gap_variance, target_variance, weights, model_id, state):
    fields = dict(family=family, probability=float(probability),
                  log_event_probability=_logs(probability)[0],
                  log_no_event_probability=_logs(probability)[1],
                  gap_variance=float(gap_variance), target_variance=float(target_variance),
                  target_weights=tuple(weights), model_sha256=model_id,
                  state_sha256=digest(asdict(state)), source_sha256=_source(),
                  threshold_fraction=.005, horizon_steps=2,
                  formula_version='two-step-absolute-return-risk-v1')
    return BridgeResult(**fields, seal=digest(fields))


def bridge_ewma(model, state):
    """Zero-mean EWMA: expected target variance equals the unobserved gap variance."""
    try:
        _need(type(model) is regime_baselines.BaselineModel
              and type(state) is regime_baselines.ObservedState and model.decay == .94)
        model_id = regime_baselines._model_identity(model)
        _need(state.model_sha256 == model_id and state.last_ends_at >= model.training_ends_at
              and state.updated_at >= model.fitted_at)
        variance = float(state.variance)
        return _result('ewma', _normal_tail(0., variance), variance, variance,
                       (), model_id, state)
    except BridgeError:
        raise
    except Exception:
        raise BridgeError('event_bridge_unavailable') from None


def bridge_garch(model, state):
    """GARCH expected second-step variance before observing the gap shock."""
    try:
        _need(type(model) is garch.GarchModel and type(state) is garch.GarchState
              and model.alpha == .10 and model.beta == .85)
        garch._state(model, state)
        gap_variance = float(state.variance)
        target_variance = float(model.omega + (model.alpha + model.beta) * gap_variance)
        return _result('garch', _normal_tail(float(model.mean), target_variance),
                       gap_variance, target_variance, (), garch._identity(model), state)
    except BridgeError:
        raise
    except Exception:
        raise BridgeError('event_bridge_unavailable') from None


def bridge_hmm(model, state):
    """Filtered state posterior advanced twice without observing the gap."""
    try:
        _need(type(model) is hmm.HMMModel and type(state) is hmm.HMMState)
        hmm._state(model, state)
        params = model.parameters
        current = _unit_weights(tuple(math.exp(v) for v in state.log_posterior))
        gap = _unit_weights(tuple(math.fsum(current[i] * params.transition[i][j] for i in range(2))
                                  for j in range(2)))
        target = _unit_weights(tuple(math.fsum(gap[i] * params.transition[i][j] for i in range(2))
                                     for j in range(2)))
        tails = tuple(_normal_tail(float(m), float(v))
                      for m, v in zip(params.means, params.variances))
        probability = math.fsum(w * p for w, p in zip(target, tails))
        return _result('hmm', probability,
                       _mixture_variance(gap, params.means, params.variances),
                       _mixture_variance(target, params.means, params.variances),
                       target, hmm._identity(model), state)
    except BridgeError:
        raise
    except Exception:
        raise BridgeError('event_bridge_unavailable') from None
