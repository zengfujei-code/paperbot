"""Two-state Gaussian HMM: bounded offline EM and strictly causal filtering.

Training backward probabilities never become online decisions. Log posterior
state preserves very small paths rather than permanently rounding them to zero.
States are statistical components, not profitable/bullish/bearish market labels.
"""
from dataclasses import asdict, dataclass
from datetime import datetime
import math
import statistics

from .contracts import (ChallengerError, RiskForecast, aware, digest, is_sha, need,
                        number, row_update, rows_digest, source_digest, training_rows)


def _probabilities(values):
    need(type(values) is tuple and len(values) == 2)
    result = tuple(number(v, minimum=0., maximum=1.) for v in values)
    need(math.isclose(math.fsum(result), 1., rel_tol=0., abs_tol=1e-12), 'invalid_probability_vector')
    return result


@dataclass(frozen=True)
class GaussianParameters:
    initial: tuple
    transition: tuple
    means: tuple
    variances: tuple

    def __post_init__(self):
        object.__setattr__(self, 'initial', _probabilities(self.initial))
        need(type(self.transition) is tuple and len(self.transition) == 2)
        object.__setattr__(self, 'transition', tuple(_probabilities(row) for row in self.transition))
        for name in ('means', 'variances'):
            value = getattr(self, name)
            need(type(value) is tuple and len(value) == 2)
            object.__setattr__(self, name, tuple(number(v, minimum=1e-12 if name == 'variances' else -100.,
                                                       maximum=1e6 if name == 'variances' else 100.) for v in value))


def _log(value):
    return math.log(value) if value else -math.inf


def _lse(values):
    maximum = max(values)
    if maximum == -math.inf:
        return maximum
    return maximum + math.log(math.fsum(math.exp(v - maximum) for v in values))


def _normalize(values):
    maximum = max(values)
    need(math.isfinite(maximum), 'impossible_hmm_observation')
    shifted = tuple(v - maximum for v in values)
    normalizer = math.log(math.fsum(math.exp(v) for v in shifted))
    return tuple(v - normalizer for v in shifted), maximum + normalizer


def _log_posterior(values):
    need(type(values) is tuple and len(values) == 2)
    need(all(type(v) in (int, float) and (math.isfinite(v) and abs(v) <= 1e18 or v == -math.inf)
             and v <= 1e-8 for v in values), 'invalid_log_probability')
    need(abs(_lse(values)) <= 1e-10, 'unnormalized_log_probability')
    return values


def _emissions(params, value):
    return tuple(-.5 * (math.log(2 * math.pi * var) + (value - mean) ** 2 / var)
                 for mean, var in zip(params.means, params.variances))


def _prior(params, posterior):
    return tuple(_lse(tuple(posterior[i] + _log(params.transition[i][j]) for i in range(2))) for j in range(2))


def filter_hmm(parameters, log_posterior, value, *, initial=False):
    need(type(parameters) is GaussianParameters and type(initial) is bool)
    value = number(value, minimum=-.9999999999999999)
    if initial:
        need(log_posterior is None)
        prior = tuple(_log(p) for p in parameters.initial)
    else:
        prior = _prior(parameters, _log_posterior(log_posterior))
    return _normalize(tuple(p + e for p, e in zip(prior, _emissions(parameters, value))))


def predictive_hmm(parameters, log_posterior):
    need(type(parameters) is GaussianParameters)
    prior, _ = _normalize(_prior(parameters, _log_posterior(log_posterior)))
    weights = tuple(math.exp(v) for v in prior)
    mean = math.fsum(w * m for w, m in zip(weights, parameters.means))
    # Stable law of total variance; avoid cancellation of large squared means.
    variance = math.fsum(w * (v + (m - mean) ** 2) for w, v, m in
                         zip(weights, parameters.variances, parameters.means))
    return mean, variance, variance + mean ** 2


def _forward(parameters, values):
    alphas, scales = [], []
    posterior = None
    for value in values:
        posterior, score = filter_hmm(parameters, posterior, value, initial=posterior is None)
        alphas.append(posterior)
        scales.append(score)
    return alphas, scales, math.fsum(scales)


def _expectation(parameters, values, alphas, scales):
    count = len(values)
    emissions = tuple(_emissions(parameters, value) for value in values)
    betas = [(0., 0.)] * count
    for t in range(count - 2, -1, -1):
        betas[t] = tuple(_lse(tuple(_log(parameters.transition[i][j]) + emissions[t + 1][j]
                                   + betas[t + 1][j] for j in range(2))) - scales[t + 1] for i in range(2))
    gamma = []
    for alpha, beta in zip(alphas, betas):
        normalized, _ = _normalize(tuple(a + b for a, b in zip(alpha, beta)))
        gamma.append(tuple(math.exp(v) for v in normalized))
    xi = []
    for t in range(count - 1):
        weights = tuple(alphas[t][i] + _log(parameters.transition[i][j]) + emissions[t + 1][j]
                        + betas[t + 1][j] for i in range(2) for j in range(2))
        normalized, _ = _normalize(weights)
        xi.append(tuple(math.exp(v) for v in normalized))
    return gamma, xi


def _maximize(values, gamma, xi):
    occupancy = tuple(math.fsum(g[j] for g in gamma) for j in range(2))
    need(min(occupancy) >= 2., 'collapsed_hmm_effective_occupancy')
    means = tuple(math.fsum(g[j] * value for g, value in zip(gamma, values)) / occupancy[j] for j in range(2))
    variances = tuple(math.fsum(g[j] * (value - means[j]) ** 2 for g, value in zip(gamma, values))
                      / occupancy[j] for j in range(2))
    need(min(variances) >= 1e-12, 'singular_hmm_component_variance')
    transitions = []
    for i in range(2):
        amounts = tuple(math.fsum(x[i * 2 + j] for x in xi) for j in range(2))
        total = math.fsum(amounts)
        need(total >= 1e-8, 'collapsed_hmm_transition_occupancy')
        transitions.append(tuple(v / total for v in amounts))
    return GaussianParameters(gamma[0], tuple(transitions), means, variances)


@dataclass(frozen=True)
class HMMModel:
    parameters: GaussianParameters
    final_log_posterior: tuple
    training_count: int
    training_ends_at: datetime
    fitted_at: datetime
    training_sha256: str
    source_sha256: str
    iterations: int
    log_likelihood_trace: tuple
    tolerance: float
    converged: bool = True
    formula_version: str = 'two-state-gaussian-em-log-filter-v1'

    def __post_init__(self):
        need(type(self.parameters) is GaussianParameters)
        _log_posterior(self.final_log_posterior)
        for name in ('training_ends_at', 'fitted_at'):
            object.__setattr__(self, name, aware(getattr(self, name)))
        object.__setattr__(self, 'tolerance', number(self.tolerance, minimum=1e-10, maximum=.01))
        need(type(self.training_count) is int and 64 <= self.training_count <= 2000
             and self.training_ends_at <= self.fitted_at and is_sha(self.training_sha256)
             and self.source_sha256 == source_digest('hmm') and type(self.iterations) is int and 1 <= self.iterations <= 80
             and self.converged is True and self.formula_version == 'two-state-gaussian-em-log-filter-v1'
             and type(self.log_likelihood_trace) is tuple and len(self.log_likelihood_trace) == self.iterations + 1
             and all(type(v) is float and math.isfinite(v) for v in self.log_likelihood_trace)
             and self.parameters.variances[0] <= self.parameters.variances[1])


def fit_hmm(rows, *, as_of, max_iterations=80, tolerance=1e-7):
    rows, at = training_rows(rows, as_of)
    need(type(max_iterations) is int and 1 <= max_iterations <= 80, 'bounded_hmm_iterations_required')
    tolerance = number(tolerance, minimum=1e-10, maximum=.01)
    values = tuple(row.return_fraction for row in rows)
    mean = statistics.fmean(values)
    variance = statistics.fmean((v - mean) ** 2 for v in values)
    need(variance >= 1e-12, 'singular_training_variance')
    parameters = GaussianParameters((.5, .5), ((.95, .05), (.05, .95)), (mean, mean),
                                    (max(1e-12, .25 * variance), 1.75 * variance))
    alphas, scales, likelihood = _forward(parameters, values)
    trace = [likelihood]
    for iteration in range(1, max_iterations + 1):
        gamma, xi = _expectation(parameters, values, alphas, scales)
        proposed = _maximize(values, gamma, xi)
        alphas, scales, next_likelihood = _forward(proposed, values)
        need(next_likelihood >= likelihood - 1e-8 * max(1., abs(likelihood)), 'hmm_likelihood_decreased')
        trace.append(next_likelihood)
        gain = (next_likelihood - likelihood) / len(values)
        parameters, likelihood = proposed, next_likelihood
        if gain <= tolerance:
            order = tuple(sorted(range(2), key=lambda j: (parameters.variances[j], parameters.means[j])))
            separation = abs(parameters.means[0] - parameters.means[1]) / math.sqrt(max(parameters.variances))
            separation += abs(math.log(parameters.variances[0] / parameters.variances[1]))
            need(separation > 1e-3, 'unidentified_hmm_components')
            canonical = GaussianParameters(tuple(parameters.initial[i] for i in order),
                tuple(tuple(parameters.transition[i][j] for j in order) for i in order),
                tuple(parameters.means[i] for i in order), tuple(parameters.variances[i] for i in order))
            return HMMModel(canonical, tuple(alphas[-1][j] for j in order), len(rows), rows[-1].ends_at,
                            at, rows_digest(rows), source_digest('hmm'), iteration, tuple(trace), tolerance)
    raise ChallengerError('hmm_not_converged', diagnostics=dict(iterations=max_iterations,
        log_likelihood_trace=tuple(trace), training_sha256=rows_digest(rows), source_sha256=source_digest('hmm')))


@dataclass(frozen=True)
class HMMState:
    model_sha256: str
    log_posterior: tuple
    last_ends_at: datetime
    updated_at: datetime
    observations: int

    def __post_init__(self):
        _log_posterior(self.log_posterior)
        for name in ('last_ends_at', 'updated_at'):
            object.__setattr__(self, name, aware(getattr(self, name)))
        need(is_sha(self.model_sha256) and self.last_ends_at <= self.updated_at
             and type(self.observations) is int and 0 <= self.observations <= 100000)


def _identity(model):
    need(type(model) is HMMModel and model.source_sha256 == source_digest('hmm'))
    return digest(asdict(model))


def _state(model, state):
    need(type(state) is HMMState and state.model_sha256 == _identity(model)
         and state.last_ends_at >= model.training_ends_at and state.updated_at >= model.fitted_at)


def initial_hmm_state(model):
    return HMMState(_identity(model), model.final_log_posterior, model.training_ends_at, model.fitted_at, 0)


def observe_hmm(model, state, row, *, observed_at):
    _state(model, state)
    at = row_update(row, last_ends_at=state.last_ends_at, updated_at=state.updated_at, observed_at=observed_at)
    posterior, _ = filter_hmm(model.parameters, state.log_posterior, row.return_fraction)
    return HMMState(state.model_sha256, posterior, row.ends_at, at, state.observations + 1)


def forecast_hmm(model, state, *, issued_at):
    _state(model, state)
    at = aware(issued_at)
    need(at >= state.updated_at, 'forecast_before_state_available')
    mean, variance, moment = predictive_hmm(model.parameters, state.log_posterior)
    return RiskForecast(state.model_sha256, at, state.last_ends_at, mean, variance, moment)
