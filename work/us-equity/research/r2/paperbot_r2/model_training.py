"""Fixed per-symbol research fits with actual completion-time provenance.

Only a provenance-bearing ready training snapshot is accepted. Fifteen attempts
are retained, including failures; there is no optimizer, replacement model,
forecast, registry registration, broker authority or automatic Pilot promotion.
"""
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
from pathlib import Path

from paperbot_us import regime_baselines
from . import classifier, garch, hmm, training_data
from .contracts import ChallengerError, aware, digest, is_sha, source_digest

FAMILIES = ('prevalence', 'ewma', 'garch', 'hmm', 'gaussian_nb')
SAFE_FAILURES = frozenset(('bounded_training_rows_required', 'insufficient_training_labels',
    'insufficient_class_evidence', 'singular_training_variance', 'constant_training_feature',
    'hmm_not_converged', 'unidentified_hmm_components', 'hmm_likelihood_decreased',
    'feature_outside_training_support'))


class ModelTrainingError(ValueError):
    pass


def _need(value, code='training_fit_unavailable'):
    if not value:
        raise ModelTrainingError(code)


def source_sha256():
    path = Path(__file__)
    _need(path.is_file() and not path.is_symlink())
    return digest(dict(fitter=hashlib.sha256(path.read_bytes()).hexdigest(),
        training=training_data.source_sha256(), garch=source_digest('garch'),
        hmm=source_digest('hmm'), classifier=classifier._source(),
        baseline=hashlib.sha256(Path(regime_baselines.__file__).read_bytes()).hexdigest()))


def _seal(value):
    return digest({k: v for k, v in asdict(value).items() if k != 'seal'})


@dataclass(frozen=True)
class PrevalenceModel:
    class_counts: tuple
    probability: float
    training_sha256: str

    def __post_init__(self):
        _need(type(self.class_counts) is tuple and len(self.class_counts) == 2
              and all(type(n) is int and n >= 0 for n in self.class_counts)
              and sum(self.class_counts) > 0 and is_sha(self.training_sha256)
              and self.probability == (self.class_counts[1] + 1) / (sum(self.class_counts) + 2))


@dataclass(frozen=True)
class FittedPayload:
    model: object
    state: object


@dataclass(frozen=True)
class FitAttempt:
    symbol: str
    family: str
    price_identity: tuple
    snapshot_sha256: str
    input_rows_sha256: str
    input_count: int
    started_at: datetime
    completed_at: datetime
    status: str
    failure_code: object
    payload: object
    source_sha256: str
    seal: str

    def __post_init__(self):
        _need(self.symbol in training_data.SYMBOLS and self.family in FAMILIES
              and self.price_identity == (self.symbol, 'alpaca', 'iex', 'raw', 900, 'USD', 'shares')
              and all(is_sha(s) for s in (self.snapshot_sha256, self.input_rows_sha256))
              and type(self.input_count) is int and 0 <= self.input_count <= 2000
              and aware(self.completed_at) >= aware(self.started_at)
              and self.status in ('succeeded', 'failed'))
        if self.status == 'failed':
            _need(self.payload is None and self.failure_code in SAFE_FAILURES | {'fixed_fit_failed'})
        else:
            _need(self.failure_code is None and type(self.payload) is FittedPayload)
            model, state = self.payload.model, self.payload.state
            expected = {'prevalence': PrevalenceModel, 'ewma': regime_baselines.BaselineModel,
                        'garch': garch.GarchModel, 'hmm': hmm.HMMModel,
                        'gaussian_nb': classifier.RegimeClassifier}[self.family]
            _need(type(model) is expected and model.training_sha256 == self.input_rows_sha256)
            if self.family == 'prevalence':
                _need(state is None and sum(model.class_counts) == self.input_count)
            elif self.family == 'gaussian_nb':
                _need(state is None and model.variance_floor == 1e-6
                      and model.fitted_at == self.started_at and len(model.reference_ids) == self.input_count)
            else:
                _need(model.fitted_at == self.started_at and model.training_count == self.input_count)
                if self.family == 'ewma':
                    _need(model.decay == .94 and type(state) is regime_baselines.ObservedState
                          and state.model_sha256 == regime_baselines._model_identity(model))
                elif self.family == 'garch':
                    _need(model.alpha == .1 and model.beta == .85 and type(state) is garch.GarchState)
                    garch._state(model, state)
                else:
                    _need(type(state) is hmm.HMMState and model.tolerance == 1e-7)
                    hmm._state(model, state)
                _need(state.last_ends_at == model.training_ends_at and state.updated_at == self.started_at)
        _need(self.source_sha256 == source_sha256() and self.seal == _seal(self),
              'training_fit_binding_mismatch')


@dataclass(frozen=True)
class TrainingFits:
    snapshot_sha256: str
    attempts: tuple
    available_at: datetime
    source_sha256: str
    seal: str

    def __post_init__(self):
        _need(type(self.attempts) is tuple and len(self.attempts) == 15
              and tuple((a.symbol, a.family) for a in self.attempts)
              == tuple((s, f) for s in training_data.SYMBOLS for f in FAMILIES)
              and all(type(a) is FitAttempt and a.snapshot_sha256 == self.snapshot_sha256
                      and a.source_sha256 == self.source_sha256 and a.completed_at <= aware(self.available_at)
                      for a in self.attempts)
              and is_sha(self.snapshot_sha256) and self.source_sha256 == source_sha256()
              and self.seal == _seal(self), 'training_fits_binding_mismatch')


def _fit(family, symbol, rows, at):
    if family == 'prevalence':
        if not rows:
            raise ChallengerError('insufficient_training_labels')
        counts = tuple(sum(r.label == k for r in rows) for k in (0, 1))
        return FittedPayload(PrevalenceModel(counts, (counts[1] + 1) / (len(rows) + 2),
                                            digest([asdict(r) for r in rows])), None)
    if family == 'ewma':
        model = regime_baselines.fit_baseline(rows, as_of=at, decay=.94, cusum_k=.5, cusum_h=5.)
        variance = model.initial_variance
        for row in rows:
            variance = .94 * variance + .06 * row.return_fraction ** 2
        state = replace(regime_baselines.initial_state(model), variance=variance)
    elif family == 'garch':
        model = garch.fit_garch(rows, as_of=at, alpha=.1, beta=.85)
        state = garch.initial_garch_state(model)
    elif family == 'hmm':
        model = hmm.fit_hmm(rows, as_of=at, max_iterations=80, tolerance=1e-7)
        state = hmm.initial_hmm_state(model)
    else:
        model = classifier.fit_classifier(rows, as_of=at, variance_floor=1e-6)
        state = None
    return FittedPayload(model, state)


def fit_training_snapshot(snapshot, *, timezone_file, clock):
    """Fit all fixed candidates once; callers must durably journal before use."""
    try:
        _need(type(snapshot) is training_data.TrainingSnapshot)
        snapshot.__post_init__()
        _need(snapshot.ready, 'training_snapshot_not_ready')
        guard = training_data.TrainingClock(timezone_file, clock)
        _need(guard() >= snapshot.available_at, 'training_fit_before_data_available')
        source = source_sha256()
        attempts = []
        for symbol in snapshot.symbols:
            for family in FAMILIES:
                started = guard()
                rows = symbol.examples if family in ('prevalence', 'gaussian_nb') else symbol.returns
                rows_hash = digest([asdict(r) for r in rows])
                payload, failure = None, None
                try:
                    payload = _fit(family, symbol, rows, started)
                except Exception as exc:
                    failure = str(exc) if type(exc) is ChallengerError and str(exc) in SAFE_FAILURES else 'fixed_fit_failed'
                completed = guard()
                _need(source_sha256() == source, 'training_source_changed_during_fit')
                fields = dict(symbol=symbol.symbol, family=family, price_identity=symbol.price_identity,
                    snapshot_sha256=snapshot.seal, input_rows_sha256=rows_hash, input_count=len(rows),
                    started_at=started, completed_at=completed, status='failed' if failure else 'succeeded',
                    failure_code=failure, payload=payload, source_sha256=source)
                attempts.append(FitAttempt(**fields, seal=digest({k: asdict(v) if k == 'payload' and v is not None
                                                                 else v for k, v in fields.items()})))
        available = guard()
        _need(source_sha256() == source)
        fields = dict(snapshot_sha256=snapshot.seal, attempts=tuple(attempts),
                      available_at=available, source_sha256=source)
        return TrainingFits(**fields, seal=digest({k: [asdict(a) for a in v] if k == 'attempts' else v
                                                  for k, v in fields.items()}))
    except ModelTrainingError:
        raise
    except Exception:
        raise ModelTrainingError('training_fit_unavailable') from None


def bound_model(attempt, *, symbol, snapshot_sha256, as_of):
    """Validate wrapper provenance before handing a payload to future issue code.

    The caller still needs a durable first-fit/first-issue journal and canonical
    registry admission. A caller-supplied as_of is not an actual issue receipt.
    """
    try:
        _need(type(attempt) is FitAttempt)
        attempt.__post_init__()
        _need(attempt.status == 'succeeded' and attempt.symbol == symbol
              and attempt.snapshot_sha256 == snapshot_sha256 and aware(as_of) >= attempt.completed_at,
              'training_model_symbol_or_availability_mismatch')
        return attempt.payload
    except ModelTrainingError:
        raise
    except Exception:
        raise ModelTrainingError('training_fit_unavailable') from None
