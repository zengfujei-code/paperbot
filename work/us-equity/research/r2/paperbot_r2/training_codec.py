"""Bounded, explicit JSON replay of known training records; never executable data."""
from dataclasses import asdict, is_dataclass
import json
import math

from paperbot_us import alpaca, regime_baselines
from . import classifier, garch, hmm, model_training as models, training_data as training
from .contracts import aware, json_value

MAX_BYTES = 8388608


class TrainingCodecError(ValueError):
    pass


def encode(value):
    try:
        if is_dataclass(value) and not isinstance(value, type):
            value = asdict(value)
        encoded = json.dumps(json_value(value), sort_keys=True, separators=(',', ':'), allow_nan=False)
        if len(encoded.encode('utf-8')) > MAX_BYTES:
            raise ValueError('size')
        return encoded
    except Exception:
        raise TrainingCodecError('training_record_encoding_failed') from None


def _raw(value):
    if type(value) is str:
        if len(value.encode('utf-8')) > MAX_BYTES:
            raise ValueError('size')
        value = alpaca.decode_json(value)
    if type(value) is not dict:
        raise ValueError('object required')
    encode(value)  # Also bound caller-supplied objects, not only strings.
    return dict(value)


def _tuples(value):
    if type(value) is list or type(value) is tuple:
        return tuple(_tuples(v) for v in value)
    if type(value) is dict:
        if value == {'log_probability': 'negative_infinity'}:
            return -math.inf
        return {k: _tuples(v) for k, v in value.items()}
    return value


def _example(value):
    raw = dict(value)
    vector = _tuples(raw['features'])
    raw['features'] = classifier.FeatureVector(**vector)
    return classifier.ClassifierExample(**raw)


def decode_snapshot(value):
    try:
        raw = _raw(value)
        symbols = []
        for value in raw['symbols']:
            symbol = dict(value)
            symbol['price_identity'] = tuple(symbol['price_identity'])
            symbol['returns'] = tuple(regime_baselines.ReturnObservation(**r) for r in symbol['returns'])
            symbol['return_witnesses'] = tuple(training.ReturnWitness(**{
                **w, 'previous_recorded_at': aware(w['previous_recorded_at']),
                'current_recorded_at': aware(w['current_recorded_at'])}) for w in symbol['return_witnesses'])
            symbol['examples'] = tuple(_example(e) for e in symbol['examples'])
            symbol['example_witnesses'] = tuple(training.ExampleWitness(**{
                **w, 'acknowledged_at': aware(w['acknowledged_at'])}) for w in symbol['example_witnesses'])
            symbol['coverage'] = tuple(training.SessionCoverage(**c) for c in symbol['coverage'])
            symbols.append(training.SymbolTraining(**symbol))
        raw['symbols'] = tuple(symbols)
        raw['bindings'] = tuple(training.SourceBinding(**b) for b in raw['bindings'])
        for key in ('read_started_at', 'available_at'):
            raw[key] = aware(raw[key])
        return training.TrainingSnapshot(**raw)
    except Exception:
        raise TrainingCodecError('training_snapshot_decode_failed') from None


def _payload(family, value):
    raw = _tuples(value)
    if set(raw) != {'model', 'state'}:
        raise ValueError('payload fields')
    model, state = raw['model'], raw['state']
    if family == 'prevalence':
        model = models.PrevalenceModel(**model)
    elif family == 'ewma':
        model = regime_baselines.BaselineModel(**model)
        state = regime_baselines.ObservedState(**state)
    elif family == 'garch':
        model = garch.GarchModel(**model)
        state = garch.GarchState(**state)
    elif family == 'hmm':
        model['parameters'] = hmm.GaussianParameters(**model['parameters'])
        model = hmm.HMMModel(**model)
        state = hmm.HMMState(**state)
    elif family == 'gaussian_nb':
        model = classifier.RegimeClassifier(**model)
    else:
        raise ValueError('family')
    return models.FittedPayload(model, state)


def decode_fits(value):
    try:
        raw = _raw(value)
        attempts = []
        for value in raw['attempts']:
            attempt = dict(value)
            attempt['price_identity'] = tuple(attempt['price_identity'])
            for key in ('started_at', 'completed_at'):
                attempt[key] = aware(attempt[key])
            if attempt['payload'] is not None:
                attempt['payload'] = _payload(attempt['family'], attempt['payload'])
            attempts.append(models.FitAttempt(**attempt))
        raw['attempts'] = tuple(attempts)
        raw['available_at'] = aware(raw['available_at'])
        return models.TrainingFits(**raw)
    except Exception:
        raise TrainingCodecError('training_fits_decode_failed') from None
