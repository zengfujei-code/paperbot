"""Five fixed family computations on frozen inputs, without issue authority.

Native formula timestamps use the captured input basis. The outer actual
completion timestamp is separate; neither creates a prospective prediction.
"""
from dataclasses import asdict,dataclass
from datetime import datetime
import hashlib
import math
from pathlib import Path

from paperbot_us import regime_baselines,alpaca
from . import classifier,event_bridges,garch,hmm,model_training,training_journal
from . import forecast_inputs,training_codec as codec
from .contracts import aware,digest,is_sha,ChallengerError


class CandidateForecastError(ValueError):pass


def _need(value):
    if not value:raise CandidateForecastError('candidate_computation_unavailable')


def source_sha256():
    return digest(dict(calculator=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs=forecast_inputs.source_sha256(),bridge=event_bridges._source(),
        training=training_journal.source_sha256()))


def probability_logs(probability):
    _need(type(probability) is float and math.isfinite(probability) and 0<=probability<=1)
    return (math.log(probability) if probability else -math.inf,
            math.log1p(-probability) if probability<1 else -math.inf)


@dataclass(frozen=True)
class CandidateRow:
    family:str
    status:str
    failure_code:object
    model_sha256:str
    probability:object
    log_event_probability:object
    log_no_event_probability:object
    gap_variance:object
    target_variance:object
    target_weights:tuple
    state_sha256:object
    state_json:object
    native_json:object
    native_sha256:object
    seal:str

    def __post_init__(self):
        _need(self.family in model_training.FAMILIES and self.status in ('computed','failed')
              and is_sha(self.model_sha256) and type(self.target_weights) is tuple)
        if self.status=='failed':
            _need(self.failure_code in ('training_fit_failed','feature_outside_training_support','fixed_numeric_failure')
                  and all(v is None for v in (self.probability,self.log_event_probability,self.log_no_event_probability,
                      self.gap_variance,self.target_variance,self.state_sha256,self.state_json,self.native_json,self.native_sha256))
                  and self.target_weights==())
        else:
            _need(self.failure_code is None and
                  (self.log_event_probability,self.log_no_event_probability)==probability_logs(self.probability)
                  and type(self.native_json) is str and is_sha(self.native_sha256)
                  and hashlib.sha256(self.native_json.encode()).hexdigest()==self.native_sha256)
            if self.family in ('ewma','garch','hmm'):
                _need(all(type(v) is float and math.isfinite(v) and v>0 for v in (self.gap_variance,self.target_variance))
                      and type(self.state_json) is str and is_sha(self.state_sha256)
                      and digest(alpaca.decode_json(self.state_json))==self.state_sha256)
            else:
                _need(self.gap_variance is self.target_variance is self.state_json is self.state_sha256 is None
                      and self.target_weights==())
        _need(self.seal==digest({k:v for k,v in asdict(self).items() if k!='seal'}))


@dataclass(frozen=True)
class CandidatePanel:
    symbol:str
    input_sha256:str
    fits_sha256:str
    feature_sha256:str
    target_start:datetime
    target_end:datetime
    started_at:datetime
    completed_at:datetime
    rows:tuple
    math_only:bool
    source_sha256:str
    seal:str

    def __post_init__(self):
        _need(self.symbol in forecast_inputs.SYMBOLS and self.math_only is True
              and all(is_sha(v) for v in (self.input_sha256,self.fits_sha256,self.feature_sha256))
              and self.completed_at>=self.started_at and self.target_end>self.target_start
              and (self.target_end-self.target_start).total_seconds()==900
              and type(self.rows) is tuple and tuple(r.family for r in self.rows)==model_training.FAMILIES
              and all(type(r) is CandidateRow for r in self.rows)
              and self.source_sha256==source_sha256()
              and self.seal==digest({k:v for k,v in asdict(self).items() if k!='seal'}))


def _make_row(family,attempt,*,native=None,state=None,probability=None,failure=None):
    logs=probability_logs(float(probability)) if failure is None else (None,None)
    native_json=codec.encode(native) if native is not None else None
    fields=dict(family=family,status='failed' if failure else 'computed',failure_code=failure,
        model_sha256=attempt.seal,probability=float(probability) if failure is None else None,
        log_event_probability=logs[0],log_no_event_probability=logs[1],
        gap_variance=native.gap_variance if state is not None else None,
        target_variance=native.target_variance if state is not None else None,
        target_weights=native.target_weights if state is not None else (),
        state_sha256=digest(asdict(state)) if state is not None else None,
        state_json=codec.encode(state) if state is not None else None,native_json=native_json,
        native_sha256=hashlib.sha256(native_json.encode()).hexdigest() if native_json else None)
    return CandidateRow(**fields,seal=digest(fields))


def compute_candidates(training_receipt,issue_input,*,clock):
    try:
        _need(type(training_receipt) is training_journal.TrainingReceipt and type(issue_input) is forecast_inputs.IssueInput)
        training_receipt.__post_init__();issue_input.__post_init__()
        started=aware(clock());_need(started>=issue_input.available_at and started>=training_receipt.loaded_at
            and issue_input.training_snapshot_sha256==training_receipt.training.seal
            and issue_input.fits_sha256==training_receipt.fits.seal)
        source=source_sha256();rows=[]
        for family in model_training.FAMILIES:
            attempt=next(a for a in training_receipt.fits.attempts if a.symbol==issue_input.symbol and a.family==family)
            attempt.__post_init__()
            if attempt.status!='succeeded':
                rows.append(_make_row(family,attempt,failure='training_fit_failed'));continue
            payload=model_training.bound_model(attempt,symbol=issue_input.symbol,
                snapshot_sha256=training_receipt.training.seal,as_of=started)
            try:
                state=payload.state;model=payload.model
                if family=='prevalence':
                    native=model;probability=model.probability
                elif family=='gaussian_nb':
                    native=classifier.classify_regime(model,issue_input.bundle.vector,
                        issued_at=issue_input.available_at,target_start=issue_input.bundle.target_start,
                        target_end=issue_input.bundle.target_end)
                    probability=native.probabilities[1]
                else:
                    for observation in issue_input.returns:
                        if family=='ewma':
                            state=regime_baselines.advance_observed(model,state,observation,observed_at=issue_input.available_at)
                        elif family=='garch':state=garch.observe_garch(model,state,observation,observed_at=issue_input.available_at)
                        else:state=hmm.observe_hmm(model,state,observation,observed_at=issue_input.available_at)
                    native={'ewma':event_bridges.bridge_ewma,'garch':event_bridges.bridge_garch,
                            'hmm':event_bridges.bridge_hmm}[family](model,state)
                    probability=native.probability
                rows.append(_make_row(family,attempt,native=native,state=state,probability=probability))
            except Exception as error:
                failure='feature_outside_training_support' if type(error) is ChallengerError and str(error)=='feature_outside_training_support' else 'fixed_numeric_failure'
                rows.append(_make_row(family,attempt,failure=failure))
        completed=aware(clock());_need(completed>=started and source_sha256()==source)
        fields=dict(symbol=issue_input.symbol,input_sha256=issue_input.seal,fits_sha256=training_receipt.fits.seal,
            feature_sha256=issue_input.bundle.seal,target_start=issue_input.bundle.target_start,
            target_end=issue_input.bundle.target_end,started_at=started,completed_at=completed,
            rows=tuple(rows),math_only=True,source_sha256=source)
        return CandidatePanel(**fields,seal=digest({**fields,'rows':[asdict(r) for r in rows]}))
    except Exception:
        raise CandidateForecastError('fixed_candidate_computation_failed') from None


def decode_panel(value):
    try:
        raw=codec._raw(value)
        rows=[]
        for item in raw['rows']:
            item=codec._tuples(item)
            rows.append(CandidateRow(**item))
        raw['rows']=tuple(rows)
        for key in ('target_start','target_end','started_at','completed_at'):raw[key]=aware(raw[key])
        return CandidatePanel(**raw)
    except Exception:
        raise CandidateForecastError('fixed_candidate_panel_decode_failed') from None
