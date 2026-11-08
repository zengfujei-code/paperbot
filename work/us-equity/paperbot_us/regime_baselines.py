"""Causal research risk forecasts and mean-shift warnings; no order authority.

Returns are simple fractions for observed15-minute intervals. Decay advances
once per observation, including across calendar gaps; it is not wall-time decay
or an overnight-risk model. The detector is a standardized mean-shift baseline,
not HMM probabilities, a volatility-change test or evidence of trading advantage.
"""
from dataclasses import dataclass,asdict
from datetime import datetime,timedelta
import hashlib,json,math,re,statistics,sys

from .data import utc_time


class BaselineError(ValueError):pass


def _number(value,*,minimum=None,maximum=1e6):
    if type(value) not in (int,float) or abs(value)>maximum or not math.isfinite(value):
        raise BaselineError('finite_bounded_number_required')
    value=float(value)
    if minimum is not None and value<minimum:raise BaselineError('number_outside_bounds')
    return value


def _time(value):
    try:return utc_time(value)
    except Exception:raise BaselineError('aware_time_required') from None


def _variance(value,*,positive_expected):
    if (not math.isfinite(value) or value<0
            or positive_expected and value<sys.float_info.min):
        raise BaselineError('variance_outside_supported_float_precision')
    return value


@dataclass(frozen=True)
class ReturnObservation:
    ends_at:datetime
    available_at:datetime
    return_fraction:float
    interval_seconds:int=900

    def __post_init__(self):
        object.__setattr__(self,'ends_at',_time(self.ends_at))
        object.__setattr__(self,'available_at',_time(self.available_at))
        object.__setattr__(self,'return_fraction',_number(self.return_fraction))
        if (type(self.interval_seconds) is not int or self.interval_seconds!=900
                or self.available_at<self.ends_at or self.return_fraction<=-1):
            raise BaselineError('invalid_15m_simple_return_observation')
        _variance(self.return_fraction*self.return_fraction,positive_expected=self.return_fraction!=0)

    @property
    def starts_at(self):return self.ends_at-timedelta(seconds=self.interval_seconds)


def _observations(rows,as_of,minimum):
    if type(rows) not in (tuple,list) or not minimum<=len(rows)<=100000:
        raise BaselineError('bounded_observations_required')
    previous=None
    for row in rows:
        if (type(row) is not ReturnObservation or row.available_at>as_of
                or previous is not None and row.starts_at<previous):
            raise BaselineError('unavailable_or_unordered_observations')
        previous=row.ends_at
    return tuple(rows)


@dataclass(frozen=True)
class BaselineModel:
    decay:float
    cusum_k:float
    cusum_h:float
    initial_variance:float
    training_mean:float
    training_sigma:float
    training_count:int
    training_ends_at:datetime
    fitted_at:datetime
    training_sha256:str
    formula_version:str='ewma-second-moment-cusum-v1'

    def __post_init__(self):
        for name in ('decay','cusum_k','cusum_h','initial_variance','training_mean','training_sigma'):
            object.__setattr__(self,name,_number(getattr(self,name),maximum=1e12))
        for name in ('training_ends_at','fitted_at'):object.__setattr__(self,name,_time(getattr(self,name)))
        if (not 0<=self.decay<1 or self.cusum_k<0 or self.cusum_h<=0 or self.initial_variance<0
                or self.training_sigma<0 or type(self.training_count) is not int or not 2<=self.training_count<=100000
                or self.training_ends_at>self.fitted_at or not isinstance(self.training_sha256,str)
                or not re.fullmatch('[0-9a-f]{64}',self.training_sha256)
                or self.formula_version!='ewma-second-moment-cusum-v1'):
            raise BaselineError('invalid_frozen_baseline_model')
        _variance(self.initial_variance,positive_expected=self.training_mean!=0 or self.training_sigma!=0)
        if not math.isclose(math.sqrt(self.initial_variance),math.hypot(self.training_mean,self.training_sigma),rel_tol=1e-12):
            raise BaselineError('inconsistent_training_second_moment')


def fit_baseline(observations,*,as_of,decay,cusum_k,cusum_h):
    at=_time(as_of);rows=_observations(observations,at,2)
    values=tuple(row.return_fraction for row in rows)
    payload=json.dumps([dict(ends_at=r.ends_at.isoformat(),available_at=r.available_at.isoformat(),
        return_fraction=r.return_fraction,interval_seconds=r.interval_seconds) for r in rows],
        sort_keys=True,separators=(',',':'),allow_nan=False)
    # Zero-conditional-mean second moment, not a demeaned sample variance.
    variance=_variance(statistics.fmean(v*v for v in values),positive_expected=any(v!=0 for v in values))
    return BaselineModel(decay,cusum_k,cusum_h,variance,statistics.fmean(values),statistics.pstdev(values),
        len(rows),rows[-1].ends_at,at,hashlib.sha256(payload.encode()).hexdigest())


@dataclass(frozen=True)
class Forecast:
    forecast_at:datetime
    target_ends_at:datetime
    observation_available_at:datetime
    variance_forecast:float
    volatility_forecast:float
    next_variance_forecast:float
    cusum_up:float|None
    cusum_down:float|None
    alarms:tuple
    detector_status:str
    target_starts_at:datetime
    forecast_horizon_seconds:float
    target_elapsed_seconds_at_forecast:float
    forecast_scope:str='conditional_15m_return_not_remaining_time_variance'
    variance_unit:str='fraction_squared_per_15m'
    annualized:bool=False


def _step(model,variance,up,down,value):
    variance=_variance(model.decay*variance+(1-model.decay)*value**2,
        positive_expected=model.decay>0 and variance>0 or value!=0)
    alarms=();status='monitoring';positive=negative=None
    if model.training_sigma==0:status='constant_training_baseline'
    else:
        z=(value-model.training_mean)/model.training_sigma
        if not math.isfinite(z):raise BaselineError('standardized_return_overflow')
        up=max(0.,up+z-model.cusum_k);down=max(0.,down-z-model.cusum_k)
        positive,negative=up,down
        alarms=tuple(name for name,v in (('up',up),('down',down)) if v>model.cusum_h)
        if alarms:up=down=0.
    return variance,up,down,positive,negative,alarms,status


def _model_identity(model):
    if type(model) is not BaselineModel:raise BaselineError('exact_frozen_model_required')
    value=json.dumps(asdict(model),sort_keys=True,separators=(',',':'),allow_nan=False,
        default=lambda v:v.isoformat())
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class ObservedState:
    model_sha256:str
    variance:float
    cusum_up:float
    cusum_down:float
    last_ends_at:datetime
    updated_at:datetime
    observations:int
    alarms:tuple=()
    detector_status:str='monitoring'

    def __post_init__(self):
        for name in ('variance','cusum_up','cusum_down'):
            object.__setattr__(self,name,_number(getattr(self,name),minimum=0,maximum=1e12))
        for name in ('last_ends_at','updated_at'):object.__setattr__(self,name,_time(getattr(self,name)))
        if (not isinstance(self.model_sha256,str) or not re.fullmatch('[0-9a-f]{64}',self.model_sha256)
                or self.last_ends_at>self.updated_at or type(self.observations) is not int
                or not 0<=self.observations<=100000 or type(self.alarms) is not tuple
                or self.alarms not in ((),('up',),('down',),('up','down'))
                or self.detector_status not in ('monitoring','constant_training_baseline')):
            raise BaselineError('invalid_observed_state')
        _variance(self.variance,positive_expected=self.variance!=0)


def initial_state(model):
    identity=_model_identity(model)
    return ObservedState(identity,model.initial_variance,0.,0.,model.training_ends_at,model.fitted_at,0,
        detector_status='constant_training_baseline' if model.training_sigma==0 else 'monitoring')


def advance_observed(model,state,observation,*,observed_at):
    """Consume known returns at actual observation time, emitting no forecast.

    Several late historical observations can update state at the same current
    time. This does not claim any of them were received or predicted earlier.
    """
    now=_time(observed_at)
    if (type(state) is not ObservedState or type(observation) is not ReturnObservation
            or state.model_sha256!=_model_identity(model) or now<state.updated_at
            or observation.available_at>now or observation.starts_at<state.last_ends_at
            or observation.ends_at<=state.last_ends_at):
        raise BaselineError('unavailable_duplicate_or_unbound_observed_state')
    variance,up,down,_,_,alarms,status=_step(model,state.variance,state.cusum_up,state.cusum_down,observation.return_fraction)
    return ObservedState(state.model_sha256,variance,up,down,observation.ends_at,now,state.observations+1,alarms,status)


def evaluate_online(model,observations,*,as_of):
    if type(model) is not BaselineModel:raise BaselineError('exact_frozen_model_required')
    at=_time(as_of)
    if model.fitted_at>at:raise BaselineError('fitted_model_not_yet_available')
    rows=_observations(observations,at,0)
    variance=model.initial_variance;up=down=0.;previous=model.fitted_at;result=[]
    for row in rows:
        if row.ends_at<=previous or row.starts_at<model.training_ends_at:
            raise BaselineError('forecast_not_available_or_overlapping_training_target')
        forecast=variance
        variance,up,down,positive,negative,alarms,status=_step(model,variance,up,down,row.return_fraction)
        if not math.isfinite(variance):raise BaselineError('variance_overflow')
        result.append(Forecast(previous,row.ends_at,row.available_at,forecast,math.sqrt(forecast),variance,
            positive,negative,alarms,status,row.starts_at,(row.ends_at-previous).total_seconds(),
            max(0.,(previous-row.starts_at).total_seconds())))
        previous=row.available_at
    return tuple(result)
