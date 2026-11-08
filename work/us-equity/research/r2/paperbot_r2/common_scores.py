"""Pure fixed-grid risk-event diagnostics, never selection or issue authority.

The caller must verify source lineage/current permission separately. Scalar
fixtures and historical mathematical replay cannot authorize a market phase.
"""
from dataclasses import asdict,dataclass
from datetime import datetime,timedelta
import hashlib
import json
import math
from pathlib import Path

from . import common_registration as registered,validation_issue_journal as issuer
from .contracts import aware,digest,is_sha
from .model_training import FAMILIES
from .training_data import SYMBOLS,TRAINING_DAYS,TIMEZONE_SHA256

RISK_FAMILIES=('ewma','garch','hmm')
FAILURES=('training_fit_failed','feature_outside_training_support','fixed_numeric_failure')
METRICS=('brier','log_loss','variance_proxy_mse','variance_proxy_qlike','second_moment_mse','second_moment_qlike')
RULES=dict(schema=1,phase='validation',common_families=list(FAMILIES),symbols=list(SYMBOLS),
    primary='brier_macro_min',macro='equal_three_symbol_weight',minimum_common_fraction=.8,
    common_set='all_five_issued_settled_on_time_unquarantined',trial_units=15,
    counts='publication_frames include failures; issued requires computed probability; settled includes late labels; usable_settled excludes missing/late/quarantine',
    tie_order=list(FAMILIES),lookback_returns=8,first_completed_bars=9,gap_seconds=900,horizon_seconds=900,
    threshold_fraction=.005,calibration_edges=[0.,.2,.4,.6,.8,1.],
    log_loss='exact_stored_p_no_clipping_positive_infinity_tag',
    variance_proxy='(variance-(return-mean)^2)^2;log(variance)+(return-mean)^2/variance',
    second_moment='(variance+mean^2-return^2)^2;log(variance+mean^2)+return^2/(variance+mean^2)',
    variance_mean='issued_frozen_model_mean_not_later_data',annualization=False,
    validation_days=list(registered.VALIDATION_DAYS),test_days=list(registered.TEST_DAYS),
    final_validation_deadline='2026-12-04T21:02:00+00:00',first_test_open='2026-12-07T14:30:00+00:00',
    requires_separate_selection_admission=True,test_admission=False,orders=False)


class CommonScoreError(ValueError):pass


def _need(value):
    if not value:raise CommonScoreError('common_comparison_score_unavailable')


def source_sha256():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


@dataclass(frozen=True)
class ScorePolicy:
    evaluation_source_sha256:str
    prospective_protocol_sha256:str
    issuer_source_sha256:str
    rules_json:str
    seal:str

    def __post_init__(self):
        _need(all(is_sha(s) for s in (self.evaluation_source_sha256,self.prospective_protocol_sha256,
            self.issuer_source_sha256,self.seal)) and self.issuer_source_sha256==issuer.source_sha256()
            and self.rules_json==_canonical(RULES)
            and self.seal==digest({k:v for k,v in asdict(self).items() if k!='seal'}))

    @property
    def configuration_sha256(self):return self.seal

    @classmethod
    def build(cls,evaluation_source_sha256,prospective_protocol_sha256):
        fields=dict(evaluation_source_sha256=evaluation_source_sha256,prospective_protocol_sha256=prospective_protocol_sha256,
                    issuer_source_sha256=issuer.source_sha256(),rules_json=_canonical(RULES))
        return cls(**fields,seal=digest(fields))


@dataclass(frozen=True)
class PlannedSlot:
    key:str
    symbol:str
    day:str
    event_at:datetime
    target_start:datetime
    target_end:datetime


def planned_slots(plan,phase='validation'):
    try:
        _need(phase=='validation' and type(plan) is dict and plan['symbols']==list(SYMBOLS)
              and plan['pairs']==[list(p) for p in registered.PAIRS]
              and plan['split']==dict(train=list(TRAINING_DAYS),validation=list(registered.VALIDATION_DAYS),
                                     test=list(registered.TEST_DAYS),context=[]))
        event=dict(formula='abs(target_first_close/gap_first_close-1)>=threshold',threshold_fraction=.005,
                   gap_seconds=900,horizon_seconds=900,feed='iex',adjustment='raw',currency='USD',volume_unit='shares')
        calendar=plan['calendar'];sessions=calendar['sessions']
        _need(plan['event']==event and calendar['timezone_sha256']==TIMEZONE_SHA256
              and calendar['sha256']==digest(sessions)
              and type(sessions) is list and len(sessions)==13)
        for value,day in zip(sessions,registered.VALIDATION_DAYS+registered.TEST_DAYS):
            _need(value==dict(day=day,opens=day+'T14:30:00+00:00',
                             closes=day+('T18:00:00+00:00' if day=='2026-11-27' else 'T21:00:00+00:00')))
        slots=[]
        for session in sessions[:8]:
            opens,closes=aware(session['opens']),aware(session['closes'])
            maximum=int((closes-opens).total_seconds()/900)-2
            for symbol in SYMBOLS:
                for completed in range(9,maximum+1):
                    at=opens+timedelta(seconds=900*completed)
                    slots.append(PlannedSlot(symbol+'|'+at.isoformat(),symbol,session['day'],at,
                                             at+timedelta(seconds=900),at+timedelta(seconds=1800)))
        _need(len(slots)==348 and all(sum(s.symbol==symbol for s in slots)==116 for symbol in SYMBOLS))
        return tuple(slots)
    except Exception:raise CommonScoreError('registered_validation_grid_invalid') from None


def _float(value,*,minimum=-math.inf,maximum=math.inf):
    return type(value) is float and math.isfinite(value) and minimum<=value<=maximum


@dataclass(frozen=True)
class ScoreRow:
    family:str
    status:str
    probability:object
    mean:object
    variance:object

    def __post_init__(self):
        _need(self.family in FAMILIES and self.status in ('computed',)+FAILURES)
        if self.status!='computed':_need(self.probability is self.mean is self.variance is None)
        else:
            _need(_float(self.probability,minimum=0.,maximum=1.))
            if self.family in RISK_FAMILIES:
                _need(_float(self.mean,minimum=-100.,maximum=100.)
                      and _float(self.variance,minimum=1e-12,maximum=1e12))
            else:_need(self.mean is self.variance is None)


@dataclass(frozen=True)
class ScoreFrame:
    key:str
    symbol:str
    event_at:datetime
    target_start:datetime
    target_end:datetime
    rows:tuple=()
    issued:bool=False
    settled:bool=False
    label:object=None
    return_fraction:object=None
    missing:bool=False
    late:bool=False
    quarantined:bool=False

    def __post_init__(self):
        for name in ('event_at','target_start','target_end'):object.__setattr__(self,name,aware(getattr(self,name)))
        _need(self.symbol in SYMBOLS and self.key==self.symbol+'|'+self.event_at.isoformat()
            and self.target_start==self.event_at+timedelta(seconds=900)
            and self.target_end==self.target_start+timedelta(seconds=900)
            and type(self.rows) is tuple and (self.rows==() or tuple(r.family for r in self.rows)==FAMILIES)
            and all(type(r) is ScoreRow for r in self.rows)
            and all(type(v) is bool for v in (self.issued,self.settled,self.missing,self.late,self.quarantined))
            and (not self.issued or len(self.rows)==5))
        if self.settled:
            _need(self.issued and type(self.label) is int and self.label in (0,1)
                  and _float(self.return_fraction,minimum=-1.,maximum=100.) and self.return_fraction>-1.)
        else:_need(self.label is self.return_fraction is None)


def point_losses(row,label,return_fraction):
    try:
        _need(type(row) is ScoreRow and row.status=='computed' and type(label) is int and label in (0,1)
              and _float(return_fraction,minimum=-1.,maximum=100.) and return_fraction>-1.)
        p=row.probability
        log=(math.log(p) if p else -math.inf) if label else (math.log1p(-p) if p<1. else -math.inf)
        result=dict(brier=(p-label)**2,log_loss=-log,calibration_bin=sum(p>=v for v in (.2,.4,.6,.8)),
                    variance_proxy_mse=None,variance_proxy_qlike=None,second_moment_mse=None,second_moment_qlike=None)
        if row.family in RISK_FAMILIES:
            centered=(return_fraction-row.mean)**2;second=row.variance+row.mean**2
            result.update(variance_proxy_mse=(row.variance-centered)**2,
                variance_proxy_qlike=math.log(row.variance)+centered/row.variance,
                second_moment_mse=(second-return_fraction**2)**2,
                second_moment_qlike=math.log(second)+return_fraction**2/second)
        return result
    except Exception:raise CommonScoreError('common_point_loss_invalid') from None


def _usable(frame,row):
    return frame.issued and frame.settled and not(frame.missing or frame.late or frame.quarantined) and row.status=='computed'


def _mean(values):return math.fsum(values)/len(values) if values else None


def _summary(values):
    losses=[point_losses(row,frame.label,frame.return_fraction) for frame,row in values]
    result={name:_mean([r[name] for r in losses if r[name] is not None]) for name in METRICS}
    bins=[]
    for index in range(5):
        found=[(frame,row) for (frame,row),loss in zip(values,losses) if loss['calibration_bin']==index]
        bins.append(dict(count=len(found),mean_probability=_mean([row.probability for _,row in found]),
                         event_fraction=_mean([frame.label for frame,_ in found])))
    return dict(count=len(values),calibration=bins,**result)


def _group(slots,by_key,as_of):
    items=[]
    for slot in slots:
        frame=by_key.get(slot.key,ScoreFrame(slot.key,slot.symbol,slot.event_at,slot.target_start,slot.target_end))
        rows={r.family:r for r in frame.rows}
        common=len(rows)==5 and all(_usable(frame,rows[f]) for f in FAMILIES)
        items.append((frame,rows,common,as_of>slot.target_end+timedelta(seconds=120)))
    counts=dict(planned=len(slots),due=sum(x[3] for x in items),not_yet_due=sum(not x[3] for x in items),
        issued=sum(x[0].issued for x in items),unissued=sum(not x[0].issued for x in items),
        settled=sum(x[0].settled for x in items),common=sum(x[2] for x in items),
        missing=sum(x[0].missing or (x[3] and x[0].issued and not x[0].settled) for x in items),
        late=sum(x[0].late for x in items),quarantined=sum(x[0].quarantined for x in items))
    families={}
    for family in FAMILIES:
        known=[(frame,rows[family],common,due) for frame,rows,common,due in items if family in rows]
        common_values=[(frame,row) for frame,row,common,_ in known if common]
        individual=[(frame,row) for frame,row,_,_ in known if _usable(frame,row)]
        successful=sum(frame.issued and row.status=='computed' for frame,row,_,_ in known)
        resolved=sum(frame.issued and frame.settled and row.status=='computed' for frame,row,_,_ in known)
        family_counts=dict(planned=len(slots),publication_frames=counts['issued'],issued=successful,unissued=len(slots)-successful,
            computed=sum(r.status=='computed' for _,r,_,_ in known),settled=resolved,usable_settled=len(individual),
            common=len(common_values),missing=counts['missing'],late=counts['late'],
            quarantined=counts['quarantined'],failed_fit=sum(r.status=='training_fit_failed' for _,r,_,_ in known),
            numeric_failure=sum(r.status=='fixed_numeric_failure' for _,r,_,_ in known),
            outside_support=sum(r.status=='feature_outside_training_support' for _,r,_,_ in known))
        families[family]=dict(counts=family_counts,individual=_summary(individual),**_summary(common_values))
    return dict(counts=counts,families=families,coverage_passed=counts['common']*5>=counts['planned']*4)


def _loss_json(value):
    if type(value) is float:
        if value==math.inf:return {'loss':'positive_infinity'}
        _need(math.isfinite(value));return value
    if type(value) in (tuple,list):return [_loss_json(v) for v in value]
    if type(value) is dict:return {k:_loss_json(v) for k,v in value.items()}
    _need(value is None or type(value) in (str,int,bool));return value


def evaluate_frames(plan,frames,*,as_of,fit_statuses):
    try:
        slots=planned_slots(plan);as_of=aware(as_of)
        _need(type(frames) is tuple and len(frames)<=len(slots) and type(fit_statuses) is tuple
              and tuple((s,f) for s,f,_ in fit_statuses)==registered.PAIRS
              and all(status in ('succeeded','failed') for _,_,status in fit_statuses))
        training={(s,f):status for s,f,status in fit_statuses}
        expected={s.key:s for s in slots};by_key={}
        for frame in frames:
            _need(type(frame) is ScoreFrame and frame.key in expected and frame.key not in by_key)
            slot=expected[frame.key]
            _need((frame.symbol,frame.event_at,frame.target_start,frame.target_end)==
                  (slot.symbol,slot.event_at,slot.target_start,slot.target_end)
                  and (not frame.issued or frame.event_at<=as_of)
                  and (not frame.settled or frame.target_end<=as_of)
                  and (not frame.missing or frame.settled or as_of>frame.target_end+timedelta(seconds=120)))
            for row in frame.rows:
                _need((row.status=='training_fit_failed')==(training[(frame.symbol,row.family)]=='failed'))
            by_key[frame.key]=frame
        groups=[]
        for day in registered.VALIDATION_DAYS:
            for symbol in SYMBOLS:
                selected=tuple(s for s in slots if s.symbol==symbol and s.day==day)
                groups.append(dict(day=day,symbol=symbol,**_group(selected,by_key,as_of)))
        symbols={symbol:_group(tuple(s for s in slots if s.symbol==symbol),by_key,as_of) for symbol in SYMBOLS}
        macro={}
        for family in FAMILIES:
            values=[symbols[s]['families'][family] for s in SYMBOLS]
            macro[family]={name:_mean([v[name] for v in values]) if all(v['count'] and v[name] is not None for v in values) else None
                           for name in METRICS}
        complete=as_of>max(s.target_end for s in slots)+timedelta(seconds=120)
        passed=complete and all(g['coverage_passed'] for g in groups)
        proposed=min(FAMILIES,key=lambda f:macro[f]['brier']) if passed else None
        result=dict(schema=1,phase='validation',as_of=as_of.isoformat(),math_only=True,selection_admission=False,
            test_admission=False,orders=False,trial_units=15,
            training_fits=[dict(symbol=s,family=f,status=status) for s,f,status in fit_statuses],
            complete=complete,engineering_gate_passed=passed,proposed_family=proposed,
            groups=groups,symbols=symbols,macro=macro,
            units=dict(return_unit='fraction',variance_unit='fraction_squared',variance_proxy_mse='fraction_fourth_power',
                       qlike='log_fraction_squared_plus_squared_error_over_predicted_variance',annualized=False),
            source_sha256=source_sha256())
        return _loss_json(result)
    except Exception:raise CommonScoreError('common_comparison_evaluation_failed') from None
