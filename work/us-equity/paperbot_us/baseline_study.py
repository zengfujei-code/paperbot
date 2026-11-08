"""Registered causal variance research, never an accepted trading strategy.

Fit on training sessions, select decay by validation MSE, freeze, then expose
test once. Intermediate observed returns update causal state, not parameters.
Historical close availability is an explicit assumption; archived receipts are
never replaced by invented earlier times. No broker/model/production access.
"""
from dataclasses import asdict,dataclass
from datetime import datetime
import hashlib,json,math
from pathlib import Path

from .data import DataQualityError
from .experiments import ExperimentRegistry,ExperimentError,digest
from .regime_baselines import ReturnObservation,BaselineError,fit_baseline,evaluate_online
from .research import ResearchDataset,SessionSplit,time_split
from .runner import _exclusive,RunnerError
from .study import _save,_read,StudyError


class BaselineStudyError(RuntimeError):pass


@dataclass(frozen=True)
class BaselineCandidate:
    decay:float
    cusum_k:float
    cusum_h:float

    def __post_init__(self):
        for name in ('decay','cusum_k','cusum_h'):
            value=getattr(self,name)
            if type(value) not in (int,float) or abs(value)>1e6 or not math.isfinite(value):
                raise BaselineStudyError('invalid_baseline_parameters')
            object.__setattr__(self,name,float(value))
        if not 0<=self.decay<1 or self.cusum_k<0 or self.cusum_h<=0:
            raise BaselineStudyError('invalid_baseline_parameters')


def _returns(dataset):
    result=[];previous=None;calendar=dataset.calendar
    for bar in dataset.bars:
        if previous is not None and previous.ends==bar.starts:
            before=calendar.at(previous.starts);current=calendar.at(bar.starts)
            if before is not None and current is not None and before.day==current.day:
                available=bar.ends if dataset.availability_policy=='vendor_history_assumed_at_close' else max(bar.received_at,previous.received_at)
                result.append(ReturnObservation(bar.ends,available,bar.close/previous.close-1))
        previous=bar
    return tuple(result)


def _code_hash():
    h=hashlib.sha256()
    for name in ('data','alpaca','quant','research','regime_baselines','temporal_validation',
                 'experiments','baseline_study','study','runner'):
        h.update(name.encode()+b'\0'+Path(__file__).with_name(name+'.py').read_bytes()+b'\0')
    return h.hexdigest()


def _normalized(value):
    def encode(v):
        if isinstance(v,datetime):return v.isoformat()
        raise TypeError('unsupported_baseline_record')
    return json.loads(json.dumps(value,default=encode,sort_keys=True,allow_nan=False))


def _score(points,rows,*,days,calendar,initial):
    selected=[(p,r) for p,r in zip(points,rows) if calendar.at(r.starts_at).day in days]
    if not selected:raise BaselineError('empty_score_window')
    result=dict(observations=len(selected),
        variance_mse=math.fsum((p.variance_forecast-r.return_fraction**2)**2 for p,r in selected)/len(selected),
        frozen_variance_mse=math.fsum((initial-r.return_fraction**2)**2 for p,r in selected)/len(selected),
        mean_variance_forecast=math.fsum(p.variance_forecast/len(selected) for p,r in selected),
        alarm_count=sum(bool(p.alarms) for p,r in selected))
    if any(not math.isfinite(v) or abs(v)>1e18 for v in result.values()):raise BaselineError('score_outside_bounds')
    return result


def run_baseline_study(dataset,*,split,candidates,registry,directory,domain='us_equity'):
    try:
        if (type(dataset) is not ResearchDataset or type(split) is not SessionSplit
                or type(registry) is not ExperimentRegistry or type(candidates) not in (tuple,list)
                or not 1<=len(candidates)<=16 or any(type(c) is not BaselineCandidate for c in candidates)
                or len(dataset.bars)>10000 or len(dataset.bars)*len(candidates)>160000
                or domain not in ('us_equity','synthetic')):
            raise BaselineStudyError('bounded_explicit_baseline_study_required')
        days=tuple(s.day for s in dataset.sessions)
        expected=time_split(days,train_sessions=len(split.train),validation_sessions=len(split.validation),
            test_sessions=len(split.test),embargo_sessions=len(split.embargo)//2)
        if split!=expected or len(split.embargo)<2:raise BaselineStudyError('complete_split_and_embargo_required')
        params={digest(asdict(c)):asdict(c) for c in candidates}
        if len(params)!=len(candidates):raise BaselineStudyError('duplicate_baseline_candidate')
        configuration=dict(schema=1,data_sha256=dataset.manifest()['data_sha256'],code_sha256=_code_hash(),
            data_origin='synthetic_fixture' if domain=='synthetic' else 'vendor_data_with_declared_availability',
            availability_policy=dataset.availability_policy,source='alpaca',feed='iex',adjustment='raw',
            symbol=dataset.bars[0].symbol,return_policy='same_session_close_to_close',candidates=params,
            split={k:[d.isoformat() for d in getattr(split,k)] for k in ('train','validation','test','embargo')},
            objective='validation_variance_mse_min_then_candidate_hash',
            cost_scope='risk_forecast_only_no_trade_returns',formula_version='ewma-second-moment-cusum-v1')
        plan=dict(schema=1,domain=domain,family='ewma_cusum',data_sha256=configuration['data_sha256'],
            code_sha256=configuration['code_sha256'],configuration_sha256=digest(configuration),
            split={k:configuration['split']['embargo' if k=='context' else k] for k in ('train','validation','test','context')},
            candidate_ids=sorted(params),trial_budget=len(params),selection_stage='validation',objective='variance_mse_min')
        directory=Path(directory).absolute()
        if directory.is_symlink():raise BaselineStudyError('research_directory_link_refused')
        directory.mkdir(mode=0o700,exist_ok=True)
        with _exclusive(directory/'baseline-study'):
            identity=registry.register(plan);state=registry.load(identity)
            if state['invalidated']:raise BaselineStudyError('late_exposure_invalidated_study')
            config_path=directory/(identity+'.configuration.json');report_path=directory/(identity+'.report.json')
            if config_path.exists():
                if _read(config_path)!=configuration:raise BaselineStudyError('configuration_changed')
            else:_save(config_path,configuration)
            if report_path.exists():
                report=_read(report_path)
                if (report['experiment_id']!=identity or report['configuration']!=configuration
                        or report['registry_trials']!=state['trials'] or report['selected_candidate_id']!=state['selected']):
                    raise BaselineStudyError('report_registry_mismatch')
                registry.finish(identity,report_sha256=digest(report),completed=report['status']=='completed')
                return report
            if state['trials'] or state['report_sha256'] is not None:
                raise BaselineStudyError('incomplete_study_requires_diagnosis_no_retry')
            models={};validation={};training={}
            for candidate,parameters in sorted(params.items()):
                registry.start_trial(identity,candidate,'train')
                try:
                    rows=_returns(dataset.subset(split.train))
                    model=fit_baseline(rows,as_of=max(r.available_at for r in rows),**parameters)
                    train_metrics=dict(observations=model.training_count,initial_variance=model.initial_variance,
                                       training_mean=model.training_mean,training_sigma=model.training_sigma)
                except (BaselineError,DataQualityError,ValueError,ArithmeticError):
                    registry.finish_trial(identity,candidate,'train',metrics={},failure='numeric_model_failure');continue
                registry.finish_trial(identity,candidate,'train',metrics=train_metrics)
                models[candidate]=model;training[candidate]=_normalized(asdict(model))
                registry.start_trial(identity,candidate,'validation')
                try:
                    through_validation=tuple(day for day in days if split.train[-1]<day<=split.validation[-1])
                    rows=_returns(dataset.subset(through_validation))
                    points=evaluate_online(model,rows,as_of=max(r.available_at for r in rows))
                    metrics=_score(points,rows,days=split.validation,calendar=dataset.calendar,initial=model.initial_variance)
                except (BaselineError,DataQualityError,ValueError,ArithmeticError):
                    registry.finish_trial(identity,candidate,'validation',metrics={},failure='evaluation_failed');continue
                registry.finish_trial(identity,candidate,'validation',metrics=metrics);validation[candidate]=metrics
            winner=None;test=None;test_points=[];status='failed_no_selection'
            if validation:
                winner=min(validation,key=lambda c:(validation[c]['variance_mse'],c));registry.select(identity,winner)
                registry.start_trial(identity,winner,'test')
                try:
                    # Only after durable selection is the test suffix read.
                    rows=_returns(dataset.subset(tuple(day for day in days if day>split.train[-1])))
                    points=evaluate_online(models[winner],rows,as_of=max(r.available_at for r in rows))
                    test=_score(points,rows,days=split.test,calendar=dataset.calendar,initial=models[winner].initial_variance)
                    calendar=dataset.calendar
                    test_points=[_normalized(asdict(p)) for p,r in zip(points,rows)
                                 if calendar.at(r.starts_at).day in split.test]
                except (BaselineError,DataQualityError,ValueError,ArithmeticError):
                    registry.finish_trial(identity,winner,'test',metrics={},failure='evaluation_failed');status='failed_test'
                else:
                    registry.finish_trial(identity,winner,'test',metrics=test);status='completed'
            report=dict(schema=1,experiment_id=identity,configuration=configuration,status=status,
                selected_candidate_id=winner,training=training,validation=validation,test=test,test_forecasts=test_points,
                registry_trials=registry.load(identity)['trials'],strategy_accepted=False,
                availability_policy=dataset.availability_policy,forecast_unit='fraction_squared_per_15m',
                cost_scope=configuration['cost_scope'],limitations=['fixed_symbol_iex_only','availability_policy_is_explicit_assumption_or_receipt',
                    'no_pit_universe_or_company_action_claim','no_market_edge_or_alarm_calibration_acceptance',
                    'registry_history_completeness_not_certified','no_trades_or_profit_claim'])
            _save(report_path,report);registry.finish(identity,report_sha256=digest(report),completed=status=='completed')
            return report
    except BaselineStudyError:raise
    except (ExperimentError,StudyError,RunnerError,DataQualityError,ValueError,TypeError,KeyError,OSError,OverflowError):
        raise BaselineStudyError('registered_baseline_study_failed') from None
