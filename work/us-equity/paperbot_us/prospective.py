"""Predeclared causal risk forecasts and actual outcomes; never an order sender.

First received prices remain immutable. Late/revised observations are evidence,
not permission to backfill a forecast. Phase completion is engineering research,
not strategy acceptance, return prediction or permission to alter the pilot.
"""
from contextlib import closing
from dataclasses import asdict
from datetime import date,datetime,timedelta,timezone
import hashlib,json,math,sqlite3
from pathlib import Path
from zoneinfo import ZoneInfo

from .baseline_study import BaselineCandidate,_returns
from .data import Bar,Calendar,Session,utc_time
from .experiments import ExperimentRegistry,ExperimentError,digest
from .monitor import ObservationArchive
from .regime_baselines import BaselineModel,ReturnObservation,fit_baseline,initial_state,advance_observed
from .research import ResearchDataset
from .runner import _exclusive
from .study import _read,_save


class CampaignError(RuntimeError):pass


def _need(condition):
    if not condition:raise CampaignError('invalid_or_incomplete_prospective_evidence')


def _normal(value):
    def encode(v):
        if isinstance(v,(datetime,date)):return v.isoformat()
        raise TypeError('unsupported_research_record')
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False,default=encode))


def _code_hash():
    paths=tuple(sorted(Path(__file__).parent.glob('*.py')))
    _need(1<=len(paths)<=64 and all(not p.is_symlink() for p in paths))
    return digest({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def _bar_identity(bar):
    return digest({k:v for k,v in _normal(asdict(bar)).items() if k!='received_at'})


def _calendar(value):
    return Calendar([Session(date.fromisoformat(s['day']),s['opens'],s['closes']) for s in value['sessions']],
                    start=value['start'],end=value['end'])


def _days(value):
    _need(type(value) in (tuple,list) and value and len(value)<=30 and all(type(d) is date for d in value))
    _need(tuple(value)==tuple(sorted(set(value))))
    return tuple(value)


def create_campaign(training,*,calendar,validation_days,test_days,context_days,candidates,registry,directory,now,domain='us_equity'):
    try:
        now=utc_time(now);directory=Path(directory).absolute()
        _need(type(training) is ResearchDataset and type(calendar) is Calendar and type(registry) is ExperimentRegistry)
        _need(domain in ('us_equity','synthetic') and registry.path==directory.parent/'shared.sqlite')
        validation_days=_days(validation_days);test_days=_days(test_days);context_days=_days(context_days)
        train_days=tuple(s.day for s in training.sessions);all_used=train_days+validation_days+test_days+context_days
        _need(len(set(all_used))==len(all_used) and set(all_used)<=set(calendar.sessions))
        _need(max(train_days)<min(validation_days) and max(validation_days)<min(test_days))
        _need(calendar.sessions[validation_days[0]].opens>now>=training.downloaded_at)
        _need(calendar.sessions[test_days[-1]].closes-now<=timedelta(days=45))
        expected_context=tuple(d for d in sorted(calendar.sessions) if max(train_days)<d<=max(test_days)
                               and d not in set(validation_days+test_days))
        _need(context_days==expected_context)
        _need(any(max(train_days)<d<min(validation_days) for d in context_days)
              and any(max(validation_days)<d<min(test_days) for d in context_days))
        _need(all(calendar.sessions[s.day]==s for s in training.sessions))
        _need(all((calendar.sessions[d].closes-calendar.sessions[d].opens).total_seconds()%900==0
                  and (calendar.sessions[d].closes-calendar.sessions[d].opens).total_seconds()>=1800
                  for d in validation_days+test_days+context_days))
        _need(type(candidates) in (tuple,list) and 1<=len(candidates)<=16
              and all(type(c) is BaselineCandidate for c in candidates))
        params={digest(asdict(c)):asdict(c) for c in candidates};_need(len(params)==len(candidates))
        _need(len(training.bars)<=10000 and len(training.bars)*len(candidates)<=160000)
        training_value=training.manifest()
        calendar_value=_normal(dict(start=calendar.start,end=calendar.end,sessions=[asdict(s) for _,s in sorted(calendar.sessions.items())]))
        configuration=dict(schema=1,kind='prospective_risk_research',created_at=now.isoformat(),code_sha256=_code_hash(),
            symbol=training.bars[0].symbol,domain=domain,training=training_value,calendar=calendar_value,candidates=params,
            validation_days=[d.isoformat() for d in validation_days],test_days=[d.isoformat() for d in test_days],
            context_days=[d.isoformat() for d in context_days],issue_delay_seconds=120,outcome_grace_seconds=3600,
            minimum_daily_coverage_fraction=.8,revision_limit=8,return_policy='same_session_close_to_close',
            outcome_policy='first_received_closed_bar',late_older_return_policy='archive_but_do_not_rewind_observed_state',
            data_origin='synthetic_fixture' if domain=='synthetic' else 'vendor_training_and_future_actual_receipts',
            future_data_binding='unknown_until_hash_chained_receipt',strategy_accepted=False)
        plan=dict(schema=1,domain=domain,family='ewma_cusum',data_sha256=digest(dict(
                training_sha256=training_value['data_sha256'],calendar_sha256=digest(calendar_value),future='receipt_archive')),
            code_sha256=configuration['code_sha256'],configuration_sha256=digest(configuration),
            split=dict(train=[d.isoformat() for d in train_days],validation=configuration['validation_days'],
                       test=configuration['test_days'],context=configuration['context_days']),
            candidate_ids=sorted(params),trial_budget=len(params),selection_stage='validation',objective='variance_mse_min')
        _need(not directory.is_symlink() and (not directory.exists() or not any(directory.iterdir())))
        directory.mkdir(mode=0o700,exist_ok=True);_save(directory/'campaign.json',configuration)
        identity=registry.register(plan);models={};rows=_returns(training)
        for candidate,parameters in sorted(params.items()):
            registry.start_trial(identity,candidate,'train')
            try:model=fit_baseline(rows,as_of=now,**parameters)
            except (ValueError,ArithmeticError):
                registry.finish_trial(identity,candidate,'train',metrics={},failure='numeric_model_failure');continue
            registry.finish_trial(identity,candidate,'train',metrics=dict(observations=model.training_count,
                initial_variance=model.initial_variance,training_mean=model.training_mean,training_sigma=model.training_sigma))
            models[candidate]=_normal(asdict(model))
        _need(models);model_artifact=dict(experiment_id=identity,models=models)
        _save(directory/'models.json',model_artifact)
        ObservationArchive(directory/'forecasts.sqlite').append(dict(generated_at=now.timestamp(),kind='created',
            configuration_sha256=digest(configuration),models_sha256=digest(model_artifact),experiment_id=identity))
        return open_campaign(directory,digest(configuration))
    except CampaignError:raise
    except Exception:raise CampaignError('prospective_creation_failed') from None


def open_campaign(directory,configuration_sha256):
    try:
        directory=Path(directory).absolute();_need(not directory.is_symlink())
        configuration=_read(directory/'campaign.json')
        _need(digest(configuration)==configuration_sha256 and configuration['schema']==1
              and configuration['kind']=='prospective_risk_research' and configuration['code_sha256']==_code_hash())
        return Campaign(directory,configuration)
    except CampaignError:raise
    except Exception:raise CampaignError('prospective_open_failed') from None


class Campaign:
    def __init__(self,directory,configuration):
        self.directory=directory;self.configuration=configuration;self.configuration_sha256=digest(configuration)
        self.calendar=_calendar(configuration['calendar']);self.registry=ExperimentRegistry(directory.parent/'shared.sqlite')
        self.archive=ObservationArchive(directory/'forecasts.sqlite');self.model_artifact=_read(directory/'models.json')
        self.identity=self.model_artifact['experiment_id']
        self.models={k:BaselineModel(**v) for k,v in self.model_artifact['models'].items()}
        state=self.registry.load(self.identity)
        _need(not state['invalidated'] and state['plan']['configuration_sha256']==self.configuration_sha256)

    def _records(self):
        _need(self.archive.path.is_file() and not self.archive.path.is_symlink())
        with closing(sqlite3.connect(self.archive.path.as_uri()+'?mode=ro',uri=True,timeout=5)) as db:
            db.execute('BEGIN');self.archive._verify_chain(db)
            return [(json.loads(p),h) for p,h in db.execute('SELECT payload,digest FROM observations ORDER BY seq')]

    def _phase(self,day):
        text=day.isoformat()
        return 'validation' if text in self.configuration['validation_days'] else 'test' if text in self.configuration['test_days'] else 'context'

    def snapshot(self):
        try:
            _need(self.configuration['code_sha256']==_code_hash())
            registry_state=self.registry.load(self.identity);_need(not registry_state['invalidated'])
            records=self._records();_need(records)
            first=records[0][0]
            _need(first['kind']=='created' and first['configuration_sha256']==self.configuration_sha256
                  and first['models_sha256']==digest(self.model_artifact) and first['experiment_id']==self.identity)
            state=dict(models=self.models,states={k:initial_state(m) for k,m in self.models.items()},bars={},
                forecasts={},revisions={},revision_count=0,selected=None,finished=False,status='awaiting_observations',
                phase_results={},calendar_proofs={},calendar_mismatch_days=set(),
                strategy_accepted=False,records=len(records),head=records[-1][1])
            prior=records[0][1]
            for record,head in records[1:]:
                now=datetime.fromtimestamp(record['generated_at'],timezone.utc);kind=record['kind']
                if kind=='calendar_checked':
                    proof=record['proof'];day=date.fromisoformat(proof['day'])
                    _need(now.astimezone(ZoneInfo('America/New_York')).date()==day)
                    expected=self.calendar.sessions.get(day)
                    expected=None if expected is None else _normal(asdict(expected))
                    verified=proof['session']==expected
                    _need(type(proof['verified']) is bool and proof['verified']==verified)
                    state['calendar_proofs'][day.isoformat()]=dict(verified=verified,received_at=now)
                    if not verified:state['calendar_mismatch_days'].add(day.isoformat())
                elif kind in ('receipt','revision'):
                    bar=Bar(**record['bar']);self._validate_bar(bar,now);key=bar.starts.isoformat()
                    if kind=='receipt':
                        _need(key not in state['bars']);state['bars'][key]=bar
                        anchor=state['bars'].get((bar.starts-timedelta(minutes=15)).isoformat())
                        if anchor is not None and self.calendar.at(anchor.starts).day==self.calendar.at(bar.starts).day:
                            row=ReturnObservation(bar.ends,max(bar.received_at,anchor.received_at),bar.close/anchor.close-1)
                            for candidate,observed in tuple(state['states'].items()):
                                if row.ends_at>observed.last_ends_at:
                                    state['states'][candidate]=advance_observed(self.models[candidate],observed,row,observed_at=now)
                    else:
                        _need(key in state['bars'] and _bar_identity(bar)!=_bar_identity(state['bars'][key]))
                        values=state['revisions'].setdefault(key,set());_need(_bar_identity(bar) not in values)
                        values.add(_bar_identity(bar));_need(len(values)<=self.configuration['revision_limit']);state['revision_count']+=1
                    self._settle(state)
                elif kind=='forecast':
                    p=record['forecast'];candidate=p['candidate_id'];start=utc_time(p['target_starts_at']);end=utc_time(p['target_ends_at'])
                    key=candidate+'@'+start.isoformat();session=self.calendar.at(start)
                    _need(candidate in self.models and key not in state['forecasts'] and session is not None)
                    phase=self._phase(session.day);anchor=state['bars'].get((start-timedelta(minutes=15)).isoformat())
                    _need(not state['finished'] and phase in ('validation','test') and (phase!='test' or state['selected']==candidate))
                    _need(anchor is not None and session.opens<start and end==start+timedelta(minutes=15) and end<=session.closes)
                    _need(0<=(now-start).total_seconds()<=self.configuration['issue_delay_seconds']
                          and p['issued_at']==now.isoformat() and p['source_prefix_sha256']==prior)
                    _need(p['variance']==state['states'][candidate].variance and p['phase']==phase)
                    proof=state['calendar_proofs'].get(session.day.isoformat(),{})
                    _need(self.configuration['domain']=='synthetic' or proof.get('verified') is True
                          and proof['received_at']<=now and session.day.isoformat() not in state['calendar_mismatch_days'])
                    _need(p['future_horizon_seconds']==(end-now).total_seconds()
                          and p['target_elapsed_seconds']==(now-start).total_seconds()
                          and p['variance_unit']=='fraction_squared_per_15m'
                          and p['variance_scope']=='conditional_full_return_not_remaining_time' and p['annualized'] is False)
                    state['forecasts'][key]=dict(candidate_id=candidate,phase=phase,target_starts_at=start,
                        target_ends_at=end,issued_at=now,variance=p['variance'],source_prefix_sha256=prior,
                        future_horizon_seconds=p['future_horizon_seconds'],target_elapsed_seconds=p['target_elapsed_seconds'],
                        variance_unit=p['variance_unit'],variance_scope=p['variance_scope'],annualized=False,
                        outcome_return=None,outcome_received_at=None)
                    self._settle(state)
                elif kind=='phase_result':
                    phase=record['phase'];_need(phase in ('validation','test') and phase not in state['phase_results'])
                    report=_read(self.directory/(phase+'.json'))
                    _need(digest(report)==record['report_sha256'] and report['archive_head']==prior
                          and report['experiment_id']==self.identity and report['phase']==phase)
                    state['phase_results'][phase]=report
                    if phase=='validation' and report['success']:
                        cutoff=self.calendar.sessions[date.fromisoformat(self.configuration['test_days'][0])].opens
                        _need(utc_time(report['created_at'])<cutoff and utc_time(report['created_at'])<=now
                              and registry_state['selected']==report['selected_candidate_id'])
                        state['selected']=report['selected_candidate_id'];state['status']='validation_complete'
                    else:state['finished']=True;state['status']='completed_evaluation' if report['success'] else report['status']
                else:raise CampaignError('unknown_research_record')
                prior=head
            return state
        except CampaignError:raise
        except Exception:raise CampaignError('prospective_replay_failed') from None

    def _validate_bar(self,bar,now):
        _need(type(bar) is Bar and bar.symbol==self.configuration['symbol'] and bar.received_at<=now)
        identity=('symbol','source','feed','adjustment','interval_seconds','currency','volume_unit')
        _need(all(getattr(bar,k)==self.configuration['training'][k] for k in identity))
        session=self.calendar.at(bar.starts);_need(session is not None)
        _need(session.day.isoformat() in set(self.configuration['validation_days']+self.configuration['test_days']+self.configuration['context_days']))
        bar.validate(as_of=now,session=session)

    def _settle(self,state):
        for forecast in state['forecasts'].values():
            if forecast['outcome_return'] is not None:continue
            bar=state['bars'].get(forecast['target_starts_at'].isoformat())
            anchor=state['bars'].get((forecast['target_starts_at']-timedelta(minutes=15)).isoformat())
            if bar is not None and anchor is not None:
                available=max(bar.received_at,anchor.received_at)
                _need(available>forecast['issued_at'])
                value=bar.close/anchor.close-1
                ReturnObservation(bar.ends,available,value)
                forecast.update(outcome_return=value,outcome_received_at=available)

    def _append(self,kind,now,**body):
        self.archive.append(dict(generated_at=now.timestamp(),kind=kind,**body))

    def record_calendar(self,calendar,*,received_at):
        try:
            now=utc_time(received_at)
            _need(type(calendar) is Calendar and calendar.start==calendar.end
                  and now.astimezone(ZoneInfo('America/New_York')).date()==calendar.start
                  and now>=utc_time(self.configuration['created_at']))
            day=calendar.start;session=calendar.sessions.get(day);expected=self.calendar.sessions.get(day)
            actual=None if session is None else _normal(asdict(session))
            planned=None if expected is None else _normal(asdict(expected));verified=actual==planned
            with _exclusive(self.directory/'collector'):
                state=self.snapshot();previous=state['calendar_proofs'].get(day.isoformat())
                if previous is None or previous['verified']!=verified:
                    self._append('calendar_checked',now,proof=dict(day=day.isoformat(),session=actual,verified=verified))
            return verified and day.isoformat() not in state['calendar_mismatch_days']
        except CampaignError:raise
        except Exception:raise CampaignError('calendar_evidence_failed') from None

    def _start(self,candidate,phase):
        state=self.registry.load(self.identity);trial=state['trials'].get(candidate+':'+phase)
        if trial is None:self.registry.start_trial(self.identity,candidate,phase)
        else:_need(trial['status']=='started')

    def _phase_report(self,state,phase,now,missed=False):
        days=[date.fromisoformat(v) for v in self.configuration[phase+'_days']]
        candidates=sorted(self.models) if phase=='validation' else [state['selected']]
        common=None;indexed={}
        for candidate in candidates:
            rows={p['target_starts_at'].isoformat():p for p in state['forecasts'].values()
                if p['candidate_id']==candidate and p['phase']==phase and p['outcome_return'] is not None
                and p['outcome_received_at']<=self.calendar.at(p['target_starts_at']).closes+timedelta(seconds=3600)}
            indexed[candidate]=rows;common=set(rows) if common is None else common&set(rows)
        common=sorted(common or ());coverage={}
        for day in days:
            session=self.calendar.sessions[day];expected=int((session.closes-session.opens).total_seconds()//900)-1
            count=sum(self.calendar.at(utc_time(t)).day==day for t in common)
            coverage[day.isoformat()]=dict(expected=expected,scored=count,
                required=math.ceil(expected*self.configuration['minimum_daily_coverage_fraction']))
        success=bool(common) and all(v['scored']>=v['required'] for v in coverage.values()) and not missed
        metrics={}
        if success:
            for candidate,rows in indexed.items():
                values=[rows[t] for t in common];initial=self.models[candidate].initial_variance
                metrics[candidate]=dict(observations=len(values),
                    variance_mse=math.fsum((p['variance']-p['outcome_return']**2)**2 for p in values)/len(values),
                    frozen_variance_mse=math.fsum((initial-p['outcome_return']**2)**2 for p in values)/len(values))
        selected=min(metrics,key=lambda c:(metrics[c]['variance_mse'],c)) if metrics and phase=='validation' else state['selected']
        return dict(schema=1,experiment_id=self.identity,configuration_sha256=self.configuration_sha256,
            phase=phase,created_at=now.isoformat(),archive_head=state['head'],success=success,
            status='completed_phase' if success else 'missed_selection_deadline' if missed else 'incomplete_data',
            selected_candidate_id=selected if success else None,metrics=metrics,daily_coverage=coverage,
            common_targets=common,common_targets_sha256=digest(common),strategy_accepted=False,
            limitations=['iex_single_symbol','first_receipt_prices','missingness_and_dependent_samples',
                         'engineering_coverage_not_statistical_acceptance','no_trade_or_profit_claim'])

    def _finalize(self,state,now):
        phase='validation' if 'validation' not in state['phase_results'] else 'test'
        if state['finished']:return state
        end_day=date.fromisoformat(self.configuration[phase+'_days'][-1])
        if now<self.calendar.sessions[end_day].closes+timedelta(seconds=3600):return state
        cutoff=self.calendar.sessions[date.fromisoformat(self.configuration['test_days'][0])].opens
        missed=phase=='validation' and now>=cutoff
        path=self.directory/(phase+'.json');previous=_read(path) if path.exists() else None
        if (missed and previous is not None and previous['success'] and utc_time(previous['created_at'])<cutoff
                and previous['configuration_sha256']==self.configuration_sha256
                and previous['archive_head']==state['head']
                and self.registry.load(self.identity)['selected']==previous['selected_candidate_id']):
            # The choice and its evidence were already committed before test;
            # complete only the missing archive marker, without a new selection.
            missed=False
        report=self._phase_report(state,phase,now,missed)
        if previous is not None:
            report['created_at']=previous['created_at'];_need(previous==report)
        else:_save(path,report)
        candidates=sorted(self.models) if phase=='validation' else [state['selected']]
        for candidate in candidates:
            trial=self.registry.load(self.identity)['trials'].get(candidate+':'+phase)
            if trial is None:self.registry.start_trial(self.identity,candidate,phase)
            self.registry.finish_trial(self.identity,candidate,phase,metrics=report['metrics'].get(candidate,{}) if report['success'] else {},
                failure=None if report['success'] else 'invalid_input')
        if phase=='validation' and report['success']:self.registry.select(self.identity,report['selected_candidate_id'])
        else:self.registry.finish(self.identity,report_sha256=digest(report),completed=report['success'])
        self._append('phase_result',now,phase=phase,report_sha256=digest(report))
        return self.snapshot()

    def observe(self,bars,*,now,calendar_verified):
        try:
            now=utc_time(now);_need(type(calendar_verified) is bool and now>=utc_time(self.configuration['created_at']))
            _need(type(bars) in (tuple,list) and len(bars)<=1000)
            for bar in bars:self._validate_bar(bar,now)
            with _exclusive(self.directory/'collector'):
                state=self.snapshot()
                day=now.astimezone(ZoneInfo('America/New_York')).date().isoformat()
                proof=state['calendar_proofs'].get(day,{})
                if (not calendar_verified or self.configuration['domain']!='synthetic'
                        and (not proof.get('verified') or day in state['calendar_mismatch_days'])):
                    return dict(status='calendar_unverified',forecasts=len(state['forecasts']),strategy_accepted=False)
                # Missing the pretest cutoff is terminal before any test input
                # can affect a choice. Otherwise ingest eligible on-time outcomes
                # before deciding the just-matured phase's completeness.
                test_open=self.calendar.sessions[date.fromisoformat(self.configuration['test_days'][0])].opens
                pending='validation' if 'validation' not in state['phase_results'] else 'test'
                if (self.directory/(pending+'.json')).exists() or 'validation' not in state['phase_results'] and now>=test_open:
                    state=self._finalize(state,now)
                if state['finished']:return dict(status=state['status'],forecasts=len(state['forecasts']),strategy_accepted=False)
                for bar in sorted(bars,key=lambda b:b.starts):
                    key=bar.starts.isoformat();known=state['bars'].get(key)
                    if known is None:self._append('receipt',now,bar=_normal(asdict(bar)));state=self.snapshot()
                    elif _bar_identity(known)!=_bar_identity(bar) and _bar_identity(bar) not in state['revisions'].get(key,set()):
                        _need(len(state['revisions'].get(key,set()))<self.configuration['revision_limit'])
                        self._append('revision',now,bar=_normal(asdict(bar)));state=self.snapshot()
                state=self._finalize(state,now)
                if state['finished']:return dict(status=state['status'],forecasts=len(state['forecasts']),strategy_accepted=False)
                session=self.calendar.at(now)
                if session is not None:
                    phase=self._phase(session.day);index=int((now-session.opens).total_seconds()//900)
                    start=session.opens+timedelta(seconds=index*900);end=start+timedelta(minutes=15)
                    anchor=state['bars'].get((start-timedelta(minutes=15)).isoformat())
                    eligible=sorted(self.models) if phase=='validation' else [state['selected']] if phase=='test' and state['selected'] else []
                    if (index>=1 and end<=session.closes and anchor is not None
                            and (now-start).total_seconds()<=self.configuration['issue_delay_seconds']):
                        for candidate in eligible:
                            key=candidate+'@'+start.isoformat()
                            if key in state['forecasts']:continue
                            self._start(candidate,phase)
                            self._append('forecast',now,forecast=dict(candidate_id=candidate,phase=phase,
                                target_starts_at=start.isoformat(),target_ends_at=end.isoformat(),issued_at=now.isoformat(),
                                variance=state['states'][candidate].variance,source_prefix_sha256=state['head'],
                                future_horizon_seconds=(end-now).total_seconds(),target_elapsed_seconds=(now-start).total_seconds(),
                                variance_unit='fraction_squared_per_15m',variance_scope='conditional_full_return_not_remaining_time',annualized=False))
                            state=self.snapshot()
                return dict(status=state['status'] if state['phase_results'] else 'collecting' if session else 'awaiting_session',
                    forecasts=len(state['forecasts']),settled=sum(p['outcome_return'] is not None for p in state['forecasts'].values()),
                    selected_candidate_id=state['selected'],strategy_accepted=False)
        except CampaignError:raise
        except Exception:raise CampaignError('prospective_observation_failed') from None
