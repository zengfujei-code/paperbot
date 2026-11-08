"""Shared research reservations and trial accounting; no trading authority.

Dates are conservative domain-wide information exposure, independent of symbol,
feed, revision or model name. Preserve this database and import known legacy
evidence. Hashes/head checks detect damage, not an owner rewriting all history.
An empty new registry does not certify historical data was never inspected.
"""
from contextlib import closing
from datetime import date
import hashlib,json,math,os,re,sqlite3
from pathlib import Path


class ExperimentError(RuntimeError):pass


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def _need(condition):
    if not condition:raise ExperimentError('invalid_or_conflicting_experiment_evidence')


def _sha(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def _days(value):
    _need(type(value) is list and len(value)<=1000)
    _need(all(type(v) is str and date.fromisoformat(v).isoformat()==v for v in value))
    _need(value==sorted(set(value)))
    return set(value)


def _plan(value):
    _need(type(value) is dict and set(value)=={'schema','domain','family','data_sha256','code_sha256',
        'configuration_sha256','split','candidate_ids','trial_budget','selection_stage','objective'})
    _need(type(value['schema']) is int and value['schema']==1)
    _need(value['domain'] in ('us_equity','synthetic'))
    _need(value['family'] in ('reference','ewma_cusum','hmm','garch','ml','drift','comparison','execution'))
    _need(all(_sha(value[k]) for k in ('data_sha256','code_sha256','configuration_sha256')))
    split=value['split'];_need(type(split) is dict and set(split)=={'train','validation','test','context'})
    groups={k:_days(v) for k,v in split.items()}
    _need(all(groups[k] for k in ('train','validation','test')))
    _need(sum(map(len,groups.values()))==len(set().union(*groups.values())))
    _need(max(split['train'])<min(split['validation']) and max(split['validation'])<min(split['test']))
    _need(not split['context'] or max(split['context'])<=max(split['test']))
    candidates=value['candidate_ids']
    _need(type(candidates) is list and 1<=len(candidates)<=16 and all(_sha(v) for v in candidates))
    _need(len(set(candidates))==len(candidates) and type(value['trial_budget']) is int
          and value['trial_budget']==len(candidates))
    _need((value['selection_stage'],value['objective']) in
          (('train','net_return_max'),('validation','variance_mse_min')))
    return value


def _metrics(value):
    _need(type(value) is dict and len(value)<=32)
    _need(all(isinstance(k,str) and re.fullmatch('[a-z_]{1,48}',k) for k in value))
    _need(all(type(v) in (int,float) and abs(v)<=1e18 and math.isfinite(v) for v in value.values()))


def _winner(experiment):
    plan=experiment['plan'];phase=plan['selection_stage'];eligible={}
    for candidate in plan['candidate_ids']:
        train=experiment['trials'].get(candidate+':train');_need(train and train['status']!='started')
        if train['status']=='failed':continue
        result=experiment['trials'].get(candidate+':'+phase);_need(result and result['status']!='started')
        if result['status']=='succeeded':eligible[candidate]=result['metrics']
    _need(eligible)
    if plan['objective']=='variance_mse_min':
        _need(all('variance_mse' in m and m['variance_mse']>=0 for m in eligible.values()))
        return min(eligible,key=lambda c:(eligible[c]['variance_mse'],c))
    _need(all('net_return_fraction' in m and 'maximum_drawdown_fraction' in m for m in eligible.values()))
    return min(eligible,key=lambda c:(-eligible[c]['net_return_fraction'],eligible[c]['maximum_drawdown_fraction'],c))


def _apply(state,event):
    _need(type(event) is dict and set(event)=={'kind','body'})
    kind=event['kind'];body=event['body'];_need(type(body) is dict)
    if kind=='created':
        _need(not state['created'] and body=={'schema':1});state['created']=True;return True
    _need(state['created'])
    if kind=='exposure':
        _need(set(body)=={'domain','days','source_sha256'})
        _need(body['domain'] in ('us_equity','synthetic') and _sha(body['source_sha256']))
        days=_days(body['days']);_need(days);identity=digest(body)
        if identity in state['imports']:return False
        state['imports'][identity]=body
        for exp in state['experiments'].values():
            plan=exp['plan']
            if plan['domain']==body['domain'] and days&set(plan['split']['validation']+plan['split']['test']):
                exp['invalidated']=True
        return True
    if kind=='registered':
        _need(set(body)=={'plan'});plan=_plan(body['plan']);identity=digest(plan)
        if identity in state['experiments']:return False
        used=set()
        for evidence in state['imports'].values():
            if evidence['domain']==plan['domain']:used.update(evidence['days'])
        for exp in state['experiments'].values():
            if exp['plan']['domain']==plan['domain']:
                used.update(day for days in exp['plan']['split'].values() for day in days)
        _need(not used&set(plan['split']['validation']+plan['split']['test']))
        state['experiments'][identity]=dict(plan=plan,trials={},selected=None,report_sha256=None,
                                            completed=None,invalidated=False)
        return True
    _need('experiment_id' in body and _sha(body['experiment_id']))
    experiment=state['experiments'].get(body['experiment_id']);_need(experiment is not None)
    _need(not experiment['invalidated'])
    if kind=='finished':
        _need(set(body)=={'experiment_id','report_sha256','completed'} and _sha(body['report_sha256'])
              and type(body['completed']) is bool)
        if experiment['report_sha256'] is not None:
            _need((experiment['report_sha256'],experiment['completed'])==(body['report_sha256'],body['completed']))
            return False
        if body['completed']:
            selected=experiment['selected'];_need(selected is not None)
            result=experiment['trials'].get(selected+':test')
            _need(result is not None and result['status']=='succeeded')
            _need(all(v['status']!='started' for v in experiment['trials'].values()))
        experiment.update(report_sha256=body['report_sha256'],completed=body['completed']);return True
    _need(experiment['report_sha256'] is None)
    if kind=='selected':
        _need(set(body)=={'experiment_id','candidate_id'} and body['candidate_id']==_winner(experiment))
        if experiment['selected'] is not None:
            _need(experiment['selected']==body['candidate_id']);return False
        experiment['selected']=body['candidate_id'];return True
    _need(kind in ('trial_started','trial_finished'))
    expected={'experiment_id','candidate_id','phase'}
    _need(set(body)==(expected if kind=='trial_started' else expected|{'metrics','failure'}))
    candidate=body['candidate_id'];phase=body['phase'];plan=experiment['plan']
    _need(candidate in plan['candidate_ids'] and phase in ('train','validation','test'))
    key=candidate+':'+phase;previous=experiment['trials'].get(key)
    if kind=='trial_started':
        _need(previous is None)
        if phase=='train':_need(experiment['selected'] is None)
        if phase=='validation':
            train=experiment['trials'].get(candidate+':train')
            _need(train is not None and train['status']=='succeeded')
            _need(experiment['selected'] is None or plan['selection_stage']=='train' and candidate==experiment['selected'])
        if phase=='test':
            _need(experiment['selected']==candidate)
            validation=experiment['trials'].get(candidate+':validation')
            _need(validation is not None and validation['status']=='succeeded')
        experiment['trials'][key]=dict(status='started');return True
    _need(previous is not None)
    metrics=body['metrics'];failure=body['failure'];_metrics(metrics)
    _need(failure in (None,'invalid_input','numeric_model_failure','evaluation_failed','process_interrupted'))
    _need(failure is None or not metrics)
    result=dict(status='succeeded' if failure is None else 'failed',metrics=metrics,failure=failure)
    if previous['status']!='started':_need(previous==result);return False
    experiment['trials'][key]=result;return True


class ExperimentRegistry:
    def __init__(self,path):self.path=Path(path).absolute()

    def _path(self):
        _need(not self.path.is_symlink() and self.path.is_file() and self.path.stat().st_size<=64*1024*1024)

    def initialize(self):
        try:
            _need(not self.path.is_symlink())
            fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600);os.close(fd)
            with closing(sqlite3.connect(self.path,timeout=5)) as db:
                db.execute('PRAGMA synchronous=FULL')
                with db:
                    db.execute('CREATE TABLE events(seq INTEGER PRIMARY KEY,payload TEXT NOT NULL,digest TEXT NOT NULL)')
                    db.execute('CREATE TABLE head(seq INTEGER NOT NULL,digest TEXT NOT NULL)')
                    db.execute('INSERT INTO head VALUES(0,?)',('0'*64,))
            self._record('created',{'schema':1})
        except ExperimentError:raise
        except (OSError,sqlite3.Error,ValueError,TypeError):raise ExperimentError('experiment_initialize_failed') from None

    def _read(self,db):
        state=dict(created=False,experiments={},imports={});previous='0'*64;sequence=0
        for seq,payload,checksum in db.execute('SELECT seq,payload,digest FROM events ORDER BY seq'):
            _need(seq==sequence+1 and seq<=10000 and isinstance(payload,str) and len(payload)<=65536)
            value=json.loads(payload);_need(type(value) is dict and set(value)=={'previous_sha256','event'})
            _need(canonical(value)==payload and value['previous_sha256']==previous and digest(value)==checksum)
            _need(_apply(state,value['event']));sequence=seq;previous=checksum
        _need(state['created'] and db.execute('SELECT seq,digest FROM head').fetchall()==[(sequence,previous)])
        return state,sequence,previous

    def _record(self,kind,body):
        try:
            self._path()
            with closing(sqlite3.connect(self.path.as_uri()+'?mode=rw',uri=True,timeout=5)) as db:
                db.execute('PRAGMA synchronous=FULL');db.execute('BEGIN IMMEDIATE')
                if kind=='created':
                    _need(db.execute('SELECT COUNT(*) FROM events').fetchone()==(0,))
                    state=dict(created=False,experiments={},imports={});sequence=0;previous='0'*64
                else:state,sequence,previous=self._read(db)
                event=dict(kind=kind,body=body);changed=_apply(state,event)
                if changed:
                    _need(sequence<10000)
                    value=dict(previous_sha256=previous,event=event);payload=canonical(value)
                    _need(len(payload)<=65536)
                    checksum=digest(value)
                    db.execute('INSERT INTO events VALUES(?,?,?)',(sequence+1,payload,checksum))
                    db.execute('UPDATE head SET seq=?,digest=?',(sequence+1,checksum))
                db.commit()
            return changed
        except ExperimentError:raise
        except (OSError,sqlite3.Error,ValueError,TypeError,KeyError,OverflowError):
            raise ExperimentError('experiment_write_failed') from None

    def register(self,plan):
        self._record('registered',dict(plan=plan));return digest(plan)

    def import_exposure(self,*,domain,days,source_sha256):
        return self._record('exposure',dict(domain=domain,days=days,source_sha256=source_sha256))

    def start_trial(self,identity,candidate,phase):
        return self._record('trial_started',dict(experiment_id=identity,candidate_id=candidate,phase=phase))

    def finish_trial(self,identity,candidate,phase,*,metrics,failure=None):
        return self._record('trial_finished',dict(experiment_id=identity,candidate_id=candidate,phase=phase,
                                                 metrics=metrics,failure=failure))

    def select(self,identity,candidate):
        return self._record('selected',dict(experiment_id=identity,candidate_id=candidate))

    def finish(self,identity,*,report_sha256,completed):
        return self._record('finished',dict(experiment_id=identity,report_sha256=report_sha256,completed=completed))

    def load(self,identity):
        try:
            self._path();_need(_sha(identity))
            with closing(sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True,timeout=5)) as db:
                # Events and retained head must describe the same committed
                # snapshot, even when another research writer commits in between.
                db.execute('BEGIN')
                state,_,_=self._read(db)
            _need(identity in state['experiments']);return state['experiments'][identity]
        except ExperimentError:raise
        except (OSError,sqlite3.Error,ValueError,TypeError,KeyError,OverflowError):
            raise ExperimentError('experiment_read_failed') from None


def import_reference_exposure(registry,evidence):
    """Import a hash-checked legacy study/commitment, without evaluating it again."""
    try:
        _need(type(registry) is ExperimentRegistry and type(evidence) is dict)
        original=evidence
        if set(evidence)=={'content','sha256'}:
            _need(digest(evidence['content'])==evidence['sha256']);evidence=evidence['content']
        config=evidence['configuration'];_need(digest(config)==evidence['study_id'])
        identity=config['data_identity']
        _need(set(identity)=={'symbol','source','feed','adjustment'} and identity['source']=='alpaca'
              and identity['feed']=='iex' and identity['adjustment']=='raw')
        split=config['split'];_need(set(split)=={'train','validation','test','embargo'})
        plan=dict(schema=1,domain='us_equity',family='reference',data_sha256=config['data_sha256'],
            code_sha256=config['code_sha256'],configuration_sha256=digest(config),
            split={k:split['embargo' if k=='context' else k] for k in ('train','validation','test','context')},
            candidate_ids=['0'*64],trial_budget=1,selection_stage='train',objective='net_return_max')
        _plan(plan)
        return registry.import_exposure(domain='us_equity',days=sorted({day for days in split.values() for day in days}),
                                         source_sha256=digest(original))
    except ExperimentError:raise
    except (ValueError,TypeError,KeyError,OverflowError):raise ExperimentError('legacy_exposure_import_failed') from None
