"""Durable first validation publications; no test, selection or order authority.

Seals detect accidental corruption, not an administrator rewriting all evidence.
Recovery uses captured math plus current permissions and opaque integrity hashes;
later source prices are never decoded into or used to recompute a forecast.
An uncertain first numeric attempt permanently consumes its five candidate slots.
"""
from contextlib import closing,contextmanager
from dataclasses import asdict,dataclass
from datetime import datetime,timedelta
import hashlib
import os
from pathlib import Path
import sqlite3

from paperbot_us import alpaca
from paperbot_us.runner import _exclusive
from . import forecast_inputs as inputs,candidate_forecasts as maths,common_registration as common
from . import canonical_history as canonical,training_journal as training,training_codec as codec
from .contracts import aware,digest,is_sha
from .training_data import _watch_versions

MAX_EVENTS=8192
MAX_EVENT_BYTES=262144


class ValidationIssueError(ValueError):pass


def _need(value):
    if not value:raise ValidationIssueError('validation_publication_unavailable')


def source_sha256():
    return digest(dict(protocol=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs=inputs.source_sha256(),maths=maths.source_sha256(),canonical=canonical.source_sha256()))


def _empty():
    return dict(binding=None,frames={},head_seq=0,head_digest='0'*64,last_at=None)


def _hash(previous,payload):return hashlib.sha256((previous+'\0'+payload).encode()).hexdigest()


class IssueClock(inputs.InputClock):
    def __call__(self):
        at=super().__call__();day=self.event_at.astimezone(self.base.zone).date().isoformat()
        _need(day in common.VALIDATION_DAYS and at.astimezone(self.base.zone).date().isoformat()==day)
        return at


def _training_binding(receipt):
    return dict(path=receipt.book_path,device=receipt.book_device,inode=receipt.book_inode,
        head_seq=receipt.head_seq,head_digest=receipt.head_digest,
        snapshot_sha256=receipt.training.seal,fits_sha256=receipt.fits.seal)


def _segment(value):
    if type(value) is inputs.SegmentProof:return value
    raw=dict(value)
    for name in ('raw_last_at','feature_last_at'):raw[name]=aware(raw[name])
    return inputs.SegmentProof(**raw)


def _same_source(a,b):
    ignored={'raw_head_seq','raw_head_digest','raw_last_at','feature_head_seq','feature_head_digest','feature_last_at'}
    return {k:v for k,v in asdict(a).items() if k not in ignored}=={k:v for k,v in asdict(b).items() if k not in ignored}


def _prefix(db,table,sequence,sha):
    _need(table in ('events','observations','canonical_v2_events'))
    _need(db.execute('SELECT digest FROM '+table+' WHERE seq=?',(sequence,)).fetchone()==(sha,))


def _current(binding,registry,training_journal,*,timezone_file,guard):
    _need(type(registry) is canonical.CanonicalRegistry and type(training_journal) is training.TrainingJournal
          and str(registry.path)==binding['canonical']['path']
          and str(training_journal.path)==binding['training']['path'])
    identity=canonical._identity(registry.path)
    _need(identity[:2]==(binding['canonical']['device'],binding['canonical']['inode']))
    receipt=common.load_registration(registry,binding['identity'],timezone_file=timezone_file,clock=guard)
    _need(receipt.plan_json==binding['plan_json'] and receipt.plan()['code_sha256']==source_sha256())
    with closing(sqlite3.connect(registry.path.as_uri()+'?mode=ro',uri=True)) as db:
        _prefix(db,'canonical_v2_events',binding['canonical']['head_seq'],binding['canonical']['head_digest'])
    trained=training_journal.load_acknowledged(clock=guard)
    _need(_training_binding(trained)==binding['training'])
    return receipt,trained


def _authority_head(binding,captured,registry,training_journal,guard):
    """Recheck previously replayed authority while held version watches remain live.

    No model cache crosses an invocation. A changed head, physical replacement,
    checksum or concurrent commit rejects; only an identical verified prefix is
    accepted, avoiding fifteen full training replays in one issue transaction.
    """
    now=guard()
    _need(canonical._identity(registry.path)[:2]==(binding['canonical']['device'],binding['canonical']['inode'])
          and training_journal._identity()[:2]==(binding['training']['device'],binding['training']['inode']))
    for path,table,names,expected in (
        (registry.path,'canonical_v2_head',('version','source_sha256','seq','digest','last_at'),
         (2,canonical.source_sha256(),captured.registration.canonical_head_seq,captured.registration.canonical_head_digest)),
        (training_journal.path,'metadata',('version','source_sha256','head_seq','head_digest','last_at'),
         (1,training.source_sha256(),binding['training']['head_seq'],binding['training']['head_digest']))):
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
            rows=db.execute('SELECT '+','.join(names)+',checksum FROM '+table).fetchall()
            _need(len(rows)==1 and tuple(rows[0][:4])==expected
                  and rows[0][-1]==digest(dict(zip(names,rows[0][:-1]))) and aware(rows[0][-2])<=now)
    guard()


@contextmanager
def _verified_epoch(paths,guard):
    """Hold versions AND pathname identities across the reads they protect."""
    identities=tuple(inputs._identity(p) for p in paths)
    with _watch_versions(paths,guard) as versions:
        def stable():
            _need(versions() and tuple(inputs._identity(p) for p in paths)==identities)
        stable()
        yield stable
        stable()


def _opaque_source_chain(pair,proof,guard):
    """Bounded integrity-only streaming; no later bar/label value is decoded.

    All timestamp/reserved-period vetoes must have passed before this call.
    Comparing stored digest cells alone cannot authenticate a descendant.
    """
    for path,table,maximum,head in (
        (pair.raw_book.path,'events',50000,(proof.raw_head_seq,proof.raw_head_digest)),
        (pair.feature_journal.path,'observations',100000,(proof.feature_head_seq,proof.feature_head_digest))):
        _need(inputs._identity(path)[2]<=268435456)
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,isolation_level=None)) as db:
            db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
            if table=='events':
                names=('version','plan_sha256','source_sha256','head_seq','head_digest','last_at')
                rows=db.execute('SELECT '+','.join(names)+',anchor_digest FROM metadata').fetchall()
                _need(len(rows)==1)
                meta=dict(zip(names,rows[0][:-1]))
                _need(meta['version']==1 and rows[0][-1]==digest(meta)
                      and meta['plan_sha256']==proof.raw_plan_sha256
                      and meta['source_sha256']==proof.raw_source_sha256
                      and aware(meta['last_at'])==proof.raw_last_at<=guard()
                      and (meta['head_seq'],meta['head_digest'])==head)
            else:
                rows=db.execute('SELECT version,started_at,head_seq,head_digest FROM metadata').fetchall()
                _need(len(rows)==1 and rows[0][0]==1 and rows[0][2:]==head)
                started=db.execute("SELECT json_extract(payload,'$.observing_started_at') FROM observations WHERE seq=1").fetchone()
                _need(started is not None and started[0]==rows[0][1]<=guard().timestamp())
            count=0;previous='0'*64
            for seq,payload,prior,sha in db.execute('SELECT seq,payload,previous_digest,digest FROM '+table+' ORDER BY seq'):
                count+=1
                _need(seq==count<=maximum and type(payload) is str and len(payload)<=16384
                      and prior==previous and sha==_hash(prior,payload))
                previous=sha
                if count%64==0:guard()
            _need((count,previous)==head);db.rollback()
        guard()


def _sources(binding,pairs,trained,guard,*,issue_input=None,exact=False):
    _need(type(pairs) is tuple and len(pairs)==2 and all(type(p) is inputs.SourcePair for p in pairs))
    proofs=tuple(inputs._metadata(p,i,trained,guard) for i,p in enumerate(pairs))
    anchors=tuple(_segment(p) for p in binding['segments'])
    for i,(pair,proof,anchor) in enumerate(zip(pairs,proofs,anchors)):
        _need(_same_source(proof,anchor))
        checks=[anchor]
        if issue_input is not None:
            captured=issue_input.segments[i];_need(_same_source(proof,captured));checks.append(captured)
            if exact:_need(proof==captured)
        _opaque_source_chain(pair,proof,guard)
        for path,table,seq_field,sha_field in ((pair.raw_book.path,'events','raw_head_seq','raw_head_digest'),
                (pair.feature_journal.path,'observations','feature_head_seq','feature_head_digest')):
            with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
                for check in checks:
                    _need(getattr(proof,seq_field)>=getattr(check,seq_field))
                    _prefix(db,table,getattr(check,seq_field),getattr(check,sha_field))
    if issue_input is not None:
        day=issue_input.bundle.session_day
        index=next(i for i,p in enumerate(proofs) if p.start_day<=day<=p.end_day)
        expected=next(s for s in issue_input.registration.plan()['calendar']['sessions'] if s['day']==day)
        with closing(sqlite3.connect(pairs[index].raw_book.path.as_uri()+'?mode=ro',uri=True)) as db:
            # Calendars only: no bar values, even during much later score loading.
            values=db.execute("SELECT json_extract(payload,'$.body.session') FROM events "
                "WHERE json_extract(payload,'$.kind')='calendar' AND json_extract(payload,'$.body.day')=?",(day,)).fetchall()
            _need(values and all(alpaca.decode_json(v[0])==expected for v in values))
        with closing(sqlite3.connect(pairs[index].feature_journal.path.as_uri()+'?mode=ro',uri=True)) as db:
            key=inputs.feature_pipeline._entry_key(issue_input.symbol,issue_input.bundle.vector.event_at)
            count=db.execute("SELECT COUNT(*) FROM observations WHERE json_extract(payload,'$.kind')='quarantined' "
                "AND json_extract(payload,'$.body.key')=?",(key,)).fetchone()[0]
            _need(count==0)
    guard();return proofs


@dataclass(frozen=True)
class IssuedReceipt:
    issue_input:inputs.IssueInput
    panel:maths.CandidatePanel
    issued_at:datetime
    commit_observed_at:datetime
    available_at:datetime
    book_path:str
    head_seq:int
    head_digest:str
    confirmation_digest:str
    validation_only:bool
    test_admission:bool
    source_sha256:str
    seal:str

    def __post_init__(self):
        _need(type(self.issue_input) is inputs.IssueInput and type(self.panel) is maths.CandidatePanel
            and self.panel.input_sha256==self.issue_input.seal and self.validation_only is True
            and self.test_admission is False
            and self.available_at>=self.commit_observed_at>self.issued_at>=self.panel.completed_at
            and self.commit_observed_at<=self.issue_input.bundle.vector.event_at+timedelta(seconds=120)
            and self.commit_observed_at<self.panel.target_start
            and all(is_sha(v) for v in (self.head_digest,self.confirmation_digest))
            and self.source_sha256==source_sha256()
            and self.seal==digest({k:v for k,v in asdict(self).items() if k!='seal'}))


class ValidationIssueJournal:
    def __init__(self,path):self.path=Path(path).absolute()

    def _identity(self):
        identity=canonical._identity(self.path)
        _need(identity[2]<=67108864);return identity

    def _apply(self,state,event,sha,source):
        _need(type(event) is dict and set(event)=={'kind','at','body','source_sha256'}
              and event['source_sha256']==source and is_sha(sha))
        at=aware(event['at']);body=event['body'];kind=event['kind']
        _need(type(body) is dict and (state['last_at'] is None or at>=state['last_at']))
        if kind=='created':
            _need(state['binding'] is None and set(body)=={'path','device','inode','identity','plan_json',
                  'canonical','training','segments','timezone_file','validation_only'}
                  and body['path']==str(self.path) and (body['device'],body['inode'])==self._identity()[:2]
                  and body['validation_only'] is True and at<common.FIRST_OPEN
                  and digest(alpaca.decode_json(body['plan_json']))==body['identity']
                  and alpaca.decode_json(body['plan_json'])['code_sha256']==source)
            state['binding']=body
        else:
            _need(state['binding'] is not None and type(body.get('key')) is str)
            key=body['key'];frame=state['frames'].get(key)
            if kind=='reserved':
                _need(frame is None and set(body)=={'key','input','candidates'})
                captured=inputs.decode_input(body['input'])
                _need(key==inputs.feature_pipeline._entry_key(captured.symbol,captured.bundle.vector.event_at)
                      and captured.registration.identity==state['binding']['identity']
                      and captured.registration.plan_json==state['binding']['plan_json']
                      and captured.fits_sha256==state['binding']['training']['fits_sha256']
                      and body['candidates']==list(maths.model_training.FAMILIES)
                      and captured.available_at<=at<=captured.bundle.vector.event_at+timedelta(seconds=120))
                plan=captured.registration.plan();day=captured.bundle.session_day
                session=next(s for s in plan['calendar']['sessions'] if s['day']==day)
                _need(day in common.VALIDATION_DAYS and aware(session['opens'])<=captured.bundle.vector.event_at
                      and captured.bundle.target_end<=aware(session['closes']))
                frame=dict(phase='reserved',input=captured,panel=None,reserved_at=at,reservation_digest=sha,
                    result_digest=None,issued_at=None,issue_digest=None,commit_observed_at=None,
                    confirmation_digest=None,consumed_slots=5)
                state['frames'][key]=frame
            elif kind=='computed':
                _need(frame is not None and frame['phase']=='reserved' and set(body)=={'key','panel','reservation_digest'}
                      and body['reservation_digest']==frame['reservation_digest'])
                panel=maths.decode_panel(body['panel']);captured=frame['input']
                _need(panel.input_sha256==captured.seal and panel.fits_sha256==captured.fits_sha256
                    and panel.feature_sha256==captured.bundle.seal and panel.symbol==captured.symbol
                    and panel.target_start==captured.bundle.target_start and panel.target_end==captured.bundle.target_end
                    and panel.started_at>=frame['reserved_at'] and panel.completed_at<=at)
                frame.update(phase='computed',panel=panel,result_digest=sha)
            elif kind=='issued':
                _need(frame is not None and frame['phase']=='computed' and set(body)=={'key','result_digest'}
                      and body['result_digest']==frame['result_digest'] and frame['panel'].completed_at<=at
                      and at<=frame['input'].bundle.vector.event_at+timedelta(seconds=120)
                      and at<frame['panel'].target_start)
                frame.update(phase='issued',issued_at=at,issue_digest=sha)
            elif kind=='confirmed':
                # at observes the preceding issue COMMIT, never this marker's COMMIT.
                _need(frame is not None and frame['phase']=='issued' and set(body)=={'key','issue_digest'}
                    and body['issue_digest']==frame['issue_digest'] and at>frame['issued_at']
                    and at<=frame['input'].bundle.vector.event_at+timedelta(seconds=120)
                    and at<frame['panel'].target_start)
                frame.update(phase='confirmed',commit_observed_at=at,confirmation_digest=sha)
            elif kind=='abandoned':
                _need(frame is not None and frame['phase']=='reserved' and set(body)=={'key','reservation_digest'}
                      and body['reservation_digest']==frame['reservation_digest'])
                frame.update(phase='abandoned')
            else:_need(False)
        state['last_at']=at

    def _replay(self,db,*,allow_empty=False):
        _need(db.execute('PRAGMA journal_mode').fetchone()[0].lower()=='delete')
        rows=db.execute('SELECT version,source_sha256,head_seq,head_digest,last_at,checksum FROM metadata').fetchall()
        _need(len(rows)==1);meta=dict(zip(('version','source_sha256','head_seq','head_digest','last_at'),rows[0][:-1]))
        source=source_sha256()
        _need(meta['version']==1 and meta['source_sha256']==source and rows[0][-1]==digest(meta)
              and type(meta['head_seq']) is int and 0<=meta['head_seq']<=MAX_EVENTS)
        state=_empty()
        for seq,payload,previous,sha in db.execute('SELECT seq,payload,previous_digest,digest FROM events ORDER BY seq'):
            _need(seq==state['head_seq']+1<=MAX_EVENTS and type(payload) is str
                  and len(payload.encode())<=MAX_EVENT_BYTES and previous==state['head_digest']
                  and sha==_hash(previous,payload))
            self._apply(state,alpaca.decode_json(payload),sha,source);state.update(head_seq=seq,head_digest=sha)
        _need(state['head_seq']==meta['head_seq'] and state['head_digest']==meta['head_digest']
              and (state['head_seq']>0 or allow_empty)
              and (state['last_at'] is None or state['last_at']==aware(meta['last_at']))
              and source_sha256()==source)
        return state,meta

    def _write(self,db,state,meta,kind,body,at):
        _need(state['head_seq']<MAX_EVENTS)
        payload=codec.encode(dict(kind=kind,at=aware(at),body=body,source_sha256=meta['source_sha256']))
        _need(len(payload.encode())<=MAX_EVENT_BYTES)
        previous=state['head_digest'];sha=_hash(previous,payload)
        self._apply(state,alpaca.decode_json(payload),sha,meta['source_sha256'])
        seq=state['head_seq']+1
        db.execute('INSERT INTO events VALUES(?,?,?,?)',(seq,payload,previous,sha))
        meta.update(head_seq=seq,head_digest=sha,last_at=aware(at).isoformat())
        db.execute('UPDATE metadata SET head_seq=?,head_digest=?,last_at=?,checksum=?',
                   (seq,sha,meta['last_at'],digest(meta)))
        return sha

    def _commit(self,db,kind):db.commit()

    def initialize(self,registry,identity,training_journal,pairs,*,timezone_file,clock):
        try:
            guard=canonical.Clock(timezone_file,clock);_need(guard()<common.FIRST_OPEN)
            _need(is_sha(identity) and self.path==registry.path.parent/('r2-validation-'+identity+'.sqlite'))
            receipt=common.load_registration(registry,identity,timezone_file=timezone_file,clock=guard)
            trained=training_journal.load_acknowledged(clock=guard)
            _need(receipt.plan()['code_sha256']==source_sha256() and receipt.plan()['data_sha256']==trained.training.seal)
            stat=canonical._identity(registry.path)
            binding=dict(identity=identity,plan_json=receipt.plan_json,
                canonical=dict(path=str(registry.path),device=stat[0],inode=stat[1],
                    head_seq=receipt.canonical_head_seq,head_digest=receipt.canonical_head_digest),
                training=_training_binding(trained),timezone_file=str(Path(timezone_file).absolute()),validation_only=True)
            _need(type(pairs) is tuple and len(pairs)==2)
            binding['segments']=[asdict(inputs._metadata(p,i,trained,guard)) for i,p in enumerate(pairs)]
            _need(guard()<common.FIRST_OPEN and self.path.parent.is_dir() and not self.path.parent.is_symlink())
            with _exclusive(self.path):
                fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600);os.close(fd)
                with closing(sqlite3.connect(self.path,isolation_level=None)) as db:
                    db.execute('PRAGMA synchronous=FULL');db.execute('BEGIN IMMEDIATE')
                    db.execute('CREATE TABLE metadata(version INTEGER,source_sha256 TEXT,head_seq INTEGER,head_digest TEXT,last_at TEXT,checksum TEXT)')
                    db.execute('CREATE TABLE events(seq INTEGER PRIMARY KEY,payload TEXT,previous_digest TEXT,digest TEXT)')
                    at=guard();_need(at<common.FIRST_OPEN)
                    meta=dict(version=1,source_sha256=source_sha256(),head_seq=0,head_digest='0'*64,last_at=at.isoformat())
                    db.execute('INSERT INTO metadata VALUES(?,?,?,?,?,?)',tuple(meta.values())+(digest(meta),))
                    stat=self._identity();binding.update(path=str(self.path),device=stat[0],inode=stat[1])
                    self._write(db,_empty(),meta,'created',binding,at);_need(guard()<common.FIRST_OPEN);db.commit()
                if os.name!='nt':
                    directory=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY)
                    try:os.fsync(directory)
                    finally:os.close(directory)
                _need(guard()<common.FIRST_OPEN);self.snapshot()
        except Exception:raise ValidationIssueError('validation_journal_initialization_failed') from None

    def snapshot(self):
        try:
            before=self._identity()
            with closing(sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True,isolation_level=None)) as db:
                db.execute('PRAGMA query_only=ON');version=db.execute('PRAGMA data_version').fetchone()[0]
                db.execute('BEGIN');state,_=self._replay(db);db.rollback()
                _need(db.execute('PRAGMA data_version').fetchone()[0]==version)
            _need(self._identity()==before);return state
        except Exception:raise ValidationIssueError('validation_journal_integrity_failed') from None

    def _before_read(self,at):
        """Timestamp/checksum only; reject future journal payloads before replay."""
        self._identity()
        with closing(sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True)) as db:
            rows=db.execute('SELECT version,source_sha256,head_seq,head_digest,last_at,checksum FROM metadata').fetchall()
            _need(len(rows)==1 and rows[0][0]==1 and rows[0][1]==source_sha256()
                  and aware(rows[0][-2])<=at and rows[0][-1]==digest(dict(zip(
                      ('version','source_sha256','head_seq','head_digest','last_at'),rows[0][:-1]))))

    def _append(self,state,kind,body,at,*,clock,check):
        with closing(sqlite3.connect(self.path.as_uri()+'?mode=rw',uri=True,isolation_level=None,timeout=5)) as db:
            db.execute('PRAGMA synchronous=FULL');db.execute('BEGIN IMMEDIATE')
            current,meta=self._replay(db)
            _need((current['head_seq'],current['head_digest'])==(state['head_seq'],state['head_digest']))
            check();stamped=clock();_need(stamped>=at)
            self._write(db,current,meta,kind,body,stamped);check();clock();self._commit(db,kind)
        check();clock()

    def issue_once(self,feature_key,*,registry,training_journal,pairs,timezone_file,clock):
        try:
            guard=IssueClock(feature_key,timezone_file,clock);at=guard()  # Before any SQL.
            with _exclusive(self.path):
                self._before_read(at)
                state=self.snapshot();binding=state['binding'];_need(at>=state['last_at'])
                frame=state['frames'].get(feature_key)
                if frame is not None and frame['phase'] in ('confirmed','abandoned'):return state
                paths=tuple(p for pair in pairs for p in (pair.raw_book.path,pair.feature_journal.path))+(registry.path,training_journal.path)
                with _verified_epoch(paths,guard) as stable:
                    receipt,trained=_current(binding,registry,training_journal,timezone_file=timezone_file,guard=guard)
                    stable();_sources(binding,pairs,trained,guard);stable()
                    if frame is not None and frame['phase']=='reserved':
                        self._append(state,'abandoned',dict(key=feature_key,reservation_digest=frame['reservation_digest']),
                                     guard(),clock=guard,check=stable)
                        return self.snapshot()
                    if frame is None:
                        captured=inputs.capture_issue_input(trained,receipt,pairs,feature_key,timezone_file=timezone_file,clock=guard)
                    else:captured=frame['input']
                    def check():
                        stable()
                        _authority_head(binding,captured,registry,training_journal,guard)
                        _sources(binding,pairs,trained,guard,issue_input=captured,exact=True)
                        stable()
                    check()
                    if frame is None:
                        self._append(state,'reserved',dict(key=feature_key,input=inputs.encode_input(captured),
                            candidates=list(maths.model_training.FAMILIES)),guard(),clock=guard,check=check)
                        state=self.snapshot();frame=state['frames'][feature_key]  # Confirm reservation COMMIT first.
                        panel=maths.compute_candidates(trained,captured,clock=guard)
                        self._append(state,'computed',dict(key=feature_key,panel=codec.encode(panel),
                            reservation_digest=frame['reservation_digest']),guard(),clock=guard,check=check)
                        state=self.snapshot();frame=state['frames'][feature_key]
                    if frame['phase']=='computed':
                        self._append(state,'issued',dict(key=feature_key,result_digest=frame['result_digest']),
                                     guard(),clock=guard,check=check)
                        state=self.snapshot();frame=state['frames'][feature_key]  # Read already committed issue.
                    _need(frame['phase']=='issued');check()
                    self._append(state,'confirmed',dict(key=feature_key,issue_digest=frame['issue_digest']),
                                 guard(),clock=guard,check=check)
                    return self.snapshot()
        except Exception:raise ValidationIssueError('validation_publication_refused') from None

    def load_issued(self,feature_key,*,clock):
        try:
            started=aware(clock());self._before_read(started);state=self.snapshot();binding=state['binding']
            guard=canonical.Clock(binding['timezone_file'],clock);guard()
            frame=state['frames'].get(feature_key);_need(frame is not None and frame['phase']=='confirmed')
            captured=frame['input']
            registry=canonical.CanonicalRegistry(binding['canonical']['path'])
            training_journal=training.TrainingJournal(binding['training']['path'])
            pairs=tuple(inputs.SourcePair(inputs.receipt_book.ReceiptBook(p['raw_path']),
                inputs.feature_pipeline.FeatureJournal(p['feature_path'])) for p in binding['segments'])
            paths=tuple(p for pair in pairs for p in (pair.raw_book.path,pair.feature_journal.path))+(registry.path,training_journal.path,self.path)
            with _verified_epoch(paths,guard) as stable:
                receipt,trained=_current(binding,registry,training_journal,timezone_file=binding['timezone_file'],guard=guard)
                stable();_sources(binding,pairs,trained,guard,issue_input=captured);stable()
                final,_=_current(binding,registry,training_journal,timezone_file=binding['timezone_file'],guard=guard)
                _need((final.canonical_head_seq,final.canonical_head_digest)==(receipt.canonical_head_seq,receipt.canonical_head_digest)
                      and self.snapshot()['head_digest']==state['head_digest'])
                stable()
                available=guard();_need(available>=started>=frame['commit_observed_at'])
            fields=dict(issue_input=captured,panel=frame['panel'],issued_at=frame['issued_at'],
                commit_observed_at=frame['commit_observed_at'],available_at=available,book_path=str(self.path),
                head_seq=state['head_seq'],head_digest=state['head_digest'],confirmation_digest=frame['confirmation_digest'],
                validation_only=True,test_admission=False,source_sha256=source_sha256())
            return IssuedReceipt(**fields,seal=digest({**fields,'issue_input':asdict(captured),'panel':asdict(frame['panel'])}))
        except Exception:raise ValidationIssueError('confirmed_validation_issue_unavailable') from None
