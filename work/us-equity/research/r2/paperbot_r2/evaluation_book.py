"""Bounded physical/source-bound SQLite storage for validation evidence.

Historical snapshots grant no current evaluation or phase permission. Domain
callers must hold all upstream watches and enforce their own clock/admission.
"""
from contextlib import closing
from datetime import datetime
import hashlib
import os
from pathlib import Path
import sqlite3

from paperbot_us import alpaca
from paperbot_us.runner import _exclusive
from . import common_registration,training_codec as codec
from .contracts import aware,digest,is_sha

MAX_EVENTS=8192
MAX_EVENT_BYTES=262144
MAX_BOOK_BYTES=67108864


class EvaluationBookError(ValueError):pass


def need(value):
    if not value:raise EvaluationBookError('validation_evidence_book_unavailable')


def _hash(previous,payload):return hashlib.sha256((previous+'\0'+payload).encode()).hexdigest()


def source_sha256():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class EvaluationBook:
    def __init__(self,path):self.path=Path(path).absolute()
    def source_sha256(self):raise NotImplementedError
    def domain_initial(self):raise NotImplementedError
    def domain_apply(self,domain,kind,body,at,sha,binding):raise NotImplementedError

    def _identity(self):
        need(self.path.is_file() and not self.path.is_symlink() and self.path.parent.is_dir()
             and not self.path.parent.is_symlink())
        stat=self.path.stat();need(stat.st_size<=MAX_BOOK_BYTES)
        return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns

    def _empty(self):
        return dict(binding=None,initialized=False,created_at=None,created_digest=None,initialized_at=None,
                    domain=self.domain_initial(),head_seq=0,head_digest='0'*64,last_at=None)

    def _apply(self,state,event,sha,source):
        need(type(event) is dict and set(event)=={'kind','at','body','source_sha256'}
             and event['source_sha256']==source and is_sha(sha))
        at=aware(event['at']);body=event['body'];kind=event['kind']
        need(type(body) is dict and (state['last_at'] is None or at>=state['last_at']))
        if kind=='created':
            need(state['binding'] is None and set(body)=={'file','binding'} and type(body['binding']) is dict
                 and body['file']==dict(path=str(self.path),device=self._identity()[0],inode=self._identity()[1])
                 and at<common_registration.FIRST_OPEN)
            state.update(binding=body['binding'],created_at=at,created_digest=sha)
        elif kind=='initialized':
            need(state['binding'] is not None and not state['initialized']
                 and body==dict(created_digest=state['created_digest'])
                 and state['created_at']<at<common_registration.FIRST_OPEN)
            state.update(initialized=True,initialized_at=at)
        else:
            need(state['initialized'])
            self.domain_apply(state['domain'],kind,body,at,sha,state['binding'])
        state['last_at']=at

    def _replay(self,db,*,allow_empty=False):
        need(db.execute('PRAGMA journal_mode').fetchone()[0].lower()=='delete')
        rows=db.execute('SELECT version,source_sha256,head_seq,head_digest,last_at,checksum FROM metadata').fetchall()
        need(len(rows)==1)
        names=('version','source_sha256','head_seq','head_digest','last_at');meta=dict(zip(names,rows[0][:-1]))
        source=self.source_sha256()
        need(is_sha(source) and meta['version']==1 and meta['source_sha256']==source and rows[0][-1]==digest(meta)
             and type(meta['head_seq']) is int and 0<=meta['head_seq']<=MAX_EVENTS)
        state=self._empty()
        for seq,payload,previous,sha in db.execute('SELECT seq,payload,previous_digest,digest FROM events ORDER BY seq'):
            need(seq==state['head_seq']+1<=MAX_EVENTS and type(payload) is str and len(payload.encode())<=MAX_EVENT_BYTES
                 and previous==state['head_digest'] and sha==_hash(previous,payload))
            self._apply(state,alpaca.decode_json(payload),sha,source);state.update(head_seq=seq,head_digest=sha)
        need(state['head_seq']==meta['head_seq'] and state['head_digest']==meta['head_digest']
             and (state['head_seq']>0 or allow_empty)
             and (state['last_at'] is None or state['last_at']==aware(meta['last_at']))
             and self.source_sha256()==source)
        return state,meta

    def _write(self,db,state,meta,kind,body,at):
        need(state['head_seq']<MAX_EVENTS)
        payload=codec.encode(dict(kind=kind,at=aware(at),body=body,source_sha256=meta['source_sha256']))
        need(len(payload.encode())<=MAX_EVENT_BYTES)
        previous=state['head_digest'];sha=_hash(previous,payload)
        self._apply(state,alpaca.decode_json(payload),sha,meta['source_sha256'])
        seq=state['head_seq']+1
        db.execute('INSERT INTO events VALUES(?,?,?,?)',(seq,payload,previous,sha))
        meta.update(head_seq=seq,head_digest=sha,last_at=aware(at).isoformat())
        db.execute('UPDATE metadata SET head_seq=?,head_digest=?,last_at=?,checksum=?',
                   (seq,sha,meta['last_at'],digest(meta)))
        return sha

    def _commit(self,db,kind):db.commit()

    def _directory_sync(self):
        if os.name!='nt':
            directory=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(directory)
            finally:os.close(directory)

    def _before_read(self,at):
        self._identity()
        with closing(sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True)) as db:
            rows=db.execute('SELECT version,source_sha256,head_seq,head_digest,last_at,checksum FROM metadata').fetchall()
            need(len(rows)==1 and rows[0][0]==1 and rows[0][1]==self.source_sha256()
                 and aware(rows[0][-2])<=at and rows[0][-1]==digest(dict(zip(
                     ('version','source_sha256','head_seq','head_digest','last_at'),rows[0][:-1]))))

    def snapshot(self,*,clock=None):
        try:
            started=aware(clock()) if clock is not None else None
            if started is not None:self._before_read(started)
            before=self._identity()
            with closing(sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True,isolation_level=None)) as db:
                db.execute('PRAGMA query_only=ON');version=db.execute('PRAGMA data_version').fetchone()[0]
                db.execute('BEGIN');state,_=self._replay(db);db.rollback()
                need(db.execute('PRAGMA data_version').fetchone()[0]==version)
            need(self._identity()==before)
            if started is not None:need(aware(clock())>=started>=state['last_at'])
            return state
        except Exception:raise EvaluationBookError('validation_evidence_replay_failed') from None

    def append(self,state,kind,body,*,clock,check):
        """Caller holds the book's exclusive lock and all upstream source watches."""
        try:
            started=aware(clock());need(started>=state['last_at'])
            self._identity()
            with closing(sqlite3.connect(self.path.as_uri()+'?mode=rw',uri=True,isolation_level=None,timeout=5)) as db:
                db.execute('PRAGMA synchronous=FULL');db.execute('BEGIN IMMEDIATE')
                current,meta=self._replay(db)
                need((current['head_seq'],current['head_digest'])==(state['head_seq'],state['head_digest']))
                check();stamped=aware(clock());need(stamped>=started)
                self._write(db,current,meta,kind,body,stamped)
                check();before_commit=aware(clock());need(before_commit>=stamped and self.source_sha256()==meta['source_sha256'])
                self._commit(db,kind)
            check();need(aware(clock())>=before_commit)
            return self.snapshot(clock=clock)
        except Exception:raise EvaluationBookError('validation_evidence_append_failed') from None

    def create(self,binding,*,clock,check):
        try:
            last=None
            def initial_clock():
                nonlocal last
                at=aware(clock())
                need((last is None or at>=last) and at<common_registration.FIRST_OPEN)
                last=at;return at
            initial_clock();need(type(binding) is dict and self.path.parent.is_dir() and not self.path.parent.is_symlink())
            with _exclusive(self.path):
                if not self.path.exists():
                    fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600);os.close(fd)
                    with closing(sqlite3.connect(self.path,isolation_level=None)) as db:
                        db.execute('PRAGMA synchronous=FULL');db.execute('BEGIN IMMEDIATE')
                        db.execute('CREATE TABLE metadata(version INTEGER,source_sha256 TEXT,head_seq INTEGER,head_digest TEXT,last_at TEXT,checksum TEXT)')
                        db.execute('CREATE TABLE events(seq INTEGER PRIMARY KEY,payload TEXT,previous_digest TEXT,digest TEXT)')
                        check();at=initial_clock();source=self.source_sha256();need(is_sha(source))
                        meta=dict(version=1,source_sha256=source,head_seq=0,head_digest='0'*64,last_at=at.isoformat())
                        db.execute('INSERT INTO metadata VALUES(?,?,?,?,?,?)',tuple(meta.values())+(digest(meta),))
                        identity=self._identity()
                        self._write(db,self._empty(),meta,'created',dict(
                            file=dict(path=str(self.path),device=identity[0],inode=identity[1]),binding=binding),at)
                        check();initial_clock();self._commit(db,'created')
                    self._directory_sync();check();initial_clock()
                state=self.snapshot(clock=initial_clock);need(state['binding']==alpaca.decode_json(codec.encode(binding)))
                if state['initialized']:return state
                self._directory_sync()
                return self.append(state,'initialized',dict(created_digest=state['created_digest']),clock=initial_clock,check=check)
        except Exception:raise EvaluationBookError('validation_evidence_initialization_failed') from None
