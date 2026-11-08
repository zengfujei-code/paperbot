"""Immutable paper activities and replayable accounting snapshots; no broker I/O."""
from contextlib import closing,contextmanager
from dataclasses import dataclass,field,fields,is_dataclass
from datetime import date,datetime
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from .alpaca import decode_json
from .accounting import AccountAnchor,Activity,ActivityWindow,OrderEvidence,AccountMark,AccountSnapshot,reconcile
from .data import utc_time
from .risk import _identity


class BookError(RuntimeError): pass


def _json(value):
    def encode(v):
        if type(v) is Decimal: return format(v,'f')
        if isinstance(v,(date,datetime)): return v.isoformat()
        if is_dataclass(v): return {f.name:getattr(v,f.name) for f in fields(v) if f.init}
        raise TypeError('unsupported accounting evidence')
    raw=json.dumps(value,default=encode,sort_keys=True,separators=(',',':'),allow_nan=False)
    if len(raw.encode())>2*1024*1024: raise BookError('accounting_record_budget')
    return raw


def _hash(raw): return hashlib.sha256(raw.encode()).hexdigest()


def code_hash():
    digest=hashlib.sha256()
    for name in ('accountbook.py','accounting.py','broker.py','risk.py','data.py'):
        digest.update(name.encode()+b'\0'+Path(__file__).with_name(name).read_bytes()+b'\0')
    return digest.hexdigest()


@dataclass(frozen=True)
class BookReceipt:
    sequence: int
    digest: str
    as_of: datetime
    result: object


@dataclass(frozen=True)
class BookSnapshot:
    receipt: BookReceipt
    anchor: AccountAnchor=field(repr=False)
    account: AccountSnapshot=field(repr=False)
    activity_until: datetime
    activity_count: int


class AccountBook:
    def __init__(self,path): self.path=Path(path).absolute()

    def initialize(self,anchor):
        try:
            if type(anchor) is not AccountAnchor or self.path.is_symlink(): raise BookError('accounting_anchor_or_path_invalid')
            if self.path.exists():
                with self._db(False) as db:
                    actual,*_=self._validate(db)
                    if actual!=anchor: raise BookError('accounting_anchor_conflict')
                return
            self.path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600);os.close(fd)
            raw=_json(anchor)
            with closing(sqlite3.connect(self.path)) as db:
                db.executescript('''PRAGMA journal_mode=DELETE;PRAGMA synchronous=FULL;BEGIN IMMEDIATE;
                    CREATE TABLE metadata(version INTEGER NOT NULL,anchor TEXT NOT NULL,digest TEXT NOT NULL);
                    CREATE TABLE activities(sequence INTEGER PRIMARY KEY,identity TEXT UNIQUE NOT NULL,payload TEXT NOT NULL,digest TEXT NOT NULL);
                    CREATE TABLE snapshots(sequence INTEGER PRIMARY KEY,payload TEXT NOT NULL,previous_digest TEXT NOT NULL,digest TEXT NOT NULL);''')
                db.execute('INSERT INTO metadata VALUES(1,?,?)',(raw,_hash(raw)));db.commit()
            if os.name!='nt':
                fd=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY)
                try: os.fsync(fd)
                finally: os.close(fd)
        except BookError: raise
        except Exception: raise BookError('accounting_initialization_failed') from None

    @contextmanager
    def _db(self,write):
        db=None
        try:
            if self.path.is_symlink() or not self.path.is_file(): raise BookError('accounting_missing_or_link_refused')
            db=sqlite3.connect(self.path.as_uri()+('?mode=rw' if write else '?mode=ro'),uri=True,timeout=10,isolation_level=None)
            db.row_factory=sqlite3.Row;db.execute('PRAGMA synchronous=FULL');db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            yield db
            db.commit()
        except BookError:
            if db is not None: db.rollback()
            raise
        except Exception:
            if db is not None: db.rollback()
            raise BookError('accounting_operation_failed') from None
        finally:
            if db is not None: db.close()

    def _activities(self,db):
        result=[];prefix={0:hashlib.sha256().hexdigest()};digest=hashlib.sha256();seen=set()
        for row in db.execute('SELECT sequence,identity,payload,digest FROM activities ORDER BY sequence'):
            sequence=len(result)+1
            if sequence>10000 or row['sequence']!=sequence or len(row['payload'])>4096 or _hash(row['payload'])!=row['digest']:
                raise BookError('accounting_activity_integrity_failed')
            event=Activity(**decode_json(row['payload']))
            if event.identity!=row['identity'] or event.identity in seen: raise BookError('accounting_activity_identity_failed')
            seen.add(event.identity);result.append((event,row['payload']))
            digest.update((str(sequence)+'\0'+event.identity+'\0'+row['digest']+'\0').encode());prefix[sequence]=digest.hexdigest()
        return result,prefix

    def _validate(self,db):
        metadata=db.execute('SELECT version,anchor,digest FROM metadata').fetchall()
        if len(metadata)!=1 or metadata[0]['version']!=1: raise BookError('accounting_schema_mismatch')
        meta=metadata[0]
        if len(meta['anchor'])>4096 or _hash(meta['anchor'])!=meta['digest']: raise BookError('accounting_anchor_integrity_failed')
        anchor=AccountAnchor(**decode_json(meta['anchor']));events,prefix=self._activities(db)
        latest=None;previous='0'*64;count=0
        for row in db.execute('SELECT sequence,payload,previous_digest,digest FROM snapshots ORDER BY sequence'):
            count+=1
            if (count>100000 or row['sequence']!=count or len(row['payload'])>2*1024*1024
                    or row['previous_digest']!=previous or _hash(previous+'\0'+row['payload'])!=row['digest']):
                raise BookError('accounting_snapshot_chain_failed')
            value=decode_json(row['payload']);number=value['activity_count'];_identity(value['code_sha256'])
            if (type(number) is not int or not 0<=number<=len(events) or value['schema']!=1
                    or value['anchor_sha256']!=meta['digest'] or value['activities_sha256']!=prefix[number]):
                raise BookError('accounting_snapshot_link_failed')
            if latest and (utc_time(value['inputs']['now'])<utc_time(latest['inputs']['now'])
                    or utc_time(value['inputs']['window']['until'])<utc_time(latest['inputs']['window']['until'])):
                raise BookError('accounting_snapshot_time_regression')
            latest=value;previous=row['digest']
        if (latest is None and events) or (latest is not None and latest['activity_count']!=len(events)):
            raise BookError('accounting_orphan_activities')
        return anchor,events,prefix,latest,count,previous

    def _append_snapshot(self,db,sequence,record,previous):
        raw=_json(record);digest=_hash(previous+'\0'+raw)
        db.execute('INSERT INTO snapshots VALUES(?,?,?,?)',(sequence,raw,previous,digest))
        return digest

    def refresh(self,*,window,orders,account,now):
        with self._db(True) as db:
            anchor,old,prefix,latest,count,previous=self._validate(db)
            if type(window) is not ActivityWindow or not window.complete or window.after!=anchor.as_of:
                raise BookError('complete_anchor_window_required')
            now=utc_time(now)
            if latest and (now<utc_time(latest['inputs']['now']) or window.until<utc_time(latest['inputs']['window']['until'])):
                raise BookError('accounting_source_time_regression')
            incoming={a.identity:_json(a) for a in window.activities}
            if any(incoming.get(event.identity)!=raw for event,raw in old): raise BookError('previous_activity_missing_or_changed')
            known={event.identity for event,_ in old};number=len(old)
            for event in window.activities:
                if event.identity not in known:
                    number+=1;raw=incoming[event.identity]
                    db.execute('INSERT INTO activities VALUES(?,?,?,?)',(number,event.identity,raw,_hash(raw)))
            all_events,prefix=self._activities(db)
            result=reconcile(anchor=anchor,window=window,orders=orders,account=account,now=now)
            normalized=decode_json(_json(dict(window=window,orders=orders,account=account,now=now)))
            del normalized['window']['activities']
            record=dict(schema=1,anchor_sha256=_hash(_json(anchor)),code_sha256=code_hash(),
                activity_count=len(all_events),activities_sha256=prefix[len(all_events)],inputs=normalized,
                result=decode_json(_json(result)))
            if latest==record: return BookReceipt(count,previous,now,result)
            if count>=100000: raise BookError('accounting_snapshot_retention_budget')
            digest=self._append_snapshot(db,count+1,record,previous)
            return BookReceipt(count+1,digest,now,result)

    def verify(self):
        with self._db(False) as db:
            _,events,_,_,count,_=self._validate(db)
            return dict(activities=len(events),snapshots=count,integrity_verified=True)

    def read_anchor(self):
        """Use an existing verified anchor; never initialize/reset during a read."""
        with self._db(False) as db:
            anchor,*_=self._validate(db)
            return anchor

    def _snapshot(self,db,sequence,validated=None):
        anchor,events,_,_,count,_=validated if validated is not None else self._validate(db)
        if sequence is None: sequence=count
        if type(sequence) is not int or sequence<1: raise BookError('invalid_accounting_snapshot')
        row=db.execute('SELECT payload,digest FROM snapshots WHERE sequence=?',(sequence,)).fetchone()
        if row is None: raise BookError('accounting_snapshot_missing')
        record=decode_json(row['payload'])
        if record['code_sha256']!=code_hash(): raise BookError('accounting_replay_source_mismatch')
        inputs=record['inputs'];w=inputs['window'];a=inputs['account']
        window=ActivityWindow(**w,activities=tuple(event for event,_ in events[:record['activity_count']]))
        account=AccountSnapshot(**{k:v for k,v in a.items() if k!='positions'},positions=tuple(AccountMark(**p) for p in a['positions']))
        result=reconcile(anchor=anchor,window=window,orders=tuple(OrderEvidence(**o) for o in inputs['orders']),account=account,now=inputs['now'])
        if decode_json(_json(result))!=record['result']: raise BookError('accounting_replay_result_mismatch')
        receipt=BookReceipt(sequence,row['digest'],utc_time(inputs['now']),result)
        return BookSnapshot(receipt,anchor,account,window.until,record['activity_count'])

    def snapshot(self,sequence=None):
        """Replayed receipt and accounting inputs from one verified read transaction."""
        with self._db(False) as db:
            return self._snapshot(db,sequence)

    def snapshot_by_digest(self,digest):
        """Replay an exact historical receipt; never substitute the latest one."""
        try:
            _identity(digest)
            with self._db(False) as db:
                validated=self._validate(db)
                rows=db.execute('SELECT sequence FROM snapshots WHERE digest=? LIMIT 2',(digest,)).fetchall()
                if len(rows)!=1:raise BookError('accounting_digest_missing_or_ambiguous')
                return self._snapshot(db,rows[0][0],validated)
        except BookError:raise
        except Exception:raise BookError('accounting_digest_lookup_failed') from None

    def snapshots_after(self,sequence,*,limit=100):
        """Replay a bounded complete suffix; truncation never silently drops peaks."""
        if type(sequence) is not int or sequence<0 or type(limit) is not int or not 1<=limit<=100:
            raise BookError('invalid_accounting_history_range')
        with self._db(False) as db:
            validated=self._validate(db);count=validated[4]
            if sequence>count or count-sequence>limit: raise BookError('accounting_history_range_or_budget')
            return tuple(self._snapshot(db,n,validated) for n in range(sequence+1,count+1))

    def replay(self,sequence):
        if sequence is None: raise BookError('invalid_accounting_snapshot')
        return self.snapshot(sequence).receipt.result
