"""Durable scan evidence and deterministic offline replay.

Hashes detect accidental changes. They are not signatures against a privileged
attacker who can rewrite both the database and its hashes. No provider secrets,
account identifiers, arbitrary metadata or provider response bodies are stored.
"""
from contextlib import closing
from dataclasses import fields
from datetime import date,datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3

from .alpaca import Asset,UniverseSnapshot,decode_json,ProviderError
from .data import Calendar,Session,Snapshot,utc_time,DataQualityError
from .scanner import ScanPolicy,ScanResult,Candidate,scan


class ArchiveError(RuntimeError):
    pass


def _json(value):
    def encode(value):
        if isinstance(value,(date,datetime)): return value.isoformat()
        raise TypeError('unsupported evidence type')
    return json.dumps(value,default=encode,sort_keys=True,separators=(',',':'),allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def code_hash():
    digest=hashlib.sha256()
    for name in ('data.py','alpaca.py','scanner.py','audit.py','runner.py'):
        digest.update(name.encode()+b'\0'+Path(__file__).with_name(name).read_bytes()+b'\0')
    return digest.hexdigest()


def _fields(value,base):
    return {field.name:getattr(value,field.name) for field in fields(base)}


def _result(value):
    result=_fields(value,ScanResult)
    result['policy']=_fields(value.policy,ScanPolicy)
    result['candidates']=[_fields(candidate,Candidate) for candidate in value.candidates]
    return result


def _pool(value):
    pool=_fields(value,UniverseSnapshot)
    pool['assets']=[_fields(a,Asset) for a in value.assets]
    return json.loads(_json(pool))


def _parse_pool(value):
    value=dict(value)
    value['assets']=tuple(Asset(**asset) for asset in value['assets'])
    value['received_at']=utc_time(value['received_at'])
    return UniverseSnapshot(**value)


def _check_selection(record,pool):
    value=record['inputs']
    selection=value['selection']
    if set(selection)!={'pool_sha256','offset','count','total'}:
        raise ArchiveError('invalid_batch_selection')
    offset,count,total=(selection[k] for k in ('offset','count','total'))
    if any(type(v) is not int for v in (offset,count,total)) or not 0<=offset<total or not 1<=count<=total-offset:
        raise ArchiveError('invalid_batch_selection_bounds')
    if selection['pool_sha256']!=_digest(pool) or total!=len(pool['assets']):
        raise ArchiveError('batch_pool_identity_mismatch')
    expected={**pool,'assets':pool['assets'][offset:offset+count]}
    if value['universe']!=expected:
        raise ArchiveError('batch_not_in_recorded_pool')


def make_scan_record(*,universe,snapshots,calendar,as_of,policy,snapshot_rejections=None,full_universe=None,batch_offset=0):
    snapshot_rejections={} if snapshot_rejections is None else snapshot_rejections
    result=scan(universe,snapshots,calendar=calendar,as_of=as_of,policy=policy,snapshot_rejections=snapshot_rejections)
    # Preserve only normalized, known inputs used by this scan. Extra symbols
    # and arbitrary API fields cannot enter the archive through this boundary.
    allowed={a.symbol for a in universe.assets}
    pool=_pool(universe if full_universe is None else full_universe)
    inputs=dict(universe=_pool(universe),snapshots={s:_fields(v,Snapshot) for s,v in sorted(snapshots.items())
                if s in allowed and isinstance(v,Snapshot)},
                calendar=dict(start=calendar.start,end=calendar.end,
                              sessions=[_fields(s,Session) for _,s in sorted(calendar.sessions.items())]),
                as_of=utc_time(as_of),policy=_fields(policy,ScanPolicy),snapshot_rejections=dict(sorted(snapshot_rejections.items())),
                selection=dict(pool_sha256=_digest(pool),offset=batch_offset,count=len(universe.assets),total=len(pool['assets'])))
    record=json.loads(_json(dict(version=3,code_sha256=code_hash(),inputs=inputs,result=_result(result))))
    record['run_id']=_digest(record)
    _check_selection(record,pool)
    # Refuse evidence that cannot explain its result, including a non-Snapshot
    # input that would otherwise be lost when serializing normalized inputs.
    replay(record)
    return record


def _verify(record):
    if not isinstance(record,dict) or set(record)!={'version','code_sha256','inputs','result','run_id'}:
        raise ArchiveError('invalid_scan_record')
    if type(record['version']) is not int or record['version']!=3 or not isinstance(record['run_id'],str) or not re.fullmatch('[0-9a-f]{64}',record['run_id']):
        raise ArchiveError('invalid_scan_version_or_identity')
    if record['run_id']!=_digest({k:v for k,v in record.items() if k!='run_id'}):
        raise ArchiveError('scan_integrity_mismatch')


def replay(record):
    try:
        _verify(record)
        if record['code_sha256']!=code_hash(): raise ArchiveError('scan_code_version_mismatch')
        value=record['inputs']
        if set(value)!={'universe','snapshots','calendar','as_of','policy','snapshot_rejections','selection'}:
            raise ArchiveError('unknown_scan_input_fields')
        universe=_parse_pool(value['universe'])
        raw_calendar=value['calendar']
        if set(raw_calendar)!={'start','end','sessions'} or any(set(s)!={'day','opens','closes'} for s in raw_calendar['sessions']):
            raise ArchiveError('unknown_calendar_fields')
        sessions=[Session(date.fromisoformat(s['day']),s['opens'],s['closes']) for s in raw_calendar['sessions']]
        calendar=Calendar(sessions,start=raw_calendar['start'],end=raw_calendar['end'])
        snapshots={}
        for symbol,raw in value['snapshots'].items():
            raw=dict(raw)
            raw['daily_session_day']=date.fromisoformat(raw['daily_session_day'])
            snapshots[symbol]=Snapshot(**raw)
        result=json.loads(_json(_result(scan(universe,snapshots,calendar=calendar,as_of=value['as_of'],policy=ScanPolicy(**value['policy']),
                                            snapshot_rejections=value['snapshot_rejections']))))
        if result!=record['result']: raise ArchiveError('scan_replay_mismatch')
        return result
    except ArchiveError:
        raise
    except (ValueError,TypeError,KeyError,OverflowError,OSError):
        raise ArchiveError('invalid_scan_evidence') from None


class ScanArchive:
    def __init__(self,path):
        self.path=Path(path).absolute()

    def append(self,record,*,full_universe=None):
        """Return only after SQLite FULL synchronous commit; idempotent by run ID."""
        try:
            replay(record)
            pool=_pool(_parse_pool(record['inputs']['universe']) if full_universe is None else full_universe)
            _check_selection(record,pool)
            payload=_json(record)
            pool_payload=_json(pool)
            if len(pool_payload.encode())>20*1024*1024: raise ArchiveError('universe_evidence_too_large')
            if len(payload.encode())>32*1024*1024: raise ArchiveError('scan_evidence_too_large')
            if self.path.is_symlink(): raise ArchiveError('archive_link_refused')
            try:
                fd=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
                os.close(fd)
            except FileExistsError:
                pass
            with closing(sqlite3.connect(self.path,timeout=10)) as db:
                db.execute('PRAGMA synchronous=FULL')
                with db:
                    db.execute('CREATE TABLE IF NOT EXISTS scans(run_id TEXT PRIMARY KEY,payload TEXT NOT NULL,digest TEXT NOT NULL)')
                    db.execute('CREATE TABLE IF NOT EXISTS universes(sha256 TEXT PRIMARY KEY,payload TEXT NOT NULL)')
                    pool_id=record['inputs']['selection']['pool_sha256']
                    existing_pool=db.execute('SELECT payload FROM universes WHERE sha256=?',(pool_id,)).fetchone()
                    if existing_pool and existing_pool[0]!=pool_payload: raise ArchiveError('universe_identity_conflict')
                    if not existing_pool: db.execute('INSERT INTO universes VALUES(?,?)',(pool_id,pool_payload))
                    existing=db.execute('SELECT payload,digest FROM scans WHERE run_id=?',(record['run_id'],)).fetchone()
                    digest=hashlib.sha256(payload.encode()).hexdigest()
                    if existing:
                        if existing!=(payload,digest): raise ArchiveError('scan_identity_conflict')
                        return False
                    db.execute('INSERT INTO scans VALUES(?,?,?)',(record['run_id'],payload,digest))
            return True
        except (sqlite3.Error,OSError,ValueError,TypeError):
            raise ArchiveError('scan_archive_write_failed') from None

    def load(self,run_id):
        try:
            if not isinstance(run_id,str) or not re.fullmatch('[0-9a-f]{64}',run_id) or self.path.is_symlink():
                raise ArchiveError('invalid_scan_identity_or_path')
            with closing(sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True,timeout=10)) as db:
                row=db.execute('SELECT payload,digest FROM scans WHERE run_id=?',(run_id,)).fetchone()
            if row is None: raise ArchiveError('scan_not_found')
            payload,digest=row
            if len(payload)>32*1024*1024 or hashlib.sha256(payload.encode()).hexdigest()!=digest:
                raise ArchiveError('scan_archive_integrity_mismatch')
            record=decode_json(payload)
            _verify(record)
            if record['run_id']!=run_id: raise ArchiveError('scan_identity_mismatch')
            _check_selection(record,self.load_universe(record['inputs']['selection']['pool_sha256']))
            return record
        except (sqlite3.Error,OSError,ValueError,TypeError,ProviderError):
            raise ArchiveError('scan_archive_read_failed') from None

    def load_universe(self,identity):
        try:
            if not isinstance(identity,str) or not re.fullmatch('[0-9a-f]{64}',identity) or self.path.is_symlink():
                raise ArchiveError('invalid_universe_identity_or_path')
            with closing(sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True,timeout=10)) as db:
                row=db.execute('SELECT payload FROM universes WHERE sha256=?',(identity,)).fetchone()
            if row is None or len(row[0])>20*1024*1024: raise ArchiveError('universe_not_found_or_oversized')
            pool=decode_json(row[0])
            if _digest(pool)!=identity or _pool(_parse_pool(pool))!=pool:
                raise ArchiveError('universe_integrity_mismatch')
            return pool
        except (sqlite3.Error,OSError,ValueError,TypeError,KeyError,ProviderError):
            raise ArchiveError('universe_archive_read_failed') from None
