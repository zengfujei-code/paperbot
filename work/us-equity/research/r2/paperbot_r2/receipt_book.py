"""Private first-receipt journal. Raw data only; no models, accounts or orders."""
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3

from paperbot_us import data, regime_baselines
from .contracts import aware, digest, is_sha, json_value


class ReceiptError(RuntimeError):
    pass


def _need(ok, code='invalid_receipt_evidence'):
    if not ok:raise ReceiptError(code)


def _json(value):
    return json.dumps(json_value(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def _source():
    research=Path(__file__).parent;core=Path(data.__file__).parent
    paths={name:research/name.split('/')[-1] for name in
           ('paperbot_r2/receipt_book.py','paperbot_r2/receipt_runner.py','paperbot_r2/contracts.py','paperbot_r2/__init__.py')}
    paths.update({name:core/name.split('/')[-1] for name in
                  ('paperbot_us/data.py','paperbot_us/alpaca.py','paperbot_us/regime_baselines.py','paperbot_us/__init__.py')})
    _need(all(p.is_file() and not p.is_symlink() and p.stat().st_size <= 524288 for p in paths.values()))
    return digest({name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in paths.items()})


@dataclass(frozen=True)
class ReceiptPlan:
    symbols: tuple
    start_day: str
    end_day: str
    timezone_sha256: str
    deadline_seconds: int = 120
    recovery_seconds: int = 3600
    max_records: int = 50000

    def __post_init__(self):
        _need(type(self.symbols) is tuple and 1 <= len(self.symbols) <= 8
              and all(type(s) is str and re.fullmatch('[A-Z][A-Z0-9.\\-/]{0,19}', s) for s in self.symbols)
              and tuple(sorted(set(self.symbols))) == self.symbols)
        try:
            start, end = date.fromisoformat(self.start_day), date.fromisoformat(self.end_day)
            _need(0 <= (end-start).days < 35 and start.isoformat() == self.start_day and end.isoformat() == self.end_day)
        except (ValueError, TypeError):raise ReceiptError('invalid_receipt_dates') from None
        _need(is_sha(self.timezone_sha256) and type(self.deadline_seconds) is int and self.deadline_seconds == 120
              and type(self.recovery_seconds) is int and self.recovery_seconds == 3600
              and type(self.max_records) is int and 2 <= self.max_records <= 50000)


def _plan(raw):
    _need(type(raw) is dict)
    value = dict(raw);value['symbols'] = tuple(value['symbols'])
    return ReceiptPlan(**value)


def _session(raw):
    if raw is None:return None
    _need(type(raw) is dict and set(raw) == {'day','opens','closes'})
    session = data.Session(date.fromisoformat(raw['day']), raw['opens'], raw['closes'])
    length = (session.closes-session.opens).total_seconds()
    _need(0 < length <= 28800 and length % 900 == 0)
    return session


def _key(symbol, start):return symbol + '|' + aware(start).isoformat()


def _price(bar):
    return digest({k:v for k,v in asdict(bar).items() if k != 'received_at'})


def _bar(state, raw, at):
    value = data.Bar(**raw)
    key = _key(value.symbol, value.starts)
    _need(key in state['slots'] and value.symbol in state['plan'].symbols
          and value.feed == 'iex', 'unrequested_or_unplanned_receipt')
    slot = state['slots'][key];proof = state['calendars'][slot['day']]
    _need(not proof['changed'], 'receipt_calendar_changed')
    value.validate(as_of=at, session=_session(proof['session']))
    return key, value


def _apply(state, event):
    _need(type(event) is dict and set(event) == {'kind','at','body'})
    at = aware(event['at']);body = event['body'];kind = event['kind']
    _need(type(body) is dict)
    if kind == 'created':
        _need(state['plan'] is None and set(body) == {'plan'})
        state['plan'] = _plan(body['plan']);return
    _need(state['plan'] is not None)
    if kind == 'calendar':
        _need(set(body) == {'day','session'})
        day = date.fromisoformat(body['day']);plan = state['plan'];session = _session(body['session'])
        _need(date.fromisoformat(plan.start_day) <= day <= date.fromisoformat(plan.end_day)
              and (session is None or session.day == day))
        prior = state['calendars'].get(body['day'])
        if prior is None:
            state['calendars'][body['day']] = dict(session=body['session'],received_at=at.isoformat(),changed=False)
            if session is not None:
                for symbol in plan.symbols:
                    for i in range(int((session.closes-session.opens).total_seconds()//900)):
                        start = session.opens + timedelta(seconds=900*i);end = start + timedelta(seconds=900)
                        key = _key(symbol,start);_need(key not in state['slots'])
                        state['slots'][key] = dict(symbol=symbol,day=body['day'],start=start.isoformat(),end=end.isoformat(),
                            deadline=(end+timedelta(seconds=plan.deadline_seconds)).isoformat(),first=None,
                            first_recorded_at=None,on_time=False,missed=False,missed_at=None,revisions=[])
        else:
            prior['changed'] = prior['changed'] or prior['session'] != body['session']
            prior['received_at'] = at.isoformat()
        return
    if kind == 'missed':
        _need(set(body) == {'key'} and body['key'] in state['slots'])
        slot = state['slots'][body['key']]
        _need(slot['first'] is None and not slot['missed'] and at > aware(slot['deadline']))
        slot.update(missed=True, missed_at=at.isoformat());return
    _need(kind in ('receipt','revision') and set(body) == {'bar'})
    key,bar = _bar(state,body['bar'],at);slot = state['slots'][key]
    if kind == 'receipt':
        _need(slot['first'] is None)
        on_time = at <= aware(slot['deadline']) and bar.received_at <= aware(slot['deadline']) and not slot['missed']
        _need(on_time or slot['missed'])
        slot.update(first=json_value(asdict(bar)),first_recorded_at=at.isoformat(),on_time=on_time)
    else:
        _need(slot['first'] is not None)
        previous = slot['revisions'][-1]['bar'] if slot['revisions'] else slot['first']
        _need(_price(bar) != _price(data.Bar(**previous)))
        slot['revisions'].append(dict(bar=json_value(asdict(bar)),recorded_at=at.isoformat()))


def _anchor(meta):return digest(meta)


class ReceiptBook:
    def __init__(self,path):self.path = Path(path).absolute()

    def _path(self):
        _need(self.path.is_file() and not self.path.is_symlink() and not self.path.parent.is_symlink()
              and self.path.stat().st_size <= 268435456, 'receipt_book_missing_or_oversized')

    def initialize(self,plan,*,now):
        _need(type(plan) is ReceiptPlan)
        at = aware(now)
        try:
            _need(self.path.parent.is_dir() and not self.path.parent.is_symlink())
            fd = os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600);os.close(fd)
            with closing(sqlite3.connect(self.path,isolation_level=None,timeout=5)) as db:
                db.execute('PRAGMA synchronous=FULL');db.execute('BEGIN IMMEDIATE')
                db.execute('CREATE TABLE metadata(version INTEGER,plan_sha256 TEXT,source_sha256 TEXT,head_seq INTEGER,head_digest TEXT,last_at TEXT,anchor_digest TEXT)')
                db.execute('CREATE TABLE events(seq INTEGER PRIMARY KEY,payload TEXT,previous_digest TEXT,digest TEXT)')
                meta = dict(version=1,plan_sha256=digest(asdict(plan)),source_sha256=_source(),head_seq=0,head_digest='0'*64,last_at=at.isoformat())
                db.execute('INSERT INTO metadata VALUES(?,?,?,?,?,?,?)',tuple(meta.values())+(_anchor(meta),))
                state = dict(plan=plan,calendars={},slots={},head_seq=0,head_digest='0'*64,last_at=at.isoformat())
                self._append(db,state,'created',dict(plan=asdict(plan)),at,apply=False)
                self._finish(db,state,meta,at);db.commit()
        except ReceiptError:raise
        except Exception:raise ReceiptError('receipt_initialization_failed') from None

    def _load(self,db):
        rows = db.execute('SELECT version,plan_sha256,source_sha256,head_seq,head_digest,last_at,anchor_digest FROM metadata').fetchall()
        _need(len(rows) == 1)
        names = ('version','plan_sha256','source_sha256','head_seq','head_digest','last_at')
        meta = dict(zip(names,rows[0][:-1]))
        _need(meta['version'] == 1 and meta['source_sha256'] == _source() and _anchor(meta) == rows[0][-1]
              and type(meta['head_seq']) is int and 1 <= meta['head_seq'] <= 50000)
        state = dict(plan=None,calendars={},slots={},head_seq=0,head_digest='0'*64,last_at=meta['last_at'])
        previous_at = None
        for seq,payload,prior,sha in db.execute('SELECT seq,payload,previous_digest,digest FROM events ORDER BY seq'):
            _need(seq == state['head_seq']+1 and len(payload) <= 16384 and prior == state['head_digest']
                  and hashlib.sha256((prior+'\0'+payload).encode()).hexdigest() == sha)
            event = json.loads(payload);at = aware(event['at'])
            _need((previous_at is None or at >= previous_at) and at <= aware(meta['last_at']))
            _apply(state,event);state.update(head_seq=seq,head_digest=sha);previous_at=at
            _need(seq <= state['plan'].max_records)
        _need(state['head_seq'] == meta['head_seq'] and state['head_digest'] == meta['head_digest']
              and digest(asdict(state['plan'])) == meta['plan_sha256'])
        return state,meta

    @contextmanager
    def _transaction(self,write=False):
        self._path()
        try:
            with closing(sqlite3.connect(self.path.as_uri()+('?mode=rw' if write else '?mode=ro'),uri=True,isolation_level=None,timeout=10)) as db:
                if write:db.execute('PRAGMA synchronous=FULL')
                db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
                state,meta = self._load(db)
                try:
                    yield db,state,meta
                    if write:db.commit()
                    else:db.rollback()
                except Exception:
                    db.rollback();raise
        except ReceiptError:raise
        except Exception:raise ReceiptError('receipt_integrity_or_storage_failed') from None

    def snapshot(self):
        with self._transaction() as (_,state,__):return state

    def _append(self,db,state,kind,body,at,*,apply=True):
        _need(state['head_seq'] < state['plan'].max_records, 'receipt_record_budget_exhausted')
        event = json.loads(_json(dict(kind=kind,at=at,body=body)));payload = _json(event)
        _need(len(payload) <= 16384)
        prior = state['head_digest'];sha = hashlib.sha256((prior+'\0'+payload).encode()).hexdigest()
        db.execute('INSERT INTO events VALUES(?,?,?,?)',(state['head_seq']+1,payload,prior,sha))
        if apply:_apply(state,event)
        state.update(head_seq=state['head_seq']+1,head_digest=sha)

    def _finish(self,db,state,meta,at):
        meta.update(head_seq=state['head_seq'],head_digest=state['head_digest'],last_at=at.isoformat())
        db.execute('UPDATE metadata SET head_seq=?,head_digest=?,last_at=?,anchor_digest=?',
                   (meta['head_seq'],meta['head_digest'],meta['last_at'],_anchor(meta)))
        state['last_at'] = at.isoformat()

    def record_calendar(self,calendar,*,day,received_at):
        _need(type(calendar) is data.Calendar)
        at = aware(received_at)
        with self._transaction(True) as (db,state,meta):
            _need(at >= aware(meta['last_at']))
            selected = date.fromisoformat(day)
            _need(calendar.start <= selected <= calendar.end)
            session = calendar.sessions.get(selected)
            raw = None if session is None else dict(day=session.day.isoformat(),opens=session.opens,closes=session.closes)
            self._append(db,state,'calendar',dict(day=day,session=raw),at)
            self._finish(db,state,meta,at)

    def observe(self,bars,*,now):
        at = aware(now)
        _need(type(bars) in (tuple,list) and len(bars) <= 1000 and all(type(b) is data.Bar for b in bars))
        with self._transaction(True) as (db,state,meta):
            _need(at >= aware(meta['last_at']), 'receipt_clock_regression')
            validated = [_bar(state,json_value(asdict(b)),at) for b in bars]
            _need(len(set(k for k,_ in validated)) == len(validated), 'duplicate_batch_receipt')
            for key,slot in state['slots'].items():
                if slot['first'] is None and not slot['missed'] and at > aware(slot['deadline']):
                    self._append(db,state,'missed',dict(key=key),at)
            for key,bar in sorted(validated):
                slot = state['slots'][key]
                if slot['first'] is None:self._append(db,state,'receipt',dict(bar=asdict(bar)),at)
                else:
                    previous = slot['revisions'][-1]['bar'] if slot['revisions'] else slot['first']
                    if _price(bar) != _price(data.Bar(**previous)):
                        self._append(db,state,'revision',dict(bar=asdict(bar)),at)
            self._finish(db,state,meta,at)
