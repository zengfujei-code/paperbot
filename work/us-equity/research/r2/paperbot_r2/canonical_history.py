"""Same-file canonical history continuation after the reserved R1 campaign.

Never creates an empty replacement authority or changes legacy source/events.
Only a terminal legacy registry in the declared future handoff window can be
extended. Hashes detect damage, not an administrator fabricating whole history.
"""
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import date, datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from paperbot_us import experiments as legacy
from paperbot_us.runner import _exclusive
from . import training_codec as codec
from .contracts import aware, digest, is_sha
from .training_data import TIMEZONE_SHA256

LEGACY_SOURCE = 'fdee9c15475f247395790a8a3afb17b8fe8d218551df67204d1b6a8173d6fd2c'
FREEZE_SQL = {f'paperbot_v2_freeze_{table}_{action.lower()}':
    f"CREATE TRIGGER paperbot_v2_freeze_{table}_{action.lower()} BEFORE {action} ON {table} "
    "BEGIN SELECT RAISE(ABORT,'legacy_registry_frozen'); END"
    for table in ('events', 'head') for action in ('INSERT', 'UPDATE', 'DELETE')}


class CanonicalHistoryError(ValueError):
    pass


def _need(value):
    if not value:
        raise CanonicalHistoryError('canonical_history_unavailable')


def source_sha256():
    old = Path(legacy.__file__); current = Path(__file__)
    _need(old.is_file() and not old.is_symlink() and current.is_file() and not current.is_symlink()
          and hashlib.sha256(old.read_bytes()).hexdigest() == LEGACY_SOURCE)
    protocol=current.with_name('common_registration.py')
    paths=(Path(codec.__file__),current.with_name('contracts.py'),Path(_exclusive.__code__.co_filename))
    _need(all(p.is_file() and not p.is_symlink() and p.stat().st_size<=524288 for p in paths))
    if protocol.is_file():
        _need(not protocol.is_symlink())
        from .common_registration import _source as protocol_source
        protocol_sha=protocol_source()
    else:
        protocol_sha=None
    return digest(dict(reader=hashlib.sha256(current.read_bytes()).hexdigest(), legacy=LEGACY_SOURCE,
        dependencies={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        protocol=protocol_sha))


class Clock:
    def __init__(self, timezone_file, clock):
        path = Path(timezone_file)
        _need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 65536 and callable(clock))
        content = path.read_bytes(); _need(hashlib.sha256(content).hexdigest() == TIMEZONE_SHA256)
        self.zone = ZoneInfo.from_file(BytesIO(content), key='America/New_York')
        self.clock, self.last = clock, None

    def __call__(self, *, handoff=False):
        at = aware(self.clock()); local = at.astimezone(self.zone)
        _need(self.last is None or at >= self.last)
        _need(not date(2026, 11, 10) <= local.date() <= date(2026, 11, 23))
        if handoff:
            _need(datetime(2026, 11, 24) <= local.replace(tzinfo=None) < datetime(2026, 11, 24, 9, 30))
        self.last = at
        return at


def _identity(path):
    _need(path.is_file() and not path.is_symlink() and not path.parent.is_symlink()
          and path.stat().st_size <= 67108864)
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _used(state):
    domains = {d: set() for d in ('us_equity', 'synthetic')}
    for item in state['imports'].values():
        domains[item['domain']].update(item['days'])
    for item in state['experiments'].values():
        for days in item['plan']['split'].values():
            domains[item['plan']['domain']].update(days)
    return {d: tuple(sorted(days)) for d, days in domains.items()}


@dataclass(frozen=True)
class LegacyHistory:
    path: str
    device: int
    inode: int
    sequence: int
    head_digest: str
    state_json: str
    state_sha256: str
    all_reports_terminal: bool
    available_at: datetime
    source_sha256: str
    seal: str

    def __post_init__(self):
        state = json.loads(self.state_json)
        _need(Path(self.path).is_absolute() and type(self.sequence) is int and self.sequence >= 1
              and is_sha(self.head_digest) and legacy.canonical(state) == self.state_json
              and digest(state) == self.state_sha256
              and self.all_reports_terminal == all(e['report_sha256'] is not None
                                                   for e in state['experiments'].values())
              and self.source_sha256 == source_sha256()
              and self.seal == digest({k: v for k, v in asdict(self).items() if k != 'seal'}))

    def legacy_state(self):
        return json.loads(self.state_json)

    def used_days(self):
        return _used(self.legacy_state())


def inspect_legacy(registry, *, timezone_file, clock):
    try:
        guard = Clock(timezone_file, clock); guard()
        _need(type(registry) is legacy.ExperimentRegistry)
        source = source_sha256(); before = _identity(registry.path)
        guard()
        with closing(sqlite3.connect(registry.path.as_uri() + '?mode=ro', uri=True, isolation_level=None)) as db:
            db.execute('PRAGMA query_only=ON'); version = db.execute('PRAGMA data_version').fetchone()[0]
            db.execute('BEGIN'); state, sequence, head = registry._read(db); db.rollback()
            _need(db.execute('PRAGMA data_version').fetchone()[0] == version)
        _need(_identity(registry.path) == before and source_sha256() == source)
        available = guard()
        fields = dict(path=str(registry.path), device=before[0], inode=before[1], sequence=sequence,
            head_digest=head, state_json=legacy.canonical(state), state_sha256=digest(state),
            all_reports_terminal=all(e['report_sha256'] is not None for e in state['experiments'].values()),
            available_at=available, source_sha256=source)
        return LegacyHistory(**fields, seal=digest(fields))
    except Exception:
        raise CanonicalHistoryError('legacy_history_inspection_failed') from None


def _checksum(previous, payload):
    return hashlib.sha256((previous + '\0' + payload).encode()).hexdigest()


def _base(state, sequence, head):
    return dict(continued=False, binding=None, legacy_state=state, legacy_sequence=sequence,
        legacy_head_digest=head, new_exposures={}, training_imports={}, reservations={},
        head_seq=0, head_digest='0' * 64, last_at=None)


class CanonicalRegistry:
    def __init__(self, path):
        self.path = Path(path).absolute()
        self.legacy = legacy.ExperimentRegistry(self.path)

    def _apply(self, state, event, sha, source):
        _need(type(event) is dict and set(event) == {'kind','body','at','source_sha256'}
              and event['source_sha256'] == source and is_sha(sha))
        kind, body, at = event['kind'], event['body'], aware(event['at'])
        _need(type(body) is dict and (state['last_at'] is None or at >= state['last_at']))
        if kind == 'continued':
            _need(not state['continued'] and set(body) == {'path','device','inode','legacy_sequence',
                'legacy_head_digest','legacy_state_sha256','anchor_sequence','anchor_digest','freeze_sha256'}
                and body['path'] == str(self.path) and (body['device'], body['inode']) == _identity(self.path)[:2]
                and body['legacy_sequence'] == state['legacy_sequence']
                and body['legacy_head_digest'] == state['legacy_head_digest']
                and body['legacy_state_sha256'] == digest(state['legacy_state'])
                and body['freeze_sha256'] == digest(FREEZE_SQL)
                and type(body['anchor_sequence']) is int and 1 <= body['anchor_sequence'] <= state['legacy_sequence']
                and is_sha(body['anchor_digest']))
            state.update(continued=True, binding=body)
        elif kind == 'exposure':
            _need(state['continued'] and set(body) == {'domain','days','source_sha256'}
                  and body['domain'] in ('us_equity','synthetic') and is_sha(body['source_sha256']))
            days = legacy._days(body['days']); _need(days)
            key = digest(body)
            if key in state['new_exposures']:
                return False
            state['new_exposures'][key] = body
            from .common_registration import invalidate_exposure
            invalidate_exposure(state,body['domain'],body['days'])
        else:
            _need(state['continued'])
            from .common_registration import apply_event
            if not apply_event(state,kind,body,at,sha):
                return False
        state['last_at'] = at
        return True

    def _read(self, db):
        _need(db.execute('PRAGMA journal_mode').fetchone()[0].lower() == 'delete')
        original, sequence, head = self.legacy._read(db)
        meta_rows = db.execute('SELECT version,source_sha256,seq,digest,last_at,checksum FROM canonical_v2_head').fetchall()
        _need(len(meta_rows) == 1)
        names = ('version','source_sha256','seq','digest','last_at')
        meta = dict(zip(names, meta_rows[0][:-1])); source = source_sha256()
        _need(meta['version'] == 2 and meta['source_sha256'] == source and meta_rows[0][-1] == digest(meta))
        triggers = dict(db.execute("SELECT name,sql FROM sqlite_schema WHERE type='trigger'"))
        _need(triggers == FREEZE_SQL)
        state = _base(original, sequence, head)
        for seq, payload, previous, sha in db.execute('SELECT seq,payload,previous_digest,digest FROM canonical_v2_events ORDER BY seq'):
            _need(seq == state['head_seq'] + 1 <= 10000 and len(payload.encode()) <= codec.MAX_BYTES
                  and previous == state['head_digest'] and sha == _checksum(previous, payload))
            _need(self._apply(state, json.loads(payload), sha, source))
            state.update(head_seq=seq, head_digest=sha)
        _need(state['continued'] and state['head_seq'] == meta['seq'] and state['head_digest'] == meta['digest']
              and state['last_at'] == aware(meta['last_at']) and source_sha256() == source)
        anchor = state['binding']
        _need(db.execute('SELECT digest FROM events WHERE seq=?', (anchor['anchor_sequence'],)).fetchone()
              == (anchor['anchor_digest'],))
        return state, meta

    def _write(self, db, state, meta, kind, body, at):
        _need(state['head_seq'] < 10000)
        payload = codec.encode(dict(kind=kind,body=body,at=aware(at).isoformat(),source_sha256=meta['source_sha256']))
        _need(self.path.stat().st_size + len(payload.encode()) + 65536 <= 67108864)
        previous = state['head_digest']; sha = _checksum(previous, payload)
        if not self._apply(state, json.loads(payload), sha, meta['source_sha256']):
            return False
        seq = state['head_seq'] + 1
        db.execute('INSERT INTO canonical_v2_events VALUES(?,?,?,?)', (seq,payload,previous,sha))
        meta.update(seq=seq,digest=sha,last_at=aware(at).isoformat())
        db.execute('UPDATE canonical_v2_head SET seq=?,digest=?,last_at=?,checksum=?',
                   (seq,sha,meta['last_at'],digest(meta)))
        return True

    def continue_legacy(self, *, expected_history, timezone_file, clock):
        try:
            guard = Clock(timezone_file, clock); guard()
            _need(type(expected_history) is LegacyHistory); expected_history.__post_init__()
            identity = _identity(self.path)
            _need((str(self.path), identity[0], identity[1]) ==
                  (expected_history.path, expected_history.device, expected_history.inode)
                  and expected_history.sequence >= 2)
            with closing(sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)) as db:
                tables = set(r[0] for r in db.execute("SELECT name FROM sqlite_schema WHERE type='table'"))
            if 'canonical_v2_head' in tables:
                return self.snapshot(timezone_file=timezone_file, clock=guard)
            guard(handoff=True)
            with _exclusive(self.path), closing(sqlite3.connect(self.path.as_uri() + '?mode=rw', uri=True, isolation_level=None, timeout=5)) as db:
                db.execute('PRAGMA synchronous=FULL'); db.execute('BEGIN IMMEDIATE')
                _need(db.execute('PRAGMA journal_mode').fetchone()[0].lower() == 'delete')
                _need(set(r[0] for r in db.execute("SELECT name FROM sqlite_schema WHERE type='table'")) == {'events','head'})
                _need(db.execute("SELECT name FROM sqlite_schema WHERE type='trigger'").fetchall() == [])
                original, sequence, head = self.legacy._read(db)
                _need(all(e['report_sha256'] is not None for e in original['experiments'].values())
                      and (original['experiments'] or original['imports'])
                      and sequence >= expected_history.sequence
                      and db.execute('SELECT digest FROM events WHERE seq=?', (expected_history.sequence,)).fetchone()
                      == (expected_history.head_digest,))
                at = guard(handoff=True); _need(at >= expected_history.available_at)
                source = source_sha256()
                db.execute('CREATE TABLE canonical_v2_events(seq INTEGER PRIMARY KEY,payload TEXT,previous_digest TEXT,digest TEXT)')
                db.execute('CREATE TABLE canonical_v2_head(version INTEGER,source_sha256 TEXT,seq INTEGER,digest TEXT,last_at TEXT,checksum TEXT)')
                meta = dict(version=2,source_sha256=source,seq=0,digest='0'*64,last_at=at.isoformat())
                db.execute('INSERT INTO canonical_v2_head VALUES(?,?,?,?,?,?)', tuple(meta.values())+(digest(meta),))
                for sql in FREEZE_SQL.values():
                    db.execute(sql)
                body = dict(path=str(self.path),device=identity[0],inode=identity[1],legacy_sequence=sequence,
                    legacy_head_digest=head,legacy_state_sha256=digest(original),anchor_sequence=expected_history.sequence,
                    anchor_digest=expected_history.head_digest,freeze_sha256=digest(FREEZE_SQL))
                self._write(db,_base(original,sequence,head),meta,'continued',body,at)
                guard(handoff=True); db.commit(); guard(handoff=True)
            return self.snapshot(timezone_file=timezone_file, clock=guard)
        except Exception:
            raise CanonicalHistoryError('canonical_continuation_refused') from None

    def snapshot(self, *, timezone_file, clock):
        try:
            guard = Clock(timezone_file, clock); guard(); before = _identity(self.path)
            with closing(sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True, isolation_level=None)) as db:
                db.execute('PRAGMA query_only=ON'); version = db.execute('PRAGMA data_version').fetchone()[0]
                db.execute('BEGIN'); state, _ = self._read(db); db.rollback()
                _need(db.execute('PRAGMA data_version').fetchone()[0] == version)
            _need(_identity(self.path) == before); _need(guard() >= state['last_at'])
            return state
        except Exception:
            raise CanonicalHistoryError('canonical_snapshot_unavailable') from None

    def append(self, kind, body, *, timezone_file, clock):
        try:
            guard = Clock(timezone_file, clock); guard(); _identity(self.path)
            _need(kind != 'continued')
            with _exclusive(self.path), closing(sqlite3.connect(self.path.as_uri() + '?mode=rw', uri=True, isolation_level=None, timeout=5)) as db:
                db.execute('PRAGMA synchronous=FULL'); db.execute('BEGIN IMMEDIATE')
                state, meta = self._read(db); at = guard()
                changed = self._write(db,state,meta,kind,body,at)
                guard(); db.commit(); guard()
            return changed
        except Exception:
            raise CanonicalHistoryError('canonical_append_refused') from None
