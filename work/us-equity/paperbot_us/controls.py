"""Durable operator pause/halt journal. No browser, broker or model interface.

OS ownership must authorize writes in deployment: an administrator-owned private
directory/database, service read access only. The Python actor label is not
authentication. This module grants no strategy acceptance or trading permission.
"""
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime
import hashlib
import os
from pathlib import Path
import sqlite3

from .alpaca import decode_json
from .data import utc_time
from .orders import _json
from .risk import _identity

REASONS = frozenset(('initial_setup', 'operator_pause', 'operator_resume', 'operator_halt',
                     'operator_release_halt', 'deployment_hold', 'fault_review'))


class ControlError(RuntimeError):
    pass


def _require_rollback_journal(db):
    # Acquire/retain the transaction's read lock before inspecting mode. A
    # concurrent mode switch then needs an exclusive lock and cannot race the
    # check-to-dispatch interval. WAL readers do not serialize writer commits.
    db.execute('SELECT name FROM sqlite_master LIMIT 1').fetchone()
    if db.execute('PRAGMA journal_mode').fetchone()[0].lower() != 'delete':
        raise ValueError('control rollback journal mode required')


def _version():
    digest = hashlib.sha256()
    for name in ('controls.py', 'data.py', 'alpaca.py', 'orders.py', 'risk.py'):
        digest.update(name.encode() + b'\0' + Path(__file__).with_name(name).read_bytes() + b'\0')
    return digest.hexdigest()


def _valid(row, previous):
    if (set(row) != {'schema', 'code_sha256', 'revision', 'updated_at', 'pause_entries', 'halt_orders', 'reason'}
            or row['schema'] != 1 or type(row['revision']) is not int or not 1 <= row['revision'] <= 10000
            or type(row['pause_entries']) is not bool or type(row['halt_orders']) is not bool
            or row['reason'] not in REASONS or (row['halt_orders'] and not row['pause_entries'])):
        raise ValueError('control contract invalid')
    at = utc_time(row['updated_at'])
    if previous is None:
        if row['revision'] != 1 or row['reason'] != 'initial_setup' or not row['pause_entries'] or row['halt_orders']:
            raise ValueError('initial pause required')
        return
    if row['revision'] != previous['revision'] + 1 or at < utc_time(previous['updated_at']):
        raise ValueError('control revision or time regression')
    if previous['halt_orders'] and not row['halt_orders']:
        if row['reason'] != 'operator_release_halt' or not row['pause_entries']:
            raise ValueError('release halt must preserve pause')
    if row['reason'] == 'operator_resume':
        if previous['halt_orders'] or row['pause_entries'] or row['halt_orders']:
            raise ValueError('explicit resume contract')
    elif row['reason'] == 'operator_halt':
        if not row['halt_orders']:
            raise ValueError('explicit halt contract')
    elif row['reason'] == 'operator_release_halt':
        if not previous['halt_orders'] or row['halt_orders'] or not row['pause_entries']:
            raise ValueError('explicit halt release contract')
    elif row['reason'] in ('operator_pause', 'deployment_hold', 'fault_review'):
        if not row['pause_entries']:
            raise ValueError('explicit pause contract')
    else:
        raise ValueError('initial state cannot be republished')


@dataclass(frozen=True)
class ControlSnapshot:
    revision: int
    updated_at: datetime
    pause_entries: bool
    halt_orders: bool
    reason: str
    digest: str
    code_sha256: str

    def __post_init__(self):
        if (type(self.revision) is not int or not 1 <= self.revision <= 10000
                or type(self.pause_entries) is not bool or type(self.halt_orders) is not bool
                or self.reason not in REASONS):
            raise ValueError('invalid control snapshot')
        object.__setattr__(self, 'updated_at', utc_time(self.updated_at))
        _identity(self.digest)
        _identity(self.code_sha256)

    def public_summary(self):
        return dict(mode='paper', execution_enabled=False, entries_paused=self.pause_entries,
                    orders_halted=self.halt_orders, control_revision=self.revision)


def _snapshot(row, digest):
    return ControlSnapshot(row['revision'], row['updated_at'], row['pause_entries'], row['halt_orders'],
                           row['reason'], digest, row['code_sha256'])


class ControlJournal:
    def __init__(self, path):
        self.path = Path(path).absolute()

    @contextmanager
    def _db(self, write):
        db = None
        try:
            if self.path.is_symlink() or not self.path.is_file():
                raise ValueError('control state missing or linked')
            db = sqlite3.connect(self.path.as_uri() + ('?mode=rw' if write else '?mode=ro'),
                                 uri=True, timeout=10, isolation_level=None)
            db.execute('PRAGMA synchronous=FULL')
            db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            _require_rollback_journal(db)
            yield db
            db.commit()
        except Exception:
            if db is not None:
                db.rollback()
            raise ControlError('operator_controls_unavailable') from None
        finally:
            if db is not None:
                db.close()

    def _read(self, db, now):
        previous = None
        digest = '0' * 64
        version = _version()
        for number, payload, prior, saved_digest in db.execute(
                'SELECT revision,payload,previous_digest,digest FROM events ORDER BY revision'):
            if (number > 10000 or len(payload) > 4096 or prior != digest
                    or hashlib.sha256((prior + '\0' + payload).encode()).hexdigest() != saved_digest):
                raise ValueError('control history integrity mismatch')
            row = decode_json(payload)
            _valid(row, previous)
            if row['revision'] != number or row['code_sha256'] != version:
                raise ValueError('control source or revision mismatch')
            previous, digest = row, saved_digest
        if previous is None or utc_time(previous['updated_at']) > now:
            raise ValueError('control state absent or future')
        return previous, digest

    def _append(self, db, row, previous_digest):
        raw = _json(row)
        digest = hashlib.sha256((previous_digest + '\0' + raw).encode()).hexdigest()
        db.execute('INSERT INTO events VALUES(?,?,?,?)', (row['revision'], raw, previous_digest, digest))
        return _snapshot(row, digest)

    def initialize(self, *, now):
        try:
            if self.path.exists() or self.path.is_symlink():
                raise ValueError('control state already exists')
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            now = utc_time(now)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
            os.close(fd)
            with closing(sqlite3.connect(self.path)) as db:
                db.executescript('''PRAGMA journal_mode=DELETE;PRAGMA synchronous=FULL;BEGIN IMMEDIATE;
                    CREATE TABLE events(revision INTEGER PRIMARY KEY,payload TEXT NOT NULL,
                                        previous_digest TEXT NOT NULL,digest TEXT NOT NULL);''')
                row = dict(schema=1, code_sha256=_version(), revision=1, updated_at=now.isoformat(),
                           pause_entries=True, halt_orders=False, reason='initial_setup')
                _valid(row, None)
                result = self._append(db, row, '0' * 64)
                db.commit()
            if os.name != 'nt':
                fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            return result
        except Exception:
            raise ControlError('operator_control_initialization_failed') from None

    def update(self, *, expected_revision, pause_entries, halt_orders, reason, now):
        try:
            if type(expected_revision) is not int:
                raise ValueError('typed revision required')
            now = utc_time(now)
            with self._db(True) as db:
                previous, digest = self._read(db, now)
                if expected_revision != previous['revision']:
                    raise ValueError('stale operator revision')
                row = dict(schema=1, code_sha256=_version(), revision=expected_revision + 1,
                           updated_at=now.isoformat(), pause_entries=pause_entries,
                           halt_orders=halt_orders, reason=reason)
                _valid(row, previous)
                return self._append(db, row, digest)
        except Exception:
            raise ControlError('operator_control_update_failed') from None

    def read(self, *, now):
        with self._db(False) as db:
            return _snapshot(*self._read(db, utc_time(now)))

    def is_current(self, snapshot, *, now):
        try:
            return type(snapshot) is ControlSnapshot and self.read(now=now) == snapshot
        except Exception:
            return False

    @contextmanager
    def dispatch_lease(self, snapshot, *, now):
        """Hold a SQLite read transaction through dispatch.

        Operator updates cannot commit until an already-started dispatch leaves
        this transaction. A halt committed before acquisition is observed here.
        This does not cancel an in-flight request. Body/network exceptions must
        propagate unchanged: they must never be mistaken for a local veto.
        """
        db = None
        try:
            if type(snapshot) is not ControlSnapshot or self.path.is_symlink() or not self.path.is_file():
                raise ValueError('control lease source invalid')
            db = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True, timeout=10, isolation_level=None)
            db.execute('BEGIN')
            _require_rollback_journal(db)
            actual = _snapshot(*self._read(db, utc_time(now)))
            current = actual == snapshot
        except Exception:
            if db is not None:
                db.close()
            raise ControlError('operator_control_lease_unavailable') from None
        try:
            yield current
        finally:
            try:
                db.rollback()
            finally:
                db.close()

    def verify(self, *, now):
        state = self.read(now=now)
        return dict(revisions=state.revision, integrity_verified=True)
