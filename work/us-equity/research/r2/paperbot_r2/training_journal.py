"""First training-batch reservation, durable result and post-commit acknowledgment.

Local consumed trial units must enter the future canonical shared registry.
This is a frozen proposal's research record, never experiment registration,
forecast admission or order authority. Seals are integrity, not admin-proof
authentication. No upstream read happens when recovering a reserved/result run.
"""
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import os
from pathlib import Path
import sqlite3

from paperbot_us import alpaca
from paperbot_us.runner import _exclusive
from . import model_training as models, training_codec as codec, training_data as training
from .contracts import aware, digest, is_sha


class TrainingJournalError(ValueError):
    pass


def _need(value):
    if not value:
        raise TrainingJournalError('first_training_journal_unavailable')


def source_sha256():
    paths = (Path(__file__), Path(codec.__file__))
    _need(all(p.is_file() and not p.is_symlink() and p.stat().st_size <= 524288 for p in paths))
    return digest(dict(journal=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        codec=hashlib.sha256(paths[1].read_bytes()).hexdigest(),
        training=training.source_sha256(), fitter=models.source_sha256()))


def _empty():
    return dict(phase=None, binding=None, training=None, fits=None, reserved_at=None,
        reserved_digest=None, result_at=None, result_digest=None, acknowledged_at=None,
        ack_digest=None, confirmation_digest=None, commit_observed_at=None,
        abandoned_reason=None, reserved_trial_units=0,
        head_seq=0, head_digest='0' * 64, last_at=None)


def _metadata(source, at):
    return dict(version=1, source_sha256=source, head_seq=0, head_digest='0' * 64,
                last_at=aware(at).isoformat())


def _hash(previous, payload):
    return hashlib.sha256((previous + '\0' + payload).encode('utf-8')).hexdigest()


def _fit_binding(snapshot, fits, reserved_at):
    _need(type(fits) is models.TrainingFits and fits.snapshot_sha256 == snapshot.seal)
    symbols = {s.symbol: s for s in snapshot.symbols}
    for attempt in fits.attempts:
        symbol = symbols[attempt.symbol]
        rows = symbol.examples if attempt.family in ('prevalence', 'gaussian_nb') else symbol.returns
        _need(attempt.price_identity == symbol.price_identity and attempt.input_count == len(rows)
              and attempt.input_rows_sha256 == digest([asdict(r) for r in rows])
              and attempt.started_at > reserved_at)


@dataclass(frozen=True)
class TrainingReceipt:
    training: training.TrainingSnapshot
    fits: models.TrainingFits
    book_path: str
    book_device: int
    book_inode: int
    head_seq: int
    head_digest: str
    reservation_digest: str
    result_digest: str
    ack_digest: str
    confirmation_digest: str
    acknowledged_at: datetime
    commit_observed_at: datetime
    loaded_at: datetime
    canonical_registration_required: bool
    source_sha256: str
    seal: str

    def __post_init__(self):
        _need(type(self.training) is training.TrainingSnapshot and type(self.fits) is models.TrainingFits
              and self.training.ready and self.fits.snapshot_sha256 == self.training.seal
              and self.canonical_registration_required is True and self.head_seq == 5
              and all(is_sha(v) for v in (self.head_digest, self.reservation_digest,
                                          self.result_digest, self.ack_digest, self.confirmation_digest))
              and self.head_digest == self.confirmation_digest
              and self.loaded_at >= self.commit_observed_at > self.acknowledged_at >= self.fits.available_at
              and self.source_sha256 == source_sha256()
              and self.seal == digest({k: v for k, v in asdict(self).items() if k != 'seal'}))


class TrainingJournal:
    def __init__(self, path):
        self.path = Path(path).absolute()

    def _identity(self):
        _need(self.path.is_file() and not self.path.is_symlink()
              and self.path.parent.is_dir() and not self.path.parent.is_symlink()
              and self.path.stat().st_size <= 67108864)
        stat = self.path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def _apply(self, state, event, sha, source):
        _need(type(event) is dict and set(event) == {'kind', 'at', 'body', 'source_sha256'}
              and event['source_sha256'] == source and is_sha(sha))
        at, body, kind = aware(event['at']), event['body'], event['kind']
        _need(type(body) is dict and (state['last_at'] is None or at >= state['last_at']))
        if kind == 'created':
            _need(state['phase'] is None and set(body) ==
                  {'book_path', 'book_device', 'book_inode', 'canonical_registration_required', 'scope'}
                  and body['book_path'] == str(self.path)
                  and (body['book_device'], body['book_inode']) == self._identity()[:2]
                  and body['canonical_registration_required'] is True
                  and body['scope'] == 'frozen_proposal_not_registered')
            state.update(phase='created', binding=dict(body))
        elif kind == 'reserved':
            _need(state['phase'] == 'created' and set(body) == {'training', 'candidates'})
            snapshot = codec.decode_snapshot(body['training'])
            _need(snapshot.ready and snapshot.available_at <= at
                  and body['candidates'] == [[s, f] for s in training.SYMBOLS for f in models.FAMILIES])
            state.update(phase='reserved', training=snapshot, reserved_at=at,
                         reserved_digest=sha, reserved_trial_units=15)
        elif kind == 'result':
            _need(state['phase'] == 'reserved' and set(body) == {'fits', 'reservation_digest'}
                  and body['reservation_digest'] == state['reserved_digest'])
            fits = codec.decode_fits(body['fits'])
            _fit_binding(state['training'], fits, state['reserved_at'])
            _need(fits.available_at <= at)
            state.update(phase='result_pending_ack', fits=fits, result_at=at, result_digest=sha)
        elif kind == 'acknowledged':
            _need(state['phase'] == 'result_pending_ack' and set(body) == {'result_digest'}
                  and body['result_digest'] == state['result_digest'] and at > state['result_at'])
            state.update(phase='ack_pending_confirmation', acknowledged_at=at, ack_digest=sha)
        elif kind == 'confirmed':
            # This timestamp observes the already durable ACK transaction. It
            # never claims to be the later confirmation transaction's commit.
            _need(state['phase'] == 'ack_pending_confirmation' and set(body) == {'ack_digest'}
                  and body['ack_digest'] == state['ack_digest'] and at > state['acknowledged_at'])
            state.update(phase='acknowledged', commit_observed_at=at, confirmation_digest=sha)
        elif kind == 'abandoned':
            _need(state['phase'] == 'reserved' and set(body) == {'reservation_digest', 'reason'}
                  and body['reservation_digest'] == state['reserved_digest']
                  and body['reason'] == 'interrupted_or_uncommitted_fit')
            state.update(phase='abandoned', abandoned_reason=body['reason'])
        else:
            _need(False)
        state['last_at'] = at

    def _replay(self, db, *, allow_empty=False):
        _need(db.execute('PRAGMA journal_mode').fetchone()[0].lower() == 'delete')
        values = db.execute('SELECT version,source_sha256,head_seq,head_digest,last_at,checksum FROM metadata').fetchall()
        _need(len(values) == 1)
        names = ('version', 'source_sha256', 'head_seq', 'head_digest', 'last_at')
        meta = dict(zip(names, values[0][:-1]))
        source = source_sha256()
        _need(meta['version'] == 1 and meta['source_sha256'] == source
              and values[0][-1] == digest(meta) and type(meta['head_seq']) is int
              and 0 <= meta['head_seq'] <= 5)
        state = _empty()
        for seq, payload, previous, sha in db.execute(
                'SELECT seq,payload,previous_digest,digest FROM events ORDER BY seq'):
            _need(seq == state['head_seq'] + 1 <= 5 and type(payload) is str
                  and len(payload.encode('utf-8')) <= codec.MAX_BYTES
                  and previous == state['head_digest'] and sha == _hash(previous, payload))
            self._apply(state, alpaca.decode_json(payload), sha, source)
            state.update(head_seq=seq, head_digest=sha)
        _need(state['head_seq'] == meta['head_seq'] and state['head_digest'] == meta['head_digest']
              and (state['head_seq'] > 0 or allow_empty)
              and (state['last_at'] is None or state['last_at'] == aware(meta['last_at']))
              and source_sha256() == source)
        return state, meta

    def _write(self, db, state, meta, kind, body, at):
        _need(state['head_seq'] < 5)
        event = dict(kind=kind, at=aware(at).isoformat(), body=body, source_sha256=meta['source_sha256'])
        payload = codec.encode(event)
        previous = state['head_digest']; sha = _hash(previous, payload)
        self._apply(state, alpaca.decode_json(payload), sha, meta['source_sha256'])
        sequence = state['head_seq'] + 1
        db.execute('INSERT INTO events VALUES(?,?,?,?)', (sequence, payload, previous, sha))
        meta.update(head_seq=sequence, head_digest=sha, last_at=aware(at).isoformat())
        db.execute('UPDATE metadata SET head_seq=?,head_digest=?,last_at=?,checksum=?',
                   (sequence, sha, meta['last_at'], digest(meta)))
        return sha

    def initialize(self, *, timezone_file, clock):
        try:
            guard = training.TrainingClock(timezone_file, clock); at = guard()
            _need(self.path.parent.is_dir() and not self.path.parent.is_symlink())
            with _exclusive(self.path):
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
                os.close(fd)
                with closing(sqlite3.connect(self.path, isolation_level=None, timeout=5)) as db:
                    db.execute('PRAGMA synchronous=FULL'); db.execute('BEGIN IMMEDIATE')
                    db.execute('CREATE TABLE metadata(version INTEGER,source_sha256 TEXT,head_seq INTEGER,head_digest TEXT,last_at TEXT,checksum TEXT)')
                    db.execute('CREATE TABLE events(seq INTEGER PRIMARY KEY,payload TEXT,previous_digest TEXT,digest TEXT)')
                    meta = _metadata(source_sha256(), at)
                    db.execute('INSERT INTO metadata VALUES(?,?,?,?,?,?)', tuple(meta.values()) + (digest(meta),))
                    identity = self._identity()
                    body = dict(book_path=str(self.path), book_device=identity[0], book_inode=identity[1],
                        canonical_registration_required=True, scope='frozen_proposal_not_registered')
                    self._write(db, _empty(), meta, 'created', body, at)
                    db.commit()
                if os.name != 'nt':
                    directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                guard()
                _need(self.snapshot()['phase'] == 'created')
        except Exception:
            raise TrainingJournalError('first_training_journal_initialization_failed') from None

    def snapshot(self):
        try:
            before = self._identity()
            with closing(sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True, isolation_level=None)) as db:
                db.execute('PRAGMA query_only=ON')
                version = db.execute('PRAGMA data_version').fetchone()[0]
                db.execute('BEGIN'); state, _ = self._replay(db); db.rollback()
                _need(db.execute('PRAGMA data_version').fetchone()[0] == version)
            _need(self._identity() == before)
            return state
        except Exception:
            raise TrainingJournalError('first_training_journal_integrity_failed') from None

    def _append(self, state, kind, body, at, *, clock):
        self._identity()
        with closing(sqlite3.connect(self.path.as_uri() + '?mode=rw', uri=True, isolation_level=None, timeout=5)) as db:
            db.execute('PRAGMA synchronous=FULL'); db.execute('BEGIN IMMEDIATE')
            current, meta = self._replay(db)
            _need(current['head_seq'] == state['head_seq'] and current['head_digest'] == state['head_digest'])
            stamped = clock()  # After lock/transaction/replay, never a stale caller timestamp.
            _need(stamped >= at)
            result = self._write(db, current, meta, kind, body, stamped)
            clock()  # Final boundary guard after serialization/validation, before COMMIT.
            db.commit()
        clock()  # If COMMIT crossed the cutoff, no post-ACK confirmation can follow.
        return result

    def run_once(self, raw_book, feature_journal, outcome_journal, *, timezone_file, clock):
        try:
            guard = training.TrainingClock(timezone_file, clock); at = guard()
            with _exclusive(self.path):
                state = self.snapshot(); _need(at >= state['last_at'])
                if state['phase'] == 'reserved':
                    self._append(state, 'abandoned', dict(reservation_digest=state['reserved_digest'],
                        reason='interrupted_or_uncommitted_fit'), guard(), clock=guard)
                    return self.snapshot()
                if state['phase'] in ('acknowledged', 'abandoned'):
                    return state
                if state['phase'] == 'created':
                    snapshot = training.read_training_snapshot(raw_book, feature_journal, outcome_journal,
                                                               timezone_file=timezone_file, clock=guard)
                    if not snapshot.ready:
                        return state
                    self._append(state, 'reserved', dict(training=asdict(snapshot),
                        candidates=[[s, f] for s in training.SYMBOLS for f in models.FAMILIES]), guard(), clock=guard)
                    state = self.snapshot()  # Acknowledge reservation COMMIT before fitting.
                    fits = models.fit_training_snapshot(state['training'], timezone_file=timezone_file, clock=guard)
                    self._append(state, 'result', dict(fits=asdict(fits),
                        reservation_digest=state['reserved_digest']), guard(), clock=guard)
                    state = self.snapshot()
                if state['phase'] == 'result_pending_ack':
                    self._append(state, 'acknowledged', dict(result_digest=state['result_digest']), guard(), clock=guard)
                    state = self.snapshot()  # Replay the ACK only after its COMMIT returned.
                _need(state['phase'] == 'ack_pending_confirmation')
                self._append(state, 'confirmed', dict(ack_digest=state['ack_digest']), guard(), clock=guard)
                return self.snapshot()
        except Exception:
            raise TrainingJournalError('first_training_journal_run_failed') from None

    def load_acknowledged(self, *, clock):
        """Read stored training only; future issue still needs canonical admission."""
        try:
            started = aware(clock()); state = self.snapshot(); loaded = aware(clock())
            _need(state['phase'] == 'acknowledged' and loaded >= started >= state['acknowledged_at'])
            binding = state['binding']
            fields = dict(training=state['training'], fits=state['fits'], book_path=str(self.path),
                book_device=binding['book_device'], book_inode=binding['book_inode'], head_seq=state['head_seq'],
                head_digest=state['head_digest'], reservation_digest=state['reserved_digest'],
                result_digest=state['result_digest'], ack_digest=state['ack_digest'],
                confirmation_digest=state['confirmation_digest'], acknowledged_at=state['acknowledged_at'],
                commit_observed_at=state['commit_observed_at'], loaded_at=loaded,
                canonical_registration_required=True, source_sha256=source_sha256())
            return TrainingReceipt(**fields, seal=digest({k: asdict(v) if k in ('training', 'fits') else v
                                                         for k, v in fields.items()}))
        except Exception:
            raise TrainingJournalError('acknowledged_training_unavailable') from None
