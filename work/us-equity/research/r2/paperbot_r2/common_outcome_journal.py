"""Durable first-observation journal for R2 common risk-event labels.

The first event records a result derived from immutable raw receipts. A second
event, written only after the first transaction commits, acknowledges that the
result is durable. Acknowledgment is necessary but not sufficient for scoring:
later code must recheck current calendar/feature quarantine and issue lineage.
This module has no network, broker, model-fit, forecast or order capability.
"""

from contextlib import closing
from dataclasses import asdict
from datetime import date, timedelta
import hashlib
from pathlib import Path
import sqlite3
import zoneinfo

from paperbot_us import alpaca, monitor, runner
from paperbot_us.monitor import ObservationArchive
from paperbot_us.runner import _exclusive
from . import classifier, common_outcome, contracts, feature_pipeline, receipt_book
from .contracts import aware, digest, is_sha, json_value
from .source_views import VerifiedFeatureView


class OutcomeJournalError(ValueError):
    pass


def _need(value):
    if not value:
        raise OutcomeJournalError('common_outcome_journal_unavailable')


def _source():
    paths = (Path(__file__), Path(common_outcome.__file__), Path(classifier.__file__),
             Path(contracts.__file__), Path(feature_pipeline.__file__),
             Path(receipt_book.__file__), Path(monitor.__file__),
             Path(runner.__file__), Path(alpaca.__file__), Path(__file__).with_name('source_views.py'))
    _need(all(p.is_file() and not p.is_symlink() and p.stat().st_size <= 524288 for p in paths))
    return digest({str(i) + ':' + p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for i, p in enumerate(paths)})


def _guard_reserved(clock):
    _need(callable(clock))
    at = aware(clock())
    local_day = at.astimezone(zoneinfo.ZoneInfo('America/New_York')).date()
    _need(not date(2026, 11, 10) <= local_day <= date(2026, 11, 23))
    return at


def _ancestor(path, table, seq, *, clock):
    _need(type(seq) is int and seq >= 1)
    _guard_reserved(clock)
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro',
                                 uri=True, isolation_level=None)) as db:
        db.execute('BEGIN')
        row = db.execute('SELECT digest FROM ' + table + ' WHERE seq=?', (seq,)).fetchone()
        db.rollback()
    _need(row is not None and is_sha(row[0]))
    return row[0]


def _binding(feature_journal, raw_book, expected=None, *, clock):
    _need(type(feature_journal) in (feature_pipeline.FeatureJournal, VerifiedFeatureView)
          and type(raw_book) is receipt_book.ReceiptBook)
    if expected is not None:
        _need(type(expected) is dict
              and str(feature_journal.path) == expected.get('feature_path')
              and str(raw_book.path) == expected.get('raw_path'))
    _guard_reserved(clock)
    feature = feature_journal.snapshot(deep=True)
    _guard_reserved(clock)
    raw = raw_book.snapshot()
    plan = feature['plan']
    _need(digest(asdict(raw['plan'])) == plan.raw_plan_sha256
          and receipt_book._source() == plan.raw_source_sha256
          and feature_pipeline._source() == feature['source_sha256']
          and raw['plan'].timezone_sha256 == plan.timezone_sha256
          and raw['plan'].symbols == plan.symbols
          and raw['plan'].start_day == plan.start_day
          and raw['plan'].end_day == plan.end_day)
    reserved = (date(2026, 11, 10) + timedelta(days=i) for i in range(14))
    _need(all(day.isoformat() in plan.paused_days for day in reserved
              if plan.start_day <= day.isoformat() <= plan.end_day))
    fstat, rstat = feature_journal.path.stat(), raw_book.path.stat()
    base = dict(feature_plan_sha256=digest(asdict(plan)),
                feature_source_sha256=feature['source_sha256'],
                raw_plan_sha256=plan.raw_plan_sha256,
                raw_source_sha256=plan.raw_source_sha256,
                timezone_sha256=plan.timezone_sha256,
                symbols=list(plan.symbols), start_day=plan.start_day,
                end_day=plan.end_day, paused_days=list(plan.paused_days),
                feature_path=str(feature_journal.path), raw_path=str(raw_book.path),
                feature_device=fstat.st_dev, feature_inode=fstat.st_ino,
                raw_device=rstat.st_dev, raw_inode=rstat.st_ino)
    if expected is None:
        return dict(base, feature_anchor_seq=feature['head_seq'],
                    feature_anchor_digest=feature['head_digest'],
                    raw_anchor_seq=raw['head_seq'], raw_anchor_digest=raw['head_digest'])
    _need(type(expected) is dict and all(expected.get(k) == v for k, v in base.items())
          and feature['head_seq'] >= expected['feature_anchor_seq']
          and raw['head_seq'] >= expected['raw_anchor_seq']
          and _ancestor(feature_journal.path, 'observations', expected['feature_anchor_seq'], clock=clock)
          == expected['feature_anchor_digest']
          and _ancestor(raw_book.path, 'events', expected['raw_anchor_seq'], clock=clock)
          == expected['raw_anchor_digest'])
    return expected


def _outcome(raw):
    _need(type(raw) is dict)
    value = dict(raw)
    example = dict(value['example'])
    vector = dict(example['features'])
    vector['feature_names'] = tuple(vector['feature_names'])
    vector['values'] = tuple(vector['values'])
    example['features'] = classifier.FeatureVector(**vector)
    value['example'] = classifier.ClassifierExample(**example)
    return common_outcome.CommonOutcome(**value)


def _verify_consumed(feature_journal, raw_book, entries, *, clock):
    # The highest consumed head commits its entire preceding chain, including
    # source history consumed after this journal was initialized.
    for field, path, table in (('feature', feature_journal.path, 'observations'),
                               ('raw', raw_book.path, 'events')):
        checkpoints = {}
        for entry in entries.values():
            outcome = entry['outcome']
            seq, sha = outcome[field + '_head_seq'], outcome[field + '_head_digest']
            _need(seq not in checkpoints or checkpoints[seq] == sha)
            checkpoints[seq] = sha
        if checkpoints:
            seq = max(checkpoints)
            _need(_ancestor(path, table, seq, clock=clock) == checkpoints[seq])


def _verify_first_lineage(feature_journal, raw_book, entry, *, clock):
    """Reject another same-plan archive without rewriting the stored first result."""
    result = _outcome(entry['outcome'])
    _guard_reserved(clock)
    feature = feature_journal.snapshot(deep=True)
    found = feature['entries'].get(result.feature_key)
    _need(found is not None and found['accepted_bundle'] is not None
          and found['accepted_bundle']['seal'] == result.feature_bundle_sha256
          and feature['head_seq'] >= result.feature_head_seq)
    _guard_reserved(clock)
    with closing(sqlite3.connect(feature_journal.path.as_uri() + '?mode=ro',
                                 uri=True, isolation_level=None)) as db:
        db.execute('BEGIN')
        old = db.execute('SELECT digest FROM observations WHERE seq=?',
                         (result.feature_head_seq,)).fetchone()
        _need(old is not None and old[0] == result.feature_head_digest)
        db.rollback()
    _guard_reserved(clock)
    with raw_book._transaction() as (db, raw, _):
        _need(raw['head_seq'] >= result.raw_head_seq)
        old = db.execute('SELECT digest FROM events WHERE seq=?',
                         (result.raw_head_seq,)).fetchone()
        _need(old is not None and old[0] == result.raw_head_digest)
        for start, expected_sha, recorded_at in (
                (result.example.features.event_at, result.gap_bar_sha256,
                 result.gap_recorded_at),
                (result.target_start, result.target_bar_sha256,
                 result.target_recorded_at)):
            slot = raw['slots'].get(receipt_book._key(result.symbol, start))
            _need(slot is not None and slot['first'] is not None
                  and digest(slot['first']) == expected_sha
                  and aware(slot['first_recorded_at']) == recorded_at)


def _event(kind, body, at):
    at = aware(at)
    return dict(kind=kind, body=json_value(body), at=at.isoformat(),
                generated_at=at.timestamp(), source_sha256=_source())


def _apply(state, event, sha, *, expected_source=None):
    _need(type(event) is dict and set(event) ==
          {'kind', 'body', 'at', 'generated_at', 'source_sha256', 'observing_started_at'}
          and is_sha(sha) and event['source_sha256'] == (expected_source or _source()))
    at = aware(event['at'])
    _need(event['generated_at'] == at.timestamp()
          and (state['last_at'] is None or at >= state['last_at']))
    kind, body = event['kind'], event['body']
    _need(type(body) is dict)
    if kind == 'created':
        _need(state['binding'] is None and set(body) == {'binding'})
        binding = body['binding']
        _need(type(binding) is dict and set(binding) ==
              {'feature_plan_sha256', 'feature_source_sha256', 'raw_plan_sha256',
               'raw_source_sha256', 'timezone_sha256', 'symbols', 'start_day',
               'end_day', 'paused_days', 'feature_path', 'raw_path',
               'feature_device', 'feature_inode', 'raw_device', 'raw_inode',
               'feature_anchor_seq', 'feature_anchor_digest',
               'raw_anchor_seq', 'raw_anchor_digest'}
              and all(is_sha(binding[name]) for name in
                      ('feature_plan_sha256', 'feature_source_sha256',
                       'raw_plan_sha256', 'raw_source_sha256', 'timezone_sha256',
                       'feature_anchor_digest', 'raw_anchor_digest'))
              and type(binding['symbols']) is list and 1 <= len(binding['symbols']) <= 20
              and all(type(v) is str for v in binding['symbols'])
              and type(binding['paused_days']) is list
              and all(type(binding[k]) is int and binding[k] >= 0 for k in
                      ('feature_device', 'feature_inode', 'raw_device', 'raw_inode'))
              and all(type(binding[k]) is int and binding[k] >= 1 for k in
                      ('feature_anchor_seq', 'raw_anchor_seq'))
              and all(type(binding[k]) is str and Path(binding[k]).is_absolute()
                      for k in ('feature_path', 'raw_path')))
        state['binding'] = binding
    elif kind == 'observed':
        _need(state['binding'] is not None and set(body) == {'key', 'outcome'}
              and type(body['key']) is str and body['key'] not in state['entries'])
        result = _outcome(body['outcome'])
        binding = state['binding']
        _need(result.feature_key == body['key'] and result.symbol in binding['symbols']
              and result.raw_plan_sha256 == binding['raw_plan_sha256']
              and result.raw_source_sha256 == binding['raw_source_sha256']
              and at >= result.observed_at)
        state['entries'][body['key']] = dict(status='pending_ack',
            outcome=body['outcome'], observed_event_at=at.isoformat(),
            observed_event_digest=sha, acknowledged_at=None,
            ack_event_digest=None, effective_late=None, score_eligible=False)
    elif kind == 'acknowledged':
        _need(state['binding'] is not None and set(body) == {'key', 'observed_event_digest'}
              and type(body['key']) is str and body['key'] in state['entries'])
        entry = state['entries'][body['key']]
        _need(entry['status'] == 'pending_ack'
              and body['observed_event_digest'] == entry['observed_event_digest']
              and at > aware(entry['observed_event_at']))
        entry.update(status='acknowledged', acknowledged_at=at.isoformat(),
                     ack_event_digest=sha,
                     effective_late=entry['outcome']['late'] or
                     (at - aware(entry['outcome']['target_end'])).total_seconds() > 120)
    else:
        raise OutcomeJournalError('common_outcome_journal_unavailable')
    state['last_at'] = at


class OutcomeJournal:
    def __init__(self, path):
        self.path = Path(path).absolute()

    def _path(self):
        _need(self.path.is_file() and not self.path.is_symlink()
              and self.path.parent.is_dir() and not self.path.parent.is_symlink()
              and self.path.stat().st_size <= 268435456)

    def initialize(self, feature_journal, raw_book, *, clock):
        try:
            _guard_reserved(clock)
            _need(self.path.parent.is_dir()
                  and not self.path.parent.is_symlink())
            binding = _binding(feature_journal, raw_book, clock=clock)
            at = _guard_reserved(clock)
            with _exclusive(self.path):
                _need(not self.path.exists() and not self.path.is_symlink())
                ObservationArchive(self.path).append(_event('created', dict(binding=binding), at))
                _need(self.snapshot()['binding'] == binding)
        except Exception:
            raise OutcomeJournalError('common_outcome_journal_initialization_failed') from None

    def snapshot(self):
        try:
            self._path()
            source = _source()
            state = dict(binding=None, entries={}, head_seq=0, head_digest=None, last_at=None)
            with closing(sqlite3.connect(self.path.as_uri() + '?mode=ro',
                                         uri=True, isolation_level=None)) as db:
                db.execute('BEGIN')
                _, count, head = ObservationArchive(self.path)._verify_chain(db)
                _need(1 <= count <= 20001)
                for seq, payload, sha in db.execute(
                        'SELECT seq,payload,digest FROM observations ORDER BY seq'):
                    _need(seq <= count and len(payload) <= 16384)
                    _apply(state, alpaca.decode_json(payload), sha, expected_source=source)
                state.update(head_seq=count, head_digest=head)
                db.rollback()
            _need(state['binding'] is not None and _source() == source)
            return state
        except Exception:
            raise OutcomeJournalError('common_outcome_journal_integrity_failed') from None

    def _append(self, state, kind, body, at):
        _need(state['head_seq'] < 20001)
        event = _event(kind, body, at)
        _apply(dict(binding=state['binding'], entries={k: dict(v) for k, v in
                     state['entries'].items()}, last_at=state['last_at']),
               {**event, 'observing_started_at': 0}, '0' * 64)
        return ObservationArchive(self.path).append(event)

    def record_first(self, feature_journal, raw_book, key, *, clock):
        try:
            _need(type(key) is str)
            _guard_reserved(clock)
            with _exclusive(self.path):
                state = self.snapshot()
                _binding(feature_journal, raw_book, state['binding'], clock=clock)
                _verify_consumed(feature_journal, raw_book, state['entries'], clock=clock)
                entry = state['entries'].get(key)
                if entry is None:
                    outcome = common_outcome.observe_common_outcome(
                        feature_journal, raw_book, key, clock=clock,
                        read_guard=lambda: _guard_reserved(clock))
                    at = aware(clock())
                    _need(at >= outcome.observed_at and state['head_seq'] <= 19999)
                    self._append(state, 'observed',
                                 dict(key=key, outcome=asdict(outcome)), at)
                    state = self.snapshot()
                    entry = state['entries'][key]
                _verify_first_lineage(feature_journal, raw_book, entry, clock=clock)
                if entry['status'] == 'pending_ack':
                    at = aware(clock())
                    _need(at > aware(entry['observed_event_at']))
                    self._append(state, 'acknowledged',
                                 dict(key=key, observed_event_digest=entry['observed_event_digest']), at)
                    state = self.snapshot()
                    entry = state['entries'][key]
                _need(entry['status'] == 'acknowledged')
                return entry
        except Exception:
            raise OutcomeJournalError('common_outcome_journal_record_failed') from None
