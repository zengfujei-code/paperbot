"""Read-only training context from original first-receipt evidence.

This is a frozen proposal's input adapter, not a registry, scheduled trainer or
forecast. Seals prove content consistency, not authenticity against an operator
able to fabricate whole archives. No account, network or order capability.
"""
from contextlib import closing, contextmanager, ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from fractions import Fraction
import hashlib
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from paperbot_us import data, regime_baselines
from . import common_outcome_journal as outcomes, feature_pipeline, receipt_book
from .contracts import aware, digest, is_sha
from .source_views import VerifiedFeatureView

SYMBOLS = ('AAPL', 'MSFT', 'SPY')
TRAINING_DAYS = ('2026-10-27', '2026-10-28', '2026-10-29', '2026-10-30',
                 '2026-11-02', '2026-11-03', '2026-11-04', '2026-11-05', '2026-11-06', '2026-11-09')
TIMEZONE_SHA256 = 'e9ed07d7bee0c76a9d442d091ef1f01668fee7c4f26014c0a868b19fe6c18a95'


class TrainingDataError(ValueError):
    pass


def _need(value, code='training_evidence_unavailable'):
    if not value:
        raise TrainingDataError(code)


def source_sha256():
    path = Path(__file__)
    _need(path.is_file() and not path.is_symlink())
    return digest(dict(adapter=hashlib.sha256(path.read_bytes()).hexdigest(),
                       raw=receipt_book._source(), feature=feature_pipeline._source(),
                       outcome=outcomes._source()))


def _seal(value):
    return digest({k: v for k, v in asdict(value).items() if k != 'seal'})


class TrainingClock:
    """Check the pinned zone and monotone actual clock before each read/fit."""
    def __init__(self, timezone_file, clock):
        path = Path(timezone_file)
        _need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 65536)
        content = path.read_bytes()
        _need(hashlib.sha256(content).hexdigest() == TIMEZONE_SHA256 and callable(clock))
        with path.open('rb') as stream:
            self.zone = ZoneInfo.from_file(stream, key='America/New_York')
        self.clock, self.previous = clock, None

    def __call__(self):
        at = aware(self.clock())
        _need(self.previous is None or at >= self.previous, 'training_clock_regression')
        local = at.astimezone(self.zone)
        _need(local.replace(tzinfo=None) < datetime(2026, 11, 9, 23, 55),
              'training_read_window_closed')
        self.previous = at
        return at


@dataclass(frozen=True)
class SourceBinding:
    kind: str
    path: str
    device: int
    inode: int
    head_seq: int
    head_digest: str
    plan_sha256: str
    source_sha256: str


@dataclass(frozen=True)
class ReturnWitness:
    day: str
    previous_bar_sha256: str
    current_bar_sha256: str
    previous_recorded_at: datetime
    current_recorded_at: datetime
    late: bool


@dataclass(frozen=True)
class ExampleWitness:
    feature_key: str
    outcome_seal: str
    ack_event_digest: str
    acknowledged_at: datetime
    late: bool


@dataclass(frozen=True)
class SessionCoverage:
    day: str
    calendar_status: str
    planned_returns: object
    planned_examples: object
    returns: int
    examples: int
    late_returns: int
    late_examples: int
    pending_examples: int
    quarantined_examples: int
    missing_examples: object


@dataclass(frozen=True)
class SymbolTraining:
    symbol: str
    price_identity: tuple
    returns: tuple
    return_witnesses: tuple
    examples: tuple
    example_witnesses: tuple
    coverage: tuple


@dataclass(frozen=True)
class TrainingSnapshot:
    symbols: tuple
    bindings: tuple
    read_started_at: datetime
    available_at: datetime
    timezone_sha256: str
    ready: bool
    source_sha256: str
    seal: str

    def __post_init__(self):
        _need(type(self.symbols) is tuple and tuple(s.symbol for s in self.symbols) == SYMBOLS
              and all(type(s) is SymbolTraining for s in self.symbols)
              and type(self.bindings) is tuple and len(self.bindings) == 3
              and tuple(b.kind for b in self.bindings) == ('raw', 'feature', 'outcome')
              and all(type(b) is SourceBinding and b.head_seq >= 1
                      and all(is_sha(v) for v in (b.head_digest, b.plan_sha256, b.source_sha256))
                      for b in self.bindings))
        _need(type(self.ready) is bool and self.timezone_sha256 == TIMEZONE_SHA256
              and aware(self.available_at) >= aware(self.read_started_at)
              and self.source_sha256 == source_sha256() and self.seal == _seal(self),
              'training_snapshot_binding_mismatch')
        for symbol in self.symbols:
            _need(symbol.price_identity == (symbol.symbol, 'alpaca', 'iex', 'raw', 900, 'USD', 'shares')
                  and all(type(v) is tuple for v in (symbol.returns, symbol.return_witnesses,
                                                    symbol.examples, symbol.example_witnesses, symbol.coverage))
                  and len(symbol.returns) == len(symbol.return_witnesses)
                  and len(symbol.examples) == len(symbol.example_witnesses)
                  and tuple(c.day for c in symbol.coverage) == TRAINING_DAYS)
            previous = None
            for row, witness in zip(symbol.returns, symbol.return_witnesses):
                _need(type(row) is regime_baselines.ReturnObservation and type(witness) is ReturnWitness
                      and witness.day in TRAINING_DAYS and row.available_at == self.available_at
                      and witness.previous_recorded_at <= self.available_at
                      and witness.current_recorded_at <= self.available_at
                      and (previous is None or row.starts_at >= previous))
                previous = row.ends_at
            _need(all(w.acknowledged_at <= self.available_at and e.outcome_available_at <= w.acknowledged_at
                      for e, w in zip(symbol.examples, symbol.example_witnesses)))


def _identity(path):
    _need(path.is_file() and not path.is_symlink() and not path.parent.is_symlink())
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _heads(path, guard):
    guard()
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, isolation_level=None)) as db:
        return dict(db.execute('SELECT seq,digest FROM observations'))


@contextmanager
def _watch_versions(paths, guard):
    # Keep the same connections across the complete read. Main-file metadata
    # alone cannot observe commits confined to SQLite's write-ahead log.
    with ExitStack() as stack:
        connections = []
        for path in paths:
            guard()
            db = stack.enter_context(closing(sqlite3.connect(
                path.as_uri() + '?mode=ro', uri=True, isolation_level=None)))
            db.execute('PRAGMA query_only=ON')
            connections.append(db)
        def versions():
            guard()
            return tuple(db.execute('PRAGMA data_version').fetchone()[0] for db in connections)
        initial = versions()
        yield lambda: versions() == initial


def _binding(raw, feature, common, paths, identities, raw_heads, feature_heads):
    plan, binding = feature['plan'], common['binding']
    _need(plan.symbols == SYMBOLS and raw['plan'].symbols == SYMBOLS
          and plan.start_day == raw['plan'].start_day == TRAINING_DAYS[0]
          and plan.end_day == raw['plan'].end_day == '2026-11-25'
          and plan.lookback_returns == 8 and plan.threshold_fraction == .005
          and plan.timezone_sha256 == raw['plan'].timezone_sha256 == TIMEZONE_SHA256
          and plan.raw_plan_sha256 == digest(asdict(raw['plan']))
          and plan.raw_source_sha256 == receipt_book._source()
          and feature['source_sha256'] == feature_pipeline._source())
    reserved = tuple((datetime(2026, 11, 10) + timedelta(days=i)).date().isoformat() for i in range(14))
    _need(plan.paused_days == reserved)
    expected = dict(feature_plan_sha256=digest(asdict(plan)), feature_source_sha256=feature['source_sha256'],
                    raw_plan_sha256=plan.raw_plan_sha256, raw_source_sha256=plan.raw_source_sha256,
                    timezone_sha256=TIMEZONE_SHA256, symbols=list(SYMBOLS), start_day=plan.start_day,
                    end_day=plan.end_day, paused_days=list(plan.paused_days),
                    feature_path=str(paths[1]), raw_path=str(paths[0]),
                    feature_device=identities[1][0], feature_inode=identities[1][1],
                    raw_device=identities[0][0], raw_inode=identities[0][1])
    _need(all(binding.get(k) == v for k, v in expected.items())
          and raw_heads.get(binding['raw_anchor_seq']) == binding['raw_anchor_digest']
          and feature_heads.get(binding['feature_anchor_seq']) == binding['feature_anchor_digest'])
    # Every consumed checkpoint must still belong to the original chains.
    for entry in common['entries'].values():
        value = entry['outcome']
        _need(raw_heads.get(value['raw_head_seq']) == value['raw_head_digest']
              and feature_heads.get(value['feature_head_seq']) == value['feature_head_digest'])


def _rows(raw, feature, common, *, guard):
    selected = []
    calendars_ok = True
    last_close = None
    for symbol in SYMBOLS:
        pairs, examples, witnesses, coverage = [], [], [], []
        for day in TRAINING_DAYS:
            guard()
            proof = raw['calendars'].get(day)
            status = ('missing' if proof is None or proof['session'] is None else
                      'changed' if proof['changed'] else 'verified')
            if status != 'verified':
                calendars_ok = False
                known_quarantine = 0
                for entry in feature['entries'].values():
                    bundle = entry['accepted_bundle'] or entry['prepared_bundle']
                    if bundle['session_day'] == day and bundle['price_identity'][0] == symbol:
                        known_quarantine += 1
                coverage.append(SessionCoverage(day, status, None, None, 0, 0, 0, 0,
                                                0, known_quarantine, None))
                continue
            session = receipt_book._session(proof['session'])
            last_close = max(last_close or session.closes, session.closes)
            count = int((session.closes - session.opens).total_seconds() // 900)
            day_pairs, day_examples, day_late, pending, quarantined = [], [], 0, 0, 0
            for i in range(1, count):
                start = session.opens + timedelta(seconds=900 * i)
                slots = tuple(raw['slots'][receipt_book._key(symbol, s)]
                              for s in (start - timedelta(seconds=900), start))
                if any(s['first'] is None for s in slots):
                    continue
                bars = tuple(data.Bar(**s['first']) for s in slots)
                if any(b.volume <= 0 for b in bars):
                    continue
                for b in bars:
                    b.validate(as_of=guard(), session=session)
                _need(bars[0].ends == bars[1].starts and bars[0].symbol == bars[1].symbol == symbol)
                price = tuple(Fraction(str(b.close)) for b in bars)
                witness = ReturnWitness(day, *(digest(s['first']) for s in slots),
                                        *(aware(s['first_recorded_at']) for s in slots),
                                        any(not s['on_time'] for s in slots))
                day_pairs.append((bars[1].ends, float((price[1] - price[0]) / price[0]), witness))
            for i in range(9, count - 1):
                event_at = session.opens + timedelta(seconds=900 * i)
                key = feature_pipeline._entry_key(symbol, event_at)
                entry = feature['entries'].get(key)
                if entry is None or entry['phase'] != 'active':
                    continue
                if entry['quarantined']:
                    quarantined += 1
                    continue
                bundle = feature_pipeline._bundle(entry['accepted_bundle'])
                _need(bundle.price_identity == (symbol, 'alpaca', 'iex', 'raw', 900, 'USD', 'shares')
                      and dict(entry['session']) == proof['session'])
                context = entry['raw_context']
                for original in entry['bars']:
                    slot = raw['slots'].get(receipt_book._key(symbol, aware(original['starts'])))
                    _need(slot is not None and slot['first'] == dict(original))
                outcome_entry = common['entries'].get(key)
                gap_target = tuple(raw['slots'][receipt_book._key(symbol, start)]
                                   for start in (event_at, bundle.target_start))
                usable = all(s['first'] is not None and s['first']['volume'] > 0 for s in gap_target)
                if outcome_entry is None or outcome_entry['status'] != 'acknowledged':
                    pending += int(usable)
                    continue
                value = outcomes._outcome(outcome_entry['outcome'])
                _need(usable and value.feature_bundle_sha256 == bundle.seal
                      and value.example.features == bundle.vector
                      and value.target_start == bundle.target_start and value.target_end == bundle.target_end)
                for slot, sha, recorded in zip(gap_target,
                        (value.gap_bar_sha256, value.target_bar_sha256),
                        (value.gap_recorded_at, value.target_recorded_at)):
                    _need(digest(slot['first']) == sha and aware(slot['first_recorded_at']) == recorded)
                first, second = (Fraction(str(s['first']['close'])) for s in gap_target)
                _need(value.label == int(abs(second - first) >= Fraction('0.005') * first)
                      and value.return_fraction == float((second - first) / first))
                late = bool(outcome_entry['effective_late'])
                day_examples.append(value.example)
                witnesses.append(ExampleWitness(key, value.seal, outcome_entry['ack_event_digest'],
                                                aware(outcome_entry['acknowledged_at']), late))
                day_late += int(late)
            pairs.extend(day_pairs); examples.extend(day_examples)
            coverage.append(SessionCoverage(day, status, count - 1, max(0, count - 10),
                len(day_pairs), len(day_examples), sum(int(p[2].late) for p in day_pairs), day_late,
                pending, quarantined, max(0, count - 10) - len(day_examples)))
        selected.append((symbol, pairs, examples, witnesses, tuple(coverage)))
    return selected, calendars_ok, last_close


def read_training_snapshot(raw_book, feature_journal, outcome_journal, *, timezone_file, clock):
    """Derive training context now, preserving gaps and actual availability.

    No source writes. Any concurrent source change invalidates the whole read;
    callers may try a later cycle, never relabel this one as successful.
    """
    try:
        guard = TrainingClock(timezone_file, clock)
        started = guard()  # Must precede any database operation.
        _need(type(raw_book) is receipt_book.ReceiptBook
              and type(feature_journal) is feature_pipeline.FeatureJournal
              and type(outcome_journal) is outcomes.OutcomeJournal)
        source = source_sha256()
        paths = (raw_book.path, feature_journal.path, outcome_journal.path)
        identities = tuple(_identity(p) for p in paths)
        with _watch_versions((paths[0], paths[2]), guard) as unchanged, \
                VerifiedFeatureView(feature_journal, read_guard=guard) as view:
            feature = view.snapshot(deep=True)
            guard(); common = outcome_journal.snapshot()
            feature_heads = _heads(feature_journal.path, guard)
            guard()
            with raw_book._transaction() as (db, raw, _):
                raw_heads = dict(db.execute('SELECT seq,digest FROM events'))
                _binding(raw, feature, common, paths, identities, raw_heads, feature_heads)
                for entry in feature['entries'].values():
                    context = entry['raw_context']
                    _need(raw_heads.get(context['sequence']) == context['digest'])
                selected, calendars_ok, last_close = _rows(raw, feature, common, guard=guard)
                bindings = tuple(SourceBinding(kind, str(path), ident[0], ident[1], state['head_seq'],
                    state['head_digest'], plan_hash, source_hash) for kind, path, ident, state, plan_hash, source_hash in (
                        ('raw', paths[0], identities[0], raw, feature['plan'].raw_plan_sha256, receipt_book._source()),
                        ('feature', paths[1], identities[1], feature, digest(asdict(feature['plan'])), feature['source_sha256']),
                        ('outcome', paths[2], identities[2], common, digest(common['binding']), outcomes._source())))
            view.snapshot()
            _need(tuple(_identity(p) for p in paths) == identities and unchanged()
                  and source_sha256() == source,
                  'training_sources_changed_during_read')
            available = guard()
            _need(available >= aware(raw['last_at']) and available >= aware(feature['last_at'])
                  and available >= aware(common['last_at']))
        symbols = tuple(SymbolTraining(symbol, (symbol, 'alpaca', 'iex', 'raw', 900, 'USD', 'shares'),
            tuple(regime_baselines.ReturnObservation(end, available, value) for end, value, _ in pairs),
            tuple(w for _, _, w in pairs), tuple(examples), tuple(witnesses), coverage)
            for symbol, pairs, examples, witnesses, coverage in selected)
        ready = bool(calendars_ok and available >= last_close + timedelta(seconds=3720)
                     and not any(c.pending_examples for s in symbols for c in s.coverage))
        fields = dict(symbols=symbols, bindings=bindings, read_started_at=started, available_at=available,
                      timezone_sha256=TIMEZONE_SHA256, ready=ready, source_sha256=source)
        return TrainingSnapshot(**fields, seal=digest({k: [asdict(v) for v in value]
            if k in ('symbols', 'bindings') else value for k, value in fields.items()}))
    except TrainingDataError:
        raise
    except Exception:
        raise TrainingDataError('training_evidence_unavailable') from None
