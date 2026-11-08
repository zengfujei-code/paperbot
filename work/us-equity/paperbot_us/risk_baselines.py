"""Durable observed-equity risk references. No network, order or model methods.

Risk readiness here means the daily reference is known, never permission to
trade. All account snapshots must replay, all skipped book observations are
processed (bounded at 100), and a final coordinator must recheck source age.
Near-close broker observations are not exact consolidated closing valuations.
"""
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from .accountbook import AccountBook, _json
from .alpaca import decode_json
from .data import Calendar, Session, utc_time
from .monitor import ObservationArchive
from .reconciler import ReconciliationCycle
from .risk import _identity, _money
from .runner import _exclusive


class BaselineError(RuntimeError):
    pass


def _version():
    digest = hashlib.sha256()
    for name in ('risk_baselines.py', 'accountbook.py', 'accounting.py', 'broker.py',
                 'reconciler.py', 'orders.py', 'risk.py', 'data.py', 'monitor.py', 'runner.py',
                 'native_cleanup.py', 'native_reconciler.py', 'native_orders.py', 'protection.py', 'repair_orders.py','repair_decisions.py','repair_execution.py','compensation.py','native_admission.py','native_execution.py'):
        digest.update(name.encode() + b'\0' + Path(__file__).with_name(name).read_bytes() + b'\0')
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _calendar(value):
    if type(value) is not Calendar or len(value.sessions) > 100:
        raise ValueError('bounded exact calendar required')
    if any(type(s) is not Session for s in value.sessions.values()):
        raise ValueError('exact sessions required')
    return dict(start=value.start.isoformat(), end=value.end.isoformat(), sessions=[
        dict(day=s.day.isoformat(), opens=s.opens.isoformat(), closes=s.closes.isoformat())
        for s in sorted(value.sessions.values(), key=lambda s: s.day)])


def _parse_calendar(raw):
    return Calendar([Session(date.fromisoformat(s['day']), s['opens'], s['closes']) for s in raw['sessions']],
                    start=raw['start'], end=raw['end'])


def _session_record(session):
    return dict(day=session.day.isoformat(), opens=session.opens.isoformat(), closes=session.closes.isoformat())


def _checkpoint_matches(checkpoint, session):
    if session is None or checkpoint['session'] != _session_record(session):
        return False
    at = utc_time(checkpoint['observed_at'])
    return session.opens <= at <= session.closes and 0 <= (session.closes - at).total_seconds() <= 30


def _source(snapshot, order_count):
    result = snapshot.receipt.result
    return dict(sequence=snapshot.receipt.sequence, digest=snapshot.receipt.digest,
                anchor_sha256=_digest(snapshot.anchor), account_sha256=snapshot.account.account_sha256,
                book_at=snapshot.receipt.as_of.isoformat(), until=snapshot.activity_until.isoformat(),
                account_at=snapshot.account.received_at.isoformat(),
                reconciled=result.reconciled, equity_usd=str(snapshot.account.equity_usd),
                cash_usd=str(snapshot.account.cash_usd), positions_count=len(snapshot.account.positions),
                activity_count=snapshot.activity_count, current_order_count=order_count)


def _validate_source(source):
    for name in ('digest', 'anchor_sha256', 'account_sha256'):
        _identity(source[name])
    for name in ('sequence', 'positions_count', 'activity_count', 'current_order_count'):
        if type(source[name]) is not int or not (1 if name == 'sequence' else 0) <= source[name] <= 100000:
            raise ValueError('source count invalid')
    for name in ('equity_usd', 'cash_usd'):
        _money(source[name])
    if type(source['reconciled']) is not bool:
        raise ValueError('explicit reconciliation required')
    at, until, account_at = (utc_time(source[n]) for n in ('book_at', 'until', 'account_at'))
    if not until <= at or not account_at <= at:
        raise ValueError('source times invalid')
    return at


def _initial(inputs):
    source = inputs['source']
    _validate_source(source)
    if (not source['reconciled'] or any(source[n] for n in ('positions_count', 'activity_count', 'current_order_count'))
            or _money(source['cash_usd']) != 10000 or _money(source['equity_usd']) != 10000):
        raise ValueError('verified pristine cash baseline required')
    return dict(anchor_sha256=source['anchor_sha256'], account_sha256=source['account_sha256'],
                high_water_equity_usd='10000', day_start_equity_usd='10000', session_day=None,
                reference_kind='unavailable', reference_source_day=None, reference_checkpoint=None,
                reference_conflict=False, entries_ready=False,
                findings=['session_reference_uninitialized'], ever_exposed=False, checkpoint=None,
                book_sequence=source['sequence'], book_digest=source['digest'],
                source_as_of=source['until'], book_at=source['book_at'])


def _advance(state, inputs):
    source = inputs['source']
    at = _validate_source(source)
    if (source['anchor_sha256'] != state['anchor_sha256'] or source['account_sha256'] != state['account_sha256']
            or source['sequence'] not in (state['book_sequence'], state['book_sequence'] + 1)
            or (source['sequence'] == state['book_sequence'] and source['digest'] != state['book_digest'])
            or at < utc_time(state['book_at']) or utc_time(source['until']) < utc_time(state['source_as_of'])):
        raise ValueError('risk source binding, continuity or time mismatch')
    calendar = _parse_calendar(inputs['calendar'])
    day = at.astimezone(ZoneInfo('America/New_York')).date()
    if not calendar.start <= day <= calendar.end:
        raise ValueError('risk calendar coverage unavailable')
    value = dict(state)
    value.update(book_sequence=source['sequence'], book_digest=source['digest'],
                 source_as_of=source['until'], book_at=source['book_at'])
    value['ever_exposed'] = state['ever_exposed'] or any(source[n] for n in
                                                       ('activity_count', 'positions_count', 'current_order_count'))
    if value['session_day'] != day.isoformat():
        value.update(session_day=day.isoformat(), reference_kind='unavailable', reference_source_day=None,
                     reference_checkpoint=None, reference_conflict=False)
    checkpoint = value['checkpoint']
    if checkpoint is not None:
        checkpoint_day = date.fromisoformat(checkpoint['day'])
        if (calendar.start <= checkpoint_day <= calendar.end
                and not _checkpoint_matches(checkpoint, calendar.sessions.get(checkpoint_day))):
            value['checkpoint'] = {**checkpoint, 'disputed': True}
    if not source['reconciled']:
        value.update(entries_ready=False, findings=['account_unreconciled'])
        return value
    equity = _money(source['equity_usd'])
    value['high_water_equity_usd'] = str(max(_money(state['high_water_equity_usd']), equity, Decimal(10000)))
    session = calendar.sessions.get(day)
    prior = [s for s in calendar.sessions.values() if s.day < day]
    previous_session = max(prior, key=lambda s: s.day) if prior else None
    if session is not None and value['reference_kind'] == 'unavailable':
        if (not value['ever_exposed'] and equity == _money(source['cash_usd']) == 10000):
            value.update(day_start_equity_usd='10000', reference_kind='pristine_cash', reference_source_day=None)
        else:
            checkpoint = value['checkpoint']
            if previous_session is not None and checkpoint is not None and checkpoint['day'] == previous_session.day.isoformat():
                if not checkpoint['disputed'] and _checkpoint_matches(checkpoint, previous_session):
                    value.update(day_start_equity_usd=checkpoint['equity_usd'], reference_kind='previous_observed_close',
                                 reference_source_day=checkpoint['day'], reference_checkpoint=checkpoint)
                else:
                    value['reference_conflict'] = True
    findings = []
    if session is None:
        findings.append('non_trading_day')
    elif value['reference_kind'] == 'unavailable':
        findings.append('previous_close_reference_missing')
    if value['reference_kind'] == 'previous_observed_close':
        frozen = value['reference_checkpoint']
        reference_day = date.fromisoformat(frozen['day'])
        if not calendar.start <= reference_day <= calendar.end:
            findings.append('reference_calendar_unavailable')
        elif (previous_session is None or previous_session.day != reference_day
              or not _checkpoint_matches(frozen, calendar.sessions.get(reference_day))):
            value['reference_conflict'] = True
    if value['reference_conflict']:
        findings.append('calendar_reference_conflict')
    value.update(entries_ready=not findings, findings=findings)
    account_at = utc_time(source['account_at'])
    if session is not None and session.opens <= account_at <= session.closes:
        seconds_to_close = (session.closes - account_at).total_seconds()
        checkpoint = value['checkpoint']
        if (0 <= seconds_to_close <= 30 and (checkpoint is None
                or (account_at > utc_time(checkpoint['observed_at'])
                    and not (checkpoint['disputed'] and checkpoint['day'] == day.isoformat())))):
            value['checkpoint'] = dict(day=day.isoformat(), equity_usd=str(equity), observed_at=account_at.isoformat(),
                                       session=_session_record(session), disputed=False)
    binding = inputs.get('accounting_only')
    if binding is not None:
        from .native_cleanup import _WARNINGS
        if (type(binding) is not dict or set(binding) != {'orders_sha256', 'audit_sha256', 'protection_findings'}
                or type(binding['protection_findings']) is not list
                or binding['protection_findings'] != sorted(set(binding['protection_findings']))
                or not set(binding['protection_findings']) <= _WARNINGS):
            raise ValueError('native accounting-only binding invalid')
        _identity(binding['orders_sha256']); _identity(binding['audit_sha256'])
        value['entries_ready'] = False
        value['findings'] = sorted(set(value['findings']) | set(binding['protection_findings']) | {'native_cleanup_only'})
    return value


def _account_snapshot(book, cycle, now):
    """Account binding shared by strict readiness and explicit cleanup scope."""
    if type(book) is not AccountBook or type(cycle) is not ReconciliationCycle:
        raise ValueError('exact accounting cycle required')
    if (cycle.book_receipt is None or type(cycle.orders_count) is not int or cycle.orders_count < 0
            or type(cycle.positions_count) is not int or cycle.positions_count < 0):
        raise ValueError('incomplete accounting cycle')
    _identity(cycle.orders_sha256); _identity(cycle.audit_sha256)
    snapshot = book.snapshot()
    if (snapshot.receipt != cycle.book_receipt or not snapshot.receipt.result.reconciled
            or cycle.positions_count != len(snapshot.account.positions)
            or not snapshot.receipt.as_of <= cycle.as_of <= now
            or not 0 <= (now - snapshot.activity_until).total_seconds() <= 30
            or snapshot.receipt.as_of.astimezone(ZoneInfo('America/New_York')).date()
                != now.astimezone(ZoneInfo('America/New_York')).date()):
        raise ValueError('unbound or stale reconciliation')
    return snapshot


@dataclass(frozen=True)
class BaselineReceipt:
    as_of: datetime
    source_as_of: datetime
    session_day: str | None
    day_start_equity_usd: Decimal
    high_water_equity_usd: Decimal
    reference_kind: str
    entries_ready: bool
    findings: tuple
    book_sequence: int
    book_digest: str
    audit_sha256: str

    def public_summary(self):
        return dict(mode='paper', phase='risk_baselines', generated_at=self.as_of.timestamp(),
                    execution_enabled=False, entries_ready=self.entries_ready,
                    session_day=self.session_day, reference_kind=self.reference_kind,
                    findings=list(self.findings))


def _receipt(record, state, digest):
    return BaselineReceipt(datetime.fromtimestamp(record['generated_at'], timezone.utc),
                           utc_time(state['source_as_of']), state['session_day'],
                           _money(state['day_start_equity_usd']), _money(state['high_water_equity_usd']),
                           state['reference_kind'], state['entries_ready'], tuple(state['findings']),
                           state['book_sequence'], state['book_digest'], digest)


class RiskBaselineStore:
    def __init__(self, path):
        self.path = Path(path).absolute()
        self.archive = ObservationArchive(self.path)

    def _load(self):
        self.archive.verify()
        state = latest = digest = None
        version = _version()
        with closing(sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)) as db:
            for payload, saved_digest in db.execute('SELECT payload,digest FROM observations ORDER BY seq'):
                row = decode_json(payload)
                if row['schema'] != 1 or row['code_sha256'] != version:
                    raise ValueError('risk source version mismatch')
                if latest and row['generated_at'] < latest['generated_at']:
                    raise ValueError('risk history clock regression')
                if state is None and row['kind'] == 'risk_initialize':
                    computed = _initial(row['inputs'])
                elif state is not None and row['kind'] == 'risk_observation':
                    computed = _advance(state, row['inputs'])
                else:
                    raise ValueError('risk record lifecycle mismatch')
                if computed != row['state']:
                    raise ValueError('risk replay mismatch')
                latest, state, digest = row, computed, saved_digest
        if latest is None:
            raise ValueError('risk baseline uninitialized')
        return state, latest, digest

    def _current(self, book, cycle, now):
        if type(book) is not AccountBook or type(cycle) is not ReconciliationCycle:
            raise ValueError('exact accounting cycle required')
        if (cycle.ready_for_risk is not True or cycle.status != 'reconciled' or cycle.findings
                or cycle.book_receipt is None or type(cycle.orders_count) is not int or cycle.orders_count < 0
                or type(cycle.positions_count) is not int or cycle.positions_count < 0):
            raise ValueError('unready reconciliation')
        return _account_snapshot(book, cycle, now)

    def initialize(self, *, book, cycle, now):
        try:
            now = utc_time(now)
            with _exclusive(self.path):
                if self.path.exists() or self.path.is_symlink():
                    raise ValueError('risk state already exists')
                snapshot = self._current(book, cycle, now)
                inputs = dict(source=_source(snapshot, cycle.orders_count))
                state = _initial(inputs)
                self.archive.append(dict(kind='risk_initialize', schema=1, generated_at=now.timestamp(),
                                         code_sha256=_version(), inputs=inputs, state=state))
                if book.snapshot().receipt != snapshot.receipt:
                    raise ValueError('account changed during risk initialization')
        except Exception:
            raise BaselineError('risk_baseline_initialization_failed') from None

    def observe(self, *, book, cycle, calendar, now):
        return self._observe(book=book, cycle=cycle, calendar=calendar, now=now)

    def observe_accounting(self, *, book, ledger, cycle, calendar, now):
        return self._observe(book=book, cycle=cycle, calendar=calendar, now=now, ledger=ledger)

    def _observe(self, *, book, cycle, calendar, now, ledger=None):
        try:
            now = utc_time(now)
            with _exclusive(self.path):
                state, previous, digest = self._load()
                if now.timestamp() < previous['generated_at']:
                    raise ValueError('risk clock regression')
                binding = None
                if ledger is None:
                    current = self._current(book, cycle, now)
                else:
                    from .native_cleanup import validate_accounting
                    from .orders import OrderLedger
                    if type(ledger) is not OrderLedger: raise ValueError('exact cleanup ledger required')
                    current = validate_accounting(book, ledger.all_orders(), cycle, calendar, now)
                    binding = dict(orders_sha256=cycle.orders_sha256, audit_sha256=cycle.audit_sha256,
                                   protection_findings=list(cycle.findings))
                raw_calendar = _calendar(calendar)
                snapshots = book.snapshots_after(state['book_sequence'])
                if current.receipt.sequence < state['book_sequence']:
                    raise ValueError('book history regression')
                if not snapshots:
                    if (previous['kind'] == 'risk_observation'
                            and previous['inputs']['source'] == _source(current, cycle.orders_count)
                            and previous['inputs']['calendar'] == raw_calendar
                            and previous['inputs'].get('accounting_only') == binding):
                        return _receipt(previous, state, digest)
                    snapshots = (current,)
                pending = []
                version = _version()
                for snapshot in snapshots:
                    inputs = dict(source=_source(snapshot, cycle.orders_count), calendar=raw_calendar,
                                  coordinator_sha256=cycle.audit_sha256, orders_sha256=cycle.orders_sha256,
                                  historical_catch_up=snapshot.receipt.sequence != current.receipt.sequence)
                    if binding is not None: inputs['accounting_only'] = binding
                    state = _advance(state, inputs)
                    row = dict(kind='risk_observation', schema=1, generated_at=now.timestamp(),
                               code_sha256=version, inputs=inputs, state=state)
                    pending.append(row)
                if book.snapshot().receipt != current.receipt:
                    raise ValueError('account changed during risk observation')
                for row in pending:
                    saved = self.archive.append(row)
                    digest = saved['audit_sha256']
                if book.snapshot().receipt != current.receipt:
                    raise ValueError('account changed after risk commit')
                if ledger is not None:
                    validate_accounting(book, ledger.all_orders(), cycle, calendar, now)
                return _receipt(pending[-1], state, digest)
        except Exception:
            raise BaselineError('risk_baseline_unavailable') from None

    def replay(self):
        try:
            state, row, digest = self._load()
            return _receipt(row, state, digest)
        except Exception:
            raise BaselineError('risk_baseline_replay_failed') from None

    def verify(self):
        self.replay()
        return self.archive.verify()
