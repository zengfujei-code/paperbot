"""Accounting-only native cleanup preparation and durable owned allocations.

No broker I/O or production entry point. The future coordinator still needs a
global lock, decision-before-admission, control lease, final source checks and a
dedicated cleanup sender. A prepared view is evidence, not an order capability.
"""
from dataclasses import dataclass, field, replace
from datetime import datetime

from .accountbook import AccountBook
from .data import utc_time
from .orders import (OrderLedger, LedgerError, ACTIVE, _base, _hash, _json, _normalize, _ts, _code_hash)
from .risk import OrderIntent, RiskLimits, RiskDecision, _identity
from .risk_view import PreparedRiskView, RiskViewError, _prepare_risk_view
from .risk_view import _version as view_version

_RISK_TOKEN = object()
_CLOSED = frozenset(('filled', 'canceled', 'expired', 'rejected'))
_WARNINGS = frozenset(('native_protection_unprotected_position',
    'native_protection_unprotected_partial_entry', 'native_protection_out_of_session_exposure'))
_OWNERSHIP = frozenset(('schema', 'family_intent_id', 'parent_broker_order_id', 'child_broker_order_id',
    'source_group_sha256', 'source_orders_sha256', 'allocated_at', 'allocated_quantity',
    'parent_terminal_at', 'child_terminal_at'))


@dataclass(frozen=True)
class GroupShares:
    family_id: str = field(repr=False)
    symbol: str
    remaining_quantity: int
    reserved_quantity: int
    cleanup_filled_quantity: int
    parent_terminal: bool
    child_terminal: bool
    settled: bool
    observed_at: datetime | None
    repair_filled_quantity: int=0

    @property
    def available_quantity(self): return self.remaining_quantity - self.reserved_quantity


def project_groups(records, *, exclude=None):
    """Derive group shares from verified ledger records, never mark-price netting."""
    if type(records) not in (list, tuple) or len(records) > 10000:
        raise ValueError('bounded verified order records required')
    families = {r['intent_id']: r for r in records if r.get('native_stop') is not None}
    if len({r['intent_id'] for r in records}) != len(records):
        raise ValueError('duplicate ledger identity')
    exits = {identity: [] for identity in families}
    for record in records:
        owner = record.get('native_cleanup')
        if owner is None: continue
        if type(owner) is not dict or set(owner) != _OWNERSHIP or owner['schema'] != 1:
            raise ValueError('cleanup ownership schema')
        family = families.get(owner['family_intent_id'])
        if family is None: raise ValueError('cleanup family missing')
        native = family['native_stop']; parent, child = native['parent'], native['stop']
        intent = OrderIntent(**record['intent'])
        for name in ('family_intent_id', 'source_group_sha256', 'source_orders_sha256'): _identity(owner[name])
        allocated = utc_time(owner['allocated_at'])
        if (record.get('native_stop') is not None or parent is None or child is None
                or owner['parent_broker_order_id'] != parent['broker_order_id']
                or owner['child_broker_order_id'] != child['broker_order_id']
                or intent.side != 'sell' or intent.stop_price_usd is not None
                or intent.symbol != family['intent']['symbol']
                or type(owner['allocated_quantity']) is not int or owner['allocated_quantity'] != intent.quantity
                or allocated != utc_time(record['created_at'])
                or not max(utc_time(owner['parent_terminal_at']), utc_time(owner['child_terminal_at'])) <= allocated
                or parent['status'] not in _CLOSED or child['status'] not in _CLOSED
                or utc_time(parent['updated_at']) < utc_time(owner['parent_terminal_at'])
                or utc_time(child['updated_at']) < utc_time(owner['child_terminal_at'])):
            raise ValueError('cleanup ownership or terminal source mismatch')
        filled = record['filled_quantity']
        if type(filled) is not int or not 0 <= filled <= intent.quantity:
            raise ValueError('cleanup fill quantity invalid')
        if filled and (record['broker_order_id'] is None or record['broker_created_at'] is None
                       or utc_time(record['broker_created_at']) < allocated):
            raise ValueError('cleanup fill chronology unavailable')
        exits[family['intent_id']].append(record)
    result = {}
    from .repair_orders import repair_rows,filled_quantity as repair_fills,reserved_quantity as repair_reserves,CLOSED as REPAIR_CLOSED
    for identity, record in families.items():
        native = record['native_stop']; parent, child = native['parent'], native['stop']
        rows = exits[identity]
        cleanup = sum(r['filled_quantity'] for r in rows)
        repairs=repair_rows(record);repaired=sum(repair_fills(r) for r in repairs)
        reserved = sum(r['intent']['quantity'] - r['filled_quantity'] for r in rows
                       if r['state'] in ACTIVE and r['intent_id'] != exclude)
        reserved+=sum(repair_reserves(r) for r in repairs)
        remaining = (parent['filled_quantity'] if parent else 0) - (child['filled_quantity'] if child else 0) - cleanup - repaired
        if remaining < 0 or reserved > remaining:
            raise ValueError('cleanup oversubscribed family')
        parent_terminal = parent is not None and parent['status'] in _CLOSED
        child_terminal = child is not None and native['child_visible'] is True and child['status'] in _CLOSED
        settled = bool((cleanup or repaired) and remaining == 0 and parent_terminal and child_terminal
                       and not any(r['state'] in ACTIVE for r in rows)
                       and all(r['state'] in ('retired','aborted_unsent') or r['state'] in REPAIR_CLOSED and r['lookup_state']=='observed' for r in repairs))
        result[identity] = GroupShares(identity, record['intent']['symbol'], remaining, reserved, cleanup,
            parent_terminal, child_terminal, settled,
            utc_time(native['observed_at']) if native['observed_at'] is not None else None,repaired)
    return result


def cleanup_projection(records, family_id):
    """Bound active/target ownership evidence, excluding old unrelated contexts."""
    groups=project_groups(records)
    if family_id not in groups:raise ValueError('cleanup family absent')
    selected={family_id}|{key for key,g in groups.items()
        if g.remaining_quantity or g.reserved_quantity or not g.parent_terminal}
    fields=('intent_id','intent','state','filled_quantity','filled_notional_usd','broker_order_id',
        'broker_created_at','broker_updated_at','submission_claimed_at','created_at','updated_at')
    result=[]
    for record in records:
        owner=record.get('native_cleanup')
        if (record['intent_id'] in selected or record['state'] in ACTIVE
                or owner is not None and owner['family_intent_id'] in selected):
            row={key:record[key] for key in fields}
            for key in ('native_stop','native_cleanup'):
                if record.get(key) is not None:row[key]=record[key]
            result.append(row)
    return result


def validate_accounting(book, records, cycle, calendar, now):
    """Permit only independently rederived protection warnings, never other faults."""
    from .native_reconciler import _assess_records
    from .reconciler import ReconciliationCycle
    from .risk_baselines import _account_snapshot
    if type(book) is not AccountBook or type(cycle) is not ReconciliationCycle:
        raise ValueError('exact native accounting evidence required')
    if (_hash(_json(records)) != cycle.orders_sha256 or cycle.orders_count != len(records)
            or cycle.status not in ('reconciled', 'vetoed') or type(cycle.findings) is not tuple):
        raise ValueError('native accounting binding mismatch')
    states, warnings = _assess_records(records, calendar, now)
    if (not warnings <= _WARNINGS or tuple(sorted(warnings)) != cycle.findings
            or tuple(states) != cycle.native_states
            or cycle.ready_for_risk is not (not warnings)
            or cycle.status != ('vetoed' if warnings else 'reconciled')):
        raise ValueError('non-accounting-only reconciliation findings')
    snapshot = _account_snapshot(book, cycle, now)
    held = {p.symbol: p.quantity for p in snapshot.account.positions}
    owned = {}
    for group in project_groups(records).values():
        if group.remaining_quantity:
            owned[group.symbol] = owned.get(group.symbol, 0) + group.remaining_quantity
    if held != owned:
        raise ValueError('unallocated or mismatched family positions')
    return snapshot


def _reservations(records, cycle, calendar, now):
    from .native_reconciler import _group, _settled
    from .protection import assess_protection
    from .risk import StopReservation
    from .repair_orders import repair_assessment,active_repair
    groups = project_groups(records); reservations, times = [], []
    for record in records:
        native = record.get('native_stop')
        if native is None: continue
        group = groups[record['intent_id']]
        if _settled(native) or group.settled: continue
        if native['parent'] is None:
            if record['submission_claimed_at'] is not None: raise ValueError('unknown native family')
            continue
        if group.observed_at is None or not 0 <= (now - group.observed_at).total_seconds() <= 30:
            raise ValueError('stale cleanup group')
        times.append(group.observed_at)
        state = repair_assessment(record,group,calendar=calendar,now=now) or assess_protection(_group(record), calendar=calendar, now=now)
        if state.native_stop_active:
            repaired=active_repair(record) if state.status=='repaired_protected' else None
            child = repaired['order'] if repaired is not None else native['stop']
            if repaired is not None:times.append(utc_time(repaired['observed_at']))
            reservations.append(StopReservation(record['intent_id'], _hash(child['broker_order_id']),
                group.symbol, child['quantity'] - child['filled_quantity'], group.observed_at))
        elif not (group.parent_terminal and group.child_terminal) and state.status != 'awaiting_entry':
            raise ValueError('unresolved cleanup protection')
    return tuple(reservations), tuple(times)


@dataclass(frozen=True)
class PreparedCleanupView:
    risk: PreparedRiskView
    family_id: str = field(repr=False)
    remaining_quantity: int
    available_quantity: int

    def public_summary(self):
        return dict(mode='paper', phase='cleanup_prepared', execution_enabled=False,
            entries_paused=True, remaining_quantity=self.remaining_quantity,
            available_quantity=self.available_quantity, findings=list(self.risk.findings))


def prepare_cleanup_risk_view(*, family_id, **inputs):
    try:
        _identity(family_id)
        records = inputs['ledger'].all_orders(); group = project_groups(records)[family_id]
        now = utc_time(inputs['now'])
        if (not group.parent_terminal or not group.child_terminal or group.remaining_quantity <= 0
                or group.observed_at is None or not 0 <= (now - group.observed_at).total_seconds() <= 30):
            raise ValueError('cleanup needs terminal fresh family')
        view = _prepare_risk_view(**inputs, symbol=group.symbol, reservation_builder=_reservations,
            accounting_validator=validate_accounting)
        if view.orders_digest != _hash(_json(records)): raise ValueError('cleanup source changed')
        return PreparedCleanupView(view, family_id, group.remaining_quantity, group.available_quantity)
    except Exception: raise RiskViewError('native_cleanup_risk_unavailable') from None


def _records(ledger, db):
    ledger._verify(db)
    return [ledger._load(db, r[0]) for r in db.execute('SELECT intent_id FROM orders ORDER BY intent_id')]


def _check_prepared(records, prepared, now, *, exclude=None):
    if type(prepared) is not PreparedCleanupView or type(prepared.risk) is not PreparedRiskView:
        raise ValueError('exact cleanup view required')
    view = prepared.risk; group = project_groups(records, exclude=exclude)[prepared.family_id]
    if (view.code_sha256 != view_version() or view.orders_digest != _hash(_json(records)) or not view.state.pause_entries
            or not view.portfolio.reconciled or not 0 <= (now - view.portfolio.as_of).total_seconds() <= 30
            or not group.parent_terminal or not group.child_terminal or group.remaining_quantity <= 0
            or group.observed_at is None or not 0 <= (now - group.observed_at).total_seconds() <= 30
            or group.remaining_quantity != prepared.remaining_quantity):
        raise ValueError('cleanup source or scope changed')
    return group


def _inputs(view, limits, calendar, now):
    if type(limits) is not RiskLimits: raise ValueError('exact cleanup limits required')
    return dict(portfolio=view.portfolio, quote=view.quote, state=view.state, limits=limits, calendar=calendar, now=now)


def _risk(ledger, db, intent, prepared, group, limits, calendar, now, *, exclude=None):
    decision, context = ledger._risk(db, intent, **_inputs(prepared.risk, limits, calendar, now),
        exclude=exclude, _native_cleanup=_RISK_TOKEN)
    if intent.quantity > group.available_quantity:
        decision = replace(decision, allowed=False,
            violations=tuple(dict.fromkeys((*decision.violations, 'native_cleanup_family_capacity'))))
    context['cleanup'] = dict(family_intent_id=prepared.family_id,
        remaining_quantity=group.remaining_quantity, available_quantity=group.available_quantity,
        protection_findings=list(prepared.risk.findings), orders_sha256=prepared.risk.orders_digest)
    return decision, context


def _decision_link(ledger,intent,prepared,decision,journal,limits,now):
    from .decision_journal import DecisionReceipt,DecisionJournal,_portfolio
    if type(decision) is not DecisionReceipt or type(journal) is not DecisionJournal:
        raise ValueError('committed cleanup decision required')
    actual=journal.replay(decision.sequence,ledger=ledger)
    if actual!=decision:raise ValueError('cleanup decision receipt mismatch')
    record=actual.read();inputs=record['inputs']
    if (record['outcome']['action']!='sell' or record['outcome']['vetoes'] or inputs['basis']!='native_cleanup'
            or inputs['cleanup']['intent_present'] or inputs['cleanup']['family_id']!=prepared.family_id
            or inputs['intent']!=_normalize(_base(intent,OrderIntent))
            or inputs['prepared']['orders_digest']!=prepared.risk.orders_digest
            or inputs['risk']['portfolio']!=_normalize(_portfolio(prepared.risk.portfolio))
            or inputs['limits']!=_normalize(_base(limits,RiskLimits)) or utc_time(record['recorded_at'])>now):
        raise ValueError('cleanup decision does not authorize this allocation')
    return dict(sequence=decision.sequence,digest=decision.digest)


def verify_cleanup_decision(ledger,record,journal):
    from .decision_journal import DecisionJournal
    if type(journal) is not DecisionJournal:raise ValueError('exact decision journal required')
    link=record.get('cleanup_decision');owner=record.get('native_cleanup')
    if (type(link) is not dict or set(link)!={'sequence','digest'} or type(link['sequence']) is not int
            or owner is None):raise ValueError('recorded cleanup admission required')
    receipt=journal.replay(link['sequence'],ledger=ledger);decision=receipt.read();inputs=decision['inputs']
    if (receipt.digest!=link['digest'] or decision['outcome']['action']!='sell' or decision['outcome']['vetoes']
            or inputs['basis']!='native_cleanup' or inputs['cleanup']['intent_present']
            or inputs['intent']!=record['intent'] or inputs['cleanup']['family_id']!=owner['family_intent_id']
            or inputs['prepared']['orders_digest']!=owner['source_orders_sha256']
            or utc_time(decision['recorded_at'])>utc_time(record['created_at'])):
        raise ValueError('cleanup admission decision link changed')
    return decision


def admit_cleanup(ledger, intent, *, prepared, limits, calendar, now, decision=None, journal=None):
    try:
        now = utc_time(now)
        if type(ledger) is not OrderLedger or type(intent) is not OrderIntent or type(prepared) is not PreparedCleanupView:
            raise ValueError('exact cleanup admission dependencies required')
        if intent.side != 'sell' or intent.stop_price_usd is not None:
            raise ValueError('cleanup must reduce owned long shares')
        from .broker import _tick
        _tick(intent.limit_price_usd)
        link=None
        if decision is not None or journal is not None:
            link=_decision_link(ledger,intent,prepared,decision,journal,limits,now)
        with ledger._db(True) as db:
            records = _records(ledger, db)
            existing = ledger._load(db, intent.intent_id, missing=True)
            serialized = _normalize(_base(intent, OrderIntent))
            if existing is not None:
                if (existing['intent'] != serialized or existing.get('native_cleanup', {}).get('family_intent_id') != prepared.family_id):
                    raise ValueError('cleanup idempotency conflict')
                if link is not None and existing.get('cleanup_decision')!=link:raise ValueError('cleanup decision cannot attach after admission')
                return existing
            group = _check_prepared(records, prepared, now)
            if intent.symbol != group.symbol: raise ValueError('cleanup symbol mismatch')
            decision, context = _risk(ledger, db, intent, prepared, group, limits, calendar, now)
            family = next(r for r in records if r['intent_id'] == prepared.family_id); native = family['native_stop']
            owner = dict(schema=1, family_intent_id=prepared.family_id,
                parent_broker_order_id=native['parent']['broker_order_id'], child_broker_order_id=native['stop']['broker_order_id'],
                source_group_sha256=_hash(_json(native)), source_orders_sha256=prepared.risk.orders_digest,
                allocated_at=_ts(now), allocated_quantity=intent.quantity,
                parent_terminal_at=native['parent']['updated_at'], child_terminal_at=native['stop']['updated_at'])
            record = dict(schema=1, intent_id=intent.intent_id, client_order_id='pb3-' + intent.intent_id[:40],
                intent=serialized, state='reserved' if decision.allowed else 'rejected_by_risk',
                risk=_base(decision, RiskDecision), filled_quantity=0, filled_notional_usd='0',
                broker_order_id=None, broker_created_at=None, broker_updated_at=None, submission_claimed_at=None,
                cancel_requested_at=None, created_at=_ts(now), updated_at=_ts(now), context=context,
                code_sha256=_code_hash(), native_cleanup=owner)
            if link is not None:record['cleanup_decision']=link
            return ledger._persist(db, record, 'native_cleanup_admission', now)
    except LedgerError: raise
    except Exception: raise LedgerError('native_cleanup_admission_failed') from None


@dataclass(frozen=True)
class CleanupDispatch:
    payload: str=field(repr=False)
    orders_sha256: str=field(repr=False)

    def read(self):
        from .alpaca import decode_json
        return decode_json(self.payload)


def prepare_cleanup_submission(ledger, identity, *, prepared, limits, calendar, now):
    return _prepare_cleanup_submission(ledger,identity,prepared=prepared,limits=limits,calendar=calendar,now=now)


def prepare_cleanup_dispatch(ledger, identity, *, prepared, limits, calendar, now):
    return _prepare_cleanup_submission(ledger,identity,prepared=prepared,limits=limits,calendar=calendar,now=now,dispatch=True)


def _prepare_cleanup_submission(ledger, identity, *, prepared, limits, calendar, now, dispatch=False):
    """One durable unknown claim, pinned to the future dedicated cleanup sender."""
    try:
        now = utc_time(now); _identity(identity)
        if type(ledger) is not OrderLedger: raise ValueError('exact cleanup ledger required')
        with ledger._db(True) as db:
            records = _records(ledger, db); record = ledger._load(db, identity)
            if record['state'] != 'reserved': return None
            owner = record.get('native_cleanup')
            if owner is None or owner['family_intent_id'] != prepared.family_id:
                raise ValueError('cleanup ownership required')
            group = _check_prepared(records, prepared, now, exclude=identity)
            intent = OrderIntent(**record['intent'])
            decision, context = _risk(ledger, db, intent, prepared, group, limits, calendar, now, exclude=identity)
            record.update(state='submission_unknown' if decision.allowed else 'rejected_by_risk',
                risk=_base(decision, RiskDecision), context=context, updated_at=_ts(now),
                submission_claimed_at=_ts(now) if decision.allowed else None)
            saved = ledger._persist(db, record, 'native_cleanup_claim' if decision.allowed else 'native_cleanup_veto', now)
            if decision.allowed and dispatch:
                return CleanupDispatch(_json(saved),_hash(_json(_records(ledger,db))))
            return saved if decision.allowed else None
    except LedgerError: raise
    except Exception: raise LedgerError('native_cleanup_claim_failed') from None


def retire_cleanup_reservation(ledger, identity, *, expected_sha256, reason, now):
    """Release only this unchanged, never-claimed allocation; no broker reads.

    Expiry never resolves an unknown submission. A coordinator may explicitly
    abort a known-unsent reservation when a new view is unavailable; the full
    ownership/risk history and reason remain in the same durable event chain.
    """
    try:
        now = utc_time(now); _identity(identity); _identity(expected_sha256)
        if type(ledger) is not OrderLedger or reason not in ('expired', 'pre_dispatch_abort'):
            raise ValueError('exact unsent cleanup retirement required')
        with ledger._db(True) as db:
            ledger._verify(db); record = ledger._load(db, identity)
            if record.get('native_cleanup') is None: raise ValueError('owned cleanup reservation required')
            if (record['state'] != 'reserved' or record['submission_claimed_at'] is not None
                    or record['broker_order_id'] is not None or record['filled_quantity'] != 0):
                return None
            if (_hash(_json(record)) != expected_sha256 or now < utc_time(record['updated_at'])
                    or reason == 'expired' and now < utc_time(record['intent']['expires_at'])):
                raise ValueError('cleanup retirement source or clock changed')
            finding = 'native_cleanup_expired_unsubmitted' if reason == 'expired' else 'native_cleanup_aborted_unsubmitted'
            record['risk'].update(allowed=False,
                violations=list(dict.fromkeys((*record['risk']['violations'], finding))))
            record['context']['cleanup_retired'] = dict(reason=reason, at=_ts(now), previous_sha256=expected_sha256)
            record.update(state='expired' if reason == 'expired' else 'rejected_by_risk', updated_at=_ts(now))
            return ledger._persist(db, record, 'native_cleanup_retired', now)
    except LedgerError: raise
    except Exception: raise LedgerError('native_cleanup_retirement_failed') from None
