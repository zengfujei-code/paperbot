"""Atomic family-owned standalone stop reservations and GET recovery.

No submission/cancellation method or production entry point. A claim records
uncertainty; only compatible current broker/account observations prove protection.
The original OTO child is immutable and never reused as a repair identity.
"""
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import timedelta
import re

from .broker import PaperBroker, BrokerError, _tick
from .data import utc_time
from .orders import OrderLedger, LedgerError, _hash, _json, _ts
from .risk import _identity, _price

CLOSED = frozenset(('filled', 'canceled', 'expired', 'rejected'))
_KEYS = frozenset(('schema', 'identity', 'client_order_id', 'quantity', 'stop_price_usd', 'allocated_at',
    'expires_at', 'ownership', 'state', 'submission_claimed_at', 'order', 'lookup_state', 'observed_at',
    'retired_at','retire_reason','lookup_generation','decision_link','dispatch','close','stop_cancel_requested_at'))
_OWNER = frozenset(('family_intent_id', 'parent_broker_order_id', 'original_child_broker_order_id',
    'parent_terminal_at', 'child_terminal_at', 'source_orders_sha256', 'source_book_sha256',
    'source_baseline_sha256', 'limits_sha256', 'calendar_sha256', 'code_sha256'))


def _leg(order):
    from .broker import _uuid
    from .protection import GroupLeg, STATUSES
    from .risk import _quantity
    keys={'broker_order_id','group_id','status','quantity','filled_quantity','average_fill_price_usd',
        'created_at','updated_at','filled_at'}
    if type(order) is not dict or set(order)!=keys:raise LedgerError('repair_leg_schema')
    _uuid(order['broker_order_id']);_quantity(order['quantity'])
    filled=order['filled_quantity'];created=utc_time(order['created_at']);updated=utc_time(order['updated_at'])
    completed=utc_time(order['filled_at']) if order['filled_at'] is not None else None
    if (order['group_id'] is not None or order['status'] not in STATUSES
            or type(filled) is not int or not 0<=filled<=order['quantity'] or created>updated
            or order['status'] in ('held','new','accepted','pending_new','accepted_for_bidding','rejected') and filled!=0
            or order['status']=='partially_filled' and not 0<filled<order['quantity']
            or order['status']=='filled' and (filled!=order['quantity'] or completed is None)
            or completed is not None and (filled!=order['quantity'] or not created<=completed<=updated)
            or not filled and order['average_fill_price_usd'] is not None):
        raise LedgerError('repair_leg_semantics')
    average=_price(order['average_fill_price_usd']) if filled else None
    return GroupLeg(order['broker_order_id'],None,order['status'],order['quantity'],filled,average,created,updated,completed)


def repair_rows(record):
    """Validate immutable allocation/identity structure in verified ledger rows."""
    from .protection import GroupLeg, STATUSES
    native = record.get('native_stop')
    if native is None: return ()
    rows = native.get('repairs', [])
    if type(rows) is not list or len(rows) > 20:
        raise LedgerError('bounded_repair_history_required')
    identities, clients, broker_ids = set(), set(), set()
    for row in rows:
        if type(row) is not dict or set(row) != _KEYS or type(row['schema']) is not int or row['schema'] != 1:
            raise LedgerError('repair_schema_mismatch')
        owner = row['ownership'];parent = native['parent'];child = native['stop']
        if type(owner) is not dict or set(owner) != _OWNER or parent is None or child is None:
            raise LedgerError('repair_family_missing')
        _identity(row['identity'])
        if (not isinstance(row['client_order_id'], str) or not re.fullmatch('pb3r-[0-9a-f]{40}', row['client_order_id'])
                or row['identity'] != _hash('paperbot-repair-owner-v1\0' + row['client_order_id'])
                or row['identity'] in identities or row['client_order_id'] in clients):
            raise LedgerError('repair_identity_mismatch')
        identities.add(row['identity']);clients.add(row['client_order_id'])
        for key in ('family_intent_id', 'source_orders_sha256', 'source_book_sha256', 'source_baseline_sha256',
                    'limits_sha256', 'calendar_sha256', 'code_sha256'): _identity(owner[key])
        at, expiry = utc_time(row['allocated_at']), utc_time(row['expires_at'])
        if (owner['family_intent_id'] != record['intent_id']
                or owner['parent_broker_order_id'] != parent['broker_order_id']
                or owner['original_child_broker_order_id'] != child['broker_order_id']
                or parent['status'] not in CLOSED or child['status'] not in CLOSED
                or not max(utc_time(owner['parent_terminal_at']), utc_time(owner['child_terminal_at'])) <= at < expiry
                or not 0 < (expiry - at).total_seconds() <= 20
                or utc_time(parent['updated_at']) < utc_time(owner['parent_terminal_at'])
                or utc_time(child['updated_at']) < utc_time(owner['child_terminal_at'])
                or type(row['quantity']) is not int or not 1 <= row['quantity'] <= parent['filled_quantity']
                or row['state'] not in STATUSES | {'reserved', 'submission_unknown','retired','aborted_unsent'}
                or row['lookup_state'] not in ('unsubmitted', 'unknown', 'observed')):
            raise LedgerError('repair_allocation_mismatch')
        _tick(_price(row['stop_price_usd']))
        claim = row['submission_claimed_at'];order = row['order'];generation=row['lookup_generation']
        link=row['decision_link'];dispatch=row['dispatch'];aborted=row['state']=='aborted_unsent'
        if link is not None:
            if (type(link) is not dict or set(link)!={'sequence','digest','identity'}
                    or type(link['sequence']) is not int or not 1<=link['sequence']<=100000):
                raise LedgerError('repair_decision_link_invalid')
            _identity(link['digest']);_identity(link['identity'])
        if dispatch is not None:
            keys={'identity','decision_digest','claimed_at','source_orders_sha256','outcome','http_status',
                'observed_at','vetoes','clock_regressed'}
            if (type(dispatch) is not dict or set(dispatch)!=keys or link is None or claim is None
                    or dispatch['claimed_at']!=claim or dispatch['decision_digest']!=link['digest']
                    or dispatch['outcome'] not in ('claimed','not_sent','unknown','acknowledged')
                    or type(dispatch['clock_regressed']) is not bool):raise LedgerError('repair_dispatch_schema')
            for key in ('identity','decision_digest','source_orders_sha256'):_identity(dispatch[key])
            from .decision_journal import _codes
            _codes(dispatch['vetoes']);http=dispatch['http_status']
            if http is not None and (type(http) is not int or not 100<=http<=599):raise LedgerError('repair_http_status_invalid')
            if dispatch['outcome']=='claimed':
                if dispatch['observed_at'] is not None or http is not None or dispatch['vetoes']:raise LedgerError('repair_claim_outcome_invalid')
            else:
                utc_time(dispatch['observed_at'])
                if (dispatch['outcome']=='not_sent' and (not aborted or http is not None or not dispatch['vetoes'])
                        or dispatch['outcome']=='acknowledged' and http!=201
                        or dispatch['outcome']!='not_sent' and dispatch['vetoes']):raise LedgerError('repair_dispatch_outcome_invalid')
        if aborted and (dispatch is None or dispatch['outcome']!='not_sent' or order is not None
                or generation!=0 or row['lookup_state']!='unsubmitted' or row['observed_at'] is not None):
            raise LedgerError('repair_unsent_abort_proof_invalid')
        if type(generation) is not int or not 0<=generation<=1000000:
            raise LedgerError('repair_lookup_generation_invalid')
        if claim is None:
            if row['state'] not in ('reserved','retired') or order is not None or row['lookup_state'] != 'unsubmitted' or row['observed_at'] is not None or generation!=0:
                raise LedgerError('unclaimed_repair_state_mismatch')
        elif not at <= utc_time(claim) < expiry or row['lookup_state']=='unsubmitted' and not aborted:
            raise LedgerError('repair_claim_time_mismatch')
        if row['state']=='retired':
            retired=utc_time(row['retired_at'])
            if (claim is not None or retired<at or row['retire_reason'] not in ('expired','pre_dispatch_abort')
                    or row['retire_reason']=='expired' and retired<expiry):raise LedgerError('repair_retirement_mismatch')
        elif row['retired_at'] is not None or row['retire_reason'] is not None:
            raise LedgerError('unexpected_repair_retirement')
        if order is not None:
            leg = _leg(order)
            if (leg.quantity != row['quantity'] or leg.status != row['state']
                    or leg.created_at < utc_time(claim) or leg.broker_order_id in broker_ids
                    or leg.broker_order_id in (parent['broker_order_id'], child['broker_order_id'])
                    or row['observed_at'] is None or leg.updated_at > utc_time(row['observed_at'])):
                raise LedgerError('repair_order_binding_mismatch')
            broker_ids.add(leg.broker_order_id)
        elif claim is not None and not aborted and (row['state'] != 'submission_unknown' or row['lookup_state']!='unknown'):
            raise LedgerError('repair_observation_missing')
        close=row['close'];cancel=row['stop_cancel_requested_at']
        if close is None:
            if cancel is not None:raise LedgerError('repair_close_claim_missing')
        else:
            if (order is None or type(close) is not dict or close.get('schema')!=1
                    or close.get('target_id')!=order['broker_order_id'] or close.get('outcome') not in ('claimed','not_sent','acknowledged','rejected','unknown')
                    or type(close.get('quantity')) is not int or not 0<close['quantity']<=row['quantity']
                    or utc_time(close['deadline_at'])!=utc_time(close['claimed_at'])+timedelta(seconds=30)
                    or (close['outcome']=='not_sent' and cancel is not None)
                    or (close['outcome']!='not_sent' and cancel!=close['claimed_at'])):
                raise LedgerError('repair_close_binding_invalid')
    return tuple(rows)


def filled_quantity(row):
    return row['order']['filled_quantity'] if row['order'] is not None else 0


def reserved_quantity(row):
    return (0 if row['state'] in ('retired','aborted_unsent') or row['state'] in CLOSED and row['lookup_state'] == 'observed'
        else row['quantity'] - filled_quantity(row))


def _records(ledger, db):
    ledger._verify(db)
    return [ledger._load(db, r[0]) for r in db.execute('SELECT intent_id FROM orders ORDER BY intent_id')]


def _find(record, identity):
    _identity(identity)
    rows = [r for r in repair_rows(record) if r['identity'] == identity]
    if len(rows) != 1: raise LedgerError('repair_owner_missing')
    return rows[0]


@dataclass(frozen=True)
class RepairReservation:
    family_id: str = field(repr=False)
    repair_id: str = field(repr=False)
    client_order_id: str = field(repr=False)
    orders_sha256: str = field(repr=False)

    def public_summary(self):
        return dict(mode='paper', phase='repair_reserved', protection_active=False,
            orders_submitted=0, ready_for_execution=False, execution_enabled=False)


def reserve_repair(*, ledger, plan, book, baselines, calendar, limits, now, decision=None,journal=None):
    from .repair_protection import ProtectionRepairPlan
    from .native_cleanup import project_groups
    try:
        now = utc_time(now)
        if type(ledger) is not OrderLedger or type(plan) is not ProtectionRepairPlan:
            raise ValueError('exact repair reservation required')
        body = plan.request_body(book=book, ledger=ledger, baselines=baselines, calendar=calendar, limits=limits, now=now)
        link=None
        if decision is not None or journal is not None:
            from .repair_decisions import RepairDecisionJournal
            if type(journal) is not RepairDecisionJournal:raise ValueError('committed repair decision journal required')
            link=journal.link_for_reservation(decision,ledger=ledger,plan=plan,body=body,now=now)
        with ledger._db(True) as db:
            records = _records(ledger, db)
            if _hash(_json(records)) != plan.prepared.risk.orders_digest:
                raise ValueError('repair source changed')
            if (book.snapshot().receipt.digest != plan.prepared.risk.book_digest
                    or baselines.replay().audit_sha256 != plan.prepared.risk.baseline_digest):
                raise ValueError('repair account source changed')
            record = ledger._load(db, plan.family_id);native = record['native_stop'];rows = list(repair_rows(record))
            group = project_groups(records)[plan.family_id]
            if group.available_quantity != group.remaining_quantity or group.remaining_quantity != plan.quantity or len(rows) >= 20:
                raise ValueError('repair family capacity changed')
            identity = _hash('paperbot-repair-owner-v1\0' + body['client_order_id'])
            if any(r['client_order_id']==body['client_order_id'] for family in records for r in repair_rows(family)):
                raise ValueError('repair identity reused')
            owner = dict(family_intent_id=plan.family_id, parent_broker_order_id=native['parent']['broker_order_id'],
                original_child_broker_order_id=native['stop']['broker_order_id'],
                parent_terminal_at=native['parent']['updated_at'], child_terminal_at=native['stop']['updated_at'],
                source_orders_sha256=plan.prepared.risk.orders_digest, source_book_sha256=plan.prepared.risk.book_digest,
                source_baseline_sha256=plan.prepared.risk.baseline_digest, limits_sha256=plan.limits_sha256,
                calendar_sha256=_hash(plan.calendar_json), code_sha256=plan.code_sha256)
            rows.append(dict(schema=1, identity=identity, client_order_id=body['client_order_id'], quantity=plan.quantity,
                stop_price_usd=body['stop_price'], allocated_at=_ts(now), expires_at=_ts(plan.expires_at), ownership=owner,
                state='reserved', submission_claimed_at=None, order=None, lookup_state='unsubmitted', observed_at=None,
                retired_at=None,retire_reason=None,lookup_generation=0,decision_link=link,dispatch=None,
                close=None,stop_cancel_requested_at=None))
            native['repairs'] = rows;record['updated_at'] = _ts(now)
            repair_rows(record);ledger._persist(db, record, 'repair_reserved', now)
            after = _records(ledger, db);project_groups(after)
            return RepairReservation(plan.family_id, identity, body['client_order_id'], _hash(_json(after)))
    except Exception: raise LedgerError('repair_reservation_unavailable') from None


def claim_repair(ledger, family_id, identity, *, expected_sha256, now):
    from .native_cleanup import project_groups
    try:
        now = utc_time(now);_identity(expected_sha256)
        if type(ledger) is not OrderLedger: raise ValueError('exact repair ledger required')
        with ledger._db(True) as db:
            records = _records(ledger, db);record = ledger._load(db, family_id);row = _find(record, identity)
            if row['submission_claimed_at'] is not None: return None
            group = project_groups(records)[family_id]
            if (_hash(_json(records)) != expected_sha256 or row['state'] != 'reserved'
                    or now < utc_time(record['updated_at']) or now >= utc_time(row['expires_at'])
                    or group.reserved_quantity != row['quantity'] or group.remaining_quantity != row['quantity']
                    or group.observed_at is None or not 0 <= (now - group.observed_at).total_seconds() <= 30):
                raise ValueError('repair claim sources or time changed')
            row.update(state='submission_unknown', submission_claimed_at=_ts(now), lookup_state='unknown')
            record['updated_at'] = _ts(now)
            return ledger._persist(db, record, 'repair_claim', now)
    except Exception: raise LedgerError('repair_claim_unavailable') from None


def retire_repair(ledger,family_id,identity,*,expected_sha256,reason,now):
    """Release only an unchanged allocation that has never been claimed."""
    try:
        now=utc_time(now);_identity(expected_sha256)
        if type(ledger) is not OrderLedger or reason not in ('expired','pre_dispatch_abort'):
            raise ValueError('exact repair retirement required')
        with ledger._db(True) as db:
            records=_records(ledger,db);record=ledger._load(db,family_id);row=_find(record,identity)
            if row['state']!='reserved' or row['submission_claimed_at'] is not None or row['order'] is not None:return None
            if (_hash(_json(records))!=expected_sha256 or now<utc_time(record['updated_at'])
                    or reason=='expired' and now<utc_time(row['expires_at'])):
                raise ValueError('repair retirement source or clock changed')
            row.update(state='retired',retired_at=_ts(now),retire_reason=reason)
            record['updated_at']=_ts(now);repair_rows(record)
            return ledger._persist(db,record,'repair_retired_unsubmitted',now)
    except Exception:raise LedgerError('repair_retirement_unavailable') from None


def observe_repair(ledger, family_id, identity, reference, *, expected_generation=None):
    from .native_broker import OpenOrderReference
    from .native_orders import _check_leg, _leg_record, refuse_reused_identity
    from .protection import GroupLeg
    try:
        if type(ledger) is not OrderLedger or type(reference) is not OpenOrderReference:
            raise ValueError('exact repair observation required')
        with ledger._db(True) as db:
            ledger._verify(db);record = ledger._load(db, family_id);row = _find(record, identity)
            expected=0 if expected_generation is None else expected_generation
            if (type(expected) is not int or expected!=row['lookup_generation'] or row['lookup_state']!='unknown'):
                raise ValueError('repair observation superseded by a later lookup')
            if (row['submission_claimed_at'] is None or reference.received_at < utc_time(record['updated_at'])
                    or reference.client_order_id != row['client_order_id'] or reference.symbol != record['intent']['symbol']
                    or reference.side != 'sell' or reference.order_type != 'stop' or reference.limit_price_usd is not None
                    or reference.stop_price_usd != Decimal(row['stop_price_usd']) or reference.quantity != row['quantity']
                    or reference.order_class not in ('simple', '') or reference.time_in_force != 'gtc'
                    or reference.created_at < utc_time(row['submission_claimed_at'])
                    or reference.status == 'filled' and reference.filled_at is None):
                raise ValueError('repair observation mismatch')
            leg = GroupLeg(reference.broker_order_id, None, reference.status, reference.quantity, reference.filled_quantity,
                reference.average_fill_price_usd, reference.created_at, reference.updated_at, reference.filled_at)
            normalized = _leg_record(leg)
            _check_leg(row['order'], normalized, parent=False, limit=None)
            native = record['native_stop']
            siblings = {native['parent']['broker_order_id'], native['stop']['broker_order_id']}
            siblings.update(r['order']['broker_order_id'] for r in repair_rows(record)
                if r['identity'] != identity and r['order'] is not None)
            if reference.broker_order_id in siblings: raise ValueError('repair broker identity reused')
            refuse_reused_identity(ledger, db, family_id, [reference.broker_order_id])
            row.update(state=reference.status, order=normalized, lookup_state='observed', observed_at=_ts(reference.received_at))
            record['updated_at'] = _ts(reference.received_at)
            repair_rows(record)
            return ledger._persist(db, record, 'repair_observation', reference.received_at)
    except Exception: raise LedgerError('repair_observation_unavailable') from None


class RepairPaperReader:
    def __init__(self, broker):
        if type(broker) is not PaperBroker: raise BrokerError('exact_paper_broker_required')
        self.broker = broker

    def recover(self, ledger, family_id, identity):
        from .native_broker import _reference
        try:
            if type(ledger) is not OrderLedger: raise ValueError('exact repair ledger required')
            with ledger._db(True) as db:
                ledger._verify(db);record = ledger._load(db, family_id);row = _find(record, identity)
                now = utc_time(self.broker._clock())
                if row['submission_claimed_at'] is None or row['state']=='aborted_unsent' or now < utc_time(record['updated_at']):
                    raise ValueError('claimed current repair required')
                generation=row['lookup_generation']+1
                if generation>1000000:raise ValueError('repair lookup budget')
                row.update(lookup_state='unknown',lookup_generation=generation);record['updated_at'] = _ts(now)
                ledger._persist(db, record, 'repair_lookup_started', now)
                client = row['client_order_id']
            response = self.broker._request('GET', '/v2/orders:by_client_order_id', params=dict(client_order_id=client))
            if response.status == 404: return None
            if response.status != 200: raise ValueError('repair GET unavailable')
            return observe_repair(ledger, family_id, identity, _reference(response.payload, utc_time(self.broker._clock())),
                expected_generation=generation)
        except Exception: raise BrokerError('repair_recovery_unavailable') from None


def repair_assessment(record, group, *, calendar, now):
    """A current owned stop must cover all remaining family shares exactly."""
    from .protection import ProtectionState, ACTIVE_STOP
    rows = tuple(r for r in repair_rows(record) if r['state'] not in ('retired','aborted_unsent'))
    if not rows: return None
    now = utc_time(now)
    def result(status, active=False, pause=True):
        return ProtectionState(status, group.remaining_quantity, active, pause,
            () if not pause else ('pause_entries', 'reconcile_group', 'reconcile_repair'))
    if any(r['lookup_state'] != 'observed' for r in rows): return result('repair_unconfirmed')
    if group.remaining_quantity == 0 and group.settled: return result('closed', pause=False)
    if any(r['stop_cancel_requested_at'] is not None and r['state'] not in CLOSED
            and (r['close'] is None or r['close']['outcome']!='rejected') for r in rows):
        return result('repair_close_in_progress')
    active = [r for r in rows if r['state'] not in CLOSED]
    # Current protection does not require polling every historical terminal
    # order. A new handoff with no active stop still needs fresh terminal proof.
    current=active or rows
    if any(r['observed_at'] is None or not 0 <= (now - utc_time(r['observed_at'])).total_seconds() <= 30 for r in current):
        return result('repair_evidence_stale')
    if calendar.at(now) is None: return result('out_of_session_exposure')
    if (len(active) == 1 and active[0]['state'] in ACTIVE_STOP and group.parent_terminal and group.child_terminal
            and reserved_quantity(active[0]) == group.remaining_quantity > 0):
        return result('repaired_protected', active=True, pause=False)
    return result('unprotected_position' if not active else 'repair_pending')


def active_repair(record):
    from .protection import ACTIVE_STOP
    rows = [r for r in repair_rows(record) if r['state'] in ACTIVE_STOP and r['lookup_state'] == 'observed']
    if len(rows) != 1: raise LedgerError('single_current_repair_required')
    return rows[0]
