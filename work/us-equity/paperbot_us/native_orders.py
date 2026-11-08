"""Native OTO ownership inside the existing atomic order journal. No broker I/O.

Only normalized fields are retained. Missing child observations preserve prior
fill evidence but never assert current visibility/protection. Cancellation is a
durable claim, not an acknowledgement or permission to send another sell.
"""
from decimal import Decimal
import hashlib

from .accounting import OrderEvidence
from .data import utc_time
from .orders import LedgerError, _base, _normalize, _ts, _hash, _json
from .protection import NativeStopPlan, GroupLeg, parse_oto_group
from .risk import OrderIntent, _identity

TERMINAL = frozenset(('filled', 'canceled', 'expired', 'rejected', 'replaced'))


def attach(ledger, identity, *, plan, at):
    _identity(identity);at=utc_time(at)
    if type(plan) is not NativeStopPlan or plan.intent_id!=identity:
        raise LedgerError('native_stop_plan_identity_mismatch')
    with ledger._db(True) as db:
        ledger._verify(db);record=ledger._load(db,identity)
        intent=OrderIntent(**record['intent'])
        if NativeStopPlan.from_intent(intent,time_in_force=plan.time_in_force)!=plan:
            raise LedgerError('native_stop_plan_intent_mismatch')
        serialized=_normalize(_base(plan,NativeStopPlan))
        if record.get('native_stop') is not None:
            if record['native_stop']['plan']!=serialized: raise LedgerError('native_stop_plan_immutable')
            return record
        if (record['state']!='reserved' or record['submission_claimed_at'] is not None
                or not utc_time(record['updated_at'])<=at<intent.expires_at):
            raise LedgerError('native_stop_attachment_requires_unclaimed_reservation')
        record['native_stop']=dict(schema=1,plan=serialized,parent=None,stop=None,
            observed_at=None,child_visible=False,stop_cancel_requested_at=None)
        record['updated_at']=_ts(at)
        return ledger._persist(db,record,'native_stop_attached',at)


def _leg_record(leg):
    return _normalize(_base(leg,GroupLeg))


def _check_leg(old, new, *, parent, limit):
    if old is None: return
    if (old['broker_order_id']!=new['broker_order_id'] or old['created_at']!=new['created_at']
            or old['quantity']!=new['quantity']
            or (old['group_id'] is not None and new['group_id'] is not None and old['group_id']!=new['group_id'])
            or (old['filled_at'] is not None and old['filled_at']!=new['filled_at'])):
        raise LedgerError('native_order_identity_or_completion_changed')
    old_at,new_at=utc_time(old['updated_at']),utc_time(new['updated_at'])
    if new_at<old_at or (new_at==old_at and old!=new):
        raise LedgerError('native_order_old_or_changed_event')
    if old['status'] in TERMINAL and new['status']!=old['status']:
        raise LedgerError('native_terminal_order_transition')
    delta=new['filled_quantity']-old['filled_quantity']
    old_value=Decimal(old['average_fill_price_usd'] or '0')*old['filled_quantity']
    new_value=Decimal(new['average_fill_price_usd'] or '0')*new['filled_quantity']
    if delta<0 or (delta==0 and new_value!=old_value) or (delta>0 and new_value<=old_value):
        raise LedgerError('native_cumulative_fill_regression')
    if parent and delta>0 and new_value-old_value>limit*delta:
        raise LedgerError('native_incremental_entry_limit_violation')


def _all_broker_ids(record):
    result=set()
    if record['broker_order_id'] is not None: result.add(record['broker_order_id'])
    native=record.get('native_stop')
    if native and native['stop'] is not None: result.add(native['stop']['broker_order_id'])
    if native and native.get('repairs'):
        from .repair_orders import repair_rows
        result.update(r['order']['broker_order_id'] for r in repair_rows(record) if r['order'] is not None)
    return result


def refuse_reused_identity(ledger, db, identity, broker_ids):
    for row in db.execute('SELECT intent_id FROM orders WHERE intent_id!=? LIMIT 10001',(identity,)):
        if _all_broker_ids(ledger._load(db,row[0])) & set(broker_ids):
            raise LedgerError('broker_order_owned_by_another_intent')


def observe(ledger, identity, *, row, received_at):
    _identity(identity);received=utc_time(received_at)
    with ledger._db(True) as db:
        ledger._verify(db);record=ledger._load(db,identity);native=record.get('native_stop')
        if native is None or record['submission_claimed_at'] is None:
            raise LedgerError('claimed_native_stop_plan_required')
        if native['parent'] is None and record['state']!='submission_unknown':
            raise LedgerError('native_first_observation_requires_unknown_submission')
        plan=NativeStopPlan(**native['plan'])
        group=parse_oto_group(row,plan=plan,received_at=received)
        if (received<utc_time(record['updated_at']) or group.parent.created_at<utc_time(record['submission_claimed_at'])
                or record['broker_order_id'] not in (None,group.parent.broker_order_id)):
            raise LedgerError('native_group_observation_time_or_owner_mismatch')
        parent=_leg_record(group.parent);stop=_leg_record(group.stop) if group.stop is not None else None
        _check_leg(native['parent'],parent,parent=True,limit=plan.entry_limit_usd)
        if stop is not None: _check_leg(native['stop'],stop,parent=False,limit=None)
        # An omitted child is not a deleted/terminal order. Retain its known
        # identity and fills for accounting while current visibility is false.
        retained=stop if stop is not None else native['stop']
        ids=[parent['broker_order_id']]+([retained['broker_order_id']] if retained else [])
        refuse_reused_identity(ledger,db,identity,ids)
        if retained is not None and (retained['broker_order_id']==parent['broker_order_id']
                or retained['filled_quantity']>parent['filled_quantity']):
            raise LedgerError('native_group_cross_leg_conflict')
        statuses={'new':'open','accepted':'open','pending_new':'open','accepted_for_bidding':'open',
                  'partially_filled':'partially_filled','filled':'filled','canceled':'canceled',
                  'expired':'expired','rejected':'rejected'}
        state=statuses.get(parent['status'],'submission_unknown')
        native.update(parent=parent,stop=retained,child_visible=stop is not None,observed_at=_ts(received))
        record.update(state=state,broker_order_id=parent['broker_order_id'],broker_created_at=parent['created_at'],
            broker_updated_at=parent['updated_at'],filled_quantity=parent['filled_quantity'],
            filled_notional_usd=str(Decimal(parent['average_fill_price_usd'] or '0')*parent['filled_quantity']),
            updated_at=_ts(received))
        return ledger._persist(db,record,'native_group_observation',received)


def prepare_stop_cancellation(ledger, identity, *, at):
    _identity(identity);at=utc_time(at)
    with ledger._db(True) as db:
        ledger._verify(db);record=ledger._load(db,identity);native=record.get('native_stop')
        if native is None: raise LedgerError('native_stop_ownership_required')
        stop=native['stop']
        if (stop is None or stop['status'] in TERMINAL or native['stop_cancel_requested_at'] is not None):
            return None
        if at<utc_time(record['updated_at']): raise LedgerError('native_cancellation_clock_regression')
        native['stop_cancel_requested_at']=_ts(at);record['updated_at']=_ts(at)
        ledger._persist(db,record,'native_stop_cancel_claim',at)
        return dict(intent_id=identity,broker_order_id=stop['broker_order_id'],claimed_at=_ts(at))


def entry_cancellation_safe(record):
    """A parent DELETE can cascade: require a fully observed inactive child."""
    native=record.get('native_stop')
    if native is None:return False
    parent,stop=native['parent'],native['stop']
    return (parent is not None and stop is not None and native['child_visible'] is True
        and parent['status'] in ('new','accepted','pending_new','accepted_for_bidding','partially_filled')
        and parent['filled_quantity']<parent['quantity']
        and stop['status'] in ('held','canceled','expired','rejected') and stop['filled_quantity']==0)


def prepare_entry_cancellation(ledger,identity,*,expected_sha256,control_snapshot,reason,at):
    """CAS-bound one-shot owned BUY cancellation. No reservation is released."""
    from .controls import ControlSnapshot
    _identity(identity);_identity(expected_sha256);at=utc_time(at)
    if type(control_snapshot) is not ControlSnapshot or reason not in ('partial_entry','entry_expired','out_of_session','global_halt','protection_failed'):
        raise LedgerError('native_entry_cancel_context_required')
    with ledger._db(True) as db:
        ledger._verify(db);record=ledger._load(db,identity);native=record.get('native_stop')
        if native is None or _hash(_json(record))!=expected_sha256:
            raise LedgerError('native_entry_cancel_source_changed')
        parent=native['parent'];previous=native.get('entry_cancel')
        if (not entry_cancellation_safe(record) or record['submission_claimed_at'] is None
                or record['broker_order_id']!=parent['broker_order_id'] or at<utc_time(record['updated_at'])):
            raise LedgerError('native_entry_cancel_not_eligible')
        if record['cancel_requested_at'] is not None:return None
        generation=1
        if previous is not None:
            if previous['outcome']!='not_sent':raise LedgerError('native_entry_cancel_unresolved')
            generation=previous['generation']+1
            if generation>3:return None
        if (reason=='partial_entry' and not 0<parent['filled_quantity']<parent['quantity']
                or reason=='entry_expired' and at<utc_time(native['plan']['expires_at'])
                or reason=='global_halt' and not control_snapshot.halt_orders
                or reason=='protection_failed' and native['stop']['status'] not in ('canceled','expired','rejected')):
            raise LedgerError('native_entry_cancel_reason_mismatch')
        claim=dict(schema=1,generation=generation,outcome='claimed',reason=reason,
            target_id=parent['broker_order_id'],claimed_at=_ts(at),source_sha256=expected_sha256,
            control_revision=control_snapshot.revision,control_digest=control_snapshot.digest,
            control_halt=control_snapshot.halt_orders,observed_at=_ts(at),http_status=None,veto_reason=None)
        claim['identity']=_hash(_json(claim))
        native['entry_cancel']=claim;record.update(cancel_requested_at=_ts(at),updated_at=_ts(at))
        saved=ledger._persist(db,record,'native_entry_cancel_claim',at)
        return dict(intent_id=identity,identity=claim['identity'],target_id=claim['target_id'],claimed_at=claim['claimed_at'],
                    claimed_record_sha256=_hash(_json(saved)))


def finish_entry_cancellation(ledger,ticket,*,outcome,observed_at,http_status=None,veto_reason=None):
    """Append result without overwriting concurrent fills or declaring cancellation."""
    if (type(ticket) is not dict or set(ticket)!={'intent_id','identity','target_id','claimed_at','claimed_record_sha256'}
            or outcome not in ('not_sent','acknowledged','rejected','unknown')):
        raise LedgerError('native_entry_cancel_result_invalid')
    _identity(ticket['intent_id']);_identity(ticket['identity']);_identity(ticket['claimed_record_sha256']);observed=utc_time(observed_at)
    if http_status is not None and (type(http_status) is not int or not 100<=http_status<=599):
        raise LedgerError('native_entry_cancel_http_status_invalid')
    if (outcome=='acknowledged' and http_status!=204 or outcome=='rejected' and http_status!=422
            or outcome=='not_sent' and (http_status is not None or veto_reason not in
                ('clock_regression','evidence_stale','deadline','control_changed','validation_unavailable','group_changed'))
            or outcome!='not_sent' and veto_reason is not None):
        raise LedgerError('native_entry_cancel_outcome_mismatch')
    with ledger._db(True) as db:
        ledger._verify(db);record=ledger._load(db,ticket['intent_id']);native=record.get('native_stop')
        claim=native.get('entry_cancel') if native is not None else None
        if (claim is None or claim['identity']!=ticket['identity'] or claim['outcome']!='claimed'
                or claim['target_id']!=ticket['target_id'] or claim['claimed_at']!=ticket['claimed_at']
                or record['cancel_requested_at']!=ticket['claimed_at']):
            raise LedgerError('native_entry_cancel_claim_changed')
        floor=max(utc_time(record['updated_at']),utc_time(ticket['claimed_at']))
        claim.update(outcome=outcome,observed_at=_ts(observed),http_status=http_status,veto_reason=veto_reason)
        if observed<floor:claim['clock_regression']=True
        if outcome=='not_sent':record['cancel_requested_at']=None
        record['updated_at']=_ts(max(observed,floor))
        return ledger._persist(db,record,'native_entry_cancel_result',max(observed,floor))


def accounting_evidence(records):
    """Flatten owned parents and retained stops, including terminal orders.

    Read records through OrderLedger.all_orders first. This projection neither
    proves current broker visibility nor releases cash or sell reservations.
    """
    result=[]
    for record in records:
        if record['broker_order_id'] is None: continue
        intent=record['intent'];native=record.get('native_stop')
        parent=native['parent'] if native is not None else None
        result.append(OrderEvidence(record['intent_id'],record['broker_order_id'],intent['symbol'],intent['side'],
            intent['quantity'],intent['limit_price_usd'],record['filled_quantity'],record['filled_notional_usd'],
            record['broker_created_at'],record['broker_updated_at'],filled_at=parent['filled_at'] if parent else None))
        if native is None or native['stop'] is None: continue
        stop=native['stop']
        identity=hashlib.sha256(('paperbot-native-stop-v1\0'+record['intent_id']).encode()).hexdigest()
        result.append(OrderEvidence(identity,stop['broker_order_id'],intent['symbol'],'sell',stop['quantity'],None,
            stop['filled_quantity'],Decimal(stop['average_fill_price_usd'] or '0')*stop['filled_quantity'],
            stop['created_at'],stop['updated_at'],order_type='stop',stop_price_usd=native['plan']['stop_price_usd'],
            parent_order_id=record['broker_order_id'],filled_at=stop['filled_at']))
        from .repair_orders import repair_rows
        for row in repair_rows(record):
            order=row['order']
            if order is None:continue
            result.append(OrderEvidence(row['identity'],order['broker_order_id'],intent['symbol'],'sell',row['quantity'],None,
                order['filled_quantity'],Decimal(order['average_fill_price_usd'] or '0')*order['filled_quantity'],
                order['created_at'],order['updated_at'],order_type='stop',stop_price_usd=row['stop_price_usd'],
                parent_order_id=record['broker_order_id'],filled_at=order['filled_at'],protection_kind='repair',
                parent_terminal_at=row['ownership']['parent_terminal_at'],allocation_sha256=_hash(_json(row['ownership']))))
    return tuple(result)
