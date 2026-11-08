"""Fresh protected-group share reservations for independent risk preparation.

No submission, cancellation or reservation release. Unsafe groups still require
the separate accounting-only reducing-exit coordinator; this module never fakes
a ready cycle or weakens the legacy order-admission veto.
"""
import hashlib

from .data import utc_time
from .native_reconciler import _assess_records, _group, _settled
from .protection import assess_protection
from .risk import StopReservation
from .risk_view import _prepare_risk_view


def _known_unsent(record,now):
    """Recognize the existing durable pre-network veto, never an unknown POST."""
    try:
        n=record['native_stop'];c=record['context'];risk=record['risk']
        claim=utc_time(record['submission_claimed_at']);veto=utc_time(c['veto_recorded_at'])
        observed=utc_time(c['observed_at'])
        return (record['state']=='rejected_by_risk' and risk['allowed'] is False and bool(risk['violations'])
            and record['broker_order_id'] is None and record['broker_created_at'] is None
            and record['broker_updated_at'] is None and record['filled_quantity']==0
            and record['filled_notional_usd']=='0' and record['cancel_requested_at'] is None
            and n['parent'] is None and n['stop'] is None and n['child_visible'] is False
            and n['stop_cancel_requested_at'] is None
            and claim<=veto==utc_time(record['updated_at'])<=now and observed<=veto
            and (observed>=claim or 'submission_clock_regression' in risk['violations']))
    except (KeyError,TypeError,ValueError):return False


def _reservations(records, cycle, calendar, now):
    from .native_cleanup import project_groups
    from .repair_orders import repair_assessment,active_repair
    groups = project_groups(records)
    states, findings = _assess_records(records, calendar, now)
    if findings or cycle.native_states != tuple(states):
        raise ValueError('fresh group states mismatch')
    reservations, times = [], []
    for record in records:
        native = record.get('native_stop')
        if native is None:
            continue
        if native['parent'] is None:
            if _known_unsent(record,now):continue
            if record['submission_claimed_at'] is not None or record['state'] not in ('reserved', 'rejected_by_risk', 'expired'):
                raise ValueError('unknown native submission')
            continue
        if _settled(native) or groups[record['intent_id']].settled:
            continue
        at = utc_time(native['observed_at'])
        if not 0 <= (now - at).total_seconds() <= 30:
            raise ValueError('stale native group')
        state = repair_assessment(record,groups[record['intent_id']],calendar=calendar,now=now) or assess_protection(_group(record), calendar=calendar, now=now)
        if state.pause_entries or not native['child_visible']:
            raise ValueError('unprotected native group')
        times.append(at)
        if state.status in ('protected','repaired_protected') and state.native_stop_active:
            repaired=active_repair(record) if state.status=='repaired_protected' else None
            child = repaired['order'] if repaired is not None else native['stop']
            if repaired is not None:times.append(utc_time(repaired['observed_at']))
            remaining = child['quantity'] - child['filled_quantity']
            if remaining != state.remaining_quantity or remaining <= 0:
                raise ValueError('native remaining share mismatch')
            reservations.append(StopReservation(record['intent_id'],
                hashlib.sha256(child['broker_order_id'].encode()).hexdigest(), record['intent']['symbol'], remaining, at))
        elif state.status != 'awaiting_entry':
            raise ValueError('unsupported native risk state')
    return tuple(reservations), tuple(times)


def prepare_native_risk_view(**inputs):
    return _prepare_risk_view(**inputs, reservation_builder=_reservations)
