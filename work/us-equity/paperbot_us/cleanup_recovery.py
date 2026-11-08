"""Owned cleanup-limit cancellation/recovery. No POST or replacement authority.

Terminal observation releases only unfilled reservations. Fresh group/account
reconciliation remains required before a new cleanup or protection repair.
"""
from contextlib import ExitStack
from dataclasses import dataclass,field
from datetime import datetime,timedelta
import time

from .broker import PaperBroker,BrokerError
from .controls import ControlJournal,ControlSnapshot,ControlError
from .data import utc_time
from .native_broker import _reference,OpenOrderReference
from .native_cleanup import _OWNERSHIP,retire_cleanup_reservation
from .orders import OrderLedger,LedgerError,TERMINAL,_hash,_json,_ts
from .risk import OrderIntent,_identity

_LIVE=frozenset(('new','accepted','pending_new','accepted_for_bidding','partially_filled','pending_cancel'))
_TERMINAL=frozenset(('filled','canceled','expired','rejected'))


def _owned(ledger,db,identity):
    record=ledger._load(db,identity);owner=record.get('native_cleanup');intent=OrderIntent(**record['intent'])
    if (type(owner) is not dict or set(owner)!=_OWNERSHIP or owner['schema']!=1
            or record.get('native_stop') is not None or intent.side!='sell' or intent.stop_price_usd is not None
            or type(owner['allocated_quantity']) is not int or owner['allocated_quantity']!=intent.quantity):
        raise LedgerError('owned_cleanup_limit_required')
    family=ledger._load(db,owner['family_intent_id']);native=family.get('native_stop')
    if (native is None or native['parent'] is None or native['stop'] is None
            or native['parent']['broker_order_id']!=owner['parent_broker_order_id']
            or native['stop']['broker_order_id']!=owner['child_broker_order_id']
            or family['intent']['symbol']!=intent.symbol):raise LedgerError('cleanup_owner_changed')
    # A capacity discrepancy is a reason to cancel, not a reason to prevent
    # quiescing this exact owned order. No new sell/repair can use this check.
    return record


def observe_cleanup(ledger,identity,reference,*,expected_generation=None):
    if type(ledger) is not OrderLedger or type(reference) is not OpenOrderReference:
        raise LedgerError('exact_cleanup_observation_required')
    with ledger._db(True) as db:
        ledger._verify(db);record=_owned(ledger,db,identity);intent=OrderIntent(**record['intent'])
        generation=record.get('cleanup_lookup_generation',0)
        expected=0 if expected_generation is None else expected_generation
        if (type(expected) is not int or type(generation) is not int or expected!=generation
                or generation and record.get('cleanup_lookup_state')!='unknown'):
            raise LedgerError('cleanup_observation_superseded')
        raw=reference.status
        if record['broker_updated_at'] is not None and reference.updated_at<utc_time(record['broker_updated_at']):
            raise LedgerError('stale_cleanup_observation')
        if (record['submission_claimed_at'] is None or reference.client_order_id!=record['client_order_id']
                or reference.order_type!='limit' or reference.order_class not in ('simple','')
                or reference.time_in_force!='day' or reference.side!='sell' or reference.symbol!=intent.symbol
                or reference.quantity!=intent.quantity or reference.limit_price_usd!=intent.limit_price_usd
                or raw not in _LIVE|_TERMINAL
                or record['state'] in TERMINAL and raw not in _TERMINAL):
            raise LedgerError('cleanup_observation_mismatch')
        full_pending=raw=='pending_cancel' and reference.filled_quantity==reference.quantity
        normalized=('filled' if full_pending else 'partially_filled' if reference.filled_quantity else 'open') if raw in _LIVE else raw
        saved=ledger._observe(db,identity,status=normalized,filled_quantity=reference.filled_quantity,
            average_fill_price_usd=reference.average_fill_price_usd,broker_order_id=reference.broker_order_id,
            broker_created_at=reference.created_at,broker_event_at=reference.updated_at,received_at=reference.received_at,
            _allow_cleanup_terminal_update=True)
        if full_pending:saved['state']='submission_unknown'
        saved['cleanup_broker_status']=raw;saved['cleanup_observed_at']=_ts(reference.received_at)
        saved['cleanup_lookup_state']='observed';saved['cleanup_lookup_at']=_ts(reference.received_at)
        saved['updated_at']=_ts(max(utc_time(saved['updated_at']),reference.received_at))
        return ledger._persist(db,saved,'cleanup_broker_observation',reference.received_at)


class CleanupPaperReader:
    def __init__(self,broker):
        if type(broker) is not PaperBroker:raise BrokerError('exact_paper_broker_required')
        self.broker=broker

    def recover(self,ledger,identity):
        try:
            _identity(identity)
            if type(ledger) is not OrderLedger:raise ValueError('exact cleanup ledger required')
            with ledger._db(True) as db:
                ledger._verify(db);record=_owned(ledger,db,identity)
                if record['submission_claimed_at'] is None:raise ValueError('submitted cleanup required')
                # Every lookup has a durable generation. A delayed success may
                # not clear the unknown marker of a newer request, even if the
                # earlier order was still working when that request started.
                at=utc_time(self.broker._clock());prior=record.get('cleanup_lookup_generation',0)
                if at<utc_time(record['updated_at']) or type(prior) is not int or not 0<=prior<1000000:
                    raise ValueError('cleanup lookup clock or generation invalid')
                generation=prior+1
                record.update(cleanup_lookup_state='unknown',cleanup_lookup_at=_ts(at),
                    cleanup_lookup_generation=generation,updated_at=_ts(at))
                record=ledger._persist(db,record,'cleanup_lookup_started',at)
            response=self.broker._request('GET','/v2/orders:by_client_order_id',params=dict(client_order_id=record['client_order_id']))
            if response.status==404:return None
            if response.status!=200:raise ValueError('cleanup lookup unavailable')
            reference=_reference(response.payload,utc_time(self.broker._clock()))
            return observe_cleanup(ledger,identity,reference,expected_generation=generation)
        except Exception:raise BrokerError('cleanup_lookup_unavailable') from None


@dataclass(frozen=True)
class CancelClaim:
    intent_id: str=field(repr=False)
    identity: str=field(repr=False)
    target_id: str=field(repr=False)
    claimed_at: datetime
    record_sha256: str=field(repr=False)


def claim_cancel(ledger,identity,*,expected_sha256,control_snapshot,reason,at):
    at=utc_time(at);_identity(identity);_identity(expected_sha256)
    if type(control_snapshot) is not ControlSnapshot or reason not in ('expired','global_halt'):
        raise LedgerError('cleanup_cancel_context_required')
    with ledger._db(True) as db:
        ledger._verify(db);record=_owned(ledger,db,identity);previous=record.get('cleanup_cancel')
        if (_hash(_json(record))!=expected_sha256 or record['state'] not in ('open','partially_filled')
                or record['broker_order_id'] is None or record['submission_claimed_at'] is None
                or record['filled_quantity']>=record['intent']['quantity']
                or record.get('cleanup_broker_status') not in _LIVE-{'pending_cancel'}
                or record.get('cleanup_lookup_state')!='observed'
                or at<utc_time(record['updated_at'])):raise LedgerError('cleanup_cancel_source_changed')
        if record['cancel_requested_at'] is not None:return None
        if reason=='expired' and at<utc_time(record['intent']['expires_at']) or reason=='global_halt' and not control_snapshot.halt_orders:
            raise LedgerError('cleanup_cancel_reason_invalid')
        generation=1 if previous is None else previous['generation']+1
        if previous is not None and previous['outcome']!='not_sent':return None
        if generation>3:return None
        claim=dict(schema=1,generation=generation,outcome='claimed',target_id=record['broker_order_id'],
            claimed_at=_ts(at),deadline_at=_ts(at+timedelta(seconds=30)),reason=reason,
            source_sha256=expected_sha256,control_digest=control_snapshot.digest,control_revision=control_snapshot.revision,
            observed_at=_ts(at),http_status=None,vetoes=[])
        claim['identity']=_hash(_json(claim));record['cleanup_cancel']=claim
        record.update(cancel_requested_at=_ts(at),updated_at=_ts(at))
        saved=ledger._persist(db,record,'cleanup_cancel_claim',at)
        return CancelClaim(identity,claim['identity'],claim['target_id'],at,_hash(_json(saved)))


def finish_cancel(ledger,ticket,*,outcome,observed_at,http_status=None,vetoes=()):
    observed=utc_time(observed_at)
    if type(ticket) is not CancelClaim or outcome not in ('not_sent','acknowledged','rejected','unknown'):
        raise LedgerError('cleanup_cancel_result_invalid')
    if http_status is not None and (type(http_status) is not int or not 100<=http_status<=599):
        raise LedgerError('cleanup_cancel_http_status_invalid')
    if (outcome=='not_sent' and (http_status is not None or not vetoes)
            or outcome=='acknowledged' and http_status!=204 or outcome=='rejected' and http_status!=422
            or outcome!='not_sent' and vetoes):raise LedgerError('cleanup_cancel_outcome_mismatch')
    from .decision_journal import _codes
    try:_codes(vetoes)
    except Exception:raise LedgerError('cleanup_cancel_veto_invalid') from None
    with ledger._db(True) as db:
        ledger._verify(db);record=_owned(ledger,db,ticket.intent_id);claim=record.get('cleanup_cancel')
        if (claim is None or claim['identity']!=ticket.identity or claim['outcome']!='claimed'
                or claim['target_id']!=ticket.target_id or utc_time(claim['claimed_at'])!=ticket.claimed_at
                or record['cancel_requested_at']!=_ts(ticket.claimed_at)):
            raise LedgerError('cleanup_cancel_claim_changed')
        floor=max(utc_time(record['updated_at']),ticket.claimed_at)
        claim.update(outcome=outcome,observed_at=_ts(observed),http_status=http_status,vetoes=list(vetoes))
        if observed<floor:claim['clock_regression']=True
        if outcome=='not_sent':record['cancel_requested_at']=None
        record['updated_at']=_ts(max(observed,floor))
        return ledger._persist(db,record,'cleanup_cancel_result',max(observed,floor))


@dataclass(frozen=True)
class CleanupRecoveryResult:
    status: str
    filled_quantity: int|None
    remaining_order_quantity: int|None
    required_actions: tuple
    request_sent: bool=False

    def public_summary(self):
        return dict(mode='paper',phase='cleanup_recovery',status=self.status,filled_quantity=self.filled_quantity,
            remaining_order_quantity=self.remaining_order_quantity,required_actions=list(self.required_actions),
            cancel_requests_sent=int(self.request_sent),orders_submitted=0,ready_for_execution=False,execution_enabled=False)


class CleanupRecovery:
    def __init__(self,broker,*,monotonic=None):
        if type(broker) is not PaperBroker:raise BrokerError('exact_paper_broker_required')
        self.broker=broker;self.reader=CleanupPaperReader(broker);self._monotonic=monotonic or time.monotonic

    def _result(self,record=None,*,status=None,sent=False):
        actions=('pause_entries','reconcile_cleanup','reconcile_group','reconcile_account')
        if record is None:return CleanupRecoveryResult(status or 'cleanup_unknown',None,None,actions,sent)
        raw=record.get('cleanup_broker_status');claim=record.get('cleanup_cancel')
        if status is None:
            if raw in _TERMINAL:status='cleanup_terminal'
            elif raw=='pending_cancel':status='cleanup_cancel_pending'
            elif claim is not None and claim['outcome']=='rejected':status='cleanup_cancel_rejected'
            elif record['cancel_requested_at'] is not None:status='cleanup_cancel_pending'
            else:status='cleanup_working'
        return CleanupRecoveryResult(status,record['filled_quantity'],record['intent']['quantity']-record['filled_quantity'],actions,sent)

    def recover(self,ledger,identity):
        try:record=self.reader.recover(ledger,identity)
        except Exception:record=None
        return self._result(record)

    def cancel(self,ledger,identity,*,controls,control_snapshot):
        try:
            if type(ledger) is not OrderLedger or type(controls) is not ControlJournal or type(control_snapshot) is not ControlSnapshot:
                raise ValueError('exact cleanup recovery dependencies required')
            _identity(identity);deadline=self._monotonic()+25
            initial=ledger.load(identity)
            if initial.get('native_cleanup') is None:raise ValueError('owned cleanup required')
            if initial['submission_claimed_at'] is None:
                now=utc_time(self.broker._clock())
                with controls.dispatch_lease(control_snapshot,now=now) as current:
                    if not current:return self._result(status='control_changed')
                    reason='expired' if now>=utc_time(initial['intent']['expires_at']) else 'pre_dispatch_abort' if control_snapshot.halt_orders else None
                    if reason is None:return self._result(initial,status='cleanup_working')
                    record=retire_cleanup_reservation(ledger,identity,expected_sha256=_hash(_json(initial)),reason=reason,now=now)
                # A concurrent submission can invalidate the unsent retirement.
                # Only its committed result proves release of the allocation.
                if record is None:return self._result()
                return self._result(record,status='cleanup_reserved_retired')
            try:record=self.reader.recover(ledger,identity)
            except Exception:record=None
            if record is None:return self._result()
            guard_at=utc_time(self.broker._clock());vetoed=False
            with ExitStack() as stack:
                try:current=stack.enter_context(controls.dispatch_lease(control_snapshot,now=guard_at))
                except ControlError:return self._result(status='control_changed')
                if not current:return self._result(status='control_changed')
                if (record.get('cleanup_broker_status') in _TERMINAL|{'pending_cancel'}
                        or record['cancel_requested_at'] is not None):return self._result(record)
                now=utc_time(self.broker._clock())
                reason='expired' if now>=utc_time(record['intent']['expires_at']) else 'global_halt' if control_snapshot.halt_orders else None
                if reason is None:return self._result(record)
                ticket=claim_cancel(ledger,identity,expected_sha256=_hash(_json(record)),control_snapshot=control_snapshot,reason=reason,at=now)
                if ticket is None:return self._result(record)
                latest=ledger.load(identity);checked=utc_time(self.broker._clock());reasons=[]
                if _hash(_json(latest))!=ticket.record_sha256:reasons.append('cleanup_cancel_source_changed')
                if checked<max(guard_at,ticket.claimed_at):reasons.append('cleanup_cancel_clock_regression')
                if not 0<=(checked-utc_time(record['cleanup_observed_at'])).total_seconds()<=30:reasons.append('cleanup_cancel_stale')
                if self._monotonic()>=deadline:reasons.append('cleanup_cancel_deadline')
                if reasons:
                    finish_cancel(ledger,ticket,outcome='not_sent',observed_at=checked,vetoes=tuple(reasons));vetoed=True
                else:
                    outcome='unknown';status=None
                    try:
                        response=self.broker._request('DELETE','/v2/orders/'+ticket.target_id)
                        if type(response.status) is int and 100<=response.status<=599:status=response.status
                        if status==204:outcome='acknowledged'
                        elif status==422:outcome='rejected'
                    except Exception:pass
            if vetoed:
                result=self.recover(ledger,identity)
                return CleanupRecoveryResult('cleanup_cancel_not_sent',result.filled_quantity,result.remaining_order_quantity,result.required_actions)
            finish_cancel(ledger,ticket,outcome=outcome,observed_at=utc_time(self.broker._clock()),http_status=status)
            if self._monotonic()>=deadline:return self._result(sent=True)
            try:fresh=self.reader.recover(ledger,identity)
            except Exception:fresh=None
            return self._result(fresh,sent=True)
        except Exception:raise BrokerError('cleanup_cancellation_unavailable_or_unknown') from None
