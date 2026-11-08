"""Owned native BUY-parent cancellation and GET-only restart recovery.

No POST, child DELETE, replacement stop or reducing sell is exposed here. The
future global coordinator must join the resulting group with fresh accounting
and group-aware risk before cleanup. An acknowledged DELETE never frees shares.
"""
from contextlib import ExitStack
from dataclasses import dataclass,field
from datetime import datetime
import time

from .broker import PaperBroker,BrokerError
from .controls import ControlJournal,ControlSnapshot,ControlError
from .data import Calendar,utc_time
from .native_broker import NativePaperReader
from .native_reconciler import _group
from .native_orders import TERMINAL,entry_cancellation_safe
from .orders import OrderLedger,_hash,_json
from .protection import assess_protection
from .risk import _identity


@dataclass(frozen=True)
class EntryRecoveryResult:
    as_of: datetime
    status: str
    protection_status: str
    remaining_quantity: int|None
    required_actions: tuple
    request_sent: bool
    order_digest: str=field(repr=False)

    def public_summary(self):
        return dict(mode='paper',phase='native_entry_recovery',generated_at=self.as_of.timestamp(),
            status=self.status,protection_status=self.protection_status,remaining_quantity=self.remaining_quantity,
            required_actions=list(self.required_actions),cancel_requests_sent=int(self.request_sent),
            orders_submitted=0,execution_enabled=False,ready_for_risk=False)


class NativeEntryRecovery:
    def __init__(self,broker,*,monotonic=None):
        if type(broker) is not PaperBroker:raise BrokerError('exact_paper_broker_required')
        self.broker=broker;self._monotonic=monotonic or time.monotonic

    def _result(self,record,calendar,*,status=None,sent=False):
        at=utc_time(self.broker._clock());native=record.get('native_stop');protection='unavailable';remaining=None
        actions=('pause_entries','reconcile_group','reconcile_account')
        if status is not None:
            # A failed/latest-unknown read or a veto cannot make cached group
            # evidence current, even while the old receipt is under 30 seconds.
            return EntryRecoveryResult(at,status,'unconfirmed',None,actions,sent,_hash(_json(record)))
        parent=native['parent'] if native else None
        if parent is not None:
            try:
                group=_group(record);state=assess_protection(group,calendar=calendar,now=at)
                protection=state.status;remaining=state.remaining_quantity
                actions=tuple(dict.fromkeys((*state.required_actions,'reconcile_account')))
                if status is None:
                    if state.status=='closed':status='group_closed'
                    elif parent['status']=='filled':status='entry_filled'
                    elif parent['status'] in TERMINAL:
                        if parent['filled_quantity']:
                            status='partial_cleanup_required'
                            actions=tuple(dict.fromkeys(('pause_entries','reconcile_group','reconcile_account',*actions)))
                        elif state.status=='entry_not_filled':status='entry_canceled'
                        else:status='entry_cancel_pending'
                    elif record['cancel_requested_at'] is not None or parent['status']=='pending_cancel':status='entry_cancel_pending'
                    else:status='recovery_not_required'
            except Exception:pass
        return EntryRecoveryResult(at,status or 'reconciliation_pending',protection,remaining,actions,sent,_hash(_json(record)))

    def cancel_entry(self,ledger,identity,*,controls,control_snapshot,calendar):
        """Cancel remaining owned entry once, then observe. No blind DELETE retry.

        A current global halt may cancel this incoming BUY, but this component
        can never remove an active protective sell child. All reads and dispatch
        need an outer process deadline as well as the monotonic checkpoints.
        """
        try:
            if (type(ledger) is not OrderLedger or type(controls) is not ControlJournal
                    or type(control_snapshot) is not ControlSnapshot or type(calendar) is not Calendar):
                raise ValueError('exact recovery dependencies required')
            _identity(identity);deadline=self._monotonic()+25
            def expired():return self._monotonic()>=deadline
            reader=NativePaperReader(self.broker,monotonic=self._monotonic)
            original=ledger.load(identity)
            if original.get('native_stop') is None:raise ValueError('owned native family required')
            record=reader.recover(ledger,identity)
            if record is None:return self._result(ledger.load(identity),calendar,status='unconfirmed_submission')
            guard_at=utc_time(self.broker._clock())
            with ExitStack() as stack:
                try:current=stack.enter_context(controls.dispatch_lease(control_snapshot,now=guard_at))
                except ControlError:return self._result(record,calendar,status='control_changed')
                if current is not True:return self._result(record,calendar,status='control_changed')
                # Validate controls before every no-op as well as every write.
                now=utc_time(self.broker._clock());native=record['native_stop'];parent=native['parent']
                if (parent['status'] not in ('new','accepted','pending_new','accepted_for_bidding','partially_filled')
                        or parent['filled_quantity']>=parent['quantity'] or record['cancel_requested_at'] is not None):
                    return self._result(record,calendar)
                if (not 0<=(now-utc_time(native['observed_at'])).total_seconds()<=30
                        or now<guard_at or expired() or not entry_cancellation_safe(record)):
                    return self._result(record,calendar,status='reconciliation_pending')
                if parent['filled_quantity']>0:reason='partial_entry'
                elif now>=utc_time(native['plan']['expires_at']):reason='entry_expired'
                elif control_snapshot.halt_orders:reason='global_halt'
                elif native['stop']['status'] in ('canceled','expired','rejected'):reason='protection_failed'
                elif calendar.at(now) is None:reason='out_of_session'
                else:return self._result(record,calendar)
                claimed_at=utc_time(self.broker._clock())
                ticket=ledger.prepare_native_entry_cancellation(identity,expected_sha256=_hash(_json(record)),
                    control_snapshot=control_snapshot,reason=reason,at=claimed_at)
                if ticket is None:return self._result(ledger.load(identity),calendar,status='reconciliation_pending')
                latest=ledger.load(identity)
                checked=utc_time(self.broker._clock());veto=None
                if checked<max(guard_at,claimed_at):veto='clock_regression'
                elif (_hash(_json(latest))!=ticket['claimed_record_sha256']
                        or not entry_cancellation_safe(latest)):veto='group_changed'
                elif expired():veto='deadline'
                elif not 0<=(checked-utc_time(native['observed_at'])).total_seconds()<=30:veto='evidence_stale'
                if veto:
                    record=ledger.finish_native_entry_cancellation(ticket,outcome='not_sent',observed_at=checked,veto_reason=veto)
                    return self._result(record,calendar,status='entry_cancel_not_sent')
                # The lease prevents a control commit through this request.
                # Only this owned BUY UUID is used; no group/bulk/child endpoint.
                outcome='unknown';http_status=None
                try:
                    reply=self.broker._request('DELETE','/v2/orders/'+ticket['target_id'])
                    if 100<=reply.status<=599:http_status=reply.status
                    if reply.status==204:outcome='acknowledged'
                    elif reply.status==422:outcome='rejected'
                except Exception:pass
            record=ledger.finish_native_entry_cancellation(ticket,outcome=outcome,
                observed_at=utc_time(self.broker._clock()),http_status=http_status)
            if expired():return self._result(record,calendar,status='entry_cancel_unknown' if outcome=='unknown' else 'reconciliation_pending',sent=True)
            try:
                fresh=reader.recover(ledger,identity)
                if fresh is not None:return self._result(fresh,calendar,sent=True)
            except Exception:pass
            return self._result(record,calendar,status='entry_cancel_unknown' if outcome=='unknown' else 'reconciliation_pending',sent=True)
        except Exception:raise BrokerError('native_entry_recovery_unavailable_or_unknown') from None
