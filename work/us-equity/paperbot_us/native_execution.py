"""Native paper entry dispatch. No CLI, strategy acceptance or recovery writes.

This low-level gateway needs the future coordinator's evidence/acceptance/global
lock. It never retries a submission. The separate native reader resolves unknown
outcomes; parent cancellation and protection recovery are not implied here.
"""
from contextlib import ExitStack
from dataclasses import replace
from decimal import Decimal
import time

from .broker import PaperBroker,BrokerError
from .controls import ControlJournal,ControlSnapshot,ControlError
from .data import utc_time
from .orders import OrderLedger,recheck_ticket
from .protection import NativeStopPlan
from .risk import RiskDecision,_identity,_money


class NativePaperGateway:
    def __init__(self,broker,*,monotonic=None):
        if type(broker) is not PaperBroker: raise BrokerError('exact_paper_broker_required')
        self.broker=broker
        self._monotonic=monotonic or time.monotonic

    def submit(self,ledger,identity,*,controls=None,control_snapshot=None,
               market_price_usd,market_at,entry_sources=None,**risk_inputs):
        """Claim once, recheck under control lease, then POST one owned OTO plan.

        Only failures established before network dispatch can become known-unsent
        vetoes. A bad response or post-success write failure keeps the durable
        claim unknown; a second invocation never sends another POST.
        """
        try:
            deadline=self._monotonic()+25
            if type(ledger) is not OrderLedger: raise ValueError('durable ledger required')
            if type(controls) is not ControlJournal or type(control_snapshot) is not ControlSnapshot:
                raise ValueError('durable operator controls required')
            _identity(identity)
            if 'gateway' in risk_inputs: raise ValueError('caller gateway override refused')
            initial=ledger.load(identity);native=initial.get('native_stop')
            if native is None: raise ValueError('attached native plan required')
            plan=NativeStopPlan(**native['plan'])
            envelope=None
            if initial.get('native_entry') is not None:
                from .native_admission import claim_entry
                if risk_inputs or entry_sources is None or entry_sources.control_snapshot!=control_snapshot:
                    raise ValueError('dedicated grouped entry sources required')
                envelope=claim_entry(ledger,identity,sources=entry_sources,now=utc_time(self.broker._clock()))
                ticket=envelope.read() if envelope is not None else None
                risk_inputs=dict(quote=entry_sources.prepared.quote,calendar=entry_sources.calendar)
            else:
                if entry_sources is not None:raise ValueError('legacy entry cannot gain grouped authority')
                ticket=ledger.prepare_native_submission(identity,**{**risk_inputs,'now':utc_time(self.broker._clock())})
            if ticket is None:return None
            guard_at=utc_time(self.broker._clock())
            def final_check(at,current):
                source_ok=True
                if envelope is not None:
                    from .native_admission import validate_dispatch
                    source_ok=validate_dispatch(ledger,envelope,entry_sources)
                checked_at=utc_time(self.broker._clock())
                try:decision=recheck_ticket(ticket,now=checked_at)
                except Exception:
                    # This stage has made no network request. Preserve the last
                    # completed reservation/exposure measures, explicitly marked
                    # unavailable for fresh risk, instead of inventing zeroes.
                    prior=ticket['risk']
                    decision=RiskDecision(False,('risk_recheck_unavailable',),*(
                        _money(Decimal(str(prior[name]))) for name in
                        ('reserved_cash_usd','modeled_trade_loss_usd','gross_exposure_after_usd')))
                reasons=list(decision.violations)
                if not source_ok:reasons.append('native_entry_source_changed')
                floor=max(at,guard_at,utc_time(ticket['updated_at']))
                if checked_at<floor:reasons.append('submission_clock_regression')
                if not current:reasons.append('operator_control_changed')
                elif control_snapshot.halt_orders:reasons.append('all_orders_halted')
                elif control_snapshot.pause_entries:reasons.append('entries_paused')
                body=None
                try:
                    body=plan.request_body(quote=risk_inputs['quote'],market_price_usd=market_price_usd,
                        market_at=market_at,calendar=risk_inputs['calendar'],now=checked_at)
                except Exception:reasons.append('native_protection_invalid')
                if self._monotonic()>=deadline:reasons.append('native_dispatch_deadline')
                if reasons:decision=replace(decision,allowed=False,violations=tuple(dict.fromkeys(reasons)))
                if not decision.allowed:
                    ledger.reject_unsubmitted(ticket,decision,at=max(checked_at,floor),observed_at=checked_at)
                    return None
                return body
            with ExitStack() as stack:
                try:current=stack.enter_context(controls.dispatch_lease(control_snapshot,now=guard_at))
                except ControlError:
                    final_check(guard_at,False);return None
                body=final_check(guard_at,current is True)
                if body is None:return None
                # No disk/network operation between final risk/plan validation
                # and this single dispatch. The control lease spans the request.
                reply=self.broker._request('POST','/v2/orders',body=body)
            if reply.status not in (200,201):raise ValueError('native submission outcome unknown')
            return ledger.observe_native_group(identity,row=reply.payload,received_at=utc_time(self.broker._clock()))
        except Exception:raise BrokerError('native_submission_unavailable_or_unknown') from None
