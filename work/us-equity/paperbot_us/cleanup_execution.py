"""Decision-bound paper cleanup dispatch; no CLI or production service.

The coordinator still owns a global process lock, current policy/acceptance and
outer process deadline. One durable claim never permits a second POST on restart.
"""
from contextlib import ExitStack
from dataclasses import replace
from decimal import Decimal

from . import native_cleanup
from .accountbook import AccountBook
from .broker import PaperBroker,BrokerError,_tick
from .controls import ControlJournal,ControlSnapshot,ControlError
from .data import utc_time,Calendar,Session
from .orders import OrderLedger,recheck_ticket,_hash,_json,_base,_normalize
from .risk import OrderIntent,RiskDecision,RiskLimits,_identity,_money
from .risk_baselines import RiskBaselineStore


class CleanupPaperGateway:
    def __init__(self,broker):
        if type(broker) is not PaperBroker:raise BrokerError('exact_paper_broker_required')
        self.broker=broker

    def submit(self,ledger,identity,*,prepared,limits,calendar,controls,control_snapshot,journal,book,baselines):
        try:
            if type(ledger) is not OrderLedger or type(controls) is not ControlJournal or type(control_snapshot) is not ControlSnapshot:
                raise ValueError('exact cleanup dispatch dependencies required')
            if type(book) is not AccountBook or type(baselines) is not RiskBaselineStore:
                raise ValueError('exact cleanup accounting sources required')
            _identity(identity);initial=ledger.load(identity)
            if initial['state']!='reserved':return None
            decision_record=native_cleanup.verify_cleanup_decision(ledger,initial,journal)
            intent=OrderIntent(**initial['intent']);_tick(intent.limit_price_usd)
            if type(limits) is not RiskLimits or type(calendar) is not Calendar:
                raise ValueError('exact committed dispatch parameters required')
            calendar_record=dict(start=calendar.start,end=calendar.end,
                sessions=[_base(s,Session) for _,s in sorted(calendar.sessions.items())])
            if (_normalize(_base(limits,RiskLimits))!=decision_record['inputs']['limits']
                    or _normalize(calendar_record)!=decision_record['inputs']['calendar']):
                raise ValueError('committed cleanup parameters changed')
            # Match the fixed pre-callback transport bound before making a claim.
            if intent.quantity*intent.limit_price_usd>1000:
                raise ValueError('cleanup_transport_budget_exceeded')
            envelope=native_cleanup.prepare_cleanup_dispatch(ledger,identity,prepared=prepared,limits=limits,
                calendar=calendar,now=utc_time(self.broker._clock()))
            if envelope is None:return None
            if type(envelope) is not native_cleanup.CleanupDispatch:raise ValueError('atomic dispatch claim required')
            ticket=envelope.read();guard_at=utc_time(self.broker._clock())
            def final_check(at,current):
                source_ok=False;ownership_ok=False
                try:
                    records=ledger.all_orders();snapshot=book.snapshot();baseline=baselines.replay()
                    source_ok=(_hash(_json(records))==envelope.orders_sha256
                        and snapshot.receipt.digest==prepared.risk.book_digest and snapshot.receipt.result.reconciled
                        and baseline.audit_sha256==prepared.risk.baseline_digest
                        and baseline.book_digest==snapshot.receipt.digest)
                    group=native_cleanup.project_groups(records,exclude=identity)[ticket['native_cleanup']['family_intent_id']]
                    ownership_ok=group.parent_terminal and group.child_terminal and intent.quantity<=group.available_quantity
                except Exception:pass
                checked=utc_time(self.broker._clock())
                try:result=recheck_ticket(ticket,now=checked)
                except Exception:
                    prior=ticket['risk']
                    result=RiskDecision(False,('risk_recheck_unavailable',),*(_money(Decimal(str(prior[k])))
                        for k in ('reserved_cash_usd','modeled_trade_loss_usd','gross_exposure_after_usd')))
                reasons=list(result.violations);floor=max(at,guard_at,utc_time(ticket['updated_at']))
                if checked<floor:reasons.append('submission_clock_regression')
                if not source_ok:reasons.append('cleanup_source_changed')
                if not ownership_ok:reasons.append('cleanup_ownership_unavailable')
                if not current or control_snapshot.digest!=decision_record['inputs']['controls']['digest']:
                    reasons.append('operator_control_changed')
                elif control_snapshot.halt_orders:reasons.append('all_orders_halted')
                if reasons:result=replace(result,allowed=False,violations=tuple(dict.fromkeys(reasons)))
                if not result.allowed:
                    ledger.reject_unsubmitted(ticket,result,at=max(checked,floor),observed_at=checked)
                return result.allowed
            with ExitStack() as stack:
                try:current=stack.enter_context(controls.dispatch_lease(control_snapshot,now=guard_at))
                except ControlError:
                    final_check(guard_at,False);return None
                order=self.broker._submit_limit(intent,before_send=lambda at:final_check(at,current is True))
            if order is None:return None
            return ledger.observe(identity,status=order.status,filled_quantity=order.filled_quantity,
                average_fill_price_usd=order.average_fill_price_usd,broker_order_id=order.broker_order_id,
                broker_created_at=order.created_at,broker_event_at=order.updated_at,received_at=order.received_at)
        except Exception:raise BrokerError('cleanup_submission_unavailable_or_unknown') from None
