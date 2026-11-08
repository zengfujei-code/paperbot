"""Decision/control-bound one-shot paper protective-stop sender. No CLI/service.

The global coordinator and its outer process deadline remain activation gates.
Unknown outcomes only permit GET recovery; a provably unsent final veto retains
claim history and releases only its own allocation. No entry/model authority.
"""
from contextlib import ExitStack
from dataclasses import dataclass,field
from copy import deepcopy
import time

from .accountbook import AccountBook
from .broker import PaperBroker,BrokerError
from .controls import ControlJournal,ControlSnapshot,ControlError
from .data import utc_time
from .orders import OrderLedger,LedgerError,_hash,_json,_ts
from .repair_decisions import (RepairDecisionJournal,RepairDecisionReceipt,evaluate,
    _inputs as validate_inputs,_evaluate as evaluate_validated)
from .repair_protection import _calendar,_limits
from .risk_baselines import RiskBaselineStore


def _admission(ledger,record,row,decision,records):
    value=decision.read()['inputs'];owner=row['ownership']
    if (row['decision_link']!=dict(sequence=decision.sequence,digest=decision.digest,identity=decision.read()['identity'])
            or row['quantity']!=int(value['body']['qty']) or row['stop_price_usd']!=value['body']['stop_price']
            or row['client_order_id']!=value['body']['client_order_id'] or row['expires_at']!=value['expires_at']
            or owner['source_orders_sha256']!=value['source_orders_sha256']
            or owner['source_book_sha256']!=value['source_book_digest']
            or owner['source_baseline_sha256']!=value['source_baseline_digest']
            or owner['limits_sha256']!=_hash(_json(value['limits']))
            or owner['calendar_sha256']!=_hash(_json(value['calendar']))
            or owner['code_sha256']!=value['plan_code_sha256']
            or utc_time(row['allocated_at']).timestamp()<decision.read()['generated_at']):
        raise ValueError('repair admission decision binding')
    expected=deepcopy(ledger.records_at_prefix(value['source_prefix']))
    original=next(r for r in expected if r['intent_id']==record['intent_id'])
    original['native_stop'].setdefault('repairs',[]).append(deepcopy(row))
    original['updated_at']=row['allocated_at']
    if _json(expected)!=_json(records):raise ValueError('repair admission source changed')


@dataclass(frozen=True)
class RepairDispatchClaim:
    family_id:str=field(repr=False)
    repair_id:str=field(repr=False)
    identity:str=field(repr=False)
    claimed_at:object
    orders_sha256:str=field(repr=False)
    decision:RepairDecisionReceipt=field(repr=False)


def claim_dispatch(ledger,family_id,identity,*,decision,book,baselines,now):
    from .repair_orders import _records,_find
    from .native_cleanup import project_groups
    now=utc_time(now);recorded=decision.read();value=recorded['inputs']
    with ledger._db(True) as db:
        records=_records(ledger,db);record=ledger._load(db,family_id);row=_find(record,identity)
        if row['submission_claimed_at'] is not None:return None
        prefix_head=db.execute('SELECT sequence FROM events ORDER BY sequence DESC LIMIT 1').fetchone()[0]
        if prefix_head!=value['source_prefix']['sequence']+1:raise LedgerError('repair_admission_prefix_changed')
        _admission(ledger,record,row,decision,records)
        group=project_groups(records)[family_id]
        if (not recorded['outcome']['allowed'] or not evaluate(value,now)['allowed']
                or now.timestamp()<recorded['generated_at'] or now<utc_time(recorded['evaluated_at'])
                or row['state']!='reserved' or row['dispatch'] is not None or now<utc_time(record['updated_at'])
                or group.reserved_quantity!=row['quantity'] or group.remaining_quantity!=row['quantity']
                or book.snapshot().receipt.digest!=value['source_book_digest']
                or baselines.replay().audit_sha256!=value['source_baseline_digest']):
            raise LedgerError('repair_dispatch_claim_not_current')
        prior_sha=_hash(_json(records));claim_id=_hash(_json(['repair_dispatch_v1',family_id,identity,decision.digest,prior_sha,now]))
        row['dispatch']=dict(identity=claim_id,decision_digest=decision.digest,claimed_at=_ts(now),
            source_orders_sha256=prior_sha,outcome='claimed',http_status=None,observed_at=None,vetoes=[],clock_regressed=False)
        row.update(state='submission_unknown',submission_claimed_at=_ts(now),lookup_state='unknown')
        record['updated_at']=_ts(now);ledger._persist(db,record,'repair_dispatch_claim',now)
        after=_records(ledger,db)
        return RepairDispatchClaim(family_id,identity,claim_id,now,_hash(_json(after)),decision)


def finish_dispatch(ledger,ticket,*,outcome,at,http_status=None,vetoes=()):
    from .repair_orders import _find
    from .decision_journal import _codes
    if type(ticket) is not RepairDispatchClaim or outcome not in ('not_sent','unknown','acknowledged'):
        raise LedgerError('repair_dispatch_outcome_invalid')
    observed=utc_time(at);_codes(vetoes)
    if (http_status is not None and (type(http_status) is not int or not 100<=http_status<=599)
            or outcome=='acknowledged' and http_status!=201
            or outcome=='not_sent' and (http_status is not None or not vetoes)
            or outcome!='not_sent' and vetoes):raise LedgerError('repair_dispatch_outcome_mismatch')
    with ledger._db(True) as db:
        ledger._verify(db);record=ledger._load(db,ticket.family_id);row=_find(record,ticket.repair_id);claim=row['dispatch']
        if (claim is None or claim['identity']!=ticket.identity or claim['outcome']!='claimed'
                or row['submission_claimed_at']!=_ts(ticket.claimed_at)):
            raise LedgerError('repair_dispatch_claim_changed')
        floor=max(utc_time(record['updated_at']),ticket.claimed_at)
        if outcome=='not_sent':
            if row['order'] is not None or row['lookup_generation']!=0 or row['state']!='submission_unknown':
                raise LedgerError('repair_not_sent_proof_changed')
            row.update(state='aborted_unsent',lookup_state='unsubmitted')
        claim.update(outcome=outcome,http_status=http_status,observed_at=_ts(observed),
            vetoes=list(vetoes),clock_regressed=observed<floor)
        record['updated_at']=_ts(max(floor,observed))
        return ledger._persist(db,record,'repair_dispatch_result',max(floor,observed))


@dataclass(frozen=True)
class RepairDispatchResult:
    status:str
    request_sent:bool=False
    vetoes:tuple=()

    def public_summary(self):
        return dict(mode='paper',phase='repair_dispatch',status=self.status,order_requests_sent=int(self.request_sent),
            vetoes=list(self.vetoes),protection_active=False,ready_for_execution=False,execution_enabled=False)


class RepairPaperGateway:
    def __init__(self,broker,*,monotonic=None):
        if type(broker) is not PaperBroker:raise BrokerError('exact_paper_broker_required')
        self.broker=broker;self._monotonic=monotonic or time.monotonic

    def submit(self,ledger,family_id,identity,*,journal,book,baselines,controls,control_snapshot,calendar,limits):
        from .repair_orders import _find,observe_repair
        from .native_broker import _reference
        from .native_cleanup import project_groups
        try:
            for value,exact in ((ledger,OrderLedger),(journal,RepairDecisionJournal),(book,AccountBook),
                    (baselines,RiskBaselineStore),(controls,ControlJournal),(control_snapshot,ControlSnapshot)):
                if type(value) is not exact:raise ValueError('exact repair dispatch inputs')
            deadline=self._monotonic()+25;initial=ledger.load(family_id);row=_find(initial,identity)
            if row['submission_claimed_at'] is not None:return RepairDispatchResult('already_claimed')
            if row['decision_link'] is None:raise ValueError('repair decision absent at admission')
            decision=journal.replay(row['decision_link']['sequence'],ledger=ledger);recorded=decision.read();value=recorded['inputs']
            if (decision.digest!=row['decision_link']['digest'] or _limits(limits)!=_hash(_json(value['limits']))
                    or _calendar(calendar)!=_json(value['calendar']) or _json(value['controls'])!=_json(_base_control(control_snapshot))):
                raise ValueError('recorded repair parameters changed')
            ticket=claim_dispatch(ledger,family_id,identity,decision=decision,book=book,baselines=baselines,now=self.broker._clock())
            if ticket is None:return RepairDispatchResult('already_claimed')
            guard=utc_time(self.broker._clock())
            with ExitStack() as stack:
                try:current=stack.enter_context(controls.dispatch_lease(control_snapshot,now=guard))
                except ControlError:current=False
                reasons=[];source_time=None
                try:
                    current_records=ledger.all_orders();snapshot=book.snapshot();baseline=baselines.replay()
                    group=project_groups(current_records)[family_id]
                    source_time=min(snapshot.activity_until,snapshot.account.received_at,baseline.source_as_of,
                        *(p.received_at for p in snapshot.account.positions))
                    if (_hash(_json(current_records))!=ticket.orders_sha256
                            or snapshot.receipt.digest!=value['source_book_digest'] or not snapshot.receipt.result.reconciled
                            or baseline.audit_sha256!=value['source_baseline_digest']
                            or baseline.book_digest!=snapshot.receipt.digest
                            or group.remaining_quantity!=int(value['body']['qty'])
                            or group.reserved_quantity!=group.remaining_quantity):reasons.append('repair_source_changed')
                except Exception:reasons.append('repair_source_unavailable')
                # Finish disk/source validation before sampling dispatch time.
                # Everything after the sample is bounded in-memory math until
                # the transport call, including expiry and session checks.
                validated=None;current_inputs=None
                try:
                    if _calendar(calendar)!=_json(value['calendar']) or _limits(limits)!=_hash(_json(value['limits'])):
                        reasons.append('repair_parameters_changed')
                    current_inputs=ticket.decision.read()['inputs'];validated=validate_inputs(current_inputs)
                except Exception:reasons.append('repair_decision_unavailable')
                checked=utc_time(self.broker._clock())
                if source_time is None or not 0<=(checked-source_time).total_seconds()<=min(30,limits.max_age_seconds):
                    reasons.append('repair_account_stale')
                if validated is not None:
                    reasons.extend(evaluate_validated(current_inputs,checked,validated)['vetoes'])
                if checked<max(guard,ticket.claimed_at) or checked.timestamp()<recorded['generated_at']:
                    reasons.append('repair_clock_regression')
                if current is not True:reasons.append('operator_control_changed')
                if control_snapshot.halt_orders:reasons.append('all_orders_halted')
                if self._monotonic()>=deadline:reasons.append('repair_dispatch_deadline')
                reasons=tuple(dict.fromkeys(reasons))
                if reasons:
                    finish_dispatch(ledger,ticket,outcome='not_sent',at=checked,vetoes=reasons)
                    return RepairDispatchResult('not_sent',False,reasons)
                status=None;outcome='unknown'
                try:
                    reply=self.broker._request('POST','/v2/orders',body=value['body'])
                    if type(reply.status) is int and 100<=reply.status<=599:status=reply.status
                    if status==201:
                        observe_repair(ledger,family_id,identity,_reference(reply.payload,utc_time(self.broker._clock())))
                        outcome='acknowledged'
                except Exception:pass
            finish_dispatch(ledger,ticket,outcome=outcome,at=self.broker._clock(),http_status=status)
            return RepairDispatchResult('submitted' if outcome=='acknowledged' else 'submission_unknown',True)
        except Exception:raise BrokerError('repair_dispatch_unavailable_or_unknown') from None


def _base_control(snapshot):
    from .orders import _base,_normalize
    return _normalize(_base(snapshot,ControlSnapshot))
