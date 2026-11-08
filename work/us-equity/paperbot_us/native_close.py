"""Explicit stop-to-close preview and one-shot owned child cancellation.

No sell/replacement POST, CLI or service. After observed cancellation the global
coordinator must reconcile accounting and record a separate cleanup decision.
Bounded recovery reports uncertainty; it cannot guarantee repair through outage.
"""
from contextlib import ExitStack
from dataclasses import dataclass,field,replace
from datetime import datetime,timedelta
from decimal import Decimal,localcontext
import time

from .accountbook import AccountBook
from .broker import PaperBroker,BrokerError,_tick
from .controls import ControlJournal,ControlSnapshot,ControlError
from .data import Calendar,Session,utc_time
from .native_broker import NativePaperReader
from .native_cleanup import project_groups,cleanup_projection
from .orders import OrderLedger,LedgerError,_base,_hash,_json,_normalize,_ts
from .protection import ACTIVE_STOP
from .risk import Portfolio,Position,PendingOrder,StopReservation,Quote,OrderIntent,RiskLimits
from .risk_baselines import RiskBaselineStore
from .risk_view import PreparedRiskView,RiskViewError,_version as view_version


def _portfolio(row):
    from .decision_journal import _typed,_portfolio as encode
    value=_typed(Portfolio,row,dict(positions=tuple(_typed(Position,v) for v in row['positions']),
        pending_orders=tuple(_typed(PendingOrder,v) for v in row['pending_orders']),
        stop_reservations=tuple(_typed(StopReservation,v) for v in row['stop_reservations']),
        valuation_quotes=tuple(_typed(Quote,v) for v in row['valuation_quotes'])))
    if _normalize(encode(value))!=row:raise ValueError('close portfolio normalization')
    return value


@dataclass(frozen=True)
class ClosePreparation:
    original: PreparedRiskView=field(repr=False)
    preview: PreparedRiskView=field(repr=False)
    intent: OrderIntent=field(repr=False)
    family_id: str=field(repr=False)
    reason: str
    max_unprotected_seconds: int=30
    target_repair_id: str|None=field(default=None,repr=False)

    def public_summary(self):
        return dict(mode='paper',phase='close_preparation',execution_enabled=False,
            shares_available_now=0,preview_close_quantity=self.intent.quantity,orders_submitted=0)


def _target(record,repair_id):
    native=record['native_stop']
    if repair_id is None:return native,native['stop']
    from .repair_orders import _find
    row=_find(record,repair_id)
    return row,row['order']


def _target_valid(record,group,repair_id,now):
    native=record['native_stop'];slot,child=_target(record,repair_id)
    if (child is None or not native['child_visible'] or child['status'] not in ACTIVE_STOP
            or child['quantity']-child['filled_quantity']!=group.remaining_quantity):return False
    if repair_id is None:
        return native['parent']['status']=='filled' and not group.child_terminal and group.reserved_quantity==0
    return (group.child_terminal and group.reserved_quantity==group.remaining_quantity
        and slot['lookup_state']=='observed' and slot['observed_at'] is not None
        and 0<=(now-utc_time(slot['observed_at'])).total_seconds()<=30)


def prepare_close(*,ledger,prepared,family_id,calendar,now,reason='quantitative'):
    try:
        now=utc_time(now)
        if type(ledger) is not OrderLedger or type(prepared) is not PreparedRiskView or type(calendar) is not Calendar:
            raise ValueError('exact close preparation inputs required')
        records=ledger.all_orders();group=project_groups(records)[family_id]
        family=next(r for r in records if r['intent_id']==family_id);native=family['native_stop']
        repair_id=None
        if native['stop'] is None or native['stop']['status'] not in ACTIVE_STOP:
            from .repair_orders import active_repair
            repair_id=active_repair(family)['identity']
        slot,child=_target(family,repair_id)
        session=calendar.at(now)
        if (reason not in ('quantitative','account_risk') or prepared.code_sha256!=view_version()
                or prepared.orders_digest!=_hash(_json(records)) or session is None
                or (session.closes-now).total_seconds()<60 or not group.parent_terminal or group.remaining_quantity<=0
                or not _target_valid(family,group,repair_id,now)
                or prepared.quote.symbol!=group.symbol or not prepared.portfolio.reconciled
                or not 0<=(now-prepared.portfolio.as_of).total_seconds()<=30):
            raise ValueError('fresh protected close family required')
        target=[r for r in prepared.portfolio.stop_reservations if r.parent_intent_id==family_id]
        if (len(target)!=1 or target[0].remaining_quantity!=group.remaining_quantity
                or target[0].child_order_sha256!=_hash(child['broker_order_id'])):
            raise ValueError('owned target stop reservation missing')
        price=prepared.quote.bid;_tick(price)
        if price*group.remaining_quantity>1000:raise ValueError('full close exceeds fixed transport budget')
        preview_portfolio=replace(prepared.portfolio,stop_reservations=tuple(
            r for r in prepared.portfolio.stop_reservations if r.parent_intent_id!=family_id))
        identity=_hash(_json(['native-close-v1',family_id,prepared.orders_digest,now,reason]))
        intent=OrderIntent(identity,group.symbol,'sell',group.remaining_quantity,price,None,now+timedelta(seconds=20))
        return ClosePreparation(prepared,replace(prepared,portfolio=preview_portfolio),intent,family_id,reason,
            target_repair_id=repair_id)
    except Exception:raise RiskViewError('native_close_preparation_unavailable') from None


def decision_binding(ledger,plan):
    from .decision_journal import _portfolio as encode
    if type(ledger) is not OrderLedger or type(plan) is not ClosePreparation:raise ValueError('exact close plan required')
    prefix,records=ledger.snapshot_prefix()
    if _hash(_json(records))!=plan.original.orders_digest:raise ValueError('close source changed')
    group=project_groups(records)[plan.family_id]
    return dict(schema=1,purpose='cancel_stop_then_reconcile',family_id=plan.family_id,
        remaining_quantity=group.remaining_quantity,prefix=prefix,orders_sha256=plan.original.orders_digest,
        projection=cleanup_projection(records,plan.family_id),
        intent_present=any(r['intent_id']==plan.intent.intent_id for r in records),
        original_portfolio=encode(plan.original.portfolio),reason=plan.reason,
        max_unprotected_seconds=plan.max_unprotected_seconds,target_repair_id=plan.target_repair_id)


def validate_binding(inputs):
    raw=inputs['close']
    if raw is None:
        if inputs['basis'] in ('native_close','native_risk_close'):raise ValueError('close binding absent')
        return None
    keys={'schema','purpose','family_id','remaining_quantity','prefix','orders_sha256','projection',
        'intent_present','original_portfolio','reason','max_unprotected_seconds','target_repair_id'}
    if (type(raw) is not dict or set(raw)!=keys or type(raw['schema']) is not int or raw['schema']!=1
            or raw['purpose']!='cancel_stop_then_reconcile' or raw['reason'] not in ('quantitative','account_risk')
            or inputs['basis']!=('native_close' if raw['reason']=='quantitative' else 'native_risk_close')
            or inputs['requested_action']!='sell' or inputs['cleanup'] is not None or inputs['intent'] is None
            or type(raw['intent_present']) is not bool or type(raw['max_unprotected_seconds']) is not int
            or raw['max_unprotected_seconds']!=30 or type(raw['projection']) is not list):
        raise ValueError('close binding contract')
    from .risk import _identity
    _identity(raw['family_id']);_identity(raw['orders_sha256']);prefix=raw['prefix']
    if raw['target_repair_id'] is not None:_identity(raw['target_repair_id'])
    if (type(prefix) is not dict or set(prefix)!={'sequence','digest'} or type(prefix['sequence']) is not int
            or not 1<=prefix['sequence']<=100000):raise ValueError('close source prefix')
    _identity(prefix['digest']);group=project_groups(raw['projection'])[raw['family_id']]
    family=next(r for r in raw['projection'] if r['intent_id']==raw['family_id']);native=family['native_stop']
    slot,child=_target(family,raw['target_repair_id'])
    original=_portfolio(raw['original_portfolio']);preview=_portfolio(inputs['risk']['portfolio'])
    target=[r for r in original.stop_reservations if r.parent_intent_id==raw['family_id']]
    intent=OrderIntent(**inputs['intent'])
    if (not group.parent_terminal or group.remaining_quantity<=0
            or not _target_valid(family,group,raw['target_repair_id'],utc_time(inputs['observed_at']))
            or type(raw['remaining_quantity']) is not int
            or raw['remaining_quantity']!=group.remaining_quantity or len(target)!=1
            or target[0].remaining_quantity!=group.remaining_quantity or target[0].child_order_sha256!=_hash(child['broker_order_id'])
            or intent.side!='sell' or intent.quantity!=group.remaining_quantity or intent.symbol!=group.symbol
            or intent.stop_price_usd is not None or intent.quantity*intent.limit_price_usd>1000
            or preview!=replace(original,stop_reservations=tuple(r for r in original.stop_reservations
                if r.parent_intent_id!=raw['family_id']))
            or inputs['prepared']['orders_digest']!=raw['orders_sha256']
            or _normalize(cleanup_projection(raw['projection'],raw['family_id']))!=raw['projection']):
        raise ValueError('close preview/source mismatch')
    _tick(intent.limit_price_usd)
    return group


def risk_trigger(inputs,limits):
    original=_portfolio(inputs['close']['original_portfolio'])
    quotes={q.symbol:q for q in original.valuation_quotes}
    with localcontext() as ctx:
        ctx.prec=40
        equity=min(original.equity_usd,original.cash_usd+sum((p.quantity*min(p.mark_price_usd,quotes[p.symbol].bid)
            for p in original.positions),Decimal(0)))
        return (original.day_start_equity_usd-equity>=limits.max_daily_loss_usd
            or original.high_water_equity_usd-equity>=limits.max_drawdown_usd)


@dataclass(frozen=True)
class CloseClaim:
    family_id: str=field(repr=False)
    identity: str=field(repr=False)
    target_id: str=field(repr=False)
    claimed_at: datetime
    orders_sha256: str=field(repr=False)
    decision_floor: datetime
    target_repair_id: str|None=field(default=None,repr=False)


def claim_close(ledger,plan,decision,control_snapshot,*,now):
    now=utc_time(now)
    authorized=decision.read()
    decision_floor=max(utc_time(authorized['recorded_at']),utc_time(authorized['evaluated_at']),
        utc_time(authorized['inputs']['observed_at']))
    if now<decision_floor:raise LedgerError('close_claim_before_committed_decision')
    with ledger._db(True) as db:
        ledger._verify(db)
        records=[ledger._load(db,r[0]) for r in db.execute('SELECT intent_id FROM orders ORDER BY intent_id')]
        if _hash(_json(records))!=plan.original.orders_digest:raise LedgerError('close_claim_source_changed')
        record=ledger._load(db,plan.family_id);slot,child=_target(record,plan.target_repair_id);previous=slot.get('close')
        if (slot['stop_cancel_requested_at'] is not None or previous is not None and previous['outcome']!='not_sent'):
            return None
        generation=1 if previous is None else previous['generation']+1
        if generation>3:return None
        if (now<utc_time(record['updated_at']) or now>=plan.intent.expires_at
                or plan.original.code_sha256!=view_version()):raise LedgerError('close_claim_time_or_version_changed')
        close=dict(schema=1,identity=plan.intent.intent_id,generation=generation,outcome='claimed',
            decision_sequence=decision.sequence,decision_digest=decision.digest,target_id=child['broker_order_id'],
            quantity=plan.intent.quantity,claimed_at=_ts(now),deadline_at=_ts(now+timedelta(seconds=30)),
            decision_expires_at=_ts(plan.intent.expires_at),
            decision_floor=_ts(decision_floor),
            control_digest=control_snapshot.digest,control_revision=control_snapshot.revision,
            observed_at=_ts(now),http_status=None,vetoes=[])
        slot['close']=close;slot['stop_cancel_requested_at']=_ts(now);record['updated_at']=_ts(now)
        ledger._persist(db,record,'native_close_claim',now)
        records=[ledger._load(db,r[0]) for r in db.execute('SELECT intent_id FROM orders ORDER BY intent_id')]
        return CloseClaim(plan.family_id,close['identity'],close['target_id'],now,_hash(_json(records)),decision_floor,plan.target_repair_id)


def finish_close(ledger,ticket,*,outcome,observed_at,http_status=None,vetoes=()):
    observed=utc_time(observed_at)
    if type(ticket) is not CloseClaim or outcome not in ('not_sent','acknowledged','rejected','unknown'):
        raise LedgerError('close_result_invalid')
    if http_status is not None and (type(http_status) is not int or not 100<=http_status<=599):
        raise LedgerError('close_http_status_invalid')
    if (outcome=='not_sent' and (http_status is not None or not vetoes)
            or outcome=='acknowledged' and http_status!=204 or outcome=='rejected' and http_status!=422
            or outcome!='not_sent' and vetoes):raise LedgerError('close_result_outcome_mismatch')
    from .decision_journal import _codes
    try:_codes(vetoes)
    except Exception:raise LedgerError('close_veto_codes_invalid') from None
    with ledger._db(True) as db:
        ledger._verify(db);record=ledger._load(db,ticket.family_id);slot,child=_target(record,ticket.target_repair_id);close=slot['close']
        if (close['identity']!=ticket.identity or close['outcome']!='claimed' or close['target_id']!=ticket.target_id
                or utc_time(close['claimed_at'])!=ticket.claimed_at
                or slot['stop_cancel_requested_at']!=_ts(ticket.claimed_at)):raise LedgerError('close_claim_changed')
        floor=max(utc_time(record['updated_at']),ticket.claimed_at,ticket.decision_floor)
        close.update(outcome=outcome,observed_at=_ts(observed),http_status=http_status,vetoes=list(vetoes))
        if observed<floor:close['clock_regression']=True
        if outcome=='not_sent':slot['stop_cancel_requested_at']=None
        record['updated_at']=_ts(max(observed,floor))
        return ledger._persist(db,record,'native_close_result',max(observed,floor))


@dataclass(frozen=True)
class CloseResult:
    status: str
    remaining_quantity: int|None
    required_actions: tuple
    request_sent: bool
    target_terminal: bool=False

    def public_summary(self):
        return dict(mode='paper',phase='close_preparation',status=self.status,remaining_quantity=self.remaining_quantity,
            required_actions=list(self.required_actions),cancel_requests_sent=int(self.request_sent),
            orders_submitted=0,ready_for_execution=False,execution_enabled=False)


class NativeCloseGateway:
    def __init__(self,broker,*,monotonic=None):
        if type(broker) is not PaperBroker:raise BrokerError('exact_paper_broker_required')
        self.broker=broker;self._monotonic=monotonic or time.monotonic

    def recover(self,ledger,family_id,*,target_repair_id=None):
        """GET-only recovery is independent of model/decision archive availability."""
        try:
            if type(ledger) is not OrderLedger or _target(ledger.load(family_id),target_repair_id)[0].get('close') is None:
                raise ValueError('existing owned close preparation required')
            return self._recover(ledger,family_id,target_repair_id=target_repair_id)
        except Exception:raise BrokerError('native_close_recovery_unavailable') from None

    def _recover(self,ledger,family_id,*,sent=False,target_repair_id=None):
        actions=('pause_entries','reconcile_group','reconcile_account')
        try:
            record=NativePaperReader(self.broker,monotonic=self._monotonic).recover(ledger,family_id)
            if record is None:return CloseResult('unconfirmed_group',None,actions,sent)
            if target_repair_id is not None:
                from .repair_orders import RepairPaperReader
                record=RepairPaperReader(self.broker).recover(ledger,family_id,target_repair_id)
                if record is None:return CloseResult('unconfirmed_group',None,actions,sent)
            group=project_groups(ledger.all_orders())[family_id];native=record['native_stop'];slot,child=_target(record,target_repair_id)
            close=slot.get('close')
            if not native['child_visible'] or native['stop'] is None:
                return CloseResult('unconfirmed_group',None,actions,sent)
            now=utc_time(self.broker._clock());overdue=close is not None and now>=utc_time(close['deadline_at'])
            if group.remaining_quantity==0:
                return CloseResult('group_closed',0,('reconcile_account',),sent)
            from .repair_orders import CLOSED
            terminal=group.child_terminal if target_repair_id is None else child is not None and child['status'] in CLOSED
            if terminal:
                return CloseResult('recovery_overdue' if overdue else 'cleanup_required',group.remaining_quantity,
                    (*actions,'record_new_cleanup_decision'),sent,True)
            if close is None or close['outcome']=='not_sent':status='close_not_sent'
            elif close['outcome']=='rejected' and child['status'] in ACTIVE_STOP:status='close_rejected_protected'
            elif overdue:status='recovery_overdue'
            else:status='awaiting_stop_terminal'
            return CloseResult(status,group.remaining_quantity,actions,sent)
        except Exception:return CloseResult('unconfirmed_group',None,actions,sent)

    def request(self,ledger,plan,*,decision,journal,controls,control_snapshot,limits,calendar,book,baselines,sources=None):
        try:
            from .decision_journal import DecisionJournal,DecisionReceipt,_portfolio as encode
            from .orders import recheck_ticket
            for value,exact in ((ledger,OrderLedger),(plan,ClosePreparation),(journal,DecisionJournal),
                    (decision,DecisionReceipt),(controls,ControlJournal),(control_snapshot,ControlSnapshot),
                    (limits,RiskLimits),(calendar,Calendar),(book,AccountBook),(baselines,RiskBaselineStore)):
                if type(value) is not exact:raise ValueError('exact close dispatch dependencies required')
            actual=journal.replay(decision.sequence,ledger=ledger,sources=sources)
            row=actual.read();inputs=row['inputs']
            if (actual!=decision or row['outcome']['action']!='sell' or row['outcome']['vetoes']
                    or inputs['basis'] not in ('native_close','native_risk_close')
                    or inputs['close']['family_id']!=plan.family_id or inputs['intent']!=_normalize(_base(plan.intent,OrderIntent))
                    or inputs['close']['target_repair_id']!=plan.target_repair_id
                    or inputs['close']['original_portfolio']!=_normalize(encode(plan.original.portfolio))
                    or inputs['risk']['portfolio']!=_normalize(encode(plan.preview.portfolio))
                    or inputs['limits']!=_normalize(_base(limits,RiskLimits))
                    or inputs['calendar']!=_normalize(dict(start=calendar.start,end=calendar.end,
                        sessions=[_base(s,Session) for _,s in sorted(calendar.sessions.items())]))):
                raise ValueError('recorded close decision mismatch')
            existing=_target(ledger.load(plan.family_id),plan.target_repair_id)[0].get('close')
            if existing is not None and existing['outcome']!='not_sent':
                return self._recover(ledger,plan.family_id,target_repair_id=plan.target_repair_id)
            deadline=self._monotonic()+25
            ticket=claim_close(ledger,plan,decision,control_snapshot,now=utc_time(self.broker._clock()))
            if ticket is None:return self._recover(ledger,plan.family_id,target_repair_id=plan.target_repair_id)
            guard_at=utc_time(self.broker._clock())
            with ExitStack() as stack:
                try:current=stack.enter_context(controls.dispatch_lease(control_snapshot,now=guard_at))
                except ControlError:current=False
                valid=False
                try:
                    records=ledger.all_orders();snapshot=book.snapshot();baseline=baselines.replay()
                    valid=(_hash(_json(records))==ticket.orders_sha256 and snapshot.receipt.digest==plan.original.book_digest
                        and snapshot.receipt.result.reconciled and baseline.audit_sha256==plan.original.baseline_digest
                        and baseline.book_digest==snapshot.receipt.digest)
                except Exception:pass
                checked=utc_time(self.broker._clock());reasons=[]
                context={**inputs['risk'],'limits':inputs['limits'],'calendar':inputs['calendar']}
                try:preview=recheck_ticket(dict(intent=inputs['intent'],context=context),now=checked);reasons.extend(preview.violations)
                except Exception:reasons.append('close_risk_unavailable')
                if not valid:reasons.append('close_source_changed')
                if checked<max(guard_at,ticket.claimed_at,ticket.decision_floor):reasons.append('close_clock_regression')
                session=calendar.at(checked)
                if session is None or (session.closes-checked).total_seconds()<60:reasons.append('close_session_margin')
                if self._monotonic()>=deadline:reasons.append('close_deadline')
                if not current or control_snapshot.digest!=inputs['controls']['digest']:reasons.append('operator_control_changed')
                elif control_snapshot.halt_orders:reasons.append('all_orders_halted')
                if inputs['basis']=='native_close' and (inputs['source'] is None
                        or not 0<=checked.timestamp()-inputs['source']['quant']['record']['bar_ends_at']<=1020):
                    reasons.append('close_model_stale')
                if reasons:
                    finish_close(ledger,ticket,outcome='not_sent',observed_at=checked,vetoes=tuple(dict.fromkeys(reasons)))
                    return self._recover(ledger,plan.family_id,target_repair_id=plan.target_repair_id)
                outcome='unknown';status=None
                try:
                    response=self.broker._request('DELETE','/v2/orders/'+ticket.target_id)
                    if type(response.status) is int and 100<=response.status<=599:status=response.status
                    if status==204:outcome='acknowledged'
                    elif status==422:outcome='rejected'
                except Exception:pass
            finish_close(ledger,ticket,outcome=outcome,observed_at=utc_time(self.broker._clock()),http_status=status)
            if self._monotonic()>=deadline:
                return CloseResult('unconfirmed_group',None,('pause_entries','reconcile_group','reconcile_account'),True)
            return self._recover(ledger,plan.family_id,sent=True,target_repair_id=plan.target_repair_id)
        except Exception:raise BrokerError('native_close_unavailable_or_unknown') from None
