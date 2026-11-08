"""Group-aware paper entry reservation. No provider I/O or execution service.

Committed model/quant decisions are references, not strategy acceptance. The
production coordinator must verify acceptance and serialize mutations. This
module independently reconstructs account risk before reserving or claiming.
"""
from dataclasses import dataclass,field

from .accountbook import AccountBook
from .controls import ControlSnapshot
from .data import Calendar,Session,utc_time
from .decision_journal import DecisionJournal,DecisionReceipt,ReplaySources,_portfolio
from .native_cleanup import _records
from .native_risk_view import prepare_native_risk_view
from .orders import OrderLedger,LedgerError,_base,_normalize,_hash,_json,_ts,_code_hash
from .protection import NativeStopPlan
from .reconciler import ReconciliationCycle
from .risk import OrderIntent,RiskLimits,RiskState
from .risk_baselines import RiskBaselineStore
from .risk_view import PreparedRiskView

_RISK_TOKEN=object()


@dataclass(frozen=True)
class NativeEntrySources:
    prepared:PreparedRiskView=field(repr=False)
    cycle:ReconciliationCycle=field(repr=False)
    book:AccountBook=field(repr=False)
    baselines:RiskBaselineStore=field(repr=False)
    journal:DecisionJournal=field(repr=False)
    replay_sources:ReplaySources=field(repr=False)
    control_snapshot:ControlSnapshot=field(repr=False)
    limits:RiskLimits
    calendar:Calendar=field(repr=False)


def _dependencies(ledger,sources):
    if type(ledger) is not OrderLedger or type(sources) is not NativeEntrySources:
        raise ValueError('exact native entry sources required')
    for name,exact in (('prepared',PreparedRiskView),('cycle',ReconciliationCycle),('book',AccountBook),
            ('baselines',RiskBaselineStore),('journal',DecisionJournal),('replay_sources',ReplaySources),
            ('control_snapshot',ControlSnapshot),('limits',RiskLimits),('calendar',Calendar)):
        if type(getattr(sources,name)) is not exact:raise ValueError('exact native entry dependency required')


def _check_sources(ledger,sources,now):
    _dependencies(ledger,sources)
    p=sources.prepared;c=sources.control_snapshot
    if c.pause_entries or c.halt_orders or utc_time(c.updated_at)>now:
        raise ValueError('current entry controls prohibit admission')
    rebuilt=prepare_native_risk_view(book=sources.book,ledger=ledger,baselines=sources.baselines,
        cycle=sources.cycle,quotes=p.portfolio.valuation_quotes,symbol=p.quote.symbol,
        operator_state=p.state,calendar=sources.calendar,now=now)
    if rebuilt!=p:raise ValueError('native entry preparation does not match actual sources')
    return p


def _match_decision(intent,actual,sources,now,*,compare_prepared=True):
    if type(actual) is not DecisionReceipt:raise ValueError('committed entry decision required')
    row=actual.read();inputs=row['inputs'];p=sources.prepared
    calendar=dict(start=sources.calendar.start,end=sources.calendar.end,
        sessions=[_base(s,Session) for s in sorted(sources.calendar.sessions.values(),key=lambda s:s.day)])
    prepared={k:getattr(p,k) for k in ('book_digest','baseline_digest','orders_digest','quotes_digest','code_sha256')}
    prepared['findings']=list(p.findings)
    if (row['outcome']['action']!='buy' or row['outcome']['vetoes']
            or inputs['basis']!='quant_model' or inputs['requested_action']!='buy'
            or inputs['intent']!=_normalize(_base(intent,OrderIntent))
            or compare_prepared and (inputs['prepared']!=prepared
                or inputs['risk']!=_normalize(dict(portfolio=_portfolio(p.portfolio),
                    quote=_base(p.quote,type(p.quote)),state=_base(p.state,RiskState))))
            or inputs['controls']!=_normalize(_base(sources.control_snapshot,ControlSnapshot))
            or inputs['limits']!=_normalize(_base(sources.limits,RiskLimits))
            or inputs['calendar']!=_normalize(calendar)
            or max(utc_time(row['recorded_at']),utc_time(inputs['observed_at']))>now):
        raise ValueError('entry decision does not authorize this reservation')
    return dict(sequence=actual.sequence,decision_digest=actual.digest)


def _decision(ledger,intent,receipt,sources,now):
    if type(receipt) is not DecisionReceipt:raise ValueError('committed entry decision required')
    actual=sources.journal.replay(receipt.sequence,sources=sources.replay_sources,ledger=ledger)
    if actual!=receipt:raise ValueError('committed entry receipt mismatch')
    return _match_decision(intent,actual,sources,now)


def _pending_close(records):
    for r in records:
        n=r.get('native_stop')
        if n is None:continue
        for slot in (n,*n.get('repairs',[])):
            c=slot.get('close')
            if c is not None and c['outcome'] not in ('not_sent','rejected') and c.get('recovery',{}).get('complete') is not True:
                raise ValueError('unfinished close task prohibits new entry')


def admit_entry(ledger,intent,*,decision,sources,now,time_in_force='day'):
    try:
        now=utc_time(now)
        _dependencies(ledger,sources)
        if type(intent) is not OrderIntent or intent.side!='buy':raise ValueError('exact long entry required')
        plan=NativeStopPlan.from_intent(intent,time_in_force=time_in_force)
        link=_decision(ledger,intent,decision,sources,now)
        with ledger._db(True) as db:
            records=_records(ledger,db);existing=ledger._load(db,intent.intent_id,missing=True)
            serialized=_normalize(_base(intent,OrderIntent))
            if existing is not None:
                if existing['intent']!=serialized or existing.get('native_entry')!=link:
                    raise ValueError('native entry idempotency conflict')
                return existing
            if _hash(_json(records))!=sources.prepared.orders_digest:raise ValueError('native entry ledger changed')
            _pending_close(records);p=_check_sources(ledger,sources,now)
            risk,context=ledger._risk(db,intent,portfolio=p.portfolio,quote=p.quote,state=p.state,
                limits=sources.limits,calendar=sources.calendar,now=now,_native_entry=_RISK_TOKEN)
            record=dict(schema=1,intent_id=intent.intent_id,client_order_id='pb3-'+intent.intent_id[:40],
                intent=serialized,state='reserved' if risk.allowed else 'rejected_by_risk',risk=_base(risk,type(risk)),
                filled_quantity=0,filled_notional_usd='0',broker_order_id=None,broker_created_at=None,
                broker_updated_at=None,submission_claimed_at=None,cancel_requested_at=None,
                created_at=_ts(now),updated_at=_ts(now),context=context,code_sha256=_code_hash(),native_entry=link,
                native_stop=dict(schema=1,plan=_normalize(_base(plan,NativeStopPlan)),parent=None,stop=None,
                    observed_at=None,child_visible=False,stop_cancel_requested_at=None))
            return ledger._persist(db,record,'native_entry_admission',now)
    except Exception:raise LedgerError('native_entry_admission_failed') from None


@dataclass(frozen=True)
class NativeEntryDispatch:
    payload:str=field(repr=False)
    orders_sha256:str=field(repr=False)
    book_digest:str=field(repr=False)
    baseline_digest:str=field(repr=False)
    code_sha256:str=field(repr=False)

    def read(self):
        from .alpaca import decode_json
        return decode_json(self.payload)


def claim_entry(ledger,identity,*,sources,now):
    try:
        now=utc_time(now);_dependencies(ledger,sources)
        with ledger._db(True) as db:
            records=_records(ledger,db);record=ledger._load(db,identity)
            if record.get('native_entry') is None:raise ValueError('dedicated native entry required')
            if record['state']!='reserved':return None
            if _hash(_json(records))!=sources.prepared.orders_digest:raise ValueError('entry claim ledger changed')
            _pending_close(records);p=_check_sources(ledger,sources,now)
            intent=OrderIntent(**record['intent']);link=record['native_entry']
            receipt=sources.journal.replay(link['sequence'],sources=sources.replay_sources,ledger=ledger)
            if _match_decision(intent,receipt,sources,now,compare_prepared=False)!=link:
                raise ValueError('original entry decision changed')
            if NativeStopPlan.from_intent(intent,time_in_force=record['native_stop']['plan']['time_in_force'])!=NativeStopPlan(**record['native_stop']['plan']):
                raise ValueError('attached entry protection changed')
            risk,context=ledger._risk(db,intent,portfolio=p.portfolio,quote=p.quote,state=p.state,
                limits=sources.limits,calendar=sources.calendar,now=now,exclude=identity,_native_entry=_RISK_TOKEN)
            record.update(state='submission_unknown' if risk.allowed else 'rejected_by_risk',
                risk=_base(risk,type(risk)),context=context,updated_at=_ts(now),
                submission_claimed_at=_ts(now) if risk.allowed else None)
            saved=ledger._persist(db,record,'native_entry_claim' if risk.allowed else 'native_entry_veto',now)
            if not risk.allowed:return None
            return NativeEntryDispatch(_json(saved),_hash(_json(_records(ledger,db))),p.book_digest,
                p.baseline_digest,_code_hash())
    except Exception:raise LedgerError('native_entry_claim_failed') from None


def validate_dispatch(ledger,envelope,sources):
    """Finish disk/source validation before the gateway samples dispatch time."""
    try:
        _dependencies(ledger,sources)
        if type(envelope) is not NativeEntryDispatch:return False
        ticket=envelope.read();context=ticket['context'];snapshot=sources.book.snapshot();baseline=sources.baselines.replay()
        link=ticket['native_entry'];receipt=sources.journal.replay(link['sequence'],sources=sources.replay_sources,ledger=ledger)
        if _match_decision(OrderIntent(**ticket['intent']),receipt,sources,
                utc_time(ticket['submission_claimed_at']),compare_prepared=False)!=link:return False
        calendar=dict(start=sources.calendar.start,end=sources.calendar.end,
            sessions=[_base(s,Session) for s in sorted(sources.calendar.sessions.values(),key=lambda s:s.day)])
        return (envelope.code_sha256==_code_hash()
            and _hash(_json(ledger.all_orders()))==envelope.orders_sha256
            and snapshot.receipt.digest==envelope.book_digest and snapshot.receipt.result.reconciled
            and baseline.audit_sha256==envelope.baseline_digest and baseline.book_digest==snapshot.receipt.digest
            and context['limits']==_normalize(_base(sources.limits,RiskLimits))
            and context['calendar']==_normalize(calendar))
    except Exception:return False
