"""Complete private decision traces and linked execution/account observations.

This is an audit store, not strategy acceptance or order authority. An acceptance
hash records a coordinator's artifact reference; the coordinator must actually
verify that artifact, hold the global lock and recheck current inputs/controls
before network I/O. No method here submits, cancels or releases reservations.
"""
from contextlib import closing,contextmanager
from dataclasses import dataclass,field,fields,replace
from datetime import date,datetime,timezone
from decimal import Decimal
import hashlib,os,re,sqlite3
from pathlib import Path

from .accountbook import AccountBook
from .controls import ControlSnapshot
from .data import Calendar,Session,utc_time,DataQualityError
from .execution_policy import ExecutionPolicy,PolicySnapshot
from .orders import OrderLedger,_base,_json,_normalize,recheck_ticket
from .replay_bridge import ReplayBridge,ReplayEvidence,strict_json
from .risk import Portfolio,Position,PendingOrder,StopReservation,Quote,RiskState,RiskLimits,OrderIntent,_identity,_symbol
from .risk_view import PreparedRiskView,_version as view_version

MAX_EVENTS=100000
MAX_BYTES=256*1024*1024


class DecisionError(RuntimeError):pass


def _hash(value):return hashlib.sha256(value.encode()).hexdigest()


def _version():
    digest=hashlib.sha256()
    for name in ('decision_journal.py','risk.py','risk_view.py','controls.py','execution_policy.py',
                 'orders.py','native_orders.py','native_broker.py','protection.py','accountbook.py',
                 'accounting.py','replay_bridge.py','replay_worker.py','data.py','native_cleanup.py','cleanup_execution.py','native_close.py','cleanup_recovery.py','repair_orders.py','repair_decisions.py','repair_execution.py','compensation.py','native_admission.py','native_execution.py'):
        digest.update(name.encode()+b'\0'+Path(__file__).with_name(name).read_bytes()+b'\0')
    return digest.hexdigest()


def _codes(values):
    if type(values) not in (tuple,list) or len(values)>100:
        raise ValueError('bounded reason codes required')
    if any(not isinstance(v,str) or not re.fullmatch('[a-z][a-z0-9_]{0,79}',v) for v in values):
        raise ValueError('fixed-format reason codes required')
    return list(values)


def _proof(value):
    from .replay_worker import record_shape,QUANT,MODEL,ATTEMPT
    expected={'schema','kind','release_sha256','timezone_sha256','bridge_sha256','quant','model','attempt','summary'}
    if type(value) is not dict or set(value)!=expected or type(value['schema']) is not int or value['schema']!=1:
        raise ValueError('source proof contract')
    if value['kind'] not in ('quant','model'):raise ValueError('source proof kind')
    for key in ('release_sha256','timezone_sha256','bridge_sha256'):_identity(value[key])
    for name,kind,keys in (('quant','decision',QUANT),('model','model_review',MODEL),('attempt','model_attempt',ATTEMPT)):
        witness=value[name]
        if witness is None:
            if name=='quant' or value['kind']=='model':raise ValueError('source witness required')
            continue
        if type(witness) is not dict or set(witness)!={'sequence','previous_digest','digest','payload','record','head'}:
            raise ValueError('source witness contract')
        head=witness['head']
        if type(head) is not dict or set(head)!={'sequence','digest'}:raise ValueError('source head contract')
        for number in (witness['sequence'],head['sequence']):
            if type(number) is not int or not 1<=number<=100000:raise ValueError('source sequence')
        if witness['sequence']>head['sequence']:raise ValueError('source head before record')
        for item in (witness['previous_digest'],witness['digest'],head['digest']):_identity(item)
        raw=witness['payload']
        if not isinstance(raw,str) or len(raw.encode())>65536:raise ValueError('source payload budget')
        row=strict_json(raw)
        if row!=witness['record'] or type(row) is not dict or set(row)!=keys or row['kind']!=kind:
            raise ValueError('source record shape')
        record_shape(row,kind)
        if _hash(witness['previous_digest']+'\0'+raw)!=witness['digest']:raise ValueError('source witness digest')
        if witness['sequence']==head['sequence'] and witness['digest']!=head['digest']:raise ValueError('source head digest')
    if value['kind']=='model' and value['model']['head']!=value['attempt']['head']:
        raise ValueError('model attempt head mismatch')
    q=value['quant']['record']
    if value['kind']=='quant':
        expected={key:q[key] for key in ('features','entry_candidate','exit_candidate','final_action','vetoes')}
    else:
        from .llm import validate_receipt
        model=value['model']['record'];receipt=validate_receipt(model['receipt'])
        vetoes=['strategy_not_accepted','execution_not_enabled']
        if ((receipt['action']=='buy' and not q['entry_candidate']) or (receipt['action']=='sell' and not q['exit_candidate'])):
            vetoes.append('model_outside_candidate')
        expected=dict(symbol=q['symbol'],bar_ends_at=q['bar_ends_at'],reviewed_at=model['generated_at'],
            context='historical_review' if model['historical'] else 'closed_bar_review',
            input_current_at_review=model['input_current_at_review'],signal_usable=False,action=receipt['action'],
            reason=receipt['reason'],model=receipt['model'],final_action='hold',vetoes=vetoes)
    if type(value['summary']) is not dict or value['summary']!=expected:raise ValueError('source summary schema mismatch')
    return value


@dataclass(frozen=True)
class ReplaySources:
    bridge: ReplayBridge=field(repr=False)
    quant_archive: object=field(repr=False)
    model_archive: object|None=field(default=None,repr=False)

    def __post_init__(self):
        if type(self.bridge) is not ReplayBridge:raise DecisionError('exact_original_replay_bridge_required')

    def verify(self,recorded):
        _proof(recorded)
        if recorded['kind']=='model':
            if self.model_archive is None:raise DecisionError('model_archive_required')
            actual=self.bridge.review(self.model_archive,recorded['model']['record']['identity'],
                source_archive=self.quant_archive,expected_quant_head=recorded['quant']['head'],
                expected_model_head=recorded['model']['head']).read()
        else:actual=self.bridge.quant(self.quant_archive,recorded['quant']['record']['identity'],
            expected_head=recorded['quant']['head']).read()
        _proof(actual)
        for key in ('kind','release_sha256','timezone_sha256','bridge_sha256','summary'):
            if actual[key]!=recorded[key]:raise DecisionError('original_replay_binding_mismatch')
        for name in ('quant','model','attempt'):
            old,new=recorded[name],actual[name]
            if old is None:
                if new is not None:raise DecisionError('unexpected_source_witness')
                continue
            if any(old[k]!=new[k] for k in ('sequence','previous_digest','digest','payload','record')):
                raise DecisionError('original_source_record_changed')
            if new['head']['sequence']<old['head']['sequence']:
                raise DecisionError('source_archive_truncated')
            if new['head']['sequence']==old['head']['sequence'] and new['head']['digest']!=old['head']['digest']:
                raise DecisionError('source_head_changed')


def strategy_digest(source):
    value=_proof(source.read() if type(source) is ReplayEvidence else source)
    row=value['quant']['record']
    return _hash(_json(dict(policy=row['policy'],quant_code_sha256=row['code_sha256'],formula=row['features']['formula_version'])))


def _portfolio(value):
    row=_base(value,Portfolio)
    row.update(positions=[_base(v,Position) for v in value.positions],
        pending_orders=[_base(v,PendingOrder) for v in value.pending_orders],
        stop_reservations=[_base(v,StopReservation) for v in value.stop_reservations],
        valuation_quotes=[_base(v,Quote) for v in value.valuation_quotes])
    return row


@dataclass(frozen=True)
class DecisionFrame:
    payload: str=field(repr=False)

    def read(self):return strict_json(self.payload)


def capture_decision(*,source,prepared,intent,policy,controls,limits,calendar,now,requested_action,
                     authorization_sha256=None,basis='quant_model',symbol=None,cleanup=None,close=None):
    try:
        for value,exact in ((source,ReplayEvidence),(prepared,PreparedRiskView),(intent,OrderIntent),
                            (policy,PolicySnapshot),(controls,ControlSnapshot)):
            if value is not None and type(value) is not exact:raise ValueError('exact decision contracts required')
        if type(limits) is not RiskLimits or type(calendar) is not Calendar:raise ValueError('risk specification required')
        if requested_action not in ('buy','sell','hold','wait') or basis not in ('quant_model','risk_reduction','unavailable','native_cleanup','native_close','native_risk_close'):
            raise ValueError('decision classification required')
        if authorization_sha256 is not None:_identity(authorization_sha256)
        value=_proof(source.read()) if source is not None else None
        selected=symbol or (intent.symbol if intent else value['quant']['record']['symbol'] if value else prepared.quote.symbol if prepared else None)
        _symbol(selected)
        risk=None;bindings=None
        if prepared is not None:
            bindings={key:getattr(prepared,key) for key in ('book_digest','baseline_digest','orders_digest','quotes_digest','code_sha256')}
            for digest in bindings.values():_identity(digest)
            bindings['findings']=_codes(prepared.findings)
            risk=dict(portfolio=_portfolio(prepared.portfolio),quote=_base(prepared.quote,Quote),state=_base(prepared.state,RiskState))
        inputs=dict(schema=1,observed_at=utc_time(now),symbol=selected,requested_action=requested_action,basis=basis,
            authorization_sha256=authorization_sha256,source=value,intent=_base(intent,OrderIntent) if intent else None,
            policy=dict(policy=_base(policy.policy,ExecutionPolicy),digest=policy.digest) if policy else None,
            controls=_base(controls,ControlSnapshot) if controls else None,prepared=bindings,risk=risk,cleanup=cleanup,close=close,
            limits=_base(limits,RiskLimits),calendar=dict(start=calendar.start,end=calendar.end,
                sessions=[_base(v,Session) for _,v in sorted(calendar.sessions.items())]))
        normalized=_normalize(inputs)
        if len(_json(normalized).encode())>512*1024:raise ValueError('decision frame budget')
        _validate_inputs(normalized)
        return DecisionFrame(_json(normalized))
    except Exception:raise DecisionError('invalid_decision_input_frame') from None


def capture_cleanup_decision(*,ledger,prepared,intent,policy,controls,limits,calendar,now):
    try:
        from .native_cleanup import PreparedCleanupView,cleanup_projection
        if type(ledger) is not OrderLedger or type(prepared) is not PreparedCleanupView:
            raise ValueError('exact cleanup sources required')
        prefix,records=ledger.snapshot_prefix()
        if _hash(_json(records))!=prepared.risk.orders_digest:raise ValueError('cleanup ledger changed')
        binding=dict(schema=1,family_id=prepared.family_id,remaining_quantity=prepared.remaining_quantity,
            available_quantity=prepared.available_quantity,prefix=prefix,orders_sha256=prepared.risk.orders_digest,
            projection=cleanup_projection(records,prepared.family_id),
            intent_present=any(r['intent_id']==intent.intent_id for r in records))
        return capture_decision(source=None,prepared=prepared.risk,intent=intent,policy=policy,controls=controls,
            limits=limits,calendar=calendar,now=now,requested_action='sell',basis='native_cleanup',cleanup=binding)
    except Exception:raise DecisionError('invalid_cleanup_decision_frame') from None


def capture_close_decision(*,ledger,prepared,source,policy,controls,limits,calendar,now):
    try:
        from .native_close import ClosePreparation,decision_binding
        if type(prepared) is not ClosePreparation:raise ValueError('exact close preparation required')
        binding=decision_binding(ledger,prepared)
        return capture_decision(source=source,prepared=prepared.preview,intent=prepared.intent,policy=policy,
            controls=controls,limits=limits,calendar=calendar,now=now,requested_action='sell',close=binding,
            authorization_sha256=policy.policy.acceptance_sha256 if policy is not None else None,
            basis='native_close' if prepared.reason=='quantitative' else 'native_risk_close')
    except Exception:raise DecisionError('invalid_close_decision_frame') from None


def _cleanup_binding(inputs):
    from .native_cleanup import cleanup_projection,project_groups
    raw=inputs['cleanup']
    if raw is None:
        if inputs['basis']=='native_cleanup':raise ValueError('cleanup evidence missing')
        return None
    if (inputs['basis']!='native_cleanup' or inputs['requested_action']!='sell' or inputs['intent'] is None
            or inputs['intent']['side']!='sell' or inputs['intent']['stop_price_usd'] is not None
            or type(raw) is not dict or set(raw)!={'schema','family_id','remaining_quantity','available_quantity',
                'prefix','orders_sha256','projection','intent_present'} or type(raw['schema']) is not int or raw['schema']!=1
            or type(raw['intent_present']) is not bool or type(raw['projection']) is not list):
        raise ValueError('cleanup scope contract')
    _identity(raw['family_id']);_identity(raw['orders_sha256'])
    prefix=raw['prefix']
    if (type(prefix) is not dict or set(prefix)!={'sequence','digest'} or type(prefix['sequence']) is not int
            or not 1<=prefix['sequence']<=100000):raise ValueError('cleanup prefix contract')
    _identity(prefix['digest'])
    group=project_groups(raw['projection'])[raw['family_id']]
    if (inputs['prepared'] is None or raw['orders_sha256']!=inputs['prepared']['orders_digest']
            or _normalize(cleanup_projection(raw['projection'],raw['family_id']))!=raw['projection']
            or not group.parent_terminal or not group.child_terminal or group.remaining_quantity<=0
            or type(raw['remaining_quantity']) is not int or raw['remaining_quantity']!=group.remaining_quantity
            or type(raw['available_quantity']) is not int or raw['available_quantity']!=group.available_quantity
            or group.symbol!=inputs['symbol']):raise ValueError('cleanup source contract')
    return group


def _verify_cleanup_source(inputs,ledger):
    if inputs['cleanup'] is None:return
    from .native_cleanup import cleanup_projection
    if type(ledger) is not OrderLedger:raise ValueError('original cleanup ledger required')
    raw=inputs['cleanup'];records=ledger.records_at_prefix(raw['prefix'])
    if (_hash(_json(records))!=raw['orders_sha256']
            or _normalize(cleanup_projection(records,raw['family_id']))!=raw['projection']
            or any(r['intent_id']==inputs['intent']['intent_id'] for r in records)!=raw['intent_present']):
        raise ValueError('cleanup prefix source mismatch')


def _typed(cls,row,overrides=None):
    if type(row) is not dict or set(row)!={f.name for f in fields(cls)}:raise ValueError('unknown decision fields')
    value=cls(**{**{f.name:row[f.name] for f in fields(cls) if f.init},**(overrides or {})})
    return value


def _validate_inputs(value):
    if type(value) is not dict or set(value)!={'schema','observed_at','symbol','requested_action','basis','authorization_sha256',
            'source','intent','policy','controls','prepared','risk','limits','calendar','cleanup','close'} or type(value['schema']) is not int or value['schema']!=1:
        raise ValueError('decision frame schema')
    utc_time(value['observed_at']);_symbol(value['symbol'])
    if value['requested_action'] not in ('buy','sell','hold','wait') or value['basis'] not in ('quant_model','risk_reduction','unavailable','native_cleanup','native_close','native_risk_close'):
        raise ValueError('decision frame classification')
    if value['authorization_sha256'] is not None:_identity(value['authorization_sha256'])
    if value['source'] is not None:_proof(value['source'])
    if value['intent'] is not None:_typed(OrderIntent,value['intent'])
    _typed(RiskLimits,value['limits'])
    raw=value['calendar']
    if type(raw) is not dict or set(raw)!={'start','end','sessions'} or type(raw['sessions']) is not list or len(raw['sessions'])>400:
        raise ValueError('bounded calendar required')
    for session in raw['sessions']:
        if type(session) is not dict or set(session)!={'day','opens','closes'}:raise ValueError('session contract')
    calendar=Calendar([Session(date.fromisoformat(s['day']),s['opens'],s['closes']) for s in raw['sessions']],start=raw['start'],end=raw['end'])
    if value['policy'] is not None:
        raw=value['policy']
        if type(raw) is not dict or set(raw)!={'policy','digest'}:raise ValueError('policy snapshot contract')
        _identity(raw['digest']);_typed(ExecutionPolicy,raw['policy'])
    if value['controls'] is not None:_typed(ControlSnapshot,value['controls'])
    if (value['risk'] is None)!=(value['prepared'] is None):raise ValueError('risk binding missing')
    if value['risk'] is not None:
        binding=value['prepared']
        if type(binding) is not dict or set(binding)!={'book_digest','baseline_digest','orders_digest','quotes_digest','code_sha256','findings'}:
            raise ValueError('risk binding contract')
        for key in set(binding)-{'findings'}:_identity(binding[key])
        _codes(binding['findings'])
        context=value['risk']
        if type(context) is not dict or set(context)!={'portfolio','quote','state'}:raise ValueError('risk context contract')
        p=context['portfolio']
        portfolio=_typed(Portfolio,p,dict(positions=tuple(_typed(Position,v) for v in p['positions']),
            pending_orders=tuple(_typed(PendingOrder,v) for v in p['pending_orders']),
            stop_reservations=tuple(_typed(StopReservation,v) for v in p['stop_reservations']),
            valuation_quotes=tuple(_typed(Quote,v) for v in p['valuation_quotes'])))
        if _normalize(_portfolio(portfolio))!=p:raise ValueError('portfolio normalization mismatch')
        quote=_typed(Quote,context['quote'])
        if _normalize(_base(quote,Quote))!=context['quote']:raise ValueError('quote normalization mismatch')
        _typed(RiskState,context['state'],dict(seen_intent_ids=frozenset(context['state']['seen_intent_ids'])))
    _cleanup_binding(value)
    from .native_close import validate_binding
    validate_binding(value)
    return calendar


def _outcome(inputs,at,clock_regressed):
    calendar=_validate_inputs(inputs);at=utc_time(at);vetoes=set();risk_result=None
    try:session=calendar.at(at);calendar_known=True
    except DataQualityError:session=None;calendar_known=False;vetoes.add('calendar_unavailable')
    requested=inputs['requested_action'];intent=inputs['intent'];context=inputs['risk'];control=inputs['controls'];policy=inputs['policy']
    if clock_regressed or utc_time(inputs['observed_at'])>at:vetoes.add('decision_clock_regression')
    if policy is None:vetoes.add('policy_unavailable')
    elif not policy['policy']['execution_enabled']:vetoes.add('execution_disabled')
    if control is None:vetoes.add('controls_unavailable')
    elif utc_time(control['updated_at'])>at:vetoes.add('control_time_regression')
    if context is None:vetoes.add('risk_unavailable')
    elif intent is not None:
        effective=strict_json(_json(context))
        effective['state']['pause_entries']=effective['state']['pause_entries'] or control is None or control['pause_entries']
        effective['state']['halt_orders']=effective['state']['halt_orders'] or control is None or control['halt_orders']
        effective.update(limits=inputs['limits'],calendar=inputs['calendar'])
        if not calendar_known:vetoes.add('risk_unavailable')
        else:
            decision=recheck_ticket(dict(intent=intent,context=effective),now=at)
            risk_result=_normalize(_base(decision,type(decision)))
            if not decision.allowed:vetoes.add('risk_veto')
        if inputs['prepared']['code_sha256']!=view_version():vetoes.add('risk_view_version_mismatch')
        findings=set(inputs['prepared']['findings'])
        if inputs['basis'] in ('native_cleanup','native_close','native_risk_close'):
            from .native_cleanup import _WARNINGS
            findings-=(_WARNINGS|{'native_cleanup_only','previous_close_reference_missing',
                'calendar_reference_conflict','reference_calendar_unavailable','non_trading_day'})
        if findings:vetoes.add('risk_preparation_findings')
        if inputs['prepared']['quotes_digest']!=_hash(_json(context['portfolio']['valuation_quotes'])):
            vetoes.add('quote_binding_mismatch')
        if policy is not None and policy['policy']['risk_limits_sha256']!=_hash(_json(inputs['limits'])):
            vetoes.add('risk_limits_policy_mismatch')
    if inputs['cleanup'] is not None:
        group=_cleanup_binding(inputs)
        if inputs['cleanup']['intent_present']:vetoes.add('cleanup_decision_after_admission')
        if intent['quantity']>group.available_quantity:vetoes.add('native_cleanup_family_capacity')
        if group.observed_at is None or not 0<=(at-group.observed_at).total_seconds()<=30:vetoes.add('cleanup_group_stale')
    if inputs['close'] is not None:
        from .native_close import validate_binding,risk_trigger
        group=validate_binding(inputs)
        if inputs['close']['intent_present']:vetoes.add('close_intent_already_recorded')
        if group.observed_at is None or not 0<=(at-group.observed_at).total_seconds()<=30:vetoes.add('close_group_stale')
        if session is None or (session.closes-at).total_seconds()<60:vetoes.add('close_session_margin')
        if inputs['basis']=='native_risk_close' and not risk_trigger(inputs,RiskLimits(**inputs['limits'])):
            vetoes.add('close_risk_trigger_not_met')
    source=inputs['source'];needs_model=requested=='buy' or (requested=='sell' and inputs['basis'] not in ('risk_reduction','native_cleanup','native_risk_close'))
    if requested in ('buy','sell') and (intent is None or intent['side']!=requested or intent['symbol']!=inputs['symbol']):
        vetoes.add('intent_action_mismatch')
    if needs_model:
        if source is None:vetoes.add('quantitative_input_unavailable')
        else:
            q=source['quant']['record'];model=source['model']
            if q['symbol']!=inputs['symbol']:vetoes.add('source_symbol_mismatch')
            if not q['entry_candidate' if requested=='buy' else 'exit_candidate']:vetoes.add('quantitative_candidate_missing')
            if model is None:vetoes.add('model_review_unavailable')
            else:
                row=model['record']
                if row['historical'] or row['input_current_at_review'] is not True:vetoes.add('historical_model_review')
                if (not q['generated_at']<=row['generated_at']<=at.timestamp()
                        or not 0<=at.timestamp()-q['bar_ends_at']<=1020 or session is None):vetoes.add('model_input_stale')
                if row['receipt']['action']!=requested:vetoes.add('model_does_not_support_candidate')
    if requested=='buy' or inputs['basis']=='native_close':
        if (policy is None or policy['policy']['strategy_sha256'] is None or policy['policy']['acceptance_sha256'] is None
                or inputs['authorization_sha256']!=policy['policy']['acceptance_sha256']):vetoes.add('strategy_acceptance_unavailable')
        if source is not None and policy is not None and policy['policy']['strategy_sha256'] not in (None,strategy_digest(source)):
            vetoes.add('strategy_definition_mismatch')
    held=context is not None and any(p['symbol']==inputs['symbol'] and p['quantity']>0 for p in context['portfolio']['positions'])
    action=('hold' if held else 'wait') if vetoes else requested
    return dict(action=action,vetoes=sorted(vetoes),risk=risk_result,authorization_is_reference_only=True)


def _exact(value,keys):
    if type(value) is not dict or set(value)!=set(keys):raise ValueError('unknown journal record fields')


def _decimal(value,*,signed=False):
    if (not isinstance(value,str) or len(value)>80
            or not re.fullmatch(r'-?(?:0|[1-9][0-9]{0,11})(?:\.[0-9]{1,40})?(?:E[+-]?[0-9]{1,2})?',value)):
        raise ValueError('bounded journal decimal required')
    number=Decimal(value)
    if not number.is_finite() or abs(number)>Decimal('1000000000000') or not -50<=number.as_tuple().exponent<=12:
        raise ValueError('journal decimal outside bounds')
    if not signed and number<0:raise ValueError('nonnegative journal amount required')


def _validate_event(record):
    base={'schema','identity','kind','code_sha256','recorded_at'}
    if type(record) is not dict or type(record.get('schema')) is not int or record['schema']!=1:raise ValueError('journal schema')
    _identity(record['identity']);_identity(record['code_sha256']);at=utc_time(record['recorded_at'])
    if record['kind']=='decision':
        _exact(record,base|{'evaluated_at','clock_regressed','inputs','outcome'})
        if type(record['clock_regressed']) is not bool or utc_time(record['evaluated_at'])>at:raise ValueError('decision audit time')
        _validate_inputs(record['inputs'])
        outcome=record['outcome'];_exact(outcome,{'action','vetoes','risk','authorization_is_reference_only'})
        if outcome['action'] not in ('buy','sell','hold','wait') or outcome['authorization_is_reference_only'] is not True:
            raise ValueError('decision outcome classification')
        _codes(outcome['vetoes'])
        risk=outcome['risk']
        if risk is not None:
            _exact(risk,{'allowed','violations','reserved_cash_usd','modeled_trade_loss_usd','gross_exposure_after_usd','mode'})
            if type(risk['allowed']) is not bool or risk['mode']!='paper':raise ValueError('risk outcome contract')
            _codes(risk['violations'])
            for key in ('reserved_cash_usd','modeled_trade_loss_usd','gross_exposure_after_usd'):_decimal(risk[key])
    elif record['kind']=='order_observation':
        _exact(record,base|{'observed_at','decision_identity','order','order_record_sha256','ledger_sha256','unexpected_after_veto','required_actions'})
        for key in ('decision_identity','order_record_sha256','ledger_sha256'):_identity(record[key])
        if type(record['unexpected_after_veto']) is not bool or utc_time(record['observed_at'])>at:raise ValueError('order observation contract')
        if type(record['required_actions']) is not list or not set(record['required_actions'])<={'pause_entries','reconcile_orders'}:
            raise ValueError('order recovery actions')
        order=record['order']
        _exact(order,{'intent_id','state','filled_quantity','filled_notional_usd','updated_at','broker_identity_present','stop_status','stop_filled_quantity','child_visible'})
        _identity(order['intent_id']);utc_time(order['updated_at']);_decimal(order['filled_notional_usd'])
        from .orders import ACTIVE,TERMINAL
        from .protection import STATUSES
        if order['state'] not in ACTIVE+TERMINAL or type(order['broker_identity_present']) is not bool:raise ValueError('order state contract')
        if type(order['filled_quantity']) is not int or not 0<=order['filled_quantity']<=1000000:raise ValueError('order filled quantity')
        if order['stop_status'] is not None and order['stop_status'] not in STATUSES:raise ValueError('stop state contract')
        if order['stop_filled_quantity'] is not None and (type(order['stop_filled_quantity']) is not int or not 0<=order['stop_filled_quantity']<=1000000):
            raise ValueError('stop fill quantity')
        if order['child_visible'] is not None and type(order['child_visible']) is not bool:raise ValueError('stop visibility contract')
        expected=_hash(_json(['decision_order_observation',record['decision_identity'],record['ledger_sha256'],record['order_record_sha256']]))
        if expected!=record['identity']:raise ValueError('order event identity')
    elif record['kind']=='account_observation':
        _exact(record,base|{'observed_at','decision_identity','source_book_digest','book_digest','book_sequence','account','required_actions'})
        for key in ('decision_identity','source_book_digest','book_digest'):_identity(record[key])
        if type(record['book_sequence']) is not int or not 1<=record['book_sequence']<=100000:raise ValueError('book sequence')
        if utc_time(record['observed_at'])>at:raise ValueError('account observation time')
        if type(record['required_actions']) is not list or not set(record['required_actions'])<={'pause_entries','reconcile_account'}:
            raise ValueError('account recovery actions')
        account=record['account']
        _exact(account,{'accounting_ok','source_as_of','cash_usd','equity_usd','profit_usd','realized_usd','unrealized_usd','fees_usd','scope'})
        if type(account['accounting_ok']) is not bool or account['scope']!='portfolio_observation_not_trade_attribution':raise ValueError('account outcome scope')
        utc_time(account['source_as_of'])
        for key in ('cash_usd','equity_usd','profit_usd','realized_usd','unrealized_usd','fees_usd'):
            if account['accounting_ok']:_decimal(account[key],signed=key in ('profit_usd','realized_usd','unrealized_usd'))
            elif account[key] is not None:raise ValueError('unverified account values must be absent')
        if record['identity']!=_hash(_json(['decision_account_observation',record['decision_identity'],record['book_digest']])):
            raise ValueError('account event identity')
    else:raise ValueError('unknown decision journal event')


@dataclass(frozen=True)
class DecisionReceipt:
    sequence: int
    digest: str
    payload: str=field(repr=False)

    def read(self):return strict_json(self.payload)

    def public_summary(self):
        row=self.read();decision=row['kind']=='decision'
        return dict(mode='paper',execution_enabled=False,orders_submitted=0,kind=row['kind'],
            symbol=row['inputs']['symbol'] if decision else None,action=row['outcome']['action'] if decision else None,
            vetoes=row['outcome']['vetoes'] if decision else [],audit_sha256=self.digest)


class DecisionJournal:
    def __init__(self,path,*,clock=None):
        self.path=Path(path).absolute();self._clock=clock or (lambda:datetime.now(timezone.utc))

    def initialize(self):
        try:
            if self.path.is_symlink():raise ValueError('journal link refused')
            if self.path.exists():self.verify();return
            self.path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600);os.close(fd)
            with closing(sqlite3.connect(self.path)) as db:
                db.executescript('''PRAGMA journal_mode=DELETE;PRAGMA synchronous=FULL;BEGIN IMMEDIATE;
                    CREATE TABLE metadata(version INTEGER NOT NULL);INSERT INTO metadata VALUES(1);
                    CREATE TABLE events(sequence INTEGER PRIMARY KEY,identity TEXT UNIQUE NOT NULL,kind TEXT NOT NULL,
                        recorded_at TEXT NOT NULL,payload TEXT NOT NULL,previous_digest TEXT NOT NULL,digest TEXT NOT NULL);COMMIT;''')
            if os.name!='nt':
                fd=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
        except Exception:raise DecisionError('decision_journal_initialization_failed') from None

    @contextmanager
    def _db(self,write):
        db=None
        try:
            if self.path.is_symlink() or not self.path.is_file() or self.path.stat().st_nlink!=1:raise ValueError('existing private journal required')
            db=sqlite3.connect(self.path.as_uri()+('?mode=rw' if write else '?mode=ro'),uri=True,timeout=10,isolation_level=None)
            db.row_factory=sqlite3.Row;db.execute('PRAGMA synchronous=FULL');db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            db.execute('SELECT name FROM sqlite_master LIMIT 1').fetchone()
            if db.execute('PRAGMA journal_mode').fetchone()[0]!='delete':raise ValueError('rollback journal required')
            if [r[0] for r in db.execute('SELECT version FROM metadata')]!=[1]:raise ValueError('journal schema mismatch')
            yield db;db.commit()
        except Exception:
            if db is not None:db.rollback()
            raise DecisionError('decision_journal_operation_failed') from None
        finally:
            if db is not None:db.close()

    def _verify(self,db):
        previous='0'*64;count=decisions=total=0;last_time=None;code=_version()
        for row in db.execute('SELECT * FROM events ORDER BY sequence'):
            count+=1;total+=len(row['payload'].encode())
            if count>MAX_EVENTS or total>MAX_BYTES or len(row['payload'].encode())>512*1024:
                raise ValueError('decision retention budget')
            if row['sequence']!=count or row['previous_digest']!=previous or _hash(previous+'\0'+row['payload'])!=row['digest']:
                raise ValueError('decision chain integrity mismatch')
            value=strict_json(row['payload']);_identity(row['identity']);at=utc_time(row['recorded_at'])
            _validate_event(value)
            if (value['schema']!=1 or value['code_sha256']!=code or value['identity']!=row['identity']
                    or value['kind']!=row['kind'] or value['recorded_at']!=row['recorded_at']
                    or (last_time is not None and at<last_time)):raise ValueError('decision event binding mismatch')
            if value['kind']=='decision':
                if value['identity']!=_hash(_json(['execution_decision',value['inputs']])):raise ValueError('decision input identity mismatch')
                decisions+=1
            elif value['kind'] not in ('order_observation','account_observation'):raise ValueError('unknown decision event')
            previous=row['digest'];last_time=at
        return dict(events=count,decisions=decisions,bytes=total,head=previous,last_time=last_time)

    def _persist(self,db,record,head):
        record=_normalize(record);_validate_event(record)
        payload=_json(record)
        if len(payload.encode())>512*1024 or head['bytes']+len(payload.encode())>MAX_BYTES or head['events']>=MAX_EVENTS:
            raise ValueError('decision retention budget')
        digest=_hash(head['head']+'\0'+payload);sequence=head['events']+1
        db.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?)',(sequence,record['identity'],record['kind'],record['recorded_at'],payload,head['head'],digest))
        return DecisionReceipt(sequence,digest,payload)

    @staticmethod
    def _receipt(row):return DecisionReceipt(row['sequence'],row['digest'],row['payload'])

    def append(self,frame,*,sources=None,ledger=None):
        try:
            if type(frame) is not DecisionFrame:raise ValueError('exact decision frame required')
            inputs=frame.read();_validate_inputs(inputs)
            _verify_cleanup_source(inputs,ledger)
            if inputs['close'] is not None:_verify_cleanup_source({**inputs,'cleanup':inputs['close']},ledger)
            if inputs['source'] is not None:
                if type(sources) is not ReplaySources:raise ValueError('original source replay required')
                sources.verify(inputs['source'])
            identity=_hash(_json(['execution_decision',inputs]))
            with self._db(True) as db:
                head=self._verify(db);existing=db.execute('SELECT * FROM events WHERE identity=?',(identity,)).fetchone()
                if existing is not None:return self._receipt(existing)
                now=utc_time(self._clock());floor=head['last_time'];regressed=floor is not None and now<floor
                outcome=_outcome(inputs,now,regressed)
                at=max(now,floor) if floor is not None else now
                record=dict(schema=1,identity=identity,kind='decision',code_sha256=_version(),recorded_at=at.isoformat(),
                    evaluated_at=now.isoformat(),clock_regressed=regressed,inputs=inputs,outcome=outcome)
                return self._persist(db,record,head)
        except Exception:raise DecisionError('decision_append_failed') from None

    def verify(self):
        with self._db(False) as db:
            result=self._verify(db)
            return dict(events=result['events'],decisions=result['decisions'],integrity_verified=True)

    def replay(self,sequence,*,sources=None,ledger=None):
        try:
            with self._db(False) as db:
                self._verify(db);row=db.execute('SELECT * FROM events WHERE sequence=?',(sequence,)).fetchone()
                if row is None:raise ValueError('decision event missing')
                receipt=self._receipt(row);record=receipt.read()
            if record['kind']!='decision':raise ValueError('decision replay requires decision event')
            _verify_cleanup_source(record['inputs'],ledger)
            if record['inputs']['close'] is not None:
                _verify_cleanup_source({**record['inputs'],'cleanup':record['inputs']['close']},ledger)
            if record['inputs']['source'] is not None:
                if type(sources) is not ReplaySources:raise ValueError('original source replay required')
                sources.verify(record['inputs']['source'])
            actual=_outcome(record['inputs'],record['evaluated_at'],record['clock_regressed'])
            if actual!=record['outcome']:raise ValueError('decision replay outcome mismatch')
            return receipt
        except Exception:raise DecisionError('decision_replay_failed') from None

    def observe_order(self,decision_sequence,*,ledger):
        try:
            if type(ledger) is not OrderLedger:raise ValueError('exact verified order ledger required')
            records=ledger.all_orders();now=utc_time(self._clock())
            with self._db(True) as db:
                head=self._verify(db);row=db.execute('SELECT * FROM events WHERE sequence=?',(decision_sequence,)).fetchone()
                if row is None:raise ValueError('linked decision missing')
                decision=strict_json(row['payload'])
                if decision['kind']!='decision' or decision['inputs']['intent'] is None:raise ValueError('decision intent required')
                intent=decision['inputs']['intent'];matches=[r for r in records if r['intent_id']==intent['intent_id']]
                if len(matches)!=1 or matches[0]['intent']!=intent:raise ValueError('linked order intent mismatch')
                order=matches[0];snapshot_sha=_hash(_json(records));order_sha=_hash(_json(order))
                identity=_hash(_json(['decision_order_observation',decision['identity'],snapshot_sha,order_sha]))
                old=db.execute('SELECT * FROM events WHERE identity=?',(identity,)).fetchone()
                if old is not None:return self._receipt(old)
                unexpected=decision['outcome']['action'] not in ('buy','sell') and (
                    order['state'] in ('submission_unknown','open','partially_filled','filled')
                    or order['broker_order_id'] is not None or order['filled_quantity']>0)
                floor=head['last_time'];at=max(now,floor) if floor is not None else now
                native=order.get('native_stop');stop=native['stop'] if native else None
                summary=dict(intent_id=order['intent_id'],state=order['state'],filled_quantity=order['filled_quantity'],
                    filled_notional_usd=order['filled_notional_usd'],updated_at=order['updated_at'],
                    broker_identity_present=order['broker_order_id'] is not None,
                    stop_status=stop['status'] if stop else None,stop_filled_quantity=stop['filled_quantity'] if stop else None,
                    child_visible=native['child_visible'] if native else None)
                record=dict(schema=1,identity=identity,kind='order_observation',code_sha256=_version(),recorded_at=at.isoformat(),
                    observed_at=now.isoformat(),decision_identity=decision['identity'],order=summary,
                    order_record_sha256=order_sha,ledger_sha256=snapshot_sha,unexpected_after_veto=unexpected,
                    required_actions=['pause_entries','reconcile_orders'] if unexpected else [])
                return self._persist(db,record,head)
        except Exception:raise DecisionError('decision_order_observation_failed') from None

    def observe_account(self,decision_sequence,*,book,sequence=None):
        """Record later portfolio performance, not causal PnL for one decision."""
        try:
            if type(book) is not AccountBook:raise ValueError('exact verified account book required')
            with self._db(False) as db:
                self._verify(db);row=db.execute('SELECT * FROM events WHERE sequence=?',(decision_sequence,)).fetchone()
                if row is None:raise ValueError('linked decision missing')
                decision=strict_json(row['payload'])
                if decision['kind']!='decision' or decision['inputs']['prepared'] is None:raise ValueError('bound decision book required')
            source_digest=decision['inputs']['prepared']['book_digest']
            original=book.snapshot_by_digest(source_digest);snapshot=book.snapshot(sequence)
            if (snapshot.anchor!=original.anchor or snapshot.receipt.sequence<original.receipt.sequence
                    or snapshot.activity_until<original.activity_until):raise ValueError('account history continuity mismatch')
            now=utc_time(self._clock());result=snapshot.receipt.result
            timely=utc_time(decision['inputs']['observed_at'])<=snapshot.receipt.as_of<=now
            good=result.reconciled and timely
            account=dict(accounting_ok=good,source_as_of=snapshot.receipt.as_of.isoformat(),
                cash_usd=result.expected_cash_usd if good else None,equity_usd=result.equity_usd if good else None,
                profit_usd=result.profit_usd if good else None,realized_usd=result.realized_usd if good else None,
                unrealized_usd=result.unrealized_usd if good else None,fees_usd=result.fees_usd if good else None,
                scope='portfolio_observation_not_trade_attribution')
            identity=_hash(_json(['decision_account_observation',decision['identity'],snapshot.receipt.digest]))
            with self._db(True) as db:
                head=self._verify(db);prior=db.execute('SELECT * FROM events WHERE identity=?',(identity,)).fetchone()
                if prior is not None:return self._receipt(prior)
                current=db.execute('SELECT payload FROM events WHERE sequence=?',(decision_sequence,)).fetchone()
                if current is None or strict_json(current[0])!=decision:raise ValueError('linked decision changed')
                floor=head['last_time'];at=max(now,floor) if floor is not None else now
                record=dict(schema=1,identity=identity,kind='account_observation',code_sha256=_version(),recorded_at=at.isoformat(),
                    observed_at=now.isoformat(),decision_identity=decision['identity'],source_book_digest=source_digest,
                    book_digest=snapshot.receipt.digest,book_sequence=snapshot.receipt.sequence,
                    account=_normalize(account),required_actions=[] if good else ['pause_entries','reconcile_account'])
                return self._persist(db,record,head)
        except Exception:raise DecisionError('decision_account_observation_failed') from None
