"""Private typed compensation decisions. No order authority or broker I/O."""
from contextlib import closing
from dataclasses import dataclass,field
from datetime import date,timedelta
from decimal import Decimal,ROUND_DOWN,localcontext
import hashlib,sqlite3
from pathlib import Path

from .alpaca import decode_json
from .controls import ControlSnapshot
from .data import Calendar,Session,utc_time
from .monitor import ObservationArchive
from .orders import OrderLedger,_base,_hash,_json,_normalize,_ts
from .repair_protection import ProtectionRepairPlan,_calendar,_limits,_version as plan_version
from .risk import Quote,RiskLimits,_identity,_price
from .runner import _exclusive


class RepairDecisionError(RuntimeError):pass


def _version():
    digest=hashlib.sha256()
    for name in ('repair_decisions.py','repair_execution.py','repair_orders.py','repair_protection.py',
            'native_cleanup.py','cleanup_recovery.py','orders.py','risk.py','risk_view.py',
            'risk_baselines.py','accountbook.py','accounting.py','native_reconciler.py',
            'native_orders.py','native_broker.py','protection.py','controls.py','data.py','broker.py','monitor.py','compensation.py','native_admission.py','native_execution.py'):
        digest.update(name.encode()+b'\0'+Path(__file__).with_name(name).read_bytes()+b'\0')
    return digest.hexdigest()


def _calendar_from(row):
    if type(row) is not dict or set(row)!={'start','end','sessions'}:raise ValueError('calendar contract')
    calendar=Calendar([Session(date.fromisoformat(s['day']),s['opens'],s['closes']) for s in row['sessions']],
        start=row['start'],end=row['end'])
    if decode_json(_calendar(calendar))!=row:raise ValueError('calendar normalization')
    return calendar


def _quote(row):
    if type(row) is not dict or set(row)!={'symbol','bid','ask','as_of','received_at','feed','currency'}:
        raise ValueError('quote contract')
    value=Quote(**{k:row[k] for k in ('symbol','bid','ask','as_of','received_at')})
    if _normalize(_base(value,Quote))!=row:raise ValueError('quote normalization')
    return value


def _inputs(value):
    keys={'purpose','family_id','prepared_at','expires_at','body','original_stop_usd','source_prefix',
        'source_orders_sha256','source_book_digest','source_baseline_digest','source_as_of','quotes',
        'limits','calendar','controls','plan_code_sha256','findings'}
    if type(value) is not dict or set(value)!=keys or value['purpose']!='protect_existing_owned_position':
        raise ValueError('typed compensation decision required')
    for key in ('family_id','source_orders_sha256','source_book_digest','source_baseline_digest','plan_code_sha256'):
        _identity(value[key])
    if value['plan_code_sha256']!=plan_version():raise ValueError('repair plan source mismatch')
    prefix=value['source_prefix']
    if type(prefix) is not dict or set(prefix)!={'sequence','digest'} or type(prefix['sequence']) is not int or not 1<=prefix['sequence']<=100000:
        raise ValueError('source prefix contract')
    _identity(prefix['digest']);prepared=utc_time(value['prepared_at']);expiry=utc_time(value['expires_at'])
    if expiry!=prepared+timedelta(seconds=20):raise ValueError('compensation lifetime')
    controls=ControlSnapshot(**value['controls']);limits=RiskLimits(**value['limits']);calendar=_calendar_from(value['calendar'])
    if _normalize(_base(controls,ControlSnapshot))!=value['controls'] or _normalize(_base(limits,RiskLimits))!=value['limits']:
        raise ValueError('versioned decision parameters')
    if type(value['quotes']) is not list or not 1<=len(value['quotes'])<=16:raise ValueError('bounded complete quotes required')
    quotes=tuple(_quote(q) for q in value['quotes'])
    if len({q.symbol for q in quotes})!=len(quotes):raise ValueError('duplicate compensation quote')
    body=value['body']
    fields={'symbol','side','type','qty','stop_price','time_in_force','extended_hours','order_class','client_order_id'}
    if (type(body) is not dict or set(body)!=fields or body['side']!='sell' or body['type']!='stop'
            or body['time_in_force']!='gtc' or body['extended_hours'] is not False or body['order_class']!='simple'):
        raise ValueError('standalone stop body required')
    from .broker import _shares,_tick
    quantity=_shares(body['qty']);_tick(body['stop_price']);_price(value['original_stop_usd'])
    import re
    if not isinstance(body['client_order_id'],str) or not re.fullmatch('pb3r-[0-9a-f]{40}',body['client_order_id']):
        raise ValueError('repair client scope')
    targets=[q for q in quotes if q.symbol==body['symbol']]
    if len(targets)!=1:raise ValueError('target quote absent')
    from .decision_journal import _codes
    _codes(value['findings']);utc_time(value['source_as_of'])
    return controls,limits,calendar,quotes,targets[0],quantity


def evaluate(value,now,*,previous_at=None):
    return _evaluate(value,now,_inputs(value),previous_at=previous_at)


def _evaluate(value,now,validated,*,previous_at=None):
    """Pure time/math checks on already parsed, source-validated inputs."""
    now=utc_time(now);controls,limits,calendar,quotes,target,quantity=validated;reasons=[]
    floor=max(utc_time(value['prepared_at']),utc_time(value['source_as_of']),controls.updated_at,
        *(q.received_at for q in quotes))
    if now<floor or previous_at is not None and now.timestamp()<previous_at:reasons.append('repair_clock_regression')
    if controls.halt_orders:reasons.append('all_orders_halted')
    if now>=utc_time(value['expires_at']):reasons.append('repair_decision_expired')
    age=min(30,limits.max_age_seconds)
    if not 0<=(now-utc_time(value['source_as_of'])).total_seconds()<=age:reasons.append('repair_account_stale')
    if any(not q.as_of<=q.received_at<=now or not 0<=(now-q.as_of).total_seconds()<=age for q in quotes):
        reasons.append('repair_quotes_stale')
    try:session=calendar.at(now)
    except Exception:session=None
    if session is None:reasons.append('repair_market_closed')
    elif any(not session.opens<=q.as_of<session.closes for q in quotes):reasons.append('repair_quote_session')
    with localcontext() as ctx:
        ctx.prec=40
        if quantity*target.ask>min(Decimal('1000'),limits.max_order_notional_usd):reasons.append('repair_transport_budget')
        if (target.ask-target.bid)*10000/target.bid>limits.max_spread_bps:reasons.append('repair_spread')
        candidate=min(_price(value['original_stop_usd']),target.bid-Decimal('.01'))
        trigger=candidate.quantize(Decimal('.01') if candidate>=1 else Decimal('.0001'),rounding=ROUND_DOWN)
        if trigger<=0 or trigger!=_price(value['body']['stop_price']):reasons.append('repair_trigger_changed')
    return dict(allowed=not reasons,action='protect' if not reasons else 'hold',vetoes=reasons)


def _source(value,ledger):
    from .native_cleanup import project_groups
    if type(ledger) is not OrderLedger:raise ValueError('original ledger required')
    records=ledger.records_at_prefix(value['source_prefix'])
    if _hash(_json(records))!=value['source_orders_sha256']:raise ValueError('original source digest')
    group=project_groups(records)[value['family_id']]
    family=next(r for r in records if r['intent_id']==value['family_id'])
    if (not group.parent_terminal or not group.child_terminal or group.remaining_quantity<=0
            or group.available_quantity!=group.remaining_quantity or str(group.remaining_quantity)!=value['body']['qty']
            or group.symbol!=value['body']['symbol']
            or _price(family['native_stop']['plan']['stop_price_usd'])!=_price(value['original_stop_usd'])):
        raise ValueError('original family allocation proof')
    return records


def _valid_record(record):
    keys={'schema','kind','identity','code_sha256','generated_at','evaluated_at','previous_generated_at',
        'inputs','outcome','observing_started_at'}
    if (type(record) is not dict or set(record)!=keys or type(record['schema']) is not int or record['schema']!=1
            or record['kind']!='repair_decision' or record['code_sha256']!=_version()
            or record['identity']!=_hash(_json(['repair_compensation',record['inputs']]))):
        raise ValueError('repair decision identity or source mismatch')
    outcome=evaluate(record['inputs'],record['evaluated_at'],previous_at=record['previous_generated_at'])
    if record['outcome']!=outcome or record['generated_at']<utc_time(record['evaluated_at']).timestamp():
        raise ValueError('repair decision replay mismatch')
    return record


@dataclass(frozen=True)
class RepairDecisionReceipt:
    sequence:int
    digest:str
    payload:str=field(repr=False)

    def read(self):
        try:return _valid_record(decode_json(self.payload))
        except Exception:raise RepairDecisionError('repair_decision_receipt_invalid') from None

    def public_summary(self):
        record=self.read()
        return dict(mode='paper',phase='repair_decision',action=record['outcome']['action'],
            vetoes=record['outcome']['vetoes'],execution_enabled=False,protection_active=False)


class RepairDecisionJournal:
    def __init__(self,path,*,clock):
        self.archive=ObservationArchive(path);self.path=self.archive.path;self._clock=clock

    def _read(self,sequence):
        if type(sequence) is not int or not 1<=sequence<=100000 or self.path.is_symlink() or not self.path.is_file():
            raise ValueError('existing repair decision required')
        with closing(sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True)) as db:
            self.archive._verify_chain(db)
            row=db.execute('SELECT payload,digest FROM observations WHERE seq=?',(sequence,)).fetchone()
            if row is None:raise ValueError('repair decision absent')
            record=_valid_record(decode_json(row[0]));prior=db.execute('SELECT payload FROM observations WHERE seq=?',(sequence-1,)).fetchone()
            previous=decode_json(prior[0])['generated_at'] if prior else None
            if record['previous_generated_at']!=previous:raise ValueError('repair decision clock predecessor')
        return RepairDecisionReceipt(sequence,row[1],row[0])

    def replay(self,sequence,*,ledger):
        try:
            receipt=self._read(sequence);_source(receipt.read()['inputs'],ledger);return receipt
        except Exception:raise RepairDecisionError('repair_decision_replay_unavailable') from None

    def record(self,plan,*,ledger,book,baselines,controls,calendar,limits):
        try:
            if type(plan) is not ProtectionRepairPlan or type(controls) is not ControlSnapshot:
                raise ValueError('exact typed compensation inputs')
            body=plan.request_body(book=book,ledger=ledger,baselines=baselines,calendar=calendar,limits=limits,now=plan.prepared_at)
            prefix,records=ledger.snapshot_prefix()
            if _hash(_json(records))!=plan.prepared.risk.orders_digest:raise ValueError('repair source changed')
            family=next(r for r in records if r['intent_id']==plan.family_id);view=plan.prepared.risk
            snapshot=book.snapshot();baseline=baselines.replay()
            source_as_of=min(view.portfolio.as_of,snapshot.activity_until,snapshot.account.received_at,
                baseline.source_as_of,*(p.received_at for p in snapshot.account.positions))
            inputs=_normalize(dict(purpose='protect_existing_owned_position',family_id=plan.family_id,
                prepared_at=_ts(plan.prepared_at),expires_at=_ts(plan.expires_at),body=body,
                original_stop_usd=family['native_stop']['plan']['stop_price_usd'],source_prefix=prefix,
                source_orders_sha256=view.orders_digest,source_book_digest=view.book_digest,
                source_baseline_digest=view.baseline_digest,source_as_of=_ts(source_as_of),
                quotes=[_base(q,Quote) for q in view.portfolio.valuation_quotes],limits=_base(limits,RiskLimits),
                calendar=decode_json(_calendar(calendar)),controls=_base(controls,ControlSnapshot),
                plan_code_sha256=plan.code_sha256,findings=list(view.findings)))
            _inputs(inputs);_source(inputs,ledger)
            with _exclusive(self.path):
                previous=None
                if self.path.exists():
                    count=self.archive.verify()['records']
                    if count:previous=self._read(count).read()['generated_at']
                now=utc_time(self._clock());outcome=evaluate(inputs,now,previous_at=previous)
                floor=max(now.timestamp(),utc_time(inputs['prepared_at']).timestamp(),previous or 0)
                record=dict(schema=1,kind='repair_decision',identity=_hash(_json(['repair_compensation',inputs])),
                    code_sha256=_version(),generated_at=floor,evaluated_at=_ts(now),previous_generated_at=previous,
                    inputs=inputs,outcome=outcome)
                saved=self.archive.append(record);count=self.archive.verify()['records'];receipt=self._read(count)
                if receipt.digest!=saved['audit_sha256']:raise ValueError('repair decision append race')
                return receipt
        except Exception:raise RepairDecisionError('repair_decision_record_unavailable') from None

    def link_for_reservation(self,receipt,*,ledger,plan,body,now):
        if type(receipt) is not RepairDecisionReceipt:raise RepairDecisionError('committed_repair_decision_required')
        actual=self.replay(receipt.sequence,ledger=ledger)
        if actual!=receipt:raise RepairDecisionError('repair_decision_receipt_changed')
        record=actual.read();value=record['inputs'];prefix,records=ledger.snapshot_prefix()
        if (not record['outcome']['allowed'] or not evaluate(value,now)['allowed']
                or utc_time(now)<max(utc_time(record['evaluated_at']),utc_time(value['prepared_at']))
                or utc_time(now).timestamp()<record['generated_at'] or prefix!=value['source_prefix']
                or _hash(_json(records))!=value['source_orders_sha256'] or value['body']!=body
                or plan.family_id!=value['family_id'] or plan.prepared.risk.book_digest!=value['source_book_digest']
                or plan.prepared.risk.baseline_digest!=value['source_baseline_digest']):
            raise RepairDecisionError('repair_decision_not_current')
        return dict(sequence=receipt.sequence,digest=receipt.digest,identity=record['identity'])
