"""Typed, deterministic long-only paper fill accounting. No network or orders."""
from dataclasses import dataclass,field
from datetime import date,datetime
from decimal import Decimal,localcontext
import re
import uuid
from .data import DataQualityError,utc_time
from .risk import _money,_price,_identity,_symbol


def _shares(value,zero=False):
    if type(value) is not int or not (0 if zero else 1)<=value<=1000000:
        raise DataQualityError('whole bounded shares required')
    return value


def _uuid(value):
    try:
        if not isinstance(value,str) or str(uuid.UUID(value))!=value: raise ValueError('uuid')
        return value
    except (ValueError,TypeError,AttributeError): raise DataQualityError('invalid private order identity') from None


def signed_money(value):
    if type(value) not in (str,int,float,Decimal): raise DataQualityError('invalid signed money type')
    raw=str(value);negative=raw.startswith('-')
    magnitude=_money(raw[1:] if negative else raw)
    return -magnitude if negative else magnitude


@dataclass(frozen=True)
class AccountAnchor:
    as_of: datetime
    cash_usd: Decimal
    evidence_sha256: str
    account_sha256: str=field(repr=False)
    mode: str=field(default='paper',init=False)

    def __post_init__(self):
        object.__setattr__(self,'as_of',utc_time(self.as_of))
        object.__setattr__(self,'cash_usd',_money(self.cash_usd))
        _identity(self.evidence_sha256);_identity(self.account_sha256)
        if self.cash_usd!=10000: raise DataQualityError('10000 USD empty paper anchor required')


@dataclass(frozen=True)
class Activity:
    identity: str=field(repr=False)
    kind: str
    symbol: str|None=None
    side: str|None=None
    quantity: int=0
    price_usd: Decimal=Decimal(0)
    cumulative_quantity: int=0
    leaves_quantity: int=0
    broker_order_id: str|None=field(default=None,repr=False)
    occurred_at: datetime|None=None
    net_usd: Decimal=Decimal(0)
    source_day: str|None=None
    source_time: datetime|None=None
    provider_kind: str=''

    def __post_init__(self):
        _identity(self.identity)
        if self.kind not in ('FILL','FEE','UNSUPPORTED'): raise DataQualityError('unknown normalized activity kind')
        provider=self.provider_kind or self.kind
        if not isinstance(provider,str) or not re.fullmatch('[A-Z][A-Z0-9_]{0,31}',provider):
            raise DataQualityError('invalid provider activity type')
        object.__setattr__(self,'provider_kind',provider)
        object.__setattr__(self,'price_usd',_money(self.price_usd))
        object.__setattr__(self,'net_usd',signed_money(self.net_usd))
        for name in ('quantity','cumulative_quantity','leaves_quantity'): _shares(getattr(self,name),zero=True)
        if self.kind=='FILL':
            _symbol(self.symbol);_shares(self.quantity);_uuid(self.broker_order_id)
            object.__setattr__(self,'price_usd',_price(self.price_usd))
            object.__setattr__(self,'occurred_at',utc_time(self.occurred_at))
            if self.side not in ('buy','sell') or self.net_usd!=0 or self.source_day is not None or self.source_time is not None or provider!='FILL':
                raise DataQualityError('invalid fill semantics')
        else:
            if any((self.symbol is not None,self.side is not None,self.quantity!=0,self.price_usd!=0,
                    self.cumulative_quantity!=0,self.leaves_quantity!=0,self.broker_order_id is not None,self.occurred_at is not None)):
                raise DataQualityError('unexpected nontrade fields')
            if self.kind=='FEE':
                try:
                    if date.fromisoformat(self.source_day).isoformat()!=self.source_day: raise ValueError('date')
                except (ValueError,TypeError): raise DataQualityError('fee source date required') from None
                if provider!='FEE': raise DataQualityError('fee type mismatch')
                if self.source_time is not None:
                    object.__setattr__(self,'source_time',utc_time(self.source_time))
                    if self.source_time.date().isoformat()!=self.source_day: raise DataQualityError('fee source time/date mismatch')
            elif self.source_time is not None: raise DataQualityError('unsupported source timestamp')


@dataclass(frozen=True)
class ActivityWindow:
    after: datetime
    until: datetime
    received_at: datetime
    activities: tuple
    complete: bool=True
    mode: str=field(default='paper',init=False)

    def __post_init__(self):
        for name in ('after','until','received_at'): object.__setattr__(self,name,utc_time(getattr(self,name)))
        if not self.after<self.until<=self.received_at or type(self.complete) is not bool:
            raise DataQualityError('activity window bounds required')
        if type(self.activities) is not tuple or len(self.activities)>10000 or any(type(a) is not Activity for a in self.activities):
            raise DataQualityError('bounded normalized activities required')
        if len({a.identity for a in self.activities})!=len(self.activities): raise DataQualityError('duplicate activity identity')


@dataclass(frozen=True)
class OrderEvidence:
    intent_id: str=field(repr=False)
    broker_order_id: str=field(repr=False)
    symbol: str
    side: str
    quantity: int
    limit_price_usd: Decimal|None
    filled_quantity: int
    filled_notional_usd: Decimal
    created_at: datetime
    updated_at: datetime
    order_type: str='limit'
    stop_price_usd: Decimal|None=None
    parent_order_id: str|None=field(default=None,repr=False)
    filled_at: datetime|None=None
    protection_kind: str='attached'
    parent_terminal_at: datetime|None=None
    allocation_sha256: str|None=field(default=None,repr=False)

    def __post_init__(self):
        _identity(self.intent_id);_uuid(self.broker_order_id);_symbol(self.symbol)
        _shares(self.quantity);_shares(self.filled_quantity,zero=True)
        if self.order_type=='limit':
            object.__setattr__(self,'limit_price_usd',_price(self.limit_price_usd))
            if (self.stop_price_usd is not None or self.parent_order_id is not None
                    or self.protection_kind!='attached' or self.parent_terminal_at is not None or self.allocation_sha256 is not None):
                raise DataQualityError('unexpected simple limit protection fields')
        elif self.order_type=='stop':
            if self.side!='sell' or self.limit_price_usd is not None:
                raise DataQualityError('only owned protective sell stop-market supported')
            _uuid(self.parent_order_id)
            if self.parent_order_id==self.broker_order_id: raise DataQualityError('self-parent order refused')
            object.__setattr__(self,'stop_price_usd',_price(self.stop_price_usd))
            if self.protection_kind=='repair':
                _identity(self.allocation_sha256)
                object.__setattr__(self,'parent_terminal_at',utc_time(self.parent_terminal_at))
                if self.parent_terminal_at>utc_time(self.created_at):raise DataQualityError('repair before terminal entry')
            elif self.protection_kind!='attached' or self.parent_terminal_at is not None or self.allocation_sha256 is not None:
                raise DataQualityError('invalid stop ownership kind')
        else: raise DataQualityError('unsupported accounting order type')
        object.__setattr__(self,'filled_notional_usd',_money(self.filled_notional_usd))
        for name in ('created_at','updated_at'): object.__setattr__(self,name,utc_time(getattr(self,name)))
        if self.filled_at is not None:
            object.__setattr__(self,'filled_at',utc_time(self.filled_at))
            if self.filled_quantity!=self.quantity or not self.created_at<=self.filled_at<=self.updated_at:
                raise DataQualityError('invalid complete fill timestamp')
        if (self.side not in ('buy','sell') or self.filled_quantity>self.quantity or self.created_at>self.updated_at
                or bool(self.filled_quantity)!=bool(self.filled_notional_usd)):
            raise DataQualityError('order accounting semantics')


@dataclass(frozen=True)
class AccountMark:
    symbol: str
    quantity: int
    reference_price_usd: Decimal
    market_value_usd: Decimal
    received_at: datetime

    def __post_init__(self):
        _symbol(self.symbol);_shares(self.quantity)
        object.__setattr__(self,'reference_price_usd',_price(self.reference_price_usd))
        object.__setattr__(self,'market_value_usd',_money(self.market_value_usd))
        object.__setattr__(self,'received_at',utc_time(self.received_at))


@dataclass(frozen=True)
class AccountSnapshot:
    received_at: datetime
    cash_usd: Decimal
    equity_usd: Decimal
    blocked: bool
    positions: tuple
    account_sha256: str=field(repr=False)
    mode: str=field(default='paper',init=False)

    def __post_init__(self):
        _identity(self.account_sha256)
        object.__setattr__(self,'received_at',utc_time(self.received_at))
        for name in ('cash_usd','equity_usd'): object.__setattr__(self,name,_money(getattr(self,name)))
        if (type(self.blocked) is not bool or type(self.positions) is not tuple or len(self.positions)>1000
                or any(type(p) is not AccountMark for p in self.positions)
                or len({p.symbol for p in self.positions})!=len(self.positions)):
            raise DataQualityError('explicit account positions required')


@dataclass(frozen=True)
class Inventory:
    symbol: str
    quantity: int
    average_cost_usd: Decimal


@dataclass(frozen=True)
class Reconciliation:
    reconciled: bool
    violations: tuple
    expected_cash_usd: Decimal
    equity_usd: Decimal|None
    profit_usd: Decimal|None
    realized_usd: Decimal
    unrealized_usd: Decimal|None
    fees_usd: Decimal
    inventory: tuple


def reconcile(*,anchor,window,orders,account,now):
    if any(type(v) is not base for v,base in ((anchor,AccountAnchor),(window,ActivityWindow),(account,AccountSnapshot))):
        raise DataQualityError('exact accounting contracts required')
    if type(orders) is not tuple or len(orders)>10000 or any(type(o) is not OrderEvidence for o in orders):
        raise DataQualityError('bounded owned orders required')
    if len({o.broker_order_id for o in orders})!=len(orders) or len({o.intent_id for o in orders})!=len(orders):
        raise DataQualityError('duplicate owned order identity')
    now=utc_time(now);errors=set()
    def veto(condition,reason):
        if condition: errors.add(reason)
    veto(not window.complete,'activity_window_incomplete')
    veto(window.after!=anchor.as_of,'anchor_window_mismatch')
    veto(not window.until<=window.received_at<=account.received_at<=now,'observation_time_order')
    veto(not 0<=(now-window.until).total_seconds()<=30,'stale_activity_window')
    veto(not 0<=(now-account.received_at).total_seconds()<=30,'stale_account')
    veto(account.blocked,'account_blocked')
    veto(account.account_sha256!=anchor.account_sha256,'account_identity_mismatch')
    owned={o.broker_order_id:o for o in orders};cum={};notionals={};shares={};costs={}
    last_fill={};protective_fills={}
    for event in window.activities:
        if event.kind=='FILL':last_fill[event.broker_order_id]=max(event.occurred_at,last_fill.get(event.broker_order_id,event.occurred_at))
    for order in orders:
        if order.order_type=='stop':protective_fills[order.parent_order_id]=protective_fills.get(order.parent_order_id,0)+order.filled_quantity
    parents=set()
    for order in orders:
        if order.order_type!='stop': continue
        parent=owned.get(order.parent_order_id)
        if order.protection_kind=='attached':
            veto(order.parent_order_id in parents,'duplicate_stop_parent')
            parents.add(order.parent_order_id)
            veto(parent is None or parent.order_type!='limit' or parent.side!='buy'
                 or parent.symbol!=order.symbol or parent.quantity!=order.quantity
                 or parent.created_at>order.created_at
                 or (order.filled_quantity and (parent.filled_quantity!=parent.quantity or parent.filled_at is None)),
                 'stop_parent_mismatch')
        else:
            veto(parent is None or parent.order_type!='limit' or parent.side!='buy' or parent.symbol!=order.symbol
                 or order.quantity>parent.filled_quantity or parent.created_at>order.created_at
                 or parent.updated_at<order.parent_terminal_at
                 or parent.filled_at is not None and order.parent_terminal_at<parent.filled_at
                 or last_fill.get(order.parent_order_id,order.parent_terminal_at)>order.parent_terminal_at,
                 'repair_parent_mismatch')
    for parent_id,quantity in protective_fills.items():
        parent=owned.get(parent_id)
        veto(parent is None or quantity>parent.filled_quantity,'protective_family_overfilled')
    cash=anchor.cash_usd;realized=Decimal(0);fees=Decimal(0)
    with localcontext() as context:
        context.prec=40
        fills=sorted((a for a in window.activities if a.kind=='FILL'),key=lambda a:(a.occurred_at,
            owned[a.broker_order_id].order_type=='stop' if a.broker_order_id in owned else False,
            a.broker_order_id,a.cumulative_quantity,a.identity))
        for event in window.activities:
            if event.kind=='UNSUPPORTED': veto(True,'unsupported_account_activity')
            elif event.kind=='FEE':
                veto(event.net_usd>0,'unsupported_fee_credit')
                veto(not anchor.as_of.date()<=date.fromisoformat(event.source_day)<=now.date(),'fee_date_outside_experiment')
                veto(event.source_time is not None and not anchor.as_of<event.source_time<=window.until<=window.received_at<=now,'fee_source_time_outside_window')
                cash+=event.net_usd;fees-=event.net_usd
        for fill in fills:
            order=owned.get(fill.broker_order_id)
            veto(not anchor.as_of<fill.occurred_at<=window.until,'fill_time_outside_window')
            if order is None: veto(True,'unrecognized_fill_order');continue
            if (fill.symbol,fill.side)!=(order.symbol,order.side): veto(True,'fill_intent_mismatch');continue
            veto(not order.created_at<=fill.occurred_at<=order.updated_at,'fill_order_time_mismatch')
            veto(order.filled_at is not None and fill.occurred_at>order.filled_at,'fill_after_complete_timestamp')
            if order.order_type=='stop':
                parent=owned.get(order.parent_order_id)
                if order.protection_kind=='repair':
                    veto(fill.occurred_at<order.parent_terminal_at,'repair_fill_before_terminal_entry')
                else:
                    veto(parent is None or parent.filled_at is None or fill.occurred_at<parent.filled_at,
                         'stop_fill_before_entry_completion')
            prior=cum.get(fill.broker_order_id,0);total=prior+fill.quantity
            veto(total!=fill.cumulative_quantity or total+fill.leaves_quantity!=order.quantity,'fill_cumulative_mismatch')
            veto(total>order.quantity,'order_overfilled')
            if order.order_type=='limit':
                veto((fill.side=='buy' and fill.price_usd>order.limit_price_usd) or
                     (fill.side=='sell' and fill.price_usd<order.limit_price_usd),'individual_fill_limit_violation')
            cum[fill.broker_order_id]=total
            notional=fill.price_usd*fill.quantity;notionals[fill.broker_order_id]=notionals.get(fill.broker_order_id,Decimal(0))+notional
            qty=shares.get(fill.symbol,0);cost=costs.get(fill.symbol,Decimal(0))
            if fill.side=='buy':
                shares[fill.symbol]=qty+fill.quantity;costs[fill.symbol]=cost+notional;cash-=notional
                veto(cash<0,'negative_cash_in_history')
            else:
                if fill.quantity>qty: veto(True,'short_or_missing_inventory');continue
                basis=cost*fill.quantity/qty
                realized+=notional-basis;cash+=notional;shares[fill.symbol]=qty-fill.quantity;costs[fill.symbol]=cost-basis
        for order in orders:
            veto(not anchor.as_of<=order.created_at<=order.updated_at<=window.until,'order_time_outside_window')
            veto(cum.get(order.broker_order_id,0)!=order.filled_quantity,'missing_or_extra_order_fills')
            tolerance=Decimal('0.00000001')*max(1,order.filled_quantity)
            veto(abs(notionals.get(order.broker_order_id,Decimal(0))-order.filled_notional_usd)>tolerance,'order_fill_notional_mismatch')
        actual={p.symbol:p for p in account.positions};expected={s:q for s,q in shares.items() if q}
        veto({s:p.quantity for s,p in actual.items()}!=expected,'position_quantity_mismatch')
        unrealized=Decimal(0);inventory=[]
        for symbol,quantity in sorted(expected.items()):
            inventory.append(Inventory(symbol,quantity,costs[symbol]/quantity))
            if symbol not in actual: continue
            mark=actual[symbol];unrealized+=mark.market_value_usd-costs[symbol]
        for mark in account.positions:
            veto(not mark.received_at<=account.received_at or not 0<=(now-mark.received_at).total_seconds()<=30,'stale_account_mark')
            veto(abs(mark.market_value_usd-mark.quantity*mark.reference_price_usd)>Decimal('.01'),'account_mark_value_mismatch')
        equity=cash+sum((p.market_value_usd for p in account.positions),Decimal(0))
        veto(abs(cash-account.cash_usd)>Decimal('.01'),'cash_mismatch')
        veto(abs(equity-account.equity_usd)>Decimal('.01'),'equity_mismatch')
        veto(abs((realized+unrealized-fees)-(equity-anchor.cash_usd))>Decimal('.01'),'profit_decomposition_mismatch')
        valid=not errors
        return Reconciliation(valid,tuple(sorted(errors)),cash,equity if valid else None,
            equity-anchor.cash_usd if valid else None,realized,unrealized if valid else None,fees,tuple(inventory))
