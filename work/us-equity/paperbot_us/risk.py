"""Independent, deterministic veto for cash-only USD 10,000 paper orders.

No network/model/broker methods. The execution coordinator must hold its durable
reservation lock while applying this check and recording an accepted intent.
Stop-plus-cost risk is a model; gaps and liquidity can exceed realized limits.
"""
from dataclasses import dataclass,field,fields
from datetime import datetime,timedelta
from decimal import Decimal,InvalidOperation,localcontext
import re
from .data import Calendar,DataQualityError,utc_time


def _money(value,*,positive=False,maximum=Decimal('1000000000000')):
    if type(value) not in (str,int,float,Decimal): raise DataQualityError('invalid money type')
    if isinstance(value,str) and not re.fullmatch(r'(?:0|[1-9][0-9]{0,12})(?:\.[0-9]{1,8})?',value):
        raise DataQualityError('invalid decimal money string')
    try:
        result=Decimal(str(value))
        if not result.is_finite() or not -8<=result.as_tuple().exponent<=12 or not 0<=result<=maximum or (positive and not result>0):
            raise ValueError('money outside bounds')
        return result
    except (InvalidOperation,ValueError,OverflowError): raise DataQualityError('invalid finite money') from None


def _symbol(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Z][A-Z0-9.\-/]{0,19}',value):
        raise DataQualityError('invalid risk symbol')


def _identity(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
        raise DataQualityError('invalid intent identity')


def _quantity(value):
    if type(value) is not int or not 1<=value<=1000000: raise DataQualityError('whole positive shares required')


def _price(value): return _money(value,positive=True,maximum=Decimal('1000000'))


@dataclass(frozen=True)
class RiskLimits:
    capital_usd: Decimal=Decimal('10000')
    max_order_notional_usd: Decimal=Decimal('1000')
    max_symbol_notional_usd: Decimal=Decimal('1000')
    max_gross_exposure_usd: Decimal=Decimal('2500')
    cash_reserve_usd: Decimal=Decimal('500')
    max_trade_loss_usd: Decimal=Decimal('50')
    max_daily_loss_usd: Decimal=Decimal('100')
    max_drawdown_usd: Decimal=Decimal('500')
    cost_reservation_bps: Decimal=Decimal('50')
    max_spread_bps: Decimal=Decimal('30')
    max_limit_deviation_bps: Decimal=Decimal('10')
    max_age_seconds: int=30

    def __post_init__(self):
        for f in fields(RiskLimits):
            if f.name!='max_age_seconds': object.__setattr__(self,f.name,_money(getattr(self,f.name),positive=True))
        if not (self.capital_usd==10000 and self.max_order_notional_usd<=self.max_symbol_notional_usd<=self.max_gross_exposure_usd<=self.capital_usd
                and self.cash_reserve_usd<self.capital_usd and self.max_trade_loss_usd<=self.max_daily_loss_usd<=self.max_drawdown_usd<=self.capital_usd
                and 1<=self.cost_reservation_bps<=100 and 0<self.max_spread_bps<=100 and 0<self.max_limit_deviation_bps<=100
                and type(self.max_age_seconds) is int and 1<=self.max_age_seconds<=60):
            raise DataQualityError('risk limits outside engineering envelope')


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: int
    mark_price_usd: Decimal
    valued_at: datetime

    def __post_init__(self):
        _symbol(self.symbol);_quantity(self.quantity)
        object.__setattr__(self,'mark_price_usd',_price(self.mark_price_usd))
        object.__setattr__(self,'valued_at',utc_time(self.valued_at))


@dataclass(frozen=True)
class PendingOrder:
    intent_id: str
    symbol: str
    side: str
    remaining_quantity: int
    limit_price_usd: Decimal

    def __post_init__(self):
        _identity(self.intent_id);_symbol(self.symbol);_quantity(self.remaining_quantity)
        if self.side not in ('buy','sell'): raise DataQualityError('unsupported pending side')
        object.__setattr__(self,'limit_price_usd',_price(self.limit_price_usd))


@dataclass(frozen=True)
class StopReservation:
    """Owned protective sell shares, not a limit order or an exposure hedge."""
    parent_intent_id: str
    child_order_sha256: str
    symbol: str
    remaining_quantity: int
    observed_at: datetime

    def __post_init__(self):
        _identity(self.parent_intent_id);_identity(self.child_order_sha256)
        _symbol(self.symbol);_quantity(self.remaining_quantity)
        object.__setattr__(self,'observed_at',utc_time(self.observed_at))


@dataclass(frozen=True)
class Portfolio:
    as_of: datetime
    cash_usd: Decimal
    equity_usd: Decimal
    day_start_equity_usd: Decimal
    high_water_equity_usd: Decimal
    positions: tuple=()
    pending_orders: tuple=()
    reconciled: bool=True
    valuation_quotes: tuple=()
    stop_reservations: tuple=()
    mode: str=field(default='paper',init=False)
    currency: str=field(default='USD',init=False)

    def __post_init__(self):
        object.__setattr__(self,'as_of',utc_time(self.as_of))
        for name in ('cash_usd','equity_usd','day_start_equity_usd','high_water_equity_usd'):
            object.__setattr__(self,name,_money(getattr(self,name)))
        if type(self.reconciled) is not bool: raise DataQualityError('reconciliation flag required')
        if type(self.positions) is not tuple or type(self.pending_orders) is not tuple or len(self.positions)>1000 or len(self.pending_orders)>1000:
            raise DataQualityError('bounded immutable portfolio required')
        if any(type(v) is not Position for v in self.positions) or any(type(v) is not PendingOrder for v in self.pending_orders):
            raise DataQualityError('explicit position/order contracts required')
        if len({v.symbol for v in self.positions})!=len(self.positions) or len({v.intent_id for v in self.pending_orders})!=len(self.pending_orders):
            raise DataQualityError('duplicate portfolio identity')
        if (type(self.valuation_quotes) is not tuple or len(self.valuation_quotes)>2000
                or any(type(v) is not Quote for v in self.valuation_quotes)
                or len({v.symbol for v in self.valuation_quotes})!=len(self.valuation_quotes)):
            raise DataQualityError('bounded unique portfolio valuation quotes required')
        if (type(self.stop_reservations) is not tuple or len(self.stop_reservations)>1000
                or any(type(v) is not StopReservation for v in self.stop_reservations)
                or len({v.parent_intent_id for v in self.stop_reservations})!=len(self.stop_reservations)
                or len({v.child_order_sha256 for v in self.stop_reservations})!=len(self.stop_reservations)):
            raise DataQualityError('bounded unique protective share reservations required')


@dataclass(frozen=True)
class OrderIntent:
    intent_id: str
    symbol: str
    side: str
    quantity: int
    limit_price_usd: Decimal
    stop_price_usd: Decimal|None
    expires_at: datetime

    def __post_init__(self):
        _identity(self.intent_id);_symbol(self.symbol);_quantity(self.quantity)
        if self.side not in ('buy','sell'): raise DataQualityError('unsupported order side')
        object.__setattr__(self,'limit_price_usd',_price(self.limit_price_usd))
        object.__setattr__(self,'expires_at',utc_time(self.expires_at))
        if self.stop_price_usd is not None: object.__setattr__(self,'stop_price_usd',_price(self.stop_price_usd))
        if self.side=='buy' and (self.stop_price_usd is None or not self.stop_price_usd<self.limit_price_usd):
            raise DataQualityError('buy requires a lower modeled stop price')


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: Decimal
    ask: Decimal
    as_of: datetime
    received_at: datetime
    feed: str=field(default='iex',init=False)
    currency: str=field(default='USD',init=False)

    def __post_init__(self):
        _symbol(self.symbol)
        object.__setattr__(self,'bid',_price(self.bid));object.__setattr__(self,'ask',_price(self.ask))
        object.__setattr__(self,'as_of',utc_time(self.as_of));object.__setattr__(self,'received_at',utc_time(self.received_at))
        if self.ask<self.bid: raise DataQualityError('crossed quote')


@dataclass(frozen=True)
class RiskState:
    pause_entries: bool=True
    halt_orders: bool=False
    seen_intent_ids: frozenset=frozenset()

    def __post_init__(self):
        if type(self.pause_entries) is not bool or type(self.halt_orders) is not bool:
            raise DataQualityError('typed pause/halt flags required')
        if type(self.seen_intent_ids) is not frozenset or len(self.seen_intent_ids)>1000000:
            raise DataQualityError('bounded intent history required')
        for identity in self.seen_intent_ids: _identity(identity)


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    violations: tuple
    reserved_cash_usd: Decimal
    modeled_trade_loss_usd: Decimal
    gross_exposure_after_usd: Decimal
    mode: str=field(default='paper',init=False)


def evaluate(intent,*,portfolio,quote,state,limits,calendar,now):
    for value,base in ((intent,OrderIntent),(portfolio,Portfolio),(quote,Quote),(state,RiskState),(limits,RiskLimits),(calendar,Calendar)):
        if type(value) is not base: raise DataQualityError('exact versioned risk contracts required')
    now=utc_time(now)
    errors=[]
    def veto(condition,reason):
        if condition: errors.append(reason)
    veto(state.halt_orders,'all_orders_halted')
    veto(not portfolio.reconciled,'account_unreconciled')
    session=calendar.at(now)
    veto(session is None,'market_closed')
    veto(now>=intent.expires_at,'intent_expired')
    veto(intent.expires_at>now+timedelta(seconds=60),'intent_lifetime_too_long')
    veto(intent.intent_id in state.seen_intent_ids or intent.intent_id in {p.intent_id for p in portfolio.pending_orders}
         or intent.intent_id in {p.parent_intent_id for p in portfolio.stop_reservations},'duplicate_intent')
    veto(intent.symbol!=quote.symbol,'quote_symbol_mismatch')
    veto(not 0<=(now-portfolio.as_of).total_seconds()<=limits.max_age_seconds,'stale_account')
    veto(not (quote.as_of<=quote.received_at<=now and 0<=(now-quote.as_of).total_seconds()<=limits.max_age_seconds),'stale_quote')
    veto(session is not None and not session.opens<=quote.as_of<session.closes,'off_session_quote')
    veto(any(not 0<=(now-p.valued_at).total_seconds()<=limits.max_age_seconds for p in portfolio.positions),'stale_position_mark')
    veto(any(not 0<=(now-p.observed_at).total_seconds()<=limits.max_age_seconds for p in portfolio.stop_reservations),
         'stale_stop_reservation')
    market_quotes={q.symbol:[q] for q in portfolio.valuation_quotes}
    market_quotes.setdefault(quote.symbol,[]).append(quote)
    required_symbols={p.symbol for p in portfolio.positions}|{p.symbol for p in portfolio.pending_orders if p.side=='buy'}
    veto(not required_symbols<=set(market_quotes),'portfolio_quotes_missing')
    veto(any(not (q.as_of<=q.received_at<=now and 0<=(now-q.as_of).total_seconds()<=limits.max_age_seconds)
             for q in portfolio.valuation_quotes),'stale_portfolio_quote')
    veto(session is not None and any(not session.opens<=q.as_of<session.closes for q in portfolio.valuation_quotes),
         'off_session_portfolio_quote')
    def current_ask(symbol,fallback): return max((q.ask for q in market_quotes.get(symbol,())),default=fallback)
    def current_bid(symbol,fallback): return min((q.bid for q in market_quotes.get(symbol,())),default=fallback)
    with localcontext() as ctx:
        ctx.prec=40
        gross=sum((p.mark_price_usd*p.quantity for p in portfolio.positions),Decimal(0))
        veto(abs(portfolio.cash_usd+gross-portfolio.equity_usd)>Decimal('0.01'),'equity_mismatch')
        veto(portfolio.high_water_equity_usd<max(portfolio.equity_usd,portfolio.day_start_equity_usd,limits.capital_usd) or portfolio.day_start_equity_usd<=0,'invalid_risk_baseline')
        spread=(quote.ask-quote.bid)/((quote.ask+quote.bid)/2)*10000
        veto(spread>limits.max_spread_bps,'spread_limit')
        reference=quote.ask if intent.side=='buy' else quote.bid
        veto(abs(intent.limit_price_usd/reference-1)*10000>limits.max_limit_deviation_bps,'limit_price_deviation')
        notional=intent.limit_price_usd*intent.quantity
        veto(notional>limits.max_order_notional_usd,'order_notional_limit')
        held_by_symbol={p.symbol:p.quantity for p in portfolio.positions}
        reserved_sells={}
        for pending in portfolio.pending_orders:
            if pending.side=='sell': reserved_sells[pending.symbol]=reserved_sells.get(pending.symbol,0)+pending.remaining_quantity
        for stop in portfolio.stop_reservations:
            reserved_sells[stop.symbol]=reserved_sells.get(stop.symbol,0)+stop.remaining_quantity
        veto(any(qty>held_by_symbol.get(symbol,0) for symbol,qty in reserved_sells.items()),'inconsistent_sell_reservations')
        pending_buys=[p for p in portfolio.pending_orders if p.side=='buy']
        pending_notional=sum((p.remaining_quantity*p.limit_price_usd for p in pending_buys),Decimal(0))
        cost_factor=1+limits.cost_reservation_bps/10000
        reserved_cash=modeled_risk=Decimal(0)
        # Keep provider accounting reconciliation separate from conservative
        # current-quote risk valuation across every held/pending-buy symbol.
        # Missing data vetoes; diagnostic fallbacks cannot authorize an order.
        risk_gross=sum((p.quantity*max(p.mark_price_usd,current_ask(p.symbol,p.mark_price_usd))
                        for p in portfolio.positions),Decimal(0))
        liquidation_equity=portfolio.cash_usd+sum((p.quantity*min(p.mark_price_usd,current_bid(p.symbol,p.mark_price_usd))
                                                 for p in portfolio.positions),Decimal(0))
        risk_equity=min(portfolio.equity_usd,liquidation_equity)
        pending_exposure=sum((p.remaining_quantity*max(p.limit_price_usd,current_ask(p.symbol,p.limit_price_usd))
                              for p in pending_buys),Decimal(0))
        exposure=risk_gross+pending_exposure
        if intent.side=='buy':
            veto(state.pause_entries,'entries_paused')
            veto(portfolio.day_start_equity_usd-risk_equity>=limits.max_daily_loss_usd,'daily_loss_limit')
            veto(portfolio.high_water_equity_usd-risk_equity>=limits.max_drawdown_usd,'drawdown_limit')
            reserved_cash=notional*cost_factor
            roundtrip_notional=(intent.limit_price_usd+intent.stop_price_usd)*intent.quantity
            modeled_risk=(intent.limit_price_usd-intent.stop_price_usd)*intent.quantity+roundtrip_notional*(cost_factor-1)
            symbol_exposure=sum((max(p.mark_price_usd,current_ask(p.symbol,p.mark_price_usd))*p.quantity for p in portfolio.positions if p.symbol==intent.symbol),Decimal(0))
            symbol_pending=sum((p.remaining_quantity*max(p.limit_price_usd,current_ask(p.symbol,p.limit_price_usd)) for p in pending_buys if p.symbol==intent.symbol),Decimal(0))
            entry_exposure=max(intent.limit_price_usd,current_ask(intent.symbol,quote.ask))*intent.quantity
            exposure+=entry_exposure
            veto(modeled_risk>limits.max_trade_loss_usd,'trade_risk_limit')
            veto(symbol_exposure+symbol_pending+entry_exposure>limits.max_symbol_notional_usd,'symbol_exposure_limit')
            other_exposure={}
            for position in portfolio.positions:
                if position.symbol!=intent.symbol:
                    other_exposure[position.symbol]=other_exposure.get(position.symbol,Decimal(0))+position.quantity*max(
                        position.mark_price_usd,current_ask(position.symbol,position.mark_price_usd))
            for pending in pending_buys:
                if pending.symbol!=intent.symbol:
                    other_exposure[pending.symbol]=other_exposure.get(pending.symbol,Decimal(0))+pending.remaining_quantity*max(
                        pending.limit_price_usd,current_ask(pending.symbol,pending.limit_price_usd))
            veto(any(amount>limits.max_symbol_notional_usd for amount in other_exposure.values()),'existing_symbol_exposure_limit')
            veto(exposure>limits.max_gross_exposure_usd,'portfolio_exposure_limit')
            veto(portfolio.cash_usd-pending_notional*cost_factor-reserved_cash<limits.cash_reserve_usd,'cash_reserve_limit')
        else:
            held=sum(p.quantity for p in portfolio.positions if p.symbol==intent.symbol)
            pending_sells=reserved_sells.get(intent.symbol,0)
            veto(intent.quantity>held-pending_sells,'insufficient_unreserved_shares')
        return RiskDecision(not errors,tuple(errors),reserved_cash,modeled_risk,exposure)
