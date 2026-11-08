"""Single-symbol intraday research simulator, with no broker connectivity.

Next-open prices, completed-bar participation caps and a scheduled final close
are bar-data approximations. They do not model the opening/closing auctions,
queue position, impact or real partial-fill timing. Costs are explicit scenarios,
not an asserted broker/regulatory fee schedule. Every session ends flat or the
run fails; raw prices never carry an open position into a corporate-action gap.
"""
from dataclasses import dataclass,field,asdict,fields
from decimal import Decimal
import hashlib
import math
from pathlib import Path
import statistics
import sys

from .data import DataQualityError,finite_number
from .quant import IndicatorSpec
from .research import ResearchDataset


class BacktestError(RuntimeError):
    pass


def _d(value): return Decimal(str(value))


def _f(value):
    result=float(value)
    if not math.isfinite(result): raise BacktestError('backtest_numeric_overflow')
    return result


@dataclass(frozen=True)
class CostModel:
    commission_per_share_usd: float
    slippage_bps: float

    def __post_init__(self):
        for name in ('commission_per_share_usd','slippage_bps'):
            object.__setattr__(self,name,finite_number(getattr(self,name)))
        if self.commission_per_share_usd>100 or self.slippage_bps>1000:
            raise DataQualityError('cost scenario outside engineering bounds')

    def components(self,price,shares):
        return _d(self.commission_per_share_usd)*shares,price*shares*_d(self.slippage_bps)/10000


@dataclass(frozen=True)
class ReferencePolicy:
    spec: IndicatorSpec=field(default_factory=IndicatorSpec)
    entry_rsi_min: float=45
    entry_rsi_max: float=75
    exit_rsi: float=85
    min_volume_ratio: float=1
    max_volatility_fraction: float=0.03
    max_position_fraction: float=0.1
    max_participation: float=0.01

    def __post_init__(self):
        if type(self.spec) is not IndicatorSpec: raise DataQualityError('indicator specification required')
        for name in ('entry_rsi_min','entry_rsi_max','exit_rsi','min_volume_ratio',
                     'max_volatility_fraction','max_position_fraction','max_participation'):
            object.__setattr__(self,name,finite_number(getattr(self,name)))
        if not (0<=self.entry_rsi_min<self.entry_rsi_max<=100 and 0<=self.exit_rsi<=100
                and 0<self.max_position_fraction<=1 and 0<self.max_participation<=0.1
                and 0<self.max_volatility_fraction<=1):
            raise DataQualityError('invalid reference policy')

    def action(self,features,position_open):
        if position_open:
            return 'sell' if (features.sma_fast_usd<features.sma_slow_usd or features.momentum_fraction<0
                              or features.rsi_sma>self.exit_rsi) else 'hold'
        return 'buy' if (features.sma_fast_usd>features.sma_slow_usd and features.momentum_fraction>0
                         and self.entry_rsi_min<=features.rsi_sma<=self.entry_rsi_max
                         and features.volume_ratio>=self.min_volume_ratio
                         and features.volatility_15m_fraction<=self.max_volatility_fraction) else 'hold'


def research_code_hash():
    digest=hashlib.sha256()
    for name in ('data.py','alpaca.py','quant.py','research.py','backtest.py'):
        digest.update(name.encode()+b'\0'+Path(__file__).with_name(name).read_bytes()+b'\0')
    return digest.hexdigest()


def policy_record(policy):
    value={f.name:getattr(policy,f.name) for f in fields(ReferencePolicy)}
    value['spec']={f.name:getattr(policy.spec,f.name) for f in fields(IndicatorSpec)}
    return value


def backtest(dataset,*,policy,costs,initial_cash_usd=10000):
    if type(dataset) is not ResearchDataset or type(policy) is not ReferencePolicy or type(costs) is not CostModel:
        raise BacktestError('explicit_backtest_contract_required')
    initial=_d(finite_number(initial_cash_usd,positive=True))
    cash,quantity,entry=initial,0,None
    fills,trades,decisions,curve,daily=[],[],[],[],[]
    peak,max_drawdown=initial,Decimal(0)
    previous_daily=initial
    calendar=dataset.calendar

    def fill(side,price,shares,at,reason,requested):
        nonlocal cash,quantity,entry
        gross=price*shares
        commission,slippage=costs.components(price,shares)
        charge=commission+slippage
        if side=='buy':
            if quantity or gross+charge>cash: raise BacktestError('cash_or_duplicate_position_violation')
            cash-=gross+charge
            quantity=shares
            entry=dict(at=at,basis=gross+charge,shares=shares)
        else:
            if shares!=quantity or entry is None: raise BacktestError('invalid_simulated_exit')
            proceeds=gross-charge
            cash+=proceeds
            trades.append(dict(entered_at=entry['at'],exited_at=at,shares=shares,
                               net_pnl_usd=_f(proceeds-entry['basis'])))
            quantity,entry=0,None
        if cash<0: raise BacktestError('negative_simulated_cash')
        event=dict(side=side,at=at,price_usd=_f(price),shares=shares,requested_shares=requested,
                   gross_usd=_f(gross),commission_usd=_f(commission),slippage_usd=_f(slippage),
                   cost_usd=_f(charge),reason=reason)
        fills.append(event)
        return event

    for index,bar in enumerate(dataset.bars):
        session=calendar.at(bar.starts)
        last=bar.ends==session.closes
        features=None
        action,reason='hold','indicators_unavailable'
        try:
            features=dataset.indicators_for_open(index,spec=policy.spec)
            action,reason=policy.action(features,quantity>0),'quantitative_reference_rule'
        except DataQualityError:
            pass
        open_price=_d(bar.open)
        capacity=int(_d(bar.volume)*_d(policy.max_participation))
        event=None
        if action=='buy' and not last:
            budget=min(cash,(cash+quantity*open_price)*_d(policy.max_position_fraction))
            per_share=open_price*(1+_d(costs.slippage_bps)/10000)+_d(costs.commission_per_share_usd)
            requested=int(budget//per_share)
            shares=min(requested,capacity)
            if shares>0: event=fill('buy',open_price,shares,bar.starts.isoformat(),'next_open_model',requested)
            else: reason='cash_or_liquidity_insufficient'
        elif action=='buy' and last:
            reason='session_end_entry_veto'
        elif action=='sell' and quantity:
            if quantity<=capacity: event=fill('sell',open_price,quantity,bar.starts.isoformat(),'next_open_model',quantity)
            else: reason='exit_liquidity_unavailable'
        if last and quantity:
            if quantity>capacity: raise BacktestError('daily_liquidation_not_simulatable')
            event=fill('sell',_d(bar.close),quantity,bar.ends.isoformat(),'scheduled_session_close_model',quantity)
        equity=cash+quantity*_d(bar.close)
        peak=max(peak,equity)
        max_drawdown=max(max_drawdown,(peak-equity)/peak)
        curve.append(dict(at=bar.ends.isoformat(),equity_usd=_f(equity),cash_usd=_f(cash),position_shares=quantity))
        decisions.append(dict(at=bar.starts.isoformat(),index=index,features=asdict(features) if features else None,
                              action=action,reason=reason,fill=event))
        if last:
            daily.append(dict(day=session.day.isoformat(),equity_usd=_f(equity),return_fraction=_f(equity/previous_daily-1)))
            previous_daily=equity
    if quantity: raise BacktestError('backtest_ended_with_open_position')
    if abs((cash-initial)-sum((_d(t['net_pnl_usd']) for t in trades),Decimal(0)))>Decimal('0.00000001'):
        raise BacktestError('backtest_accounting_mismatch')
    daily_returns=[r['return_fraction'] for r in daily]
    annual_volatility=statistics.stdev(daily_returns)*math.sqrt(252) if len(daily_returns)>1 else None
    manifest=dataset.manifest()
    total_cost=sum((_d(f['cost_usd']) for f in fills),Decimal(0))
    turnover=sum((_d(f['gross_usd']) for f in fills),Decimal(0))/initial
    return dict(initial_cash_usd=_f(initial),ending_equity_usd=_f(cash),net_return_fraction=_f(cash/initial-1),
                maximum_drawdown_fraction=_f(max_drawdown),drawdown_basis='15min_close_equity',
                annualized_daily_volatility_fraction=_f(annual_volatility) if annual_volatility is not None else None,
                volatility_assumption='sample_daily_returns_sqrt_252_not_high_frequency_scaling',
                total_cost_usd=_f(total_cost),turnover_fraction_of_initial_equity=_f(turnover),
                final_position_shares=0,closed_trades=len(trades),
                win_rate=sum(t['net_pnl_usd']>0 for t in trades)/len(trades) if trades else None,
                fills=fills,trades=trades,decisions=decisions,equity_curve=curve,daily_equity=daily,
                policy=policy_record(policy),costs={f.name:getattr(costs,f.name) for f in fields(CostModel)},cost_assumption='uncalibrated_friction_scenario',
                execution_model='next_open_and_scheduled_close_with_completed_bar_volume_caps',
                availability_policy=dataset.availability_policy,universe_policy='fixed_symbol_research_only',
                validation_status='research_only_not_strategy_acceptance',broker_orders_submitted=0,
                data_sha256=manifest['data_sha256'],code_sha256=research_code_hash(),
                runtime=dict(python=sys.version.split()[0],numeric_engine='stdlib_decimal_and_statistics',
                             session_bounds='explicit_utc_with_system_America_New_York_validation'))
