"""Shared deterministic indicator formulas. No network, orders or confidence.

RSI here is explicitly the simple-window variant, not Wilder's recursive RSI.
Returns/volatility are fractions per 15-minute interval, never percentages or
annualized values. Caller-specific availability checks stay outside the math.
"""
from dataclasses import dataclass,fields
import math
import statistics
from .data import DataQualityError,finite_number,validate_series


@dataclass(frozen=True)
class IndicatorSpec:
    fast: int=5
    slow: int=20
    rsi: int=14
    momentum: int=4
    volatility: int=20
    volume: int=20

    def __post_init__(self):
        if any(type(getattr(self,f.name)) is not int or not 1<=getattr(self,f.name)<=500 for f in fields(IndicatorSpec)):
            raise DataQualityError('invalid indicator window')
        if self.fast>=self.slow or self.rsi<2 or self.volatility<2:
            raise DataQualityError('invalid indicator relationship')

    @property
    def required_bars(self):
        return max(self.slow,self.rsi+1,self.momentum+1,self.volatility+1,self.volume+1)


@dataclass(frozen=True)
class Indicators:
    last_price_usd: float
    sma_fast_usd: float
    sma_slow_usd: float
    rsi_sma: float
    momentum_fraction: float
    volatility_15m_fraction: float
    volume_ratio: float
    interval_seconds: int=900
    volatility_annualized: bool=False
    formula_version: str='simple-window-v1'


def _mean(values):
    return math.fsum(value/len(values) for value in values)


def indicators(closes,volumes,*,spec):
    if not isinstance(spec,IndicatorSpec) or not isinstance(closes,(list,tuple)) or not isinstance(volumes,(list,tuple)):
        raise DataQualityError('explicit indicator inputs required')
    if not spec.required_bars<=len(closes)<=10000 or len(closes)!=len(volumes):
        raise DataQualityError('insufficient or mismatched indicator input')
    closes=[finite_number(value,positive=True) for value in closes]
    volumes=[finite_number(value) for value in volumes]
    try:
        fast,slow=_mean(closes[-spec.fast:]),_mean(closes[-spec.slow:])
        changes=[b-a for a,b in zip(closes[-spec.rsi-1:-1],closes[-spec.rsi:])]
        gain=_mean([max(change,0) for change in changes])
        loss=_mean([max(-change,0) for change in changes])
        scale=max(gain,loss)
        rsi=50.0 if scale==0 else 100*(gain/scale)/(gain/scale+loss/scale)
        momentum=closes[-1]/closes[-spec.momentum-1]-1
        returns=[b/a-1 for a,b in zip(closes[-spec.volatility-1:-1],closes[-spec.volatility:])]
        if not all(math.isfinite(r) for r in returns): raise ValueError('return overflow')
        volatility=statistics.pstdev(returns)
        volume_base=_mean(volumes[-spec.volume-1:-1])
        if volume_base<=0: raise ValueError('zero prior-volume baseline')
        ratio=volumes[-1]/volume_base
        values=(closes[-1],fast,slow,rsi,momentum,volatility,ratio)
        if not all(math.isfinite(value) for value in values): raise ValueError('feature overflow')
        return Indicators(*values)
    except (ValueError,OverflowError,ZeroDivisionError):
        raise DataQualityError('indicators unavailable for numeric input') from None


def live_indicators(bars,*,calendar,as_of,spec):
    """Only receipt-available, closed contiguous intraday bars enter live math."""
    validate_series(bars,calendar=calendar,as_of=as_of)
    return indicators([bar.close for bar in bars],[bar.volume for bar in bars],spec=spec)
