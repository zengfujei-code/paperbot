"""Deterministic current-universe liquidity screen, with no order capability.

Daily volume and the price-times-volume proxy are IEX provider-day measures,
not consolidated market volume or measured traded dollar turnover.
"""
from dataclasses import dataclass
import math
from .alpaca import UniverseSnapshot, Asset
from .data import Snapshot, Calendar, DataQualityError, finite_number, utc_time


@dataclass(frozen=True)
class ScanPolicy:
    min_price: float = 5
    min_daily_volume: float = 10000
    max_spread_bps: float = 30
    max_candidates: int = 50
    universe_max_age_seconds: int = 86400

    def __post_init__(self):
        for name in ('min_price','min_daily_volume','max_spread_bps'):
            object.__setattr__(self,name,finite_number(getattr(self,name),positive=True))
        if type(self.max_candidates) is not int or not 1 <= self.max_candidates <= 1000:
            raise DataQualityError('invalid candidate budget')
        if type(self.universe_max_age_seconds) is not int or not 1 <= self.universe_max_age_seconds <= 86400:
            raise DataQualityError('invalid universe age policy')


@dataclass(frozen=True)
class Candidate:
    symbol: str
    price: float
    spread_bps: float
    iex_daily_volume: float
    price_volume_proxy_usd: float


@dataclass(frozen=True)
class ScanResult:
    as_of: object
    status: str
    candidates: tuple
    rejected: dict
    policy: ScanPolicy
    feed: str = 'iex'
    universe_coverage: str = 'current_snapshot_only'


def scan(universe, snapshots, *, calendar, as_of, policy, snapshot_rejections=None):
    now = utc_time(as_of)
    if not isinstance(policy,ScanPolicy) or not isinstance(calendar,Calendar) or not isinstance(snapshots,dict):
        raise DataQualityError('explicit scanner contracts required')
    if not isinstance(universe,UniverseSnapshot) or (universe.source,universe.coverage) != ('alpaca','current_snapshot_only'):
        raise DataQualityError('unsupported universe contract')
    if not 0 <= (now-utc_time(universe.received_at)).total_seconds() <= policy.universe_max_age_seconds:
        raise DataQualityError('future or stale universe')
    if any(not isinstance(a,Asset) for a in universe.assets):
        raise DataQualityError('invalid universe asset')
    symbols = [a.symbol for a in universe.assets]
    if len(symbols) != len(set(symbols)):
        raise DataQualityError('duplicate universe symbol')
    snapshot_rejections={} if snapshot_rejections is None else snapshot_rejections
    if (not isinstance(snapshot_rejections,dict) or not set(snapshot_rejections).issubset(symbols)
            or any(reason not in ('missing_snapshot','invalid_snapshot') for reason in snapshot_rejections.values())
            or set(snapshot_rejections)&set(snapshots)):
        raise DataQualityError('invalid snapshot rejection contract')
    if calendar.at(now) is None:
        return ScanResult(now,'market_closed',(),{s:'market_closed' for s in sorted(symbols)},policy)
    rejected, candidates = {}, []
    for symbol in sorted(symbols):
        value = snapshots.get(symbol)
        if value is None:
            rejected[symbol] = 'snapshot_invalid_or_stale' if snapshot_rejections.get(symbol)=='invalid_snapshot' else 'snapshot_missing'
            continue
        if not isinstance(value,Snapshot) or value.symbol != symbol:
            rejected[symbol] = 'snapshot_identity'
            continue
        try:
            value.validate(as_of=now,calendar=calendar)
        except DataQualityError:
            rejected[symbol] = 'snapshot_invalid_or_stale'
            continue
        if value.bid < policy.min_price or value.ask < policy.min_price:
            rejected[symbol] = 'quote_below_price_floor'
            continue
        mid = value.bid + (value.ask-value.bid)/2
        spread = (value.ask-value.bid)/mid*10000
        proxy = value.price*value.daily_volume
        if not math.isfinite(spread) or not math.isfinite(proxy):
            rejected[symbol] = 'numeric_overflow'
        elif value.price < policy.min_price:
            rejected[symbol] = 'price_below_minimum'
        elif value.daily_volume < policy.min_daily_volume:
            rejected[symbol] = 'volume_below_minimum'
        elif spread > policy.max_spread_bps:
            rejected[symbol] = 'spread_above_maximum'
        elif value.bid_size <= 0 or value.ask_size <= 0:
            rejected[symbol] = 'empty_quote_size'
        else:
            candidates.append(Candidate(symbol,value.price,spread,value.daily_volume,proxy))
    candidates.sort(key=lambda c:(-c.price_volume_proxy_usd,c.symbol))
    for value in candidates[policy.max_candidates:]:
        rejected[value.symbol] = 'candidate_budget'
    return ScanResult(now,'ok',tuple(candidates[:policy.max_candidates]),dict(sorted(rejected.items())),policy)
