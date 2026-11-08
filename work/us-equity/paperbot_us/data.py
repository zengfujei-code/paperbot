"""Strict market-data contracts; this module cannot submit orders."""
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import math
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

UTC = timezone.utc


class DataQualityError(ValueError):
    pass


def utc_time(value):
    try:
        if isinstance(value, str):
            # Vendor events use RFC3339 nanoseconds. Python 3.10's parser only
            # accepts selected fractional lengths, unlike newer Windows Python.
            # Normalize explicitly to microseconds by flooring, never rounding
            # an event forward in time. Reject runtime-dependent ISO extensions.
            match=re.fullmatch(r'(\d{4}-\d{2}-\d{2})[Tt](\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?([Zz]|[+-]\d{2}:\d{2})',value)
            if match is None: raise DataQualityError('RFC3339 timestamp required')
            day,clock,fraction,offset=match.groups()
            if offset not in ('Z','z') and (int(offset[1:3])>23 or int(offset[4:6])>59):
                raise DataQualityError('invalid timestamp offset')
            fraction='.'+(fraction+'000000')[:6] if fraction else ''
            offset='+00:00' if offset in ('Z','z') else offset
            value = datetime.fromisoformat(day+'T'+clock+fraction+offset)
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise DataQualityError('timestamp requires an explicit timezone')
        return value.astimezone(UTC)
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataQualityError('invalid timestamp') from None


def finite_number(value, *, positive=False):
    try:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise DataQualityError('invalid numeric market field')
        value = float(value)
        if value < 0 or (positive and value == 0):
            raise DataQualityError('negative or zero market field')
        return value
    except (ValueError, TypeError, OverflowError):
        raise DataQualityError('invalid numeric market field') from None


def _ny():
    try:
        return ZoneInfo('America/New_York')
    except ZoneInfoNotFoundError:
        raise DataQualityError('IANA America/New_York timezone data is required') from None


@dataclass(frozen=True)
class Session:
    day: date
    opens: datetime
    closes: datetime

    def __post_init__(self):
        if type(self.day) is not date:
            raise DataQualityError('invalid calendar date')
        opens, closes = utc_time(self.opens), utc_time(self.closes)
        if not opens < closes or closes-opens > timedelta(hours=24):
            raise DataQualityError('invalid session bounds')
        if opens.astimezone(_ny()).date() != self.day or closes.astimezone(_ny()).date() != self.day:
            raise DataQualityError('session date mismatch')
        object.__setattr__(self, 'opens', opens)
        object.__setattr__(self, 'closes', closes)

    @classmethod
    def from_alpaca(cls, record):
        try:
            day = date.fromisoformat(record['date'])
            opened, closed = time.fromisoformat(record['open']), time.fromisoformat(record['close'])
            if opened.tzinfo is not None or closed.tzinfo is not None:
                raise DataQualityError('provider calendar times must be local exchange times')
            return cls(day, datetime.combine(day, opened, _ny()), datetime.combine(day, closed, _ny()))
        except (KeyError, TypeError, ValueError):
            raise DataQualityError('invalid provider calendar record') from None


class Calendar:
    """Absence means closed only inside the explicitly downloaded coverage."""
    def __init__(self, sessions, *, start, end):
        try:
            self.start, self.end = date.fromisoformat(start), date.fromisoformat(end)
        except (TypeError, ValueError):
            raise DataQualityError('invalid calendar coverage') from None
        if self.start > self.end:
            raise DataQualityError('inverted calendar coverage')
        self.sessions = {}
        for session in sessions:
            if not isinstance(session, Session) or not self.start <= session.day <= self.end:
                raise DataQualityError('session outside calendar coverage')
            if session.day in self.sessions:
                raise DataQualityError('duplicate calendar session')
            self.sessions[session.day] = session

    def at(self, timestamp):
        now = utc_time(timestamp)
        day = now.astimezone(_ny()).date()
        if not self.start <= day <= self.end:
            raise DataQualityError('calendar coverage unavailable')
        session = self.sessions.get(day)
        return session if session and session.opens <= now < session.closes else None


@dataclass(frozen=True)
class Bar:
    symbol: str
    starts: datetime
    received_at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str = 'alpaca'
    feed: str = 'iex'
    adjustment: str = 'raw'
    interval_seconds: int = 900
    currency: str = 'USD'
    volume_unit: str = 'shares'

    def __post_init__(self):
        if not isinstance(self.symbol, str) or not re.fullmatch(r'[A-Z][A-Z0-9.\-/]{0,19}', self.symbol):
            raise DataQualityError('invalid stock symbol')
        if self.source != 'alpaca' or self.feed not in ('iex', 'sip'):
            raise DataQualityError('unsupported source or feed')
        if self.adjustment != 'raw' or self.currency != 'USD' or self.volume_unit != 'shares':
            raise DataQualityError('unsupported price basis or units')
        if type(self.interval_seconds) is not int or self.interval_seconds != 900:
            raise DataQualityError('this research contract requires 15-minute bars')
        object.__setattr__(self, 'starts', utc_time(self.starts))
        object.__setattr__(self, 'received_at', utc_time(self.received_at))
        try:
            self.ends
        except OverflowError:
            raise DataQualityError('bar interval outside representable dates') from None
        for name in ('open', 'high', 'low', 'close', 'volume'):
            object.__setattr__(self, name, finite_number(getattr(self, name), positive=name != 'volume'))
        if not self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high:
            raise DataQualityError('inconsistent OHLC prices')

    @property
    def ends(self):
        return self.starts + timedelta(seconds=self.interval_seconds)

    @classmethod
    def from_alpaca(cls, symbol, record, *, received_at, feed='iex'):
        try:
            return cls(symbol=symbol, starts=record['t'], received_at=received_at,
                       open=record['o'], high=record['h'], low=record['l'], close=record['c'],
                       volume=record['v'], feed=feed)
        except (KeyError, TypeError):
            raise DataQualityError('incomplete provider bar') from None

    def validate(self, *, as_of, session):
        now = utc_time(as_of)
        if not isinstance(session, Session) or not session.opens <= self.starts < self.ends <= session.closes:
            raise DataQualityError('bar outside regular session')
        if (self.starts-session.opens).total_seconds() % self.interval_seconds:
            raise DataQualityError('bar misaligned with session')
        if not self.ends <= self.received_at <= now:
            raise DataQualityError('unclosed, future or unavailable bar')


def validate_series(bars, *, as_of, calendar=None, max_age_seconds=1020):
    """Validate one contiguous intraday input window, not an overnight history."""
    now = utc_time(as_of)
    if not isinstance(calendar, Calendar):
        raise DataQualityError('explicit session calendar is required')
    if not bars or type(max_age_seconds) not in (int, float) or not 0 < max_age_seconds <= 3600:
        raise DataQualityError('empty series or invalid freshness policy')
    first = bars[0]
    if not isinstance(first, Bar):
        raise DataQualityError('invalid bar type')
    identity = lambda b: (b.symbol,b.source,b.feed,b.adjustment,b.interval_seconds,b.currency,b.volume_unit)
    previous = None
    for bar in bars:
        if not isinstance(bar, Bar) or identity(bar) != identity(first):
            raise DataQualityError('mixed market-data contract')
        session = calendar.at(bar.starts)
        if session is None:
            raise DataQualityError('off-session market input')
        bar.validate(as_of=now,session=session)
        if not bar.ends <= bar.received_at <= now:
            raise DataQualityError('future or unclosed input')
        if previous and bar.starts != previous.ends:
            raise DataQualityError('duplicate, unordered or missing bar')
        previous = bar
    if not 0 <= (now-bars[-1].ends).total_seconds() <= max_age_seconds:
        raise DataQualityError('stale market input')


@dataclass(frozen=True)
class Snapshot:
    symbol: str
    price: float
    trade_at: datetime
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    quote_at: datetime
    daily_volume: float
    daily_session_day: date
    received_at: datetime
    source: str = 'alpaca'
    feed: str = 'iex'
    currency: str = 'USD'
    volume_unit: str = 'shares'
    volume_period: str = 'provider_day_including_available_sessions'

    def __post_init__(self):
        if not isinstance(self.symbol,str) or not re.fullmatch(r'[A-Z][A-Z0-9.\-/]{0,19}',self.symbol):
            raise DataQualityError('invalid snapshot symbol')
        if (self.source,self.feed,self.currency,self.volume_unit,self.volume_period) != (
                'alpaca','iex','USD','shares','provider_day_including_available_sessions'):
            raise DataQualityError('unsupported snapshot provenance or units')
        for name in ('price','bid','ask','bid_size','ask_size','daily_volume'):
            object.__setattr__(self,name,finite_number(getattr(self,name),positive=name in ('price','bid','ask')))
        for name in ('trade_at','quote_at','received_at'):
            object.__setattr__(self,name,utc_time(getattr(self,name)))
        if self.ask < self.bid:
            raise DataQualityError('crossed quote')
        if not 0 <= (self.received_at-self.trade_at).total_seconds() <= 120 or not 0 <= (self.received_at-self.quote_at).total_seconds() <= 30:
            raise DataQualityError('future or stale snapshot')
        if type(self.daily_session_day) is not date or self.daily_session_day != self.trade_at.astimezone(_ny()).date():
            raise DataQualityError('daily aggregate date mismatch')

    @classmethod
    def from_alpaca(cls, symbol, record, *, received_at):
        try:
            if not isinstance(symbol,str) or not re.fullmatch(r'[A-Z][A-Z0-9.\-/]{0,19}',symbol):
                raise DataQualityError('invalid snapshot symbol')
            trade,quote,daily = record['latestTrade'],record['latestQuote'],record['dailyBar']
            price = finite_number(trade['p'],positive=True)
            bid,ask = finite_number(quote['bp'],positive=True),finite_number(quote['ap'],positive=True)
            bid_size,ask_size = finite_number(quote['bs']),finite_number(quote['as'])
            daily_volume = finite_number(daily['v'])
            received = utc_time(received_at)
            trade_at,quote_at,day_at = utc_time(trade['t']),utc_time(quote['t']),utc_time(daily['t'])
            if not 0 <= (received-trade_at).total_seconds() <= 120 or not 0 <= (received-quote_at).total_seconds() <= 30:
                raise DataQualityError('future or stale snapshot')
            if ask < bid or day_at > received:
                raise DataQualityError('crossed quote or future daily aggregate')
            day = day_at.astimezone(_ny()).date()
            if day != trade_at.astimezone(_ny()).date():
                raise DataQualityError('daily aggregate date mismatch')
            return cls(symbol,price,trade_at,bid,ask,bid_size,ask_size,quote_at,daily_volume,day,received)
        except (KeyError,TypeError,ValueError,OverflowError):
            raise DataQualityError('invalid or stale snapshot') from None

    def validate(self, *, as_of, calendar):
        now = utc_time(as_of)
        if not isinstance(calendar,Calendar) or now < self.received_at:
            raise DataQualityError('snapshot availability or calendar missing')
        current = calendar.at(now)
        if current is None or not (current.opens <= self.trade_at < current.closes and current.opens <= self.quote_at < current.closes):
            raise DataQualityError('off-session snapshot')
        if not 0 <= (now-self.trade_at).total_seconds() <= 120 or not 0 <= (now-self.quote_at).total_seconds() <= 30:
            raise DataQualityError('stale snapshot at decision time')
