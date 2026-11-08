"""Explicit historical availability, complete sessions and chronological splits.

This initial dataset is a fixed-symbol research input, not a historical market
universe. Raw bars may only form same-session feature windows. Corporate-action
and historical membership gates remain separate; no historical return is proof
of what a live receiver would have known.
"""
from dataclasses import dataclass,fields
from datetime import date,datetime,timedelta
import hashlib
import json
from .data import Bar,Calendar,Session,DataQualityError,utc_time
from .quant import IndicatorSpec,indicators


@dataclass(frozen=True)
class SessionSplit:
    train: tuple
    validation: tuple
    test: tuple
    embargo: tuple


def time_split(days,*,train_sessions,validation_sessions,test_sessions,embargo_sessions=1):
    counts=(train_sessions,validation_sessions,test_sessions)
    if (any(type(n) is not int or n<1 for n in counts) or type(embargo_sessions) is not int or embargo_sessions<0
            or not isinstance(days,(tuple,list)) or any(type(day) is not date for day in days)):
        raise DataQualityError('invalid chronological split')
    days=tuple(days)
    if days!=tuple(sorted(set(days))) or len(days)!=sum(counts)+2*embargo_sessions:
        raise DataQualityError('split requires exactly ordered complete sessions')
    a=train_sessions
    b=a+embargo_sessions
    c=b+validation_sessions
    d=c+embargo_sessions
    return SessionSplit(days[:a],days[b:c],days[d:],days[a:b]+days[c:d])


def _record(value,base):
    return {f.name:getattr(value,f.name) for f in fields(base)}


def _canonical(value):
    def encode(value):
        if isinstance(value,(date,datetime)): return value.isoformat()
        raise TypeError('unsupported research field')
    return json.dumps(value,default=encode,allow_nan=False,sort_keys=True,separators=(',',':'))


@dataclass(frozen=True)
class ResearchDataset:
    bars: tuple
    sessions: tuple
    calendar_start: str
    calendar_end: str
    downloaded_at: datetime
    availability_policy: str

    @classmethod
    def from_bars(cls,bars,*,calendar,downloaded_at,availability_policy):
        if not isinstance(calendar,Calendar) or not isinstance(bars,(tuple,list)):
            raise DataQualityError('explicit historical dataset contract required')
        sessions=tuple(Session(s.day,s.opens,s.closes) for _,s in sorted(calendar.sessions.items()))
        return cls(tuple(bars),sessions,calendar.start.isoformat(),calendar.end.isoformat(),utc_time(downloaded_at),availability_policy)

    @property
    def calendar(self):
        return Calendar(self.sessions,start=self.calendar_start,end=self.calendar_end)

    def __post_init__(self):
        if self.availability_policy not in ('vendor_history_assumed_at_close','archived_receipts'):
            raise DataQualityError('historical availability policy required')
        object.__setattr__(self,'downloaded_at',utc_time(self.downloaded_at))
        if type(self.bars) is not tuple or not 1<=len(self.bars)<=100000 or type(self.sessions) is not tuple:
            raise DataQualityError('empty or unbounded historical dataset')
        calendar=self.calendar
        if not calendar.sessions: raise DataQualityError('historical calendar empty')
        first=self.bars[0]
        if not isinstance(first,Bar) or first.feed!='iex': raise DataQualityError('unsupported historical bar contract')
        identity=lambda b:(b.symbol,b.source,b.feed,b.adjustment,b.interval_seconds,b.currency,b.volume_unit)
        previous=None
        actual={s.day:[] for s in self.sessions}
        for bar in self.bars:
            if not isinstance(bar,Bar) or identity(bar)!=identity(first):
                raise DataQualityError('mixed historical bar identity')
            if previous is not None and bar.starts<=previous.starts:
                raise DataQualityError('duplicate or unordered historical bar')
            session=calendar.at(bar.starts)
            if session is None: raise DataQualityError('off-session historical bar')
            bar.validate(as_of=self.downloaded_at,session=session)
            actual[session.day].append(bar.starts)
            previous=bar
        for session in self.sessions:
            duration=(session.closes-session.opens).total_seconds()
            if duration%900: raise DataQualityError('session not representable by 15-minute bars')
            expected=[session.opens+timedelta(seconds=900*i) for i in range(int(duration//900))]
            if actual[session.day]!=expected:
                raise DataQualityError('incomplete historical session')

    def manifest(self):
        payload=dict(symbol=self.bars[0].symbol,source='alpaca',feed='iex',adjustment='raw',interval_seconds=900,
                     currency='USD',volume_unit='shares',downloaded_at=self.downloaded_at,
                     availability_policy=self.availability_policy,universe_policy='fixed_symbol_research_only',
                     calendar=dict(start=self.calendar_start,end=self.calendar_end,
                                   sessions=[_record(s,Session) for s in self.sessions]),
                     bars=[_record(b,Bar) for b in self.bars])
        normalized=json.loads(_canonical(payload))
        normalized['data_sha256']=hashlib.sha256(_canonical(payload).encode()).hexdigest()
        return normalized

    def indicators_for_open(self,index,*,spec):
        if type(index) is not int or not 0<=index<len(self.bars) or not isinstance(spec,IndicatorSpec):
            raise DataQualityError('invalid historical event')
        now=self.bars[index].starts
        session=self.calendar.at(now)
        available=[bar for bar in self.bars[:index] if session.opens<=bar.starts and bar.ends<=now
                   and (self.availability_policy=='vendor_history_assumed_at_close' or bar.received_at<=now)]
        if len(available)<spec.required_bars:
            raise DataQualityError('historical warmup unavailable')
        window=available[-spec.required_bars:]
        if any(a.ends!=b.starts for a,b in zip(window,window[1:])) or not 0<=(now-window[-1].ends).total_seconds()<=1020:
            raise DataQualityError('historical window missing or stale')
        return indicators([b.close for b in window],[b.volume for b in window],spec=spec)

    def subset(self,days):
        if not isinstance(days,(tuple,list)) or not days or any(type(d) is not date for d in days):
            raise DataQualityError('explicit research sessions required')
        days=tuple(days)
        if days!=tuple(sorted(set(days))) or not set(days).issubset({s.day for s in self.sessions}):
            raise DataQualityError('unknown or unordered research sessions')
        selected=[s for s in self.sessions if s.day in days]
        calendar=Calendar(selected,start=days[0].isoformat(),end=days[-1].isoformat())
        bars=[b for b in self.bars if any(s.opens<=b.starts<s.closes for s in selected)]
        return ResearchDataset.from_bars(bars,calendar=calendar,downloaded_at=self.downloaded_at,
                                         availability_policy=self.availability_policy)
