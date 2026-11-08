"""Causal immutable inputs across explicitly bound original/successor segments.

No outcome/model/client/order calls. This evidence is not prospective issue or
phase authority. Future math replay uses this captured input, not later prices.
"""
from contextlib import closing,ExitStack
from dataclasses import asdict,dataclass
from datetime import datetime,timedelta,timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sqlite3

from paperbot_us import alpaca,data,regime_baselines
from . import canonical_history,common_registration,feature_pipeline,receipt_book,training_journal
from . import training_codec as codec
from .contracts import aware,digest,is_sha
from .source_views import VerifiedFeatureView
from .training_data import SYMBOLS,TIMEZONE_SHA256,_watch_versions

RANGES=(('original','2026-10-27','2026-11-25'),('successor','2026-11-27','2026-12-11'))


class ForecastInputError(ValueError):pass


def _need(value):
    if not value:raise ForecastInputError('causal_issue_input_unavailable')


def source_sha256():
    return digest(dict(reader=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        raw=receipt_book._source(),feature=feature_pipeline._source(),
        training=training_journal.source_sha256(),canonical=canonical_history.source_sha256()))


@dataclass(frozen=True)
class SourcePair:
    raw_book:receipt_book.ReceiptBook
    feature_journal:feature_pipeline.FeatureJournal

    def __post_init__(self):
        _need(type(self.raw_book) is receipt_book.ReceiptBook
              and type(self.feature_journal) is feature_pipeline.FeatureJournal)


@dataclass(frozen=True)
class SegmentProof:
    name:str
    start_day:str
    end_day:str
    raw_path:str
    raw_device:int
    raw_inode:int
    raw_plan_sha256:str
    raw_source_sha256:str
    raw_head_seq:int
    raw_head_digest:str
    raw_last_at:datetime
    feature_path:str
    feature_device:int
    feature_inode:int
    feature_plan_sha256:str
    feature_source_sha256:str
    feature_head_seq:int
    feature_head_digest:str
    feature_last_at:datetime


@dataclass(frozen=True)
class BarWitness:
    segment:str
    starts_at:datetime
    first_recorded_at:datetime
    bar_sha256:str
    late:bool


@dataclass(frozen=True)
class IssueInput:
    symbol:str
    registration:common_registration.RegistrationReceipt
    training_snapshot_sha256:str
    fits_sha256:str
    bundle:object
    segments:tuple
    returns:tuple
    bar_witnesses:tuple
    expected_return_slots:int
    missing_return_slots:int
    late_return_slots:int
    read_started_at:datetime
    available_at:datetime
    source_sha256:str
    seal:str

    def __post_init__(self):
        from .features import FeatureBundle
        _need(self.symbol in SYMBOLS and type(self.registration) is common_registration.RegistrationReceipt
              and type(self.bundle) is FeatureBundle and self.bundle.price_identity[0]==self.symbol
              and self.available_at>=self.read_started_at and self.available_at>=self.bundle.issued_at
              and self.available_at>=self.registration.loaded_at
              and self.bundle.vector.event_at<=self.available_at<=self.bundle.vector.event_at+timedelta(seconds=120)
              and all(is_sha(v) for v in (self.training_snapshot_sha256,self.fits_sha256)))
        _need(type(self.segments) is tuple and tuple((s.name,s.start_day,s.end_day) for s in self.segments)==RANGES
              and all(type(s) is SegmentProof and s.raw_last_at<=self.available_at
                      and s.feature_last_at<=self.available_at for s in self.segments)
              and type(self.returns) is tuple and type(self.bar_witnesses) is tuple
              and type(self.expected_return_slots) is int and type(self.missing_return_slots) is int
              and type(self.late_return_slots) is int
              and self.expected_return_slots-len(self.returns)==self.missing_return_slots>=0
              and 0<=self.late_return_slots<=len(self.returns)<=500)
        previous=None
        for row in self.returns:
            _need(type(row) is regime_baselines.ReturnObservation and row.available_at==self.available_at
                  and (previous is None or row.starts_at>=previous) and row.ends_at<=self.bundle.vector.event_at)
            previous=row.ends_at
        _need(previous==self.bundle.vector.event_at and len(self.bar_witnesses)<=1000
              and all(type(w) is BarWitness and w.segment in ('original','successor')
                      and w.first_recorded_at<=self.available_at and is_sha(w.bar_sha256)
                      and type(w.late) is bool for w in self.bar_witnesses)
              and self.source_sha256==source_sha256()
              and self.seal==digest({k:v for k,v in asdict(self).items() if k!='seal'}))


class InputClock:
    def __init__(self,feature_key,timezone_file,clock):
        _need(type(feature_key) is str)
        self.symbol,raw=feature_key.split('|',1);self.event_at=aware(raw)
        _need(self.symbol in SYMBOLS and feature_key==feature_pipeline._entry_key(self.symbol,self.event_at))
        self.base=canonical_history.Clock(timezone_file,clock)

    def __call__(self):
        at=self.base();local=self.event_at.astimezone(self.base.zone)
        _need(local.date().isoformat() in common_registration.VALIDATION_DAYS+common_registration.TEST_DAYS
              and self.event_at<=at<=self.event_at+timedelta(seconds=120))
        return at


def _identity(path):
    _need(path.is_file() and not path.is_symlink() and not path.parent.is_symlink())
    stat=path.stat()
    return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns


def _metadata(pair,index,training_receipt,guard):
    name,start,end=RANGES[index];raw,feature=pair.raw_book,pair.feature_journal
    raw_identity,feature_identity=_identity(raw.path),_identity(feature.path)
    now=guard()
    with closing(sqlite3.connect(raw.path.as_uri()+'?mode=ro',uri=True,isolation_level=None)) as db:
        values=db.execute('SELECT plan_sha256,source_sha256,head_seq,head_digest,last_at FROM metadata').fetchall()
        _need(len(values)==1);plan_sha,source,seq,head,last=values[0]
        _need(aware(last)<=now and source==receipt_book._source())
        # Select timestamps/kinds only; never materialize reserved-period prices.
        forbidden=db.execute("SELECT COUNT(*) FROM events WHERE json_extract(payload,'$.kind') IN ('receipt','revision') "
            "AND substr(json_extract(payload,'$.body.bar.starts'),1,10) BETWEEN '2026-11-10' AND '2026-11-23'").fetchone()[0]
        _need(forbidden==0)
        created=alpaca.decode_json(db.execute('SELECT payload FROM events WHERE seq=1').fetchone()[0])
        _need(created['kind']=='created');raw_plan=receipt_book._plan(created['body']['plan'])
        _need(raw_plan.symbols==SYMBOLS and raw_plan.start_day==start and raw_plan.end_day==end
              and raw_plan.timezone_sha256==TIMEZONE_SHA256 and digest(asdict(raw_plan))==plan_sha)
        if index==0:
            binding=next(b for b in training_receipt.training.bindings if b.kind=='raw')
            _need((str(raw.path),raw_identity[0],raw_identity[1],plan_sha,source)==
                  (binding.path,binding.device,binding.inode,binding.plan_sha256,binding.source_sha256)
                  and db.execute('SELECT digest FROM events WHERE seq=?',(binding.head_seq,)).fetchone()==(binding.head_digest,))
    now=guard()
    with closing(sqlite3.connect(feature.path.as_uri()+'?mode=ro',uri=True,isolation_level=None)) as db:
        values=db.execute('SELECT head_seq,head_digest FROM metadata').fetchall();_need(len(values)==1)
        fseq,fhead=values[0]
        latest=db.execute("SELECT MAX(json_extract(payload,'$.generated_at')) FROM observations").fetchone()[0]
        _need(type(latest) in (float,int) and latest<=now.timestamp())
        created=alpaca.decode_json(db.execute('SELECT payload FROM observations WHERE seq=1').fetchone()[0])
        _need(created['kind']=='created' and created['source_sha256']==feature_pipeline._source())
        plan=feature_pipeline._plan(created['body']['plan'])
        expected=tuple((datetime(2026,11,10)+timedelta(days=i)).date().isoformat() for i in range(14)) if index==0 else ()
        _need(plan.symbols==SYMBOLS and plan.start_day==start and plan.end_day==end and plan.lookback_returns==8
              and plan.threshold_fraction==.005 and plan.paused_days==expected
              and plan.raw_plan_sha256==plan_sha and plan.raw_source_sha256==source
              and plan.timezone_sha256==TIMEZONE_SHA256)
        if index==0:
            binding=next(b for b in training_receipt.training.bindings if b.kind=='feature')
            _need((str(feature.path),feature_identity[0],feature_identity[1],digest(asdict(plan)),created['source_sha256'])==
                  (binding.path,binding.device,binding.inode,binding.plan_sha256,binding.source_sha256)
                  and db.execute('SELECT digest FROM observations WHERE seq=?',(binding.head_seq,)).fetchone()==(binding.head_digest,))
    return SegmentProof(name,start,end,str(raw.path),raw_identity[0],raw_identity[1],plan_sha,source,seq,head,aware(last),
        str(feature.path),feature_identity[0],feature_identity[1],digest(asdict(plan)),created['source_sha256'],fseq,fhead,
        datetime.fromtimestamp(latest,timezone.utc))


def _collect_returns(raw_states,proofs,plan,symbol,event,guard):
    rows=[];witnesses={};expected=0;late=0
    for session_value in plan['calendar']['sessions']:
        opens,closes=aware(session_value['opens']),aware(session_value['closes'])
        if opens>=event:continue
        limit=min(closes,event);count=int((limit-opens).total_seconds()//900)
        expected+=max(0,count-1)
        index=next(i for i,p in enumerate(proofs) if p.start_day<=session_value['day']<=p.end_day)
        raw=raw_states[index];calendar=raw['calendars'].get(session_value['day'])
        if calendar is None:continue
        _need(not calendar['changed'] and calendar['session']==session_value)
        session=receipt_book._session(calendar['session'])
        for i in range(1,count):
            starts=(opens+timedelta(seconds=(i-1)*900),opens+timedelta(seconds=i*900))
            slots=tuple(raw['slots'].get(receipt_book._key(symbol,s)) for s in starts)
            if any(s is None or s['first'] is None or s['first']['volume']<=0 for s in slots):continue
            bars=tuple(data.Bar(**s['first']) for s in slots)
            now=guard()
            for bar in bars:bar.validate(as_of=now,session=session)
            _need(bars[0].ends==bars[1].starts and bars[1].ends<=event and all(b.symbol==symbol for b in bars))
            a,b=(Fraction(str(bar.close)) for bar in bars)
            rows.append((bars[1].ends,float((b-a)/a)))
            late+=int(any(not s['on_time'] for s in slots))
            for bar,slot in zip(bars,slots):
                key=(proofs[index].name,bar.starts)
                witness=BarWitness(proofs[index].name,bar.starts,aware(slot['first_recorded_at']),
                                   digest(slot['first']),not slot['on_time'])
                _need(key not in witnesses or witnesses[key]==witness);witnesses[key]=witness
    return rows,tuple(witnesses[k] for k in sorted(witnesses)),expected,late


def capture_issue_input(training_receipt,registration_receipt,pairs,feature_key,*,timezone_file,clock):
    try:
        guard=InputClock(feature_key,timezone_file,clock);started=guard()
        _need(type(training_receipt) is training_journal.TrainingReceipt
              and type(registration_receipt) is common_registration.RegistrationReceipt
              and type(pairs) is tuple and len(pairs)==2 and all(type(p) is SourcePair for p in pairs))
        training_receipt.__post_init__();registration_receipt.__post_init__()
        _need(started>=training_receipt.loaded_at and started>=registration_receipt.loaded_at)
        plan=registration_receipt.plan()
        _need(plan['data_sha256']==training_receipt.training.seal)
        source=source_sha256();paths=tuple(path for pair in pairs for path in (pair.raw_book.path,pair.feature_journal.path))
        identities=tuple(_identity(p) for p in paths)
        with ExitStack() as stack:
            unchanged=stack.enter_context(_watch_versions(paths,guard))
            proofs=tuple(_metadata(pair,i,training_receipt,guard) for i,pair in enumerate(pairs))
            day=guard.event_at.astimezone(guard.base.zone).date().isoformat()
            index=next(i for i,p in enumerate(proofs) if p.start_day<=day<=p.end_day)
            view=stack.enter_context(VerifiedFeatureView(pairs[index].feature_journal,read_guard=guard))
            feature=view.snapshot(deep=True);entry=feature['entries'].get(feature_key)
            _need(entry is not None and entry['phase']=='active' and not entry['quarantined'])
            bundle=feature_pipeline._bundle(entry['accepted_bundle'])
            _need(bundle.vector.event_at==guard.event_at and bundle.price_identity[0]==guard.symbol)
            expected_calendar=next(s for s in plan['calendar']['sessions'] if s['day']==day)
            _need(dict(entry['session'])==expected_calendar)
            raw_states=[]
            for i,pair in enumerate(pairs):
                guard()
                with pair.raw_book._transaction() as (db,raw,meta):
                    _need((raw['head_seq'],raw['head_digest'],aware(meta['last_at']))==
                          (proofs[i].raw_head_seq,proofs[i].raw_head_digest,proofs[i].raw_last_at))
                    if i==index:
                        context=entry['raw_context']
                        _need(db.execute('SELECT digest FROM events WHERE seq=?',(context['sequence'],)).fetchone()==(context['digest'],))
                        calendar=raw['calendars'].get(day)
                        _need(calendar is not None and not calendar['changed'] and calendar['session']==expected_calendar)
                        for original in entry['bars']:
                            slot=raw['slots'].get(receipt_book._key(guard.symbol,aware(original['starts'])))
                            _need(slot is not None and slot['first']==dict(original) and slot['on_time'])
                    raw_states.append(raw)
            rows,witnesses,expected,late=_collect_returns(raw_states,proofs,plan,guard.symbol,guard.event_at,guard)
            view.snapshot()
            _need(unchanged() and tuple(_identity(p) for p in paths)==identities and source_sha256()==source)
            available=guard()
        fields=dict(symbol=guard.symbol,registration=registration_receipt,
            training_snapshot_sha256=training_receipt.training.seal,fits_sha256=training_receipt.fits.seal,
            bundle=bundle,segments=proofs,
            returns=tuple(regime_baselines.ReturnObservation(end,available,value) for end,value in rows),
            bar_witnesses=witnesses,expected_return_slots=expected,missing_return_slots=expected-len(rows),
            late_return_slots=late,read_started_at=started,available_at=available,source_sha256=source)
        return IssueInput(**fields,seal=digest({k:asdict(v) if k in ('registration','bundle') else
            [asdict(x) for x in v] if k in ('segments','returns','bar_witnesses') else v for k,v in fields.items()}))
    except Exception:
        raise ForecastInputError('causal_issue_input_capture_failed') from None


def encode_input(value):
    try:
        _need(type(value) is IssueInput);value.__post_init__()
        raw=asdict(value)
        raw['returns']=[[r.ends_at,r.return_fraction] for r in value.returns]
        raw['bar_witnesses']=[[w.segment,w.starts_at,w.first_recorded_at,w.bar_sha256,w.late] for w in value.bar_witnesses]
        return codec.encode(raw)
    except Exception:
        raise ForecastInputError('causal_issue_input_encoding_failed') from None


def decode_input(value):
    try:
        raw=codec._raw(value)
        registration=dict(raw['registration'])
        for name in ('confirmed_at','loaded_at'):registration[name]=aware(registration[name])
        raw['registration']=common_registration.RegistrationReceipt(**registration)
        raw['bundle']=feature_pipeline._bundle(raw['bundle'])
        raw['segments']=tuple(SegmentProof(**{**p,'raw_last_at':aware(p['raw_last_at']),
            'feature_last_at':aware(p['feature_last_at'])}) for p in raw['segments'])
        for name in ('read_started_at','available_at'):raw[name]=aware(raw[name])
        raw['returns']=tuple(regime_baselines.ReturnObservation(end,raw['available_at'],value) for end,value in raw['returns'])
        raw['bar_witnesses']=tuple(BarWitness(segment,aware(start),aware(recorded),sha,late)
            for segment,start,recorded,sha,late in raw['bar_witnesses'])
        return IssueInput(**raw)
    except Exception:
        raise ForecastInputError('causal_issue_input_decode_failed') from None
