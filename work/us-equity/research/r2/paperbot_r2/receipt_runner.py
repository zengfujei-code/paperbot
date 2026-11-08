"""Bounded GET-only raw collector; no classifier, LLM or order invocation."""
import argparse
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
import zoneinfo

from paperbot_us.alpaca import AlpacaDataClient, decode_json
from paperbot_us.data import Bar, utc_time
from .receipt_book import ReceiptBook, ReceiptError, ReceiptPlan, _plan, _session


def _projection(state,status,now):
    slots=list(state['slots'].values())
    return dict(schema=1,component='r2_raw_receipts',research_only=True,execution_enabled=False,
        models_fitted=0,orders_submitted=0,generated_at=now.isoformat(),status=status,
        symbols=list(state['plan'].symbols),slot_count=len(slots),raw_first_count=sum(s['first'] is not None for s in slots),
        on_time_count=sum(s['on_time'] for s in slots),missed_count=sum(s['missed'] for s in slots),
        late_first_count=sum(s['first'] is not None and not s['on_time'] for s in slots),
        pending_count=sum(s['first'] is None and not s['missed'] for s in slots),
        revision_count=sum(len(s['revisions']) for s in slots),
        zero_volume_first_count=sum(s['first'] is not None and s['first']['volume']==0 for s in slots),
        calendar_days=len(state['calendars']),quarantined_days=sum(p['changed'] for p in state['calendars'].values()),
        head_seq=state['head_seq'],head_digest=state['head_digest'],training_or_context_only=True,
        full_market_coverage=False,usable_model_claimed=False)


def _unavailable(now,status='evidence_unavailable'):
    return dict(schema=1,component='r2_raw_receipts',research_only=True,execution_enabled=False,
        models_fitted=0,orders_submitted=0,generated_at=now.isoformat(),status=status,
        slot_count=None,raw_first_count=None,on_time_count=None,missed_count=None)


def collect_once(book,*,client,clock=None):
    clock=clock or (lambda:datetime.now(timezone.utc));now=utc_time(clock())
    try:
        state=book.snapshot();plan=state['plan']
        if now<utc_time(state['last_at']):raise ReceiptError('receipt_clock_regression')
        day=now.astimezone(zoneinfo.ZoneInfo('America/New_York')).date()
        if not date.fromisoformat(plan.start_day)<=day<=date.fromisoformat(plan.end_day):
            book.observe([],now=now)
            return _projection(book.snapshot(),'awaiting_start' if day<date.fromisoformat(plan.start_day) else 'collection_closed',now)
        proof=state['calendars'].get(day.isoformat());session=None if proof is None else _session(proof['session'])
        near=session is not None and session.opens-timedelta(minutes=5)<=now<=session.closes+timedelta(seconds=plan.recovery_seconds)
        refresh=300 if near else 21600
        if proof is None or (now-utc_time(proof['received_at'])).total_seconds()>=refresh:
            try:
                calendar=client.calendar(day.isoformat(),day.isoformat());now=utc_time(clock())
            except Exception:
                now=utc_time(clock());book.observe([],now=now)
                return _projection(book.snapshot(),'calendar_unavailable',now)
            book.record_calendar(calendar,day=day.isoformat(),received_at=now)
            state=book.snapshot();proof=state['calendars'][day.isoformat()];session=_session(proof['session'])
        if proof['changed']:
            book.observe([],now=now);return _projection(book.snapshot(),'calendar_changed',now)
        if session is None or now<session.opens or now>session.closes+timedelta(seconds=plan.recovery_seconds):
            book.observe([],now=now);return _projection(book.snapshot(),'market_closed',now)
        if now<session.opens+timedelta(seconds=900):
            book.observe([],now=now);return _projection(book.snapshot(),'awaiting_first_bar',now)
        rows=[];status='collecting' if now<=session.closes else 'recovering'
        try:
            fetched=client.bars(list(plan.symbols),start=session.opens,end=min(now,session.closes));now=utc_time(clock())
            if type(fetched) is not dict or not set(fetched).issubset(plan.symbols):raise ValueError('bar_symbols')
            seen=set()
            for symbol,values in fetched.items():
                if type(values) not in (tuple,list):raise ValueError('bar_list')
                for bar in values:
                    if (type(bar) is not Bar or bar.symbol!=symbol or not session.opens<=bar.starts<session.closes
                            or bar.ends>session.closes or not bar.starts<=bar.received_at<=now
                            or bar.feed!='iex' or (bar.starts-session.opens).total_seconds()%900
                            or (symbol,bar.starts) in seen):raise ValueError('bar_contract')
                    seen.add((symbol,bar.starts))
                    if bar.ends<=now:
                        bar.validate(as_of=now,session=session)
                        rows.append(bar)
            if len(rows)>1000:raise ValueError('bar_count')
        except Exception:
            now=utc_time(clock());rows=[];status='data_unavailable'
        book.observe(rows,now=now)
        return _projection(book.snapshot(),status,now)
    except Exception:return _unavailable(utc_time(clock()))


def _publish(path,value):
    target=Path(path)
    if target.is_symlink() or target.parent.is_symlink() or not target.parent.is_dir():raise ReceiptError('projection_path')
    temporary=target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.tmp')
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
    with os.fdopen(fd,'w',encoding='utf-8') as handle:
        json.dump(value,handle,sort_keys=True,allow_nan=False);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,target)


def main():
    parser=argparse.ArgumentParser(description='Private GET-only raw IEX receipts')
    for field in ('plan','plan-sha256','book','publish','timezone-root'):parser.add_argument('--'+field,required=True)
    parser.add_argument('--initialize-only',action='store_true');args=parser.parse_args()
    result=_unavailable(datetime.now(timezone.utc),'collector_unavailable')
    try:
        path=Path(args.plan)
        if not path.is_file() or path.is_symlink() or path.stat().st_size>16384:raise ReceiptError('plan_file')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=args.plan_sha256:raise ReceiptError('plan_binding')
        plan=_plan(decode_json(raw))
        tzroot=Path(args.timezone_root);tzfile=tzroot/'America/New_York'
        if tzroot.is_symlink() or not tzfile.is_file() or tzfile.is_symlink() or hashlib.sha256(tzfile.read_bytes()).hexdigest()!=plan.timezone_sha256:
            raise ReceiptError('timezone_binding')
        zoneinfo.reset_tzpath([str(tzroot)]);zoneinfo.ZoneInfo.clear_cache()
        book=ReceiptBook(args.book)
        if args.initialize_only:
            book.initialize(plan,now=datetime.now(timezone.utc))
            result=_projection(book.snapshot(),'initialized',datetime.now(timezone.utc))
        else:
            if book.snapshot()['plan']!=plan:raise ReceiptError('plan_binding')
            key=os.getenv('ALPACA_PAPER_KEY_ID','');secret=os.getenv('ALPACA_PAPER_SECRET_KEY','')
            if not re.fullmatch('PK[A-Za-z0-9]{14,510}',key) or not re.fullmatch('[A-Za-z0-9]{16,512}',secret):
                result=_unavailable(datetime.now(timezone.utc),'paper_credentials_unavailable')
            else:result=collect_once(book,client=AlpacaDataClient(key,secret,max_pages=1))
        try:_publish(args.publish,result)
        except Exception:result=_unavailable(datetime.now(timezone.utc),'publication_unavailable')
    except Exception:pass
    print(json.dumps(result,sort_keys=True),flush=True)
    return 1 if result['status'] in ('collector_unavailable','evidence_unavailable','paper_credentials_unavailable','publication_unavailable') else 0


if __name__=='__main__':raise SystemExit(main())
