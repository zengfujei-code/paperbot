"""Audited bounded model sidecar. Reads shadow evidence; has no broker capability."""
import argparse
from contextlib import closing
from datetime import date,datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from zoneinfo import ZoneInfo

from .alpaca import decode_json
from .data import Calendar,Session,utc_time
from .llm import DeepSeekClient,PROMPT_VERSION,request_for,validate_receipt
from .monitor import ObservationArchive,_publish
from .runner import _exclusive
from .shadow import replay_decision,_version as quant_version

DAILY_LIMIT=60
HOURLY_LIMIT=20
MIN_INTERVAL=60


def _json(value): return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def _id(*values): return hashlib.sha256(_json(values).encode()).hexdigest()


def _version():
    digest=hashlib.sha256(quant_version().encode())
    for name in ('llm.py','model_runner.py'):
        digest.update(name.encode()+b'\0'+Path(__file__).with_name(name).read_bytes()+b'\0')
    return digest.hexdigest()


def _source_rows(path):
    archive=ObservationArchive(path);archive.verify()
    with closing(sqlite3.connect(archive.path.as_uri()+'?mode=ro',uri=True,timeout=10)) as db:
        rows=db.execute('SELECT payload,digest FROM observations WHERE instr(payload,?)>0 ORDER BY seq DESC LIMIT 300',
            ('"kind":"decision"',)).fetchall()
        pulse=db.execute('SELECT payload FROM observations WHERE instr(payload,?)>0 ORDER BY seq DESC LIMIT 1',
            ('"kind":"cycle"',)).fetchone()
    return [{**decode_json(raw),'audit_sha256':digest} for raw,digest in rows],decode_json(pulse[0]) if pulse else None


def source_decision(path,identity):
    archive=ObservationArchive(path);archive.verify()
    with closing(sqlite3.connect(archive.path.as_uri()+'?mode=ro',uri=True,timeout=10)) as db:
        rows=db.execute('SELECT payload,digest FROM observations WHERE instr(payload,?)>0 LIMIT 2',
            ('"identity":"'+identity+'"',)).fetchall()
    if len(rows)!=1: raise ValueError('source decision unavailable')
    record={**decode_json(rows[0][0]),'audit_sha256':rows[0][1]};replay_decision(record)
    return record


def _summary(record):
    receipt=validate_receipt(record['receipt']);data=decode_json(record['request']['messages'][1]['content'])
    vetoes=['strategy_not_accepted','execution_not_enabled']
    if ((receipt['action']=='buy' and not data['entry_candidate'])
            or (receipt['action']=='sell' and not data['exit_candidate'])): vetoes.append('model_outside_candidate')
    return dict(symbol=data['symbol'],bar_ends_at=data['bar_ends_at'],reviewed_at=record['generated_at'],
        context=data['context'],input_current_at_review=record['input_current_at_review'],
        signal_usable=False,action=receipt['action'],reason=receipt['reason'],
        model=receipt['model'],final_action='hold',vetoes=vetoes)


def replay_review(record,source):
    replay_decision(source)
    if (record['kind']!='model_review' or record['code_sha256']!=_version()
            or record['prompt_version']!=PROMPT_VERSION or record['source_identity']!=source['identity']
            or record['source_sha256']!=source['audit_sha256']
            or record['request']!=request_for(source,historical=record['historical'])
            or record['attempt_identity']!=_id('model-bar-v1',source['symbol'],source['bar_ends_at'])
            or record['identity']!=_id('model-result-v1',record['attempt_identity'])
            or not source['generated_at']<=record['request_started_at']<=record['generated_at']
            or not _current(source,datetime.fromtimestamp(record['request_started_at'],timezone.utc),record['historical'])
            or record['input_current_at_review']!=_current(source,datetime.fromtimestamp(record['generated_at'],timezone.utc),False)):
        raise ValueError('model replay mismatch')
    return _summary(record)


def _current(source,now,historical):
    age=now.timestamp()-source['bar_ends_at']
    if not 0<=age<=(86400 if historical else 1020): return False
    return historical or any(utc_time(s['opens'])<=now<utc_time(s['closes']) for s in source['calendar']['sessions'])


def _history(journal,now):
    attempts={};outcomes=set();reviews=[]
    if not journal.path.exists(): return attempts,reviews
    journal.verify()
    with closing(sqlite3.connect(journal.path.as_uri()+'?mode=ro',uri=True)) as db:
        for payload,digest in db.execute('SELECT payload,digest FROM observations ORDER BY seq'):
            row={**decode_json(payload),'audit_sha256':digest};at=row['generated_at']
            if type(at) not in (int,float) or not 0<at<=now: raise ValueError('model time regression')
            if row['kind']=='model_attempt':
                if row['identity'] in attempts: raise ValueError('duplicate billed attempt')
                attempts[row['identity']]=row
            elif row['kind'] in ('model_review','model_failure'):
                identity=row['attempt_identity']
                if identity not in attempts or identity in outcomes or at<attempts[identity]['generated_at']:
                    raise ValueError('model outcome mismatch')
                outcomes.add(identity)
                attempts[identity]['completed_at']=at
                if row['kind']=='model_review' and now-at<=86400: reviews.append(row)
            else: raise ValueError('model record unsupported')
    return attempts,reviews


def assistance_once(*,client,source_archive,archive,publish,clock=None,historical=False):
    clock=clock or (lambda:datetime.now(timezone.utc));now=utc_time(clock())
    result=dict(schema=1,mode='paper',phase='model_assistance',execution_enabled=False,
        generated_at=now.timestamp(),status='unavailable',decisions=[],attempts_today=0,daily_limit=DAILY_LIMIT,
        hourly_limit=HOURLY_LIMIT,min_interval_seconds=MIN_INTERVAL,audit_sha256=None)
    journal=ObservationArchive(archive)
    try:
        if type(historical) is not bool: raise ValueError('historical context required')
        with _exclusive(Path(archive)):
            attempts,reviews=_history(journal,now.timestamp())
            rows,pulse=_source_rows(source_archive);selection=[];seen=set();market_open=False;market_known=False
            local_day=now.astimezone(ZoneInfo('America/New_York')).date()
            if (pulse and pulse['session_day']==local_day.isoformat()
                    and 0<=now.timestamp()-pulse['generated_at']<=1020
                    and pulse['status'] in ('analyzing','market_closed')):
                market_known=True;market_open=pulse['status']=='analyzing'
            for row in rows:
                if row['reason']!='rules_evaluated' or row['symbol'] in seen: continue
                replay_decision(row);seen.add(row['symbol'])
                raw=row['calendar'];sessions=[Session(date.fromisoformat(s['day']),s['opens'],s['closes']) for s in raw['sessions']]
                # An actual recorded regular-session boundary, not local weekday arithmetic.
                open_now=any(s.opens<=now<s.closes for s in sessions);market_open=market_open or open_now
                market_known=market_known or any(s.day==local_day for s in sessions)
                age=now.timestamp()-row['bar_ends_at']
                if not 0<=row['generated_at']<=now.timestamp(): raise ValueError('future quant input')
                if 0<=age<=(86400 if historical else 1020) and (historical or open_now): selection.append(row)
                if len(selection)==3: break
            day_start=now.replace(hour=0,minute=0,second=0,microsecond=0).timestamp()
            today=sum(r['generated_at']>=day_start for r in attempts.values())
            result['attempts_today']=today
            eligible=[r for r in selection if _id('model-bar-v1',r['symbol'],r['bar_ends_at']) not in attempts]
            state='waiting_for_new_bar'
            if not selection: state='market_closed' if not historical and market_known and not market_open else 'input_unavailable'
            elif eligible:
                if today>=DAILY_LIMIT: state='budget_exhausted'
                elif (sum(r.get('completed_at',r['generated_at']+60)>now.timestamp()-3600 for r in attempts.values())>=HOURLY_LIMIT
                        or any(now.timestamp()-r.get('completed_at',r['generated_at']+60)<MIN_INTERVAL for r in attempts.values())):
                    state='rate_limited'
                else:
                    source=eligible[0];body=request_for(source,historical=historical)
                    identity=_id('model-bar-v1',source['symbol'],source['bar_ends_at'])
                    # Durable reservation BEFORE the billed POST. Unknown outcomes consume quota.
                    attempt=journal.append(dict(kind='model_attempt',identity=identity,generated_at=now.timestamp(),
                        code_sha256=_version(),prompt_version=PROMPT_VERSION,source_identity=source['identity'],
                        source_sha256=source['audit_sha256'],historical=historical,request=body))
                    result.update(attempts_today=today+1,audit_sha256=attempt['audit_sha256'])
                    request_at=utc_time(clock());receipt=None
                    if (not 0<=request_at.timestamp()-now.timestamp()<=30 or request_at.date()!=now.date()
                            or not _current(source,request_at,historical)):
                        state='input_unavailable'
                    else:
                        try:
                            receipt=validate_receipt(client.assess(body));state='reviewed'
                        except Exception: state='model_unavailable'
                    at=utc_time(clock()).timestamp()
                    if at<now.timestamp(): raise ValueError('model clock regression')
                    if receipt is None:
                        outcome=journal.append(dict(kind='model_failure',identity=_id('model-result-v1',identity),
                            attempt_identity=identity,generated_at=at,status=state))
                    else:
                        record=dict(kind='model_review',identity=_id('model-result-v1',identity),attempt_identity=identity,
                            generated_at=at,code_sha256=_version(),prompt_version=PROMPT_VERSION,
                            source_identity=source['identity'],source_sha256=source['audit_sha256'],
                            historical=historical,request=body,receipt=receipt,request_started_at=request_at.timestamp(),
                            input_current_at_review=_current(source,datetime.fromtimestamp(at,timezone.utc),False))
                        replay_review(record,source)
                        outcome=journal.append(record);reviews.append(outcome)
                    result['audit_sha256']=outcome['audit_sha256']
            clean=[];symbols=set()
            for review in reversed(reviews):
                summary=_summary(review)
                if summary['symbol'] in symbols: continue
                symbols.add(summary['symbol']);clean.append(summary)
                if len(clean)==3: break
            # A failed current call is explicit; earlier records retain their original review times.
            result.update(status=state,decisions=clean,generated_at=utc_time(clock()).timestamp())
            if result['audit_sha256'] is None and reviews: result['audit_sha256']=reviews[-1]['audit_sha256']
    except Exception:
        result.update(status='unavailable',decisions=[],audit_sha256=None,generated_at=utc_time(clock()).timestamp())
    _publish(publish,result)
    return result


def main():
    parser=argparse.ArgumentParser(description='Bounded DeepSeek assistance; no broker capability')
    for name in ('archive','source-archive','publish'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--historical-once',action='store_true')
    args=parser.parse_args()
    try:
        result=assistance_once(client=DeepSeekClient.from_environment(),source_archive=args.source_archive,
            archive=args.archive,publish=args.publish,historical=args.historical_once)
        print(json.dumps(dict(status=result['status'],orders_submitted=0,attempts_today=result['attempts_today'])),flush=True)
        return int(result['status'] in ('unavailable','model_unavailable'))
    except Exception:
        print(json.dumps(dict(status='unavailable',orders_submitted=0)),flush=True);return 1


if __name__=='__main__': sys.exit(main())
