"""Continuous, auditable quantitative shadow analysis. No broker mutations/LLM calls."""
import argparse
from contextlib import closing
from dataclasses import asdict
from datetime import date,datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from zoneinfo import ZoneInfo

from .alpaca import AlpacaDataClient,decode_json
from .audit import ScanArchive,replay
from .backtest import ReferencePolicy,policy_record
from .data import Bar,Calendar,Session,utc_time
from .monitor import ObservationArchive,_publish
from .quant import live_indicators
from .runner import _exclusive


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False,
                      default=lambda v:v.isoformat())


def _version():
    digest=hashlib.sha256()
    for name in ('shadow.py','quant.py','backtest.py','research.py','risk.py','data.py','alpaca.py','audit.py','scanner.py','runner.py','monitor.py'):
        digest.update(name.encode()+b'\0'+Path(__file__).with_name(name).read_bytes()+b'\0')
    return digest.hexdigest()


def _identity(*parts): return hashlib.sha256(_json(parts).encode()).hexdigest()


def _find(archive,identity):
    if not archive.path.exists(): return None
    with closing(sqlite3.connect(archive.path.as_uri()+'?mode=ro',uri=True)) as db:
        rows=db.execute('SELECT payload,digest FROM observations WHERE instr(payload,?)>0 LIMIT 2',
                        ('"identity":"'+identity+'"',)).fetchall()
    if len(rows)>1: raise ValueError('duplicate shadow identity')
    return {**decode_json(rows[0][0]),'audit_sha256':rows[0][1]} if rows else None


def _summary(record):
    return {name:record[name] for name in ('symbol','generated_at','bar_ends_at','features','entry_candidate',
        'exit_candidate','final_action','reason','vetoes','identity')}


def _last_decision(archive,identity):
    if not archive.path.exists(): return None
    with closing(sqlite3.connect(archive.path.as_uri()+'?mode=ro',uri=True)) as db:
        row=db.execute('SELECT payload,digest FROM observations WHERE instr(payload,?)>0 ORDER BY seq DESC LIMIT 1',
                        ('"bar_identity":"'+identity+'"',)).fetchone()
    return {**decode_json(row[0]),'audit_sha256':row[1]} if row else None


def replay_decision(record):
    """Recompute a successful rule decision using only preserved normalized inputs."""
    policy=ReferencePolicy()
    if (record.get('kind')!='decision' or record.get('reason')!='rules_evaluated'
            or record.get('code_sha256')!=_version() or record.get('policy')!=policy_record(policy)):
        raise ValueError('unsupported shadow replay')
    raw=record['calendar']
    sessions=[Session(date.fromisoformat(s['day']),s['opens'],s['closes']) for s in raw['sessions']]
    calendar=Calendar(sessions,start=raw['start'],end=raw['end'])
    bars=[Bar(**row) for row in record['input_bars']]
    if bars[-1].symbol!=record['symbol'] or bars[-1].ends.timestamp()!=record['bar_ends_at']:
        raise ValueError('shadow input identity mismatch')
    features=live_indicators(bars,calendar=calendar,
        as_of=datetime.fromtimestamp(record['generated_at'],timezone.utc),spec=policy.spec)
    expected=dict(features=asdict(features),entry_candidate=policy.action(features,False)=='buy',
        exit_candidate=policy.action(features,True)=='sell',final_action='hold',
        vetoes=['strategy_not_accepted','execution_not_enabled','model_not_configured'])
    if any(record[name]!=value for name,value in expected.items()): raise ValueError('shadow decision mismatch')
    return expected


def analyze_once(*,client,scan_archive,archive,publish,clock=None):
    clock=clock or (lambda:datetime.now(timezone.utc));now=utc_time(clock())
    result=dict(schema=1,mode='paper',phase='shadow',execution_enabled=False,model_status='not_configured',
        generated_at=now.timestamp(),analysis_started_at=None,status='unavailable',decisions=[],audit_sha256=None)
    journal=ObservationArchive(archive)
    try:
        with _exclusive(Path(archive)):
            if journal.path.exists(): journal.verify()
            code=_version();policy=ReferencePolicy();day=now.astimezone(ZoneInfo('America/New_York')).date().isoformat()
            calendar=client.calendar(day,day);now=utc_time(clock());session=calendar.at(now)
            selection=[];scan_id=None
            if session is not None:
                source=ScanArchive(scan_archive)
                if source.path.is_symlink(): raise ValueError('scan link refused')
                with closing(sqlite3.connect(source.path.as_uri()+'?mode=ro',uri=True,timeout=10)) as db:
                    row=db.execute('SELECT run_id FROM scans ORDER BY rowid DESC LIMIT 1').fetchone()
                if row is None: raise ValueError('missing scan')
                record=source.load(row[0]);replay(record);scan_id=record['run_id']
                if (record['result']['status']!='ok' or not 0<=(now-utc_time(record['result']['as_of'])).total_seconds()<=180):
                    raise ValueError('scan unavailable')
                selection=[r['symbol'] for r in record['result']['candidates'][:3]]
            bucket=int(now.timestamp()//900)
            heartbeat_id=_identity('cycle',code,day,bucket,'open' if session else 'closed')
            pulse=_find(journal,heartbeat_id)
            if pulse is None:
                pulse=journal.append(dict(generated_at=now.timestamp(),identity=heartbeat_id,kind='cycle',code_sha256=code,
                    session_day=day,status='market_closed' if session is None else 'analyzing'))
            result.update(analysis_started_at=pulse['observing_started_at'],audit_sha256=pulse['audit_sha256'])
            if session is None:
                result.update(generated_at=utc_time(clock()).timestamp(),status='market_closed')
            else:
                expected=session.opens.timestamp()+int((now-session.opens).total_seconds()//900)*900
                pending=[];completed={}
                for symbol in selection:
                    base_identity=_identity('decision',code,policy_record(policy),symbol,expected)
                    previous=_last_decision(journal,base_identity)
                    if previous is None or (previous['reason']!='rules_evaluated' and now.timestamp()-previous['generated_at']>=60):
                        pending.append((symbol,base_identity,_identity('attempt',base_identity,int(now.timestamp()//60))))
                    else: completed[symbol]=previous
                fetched={}
                if pending:
                    try: fetched=client.bars([s for s,_,_ in pending],start=session.opens,end=now)
                    except Exception: pass
                for symbol,base_identity,identity in pending:
                    decision=dict(generated_at=utc_time(clock()).timestamp(),identity=identity,bar_identity=base_identity,kind='decision',
                        code_sha256=code,scan_run_id=scan_id,policy=policy_record(policy),symbol=symbol,
                        calendar=json.loads(_json(dict(start=calendar.start,end=calendar.end,
                            sessions=[asdict(s) for _,s in sorted(calendar.sessions.items())]))),
                        bar_ends_at=expected,features=None,entry_candidate=False,exit_candidate=False,
                        final_action='hold',reason='input_unavailable',
                        vetoes=['strategy_not_accepted','execution_not_enabled','model_not_configured'],input_bars=[])
                    bars=fetched.get(symbol,[])
                    if bars:
                        decision['input_bars']=json.loads(_json([asdict(b) for b in bars[-policy.spec.required_bars:]]))
                        decision['reason']='insufficient_bars'
                        if len(bars)>=policy.spec.required_bars:
                            try:
                                window=bars[-policy.spec.required_bars:];at=utc_time(clock())
                                if window[-1].symbol!=symbol or window[-1].ends.timestamp()!=expected:
                                    raise ValueError('wrong or delayed final bar')
                                features=live_indicators(window,calendar=calendar,as_of=at,spec=policy.spec)
                                decision.update(generated_at=at.timestamp(),features=asdict(features),entry_candidate=policy.action(features,False)=='buy',
                                    exit_candidate=policy.action(features,True)=='sell',reason='rules_evaluated',
                                    input_bars=json.loads(_json([asdict(b) for b in window])))
                            except Exception: decision['reason']='invalid_or_stale_bars'
                    if decision['reason']=='rules_evaluated': replay_decision(decision)
                    completed[symbol]=journal.append(decision)
                result.update(status='analyzing',generated_at=utc_time(clock()).timestamp(),
                    decisions=[_summary(completed[s]) for s in selection])
                if selection: result['audit_sha256']=completed[selection[-1]]['audit_sha256']
    except Exception:
        result.update(status='unavailable',decisions=[],audit_sha256=None,generated_at=utc_time(clock()).timestamp())
    _publish(publish,result)
    return result


def main():
    parser=argparse.ArgumentParser(description='GET-only continuous quantitative shadow analysis')
    for name in ('archive','scan-archive','publish'): parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    try:
        result=analyze_once(client=AlpacaDataClient.from_environment(),scan_archive=args.scan_archive,
            archive=args.archive,publish=args.publish)
        print(json.dumps(dict(status=result['status'],phase='shadow',orders_submitted=0)),flush=True)
        return int(result['status']=='unavailable')
    except Exception:
        print(json.dumps(dict(status='unavailable',phase='shadow',orders_submitted=0)),flush=True);return 1


if __name__=='__main__': sys.exit(main())
