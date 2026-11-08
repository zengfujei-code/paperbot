"""Bounded GET-only risk-research worker. No account/LLM/order endpoints.

The service manager bounds one entire process. Secrets enter only the existing
paper environment and never the configuration, journal, projection or stdout.
"""
import argparse
from datetime import date,datetime,timedelta,timezone
import json,os,re
from zoneinfo import ZoneInfo

from .alpaca import AlpacaDataClient
from .data import utc_time
from .monitor import _publish
from .prospective import open_campaign


def _projection(campaign,state,status,now):
    return dict(schema=1,component='prospective_risk_research',mode='paper',research_only=True,
        execution_enabled=False,strategy_accepted=False,generated_at=now.timestamp(),status=status,
        symbol=campaign.configuration['symbol'],feed=campaign.configuration['training']['feed'],forecast_count=len(state['forecasts']),
        settled_count=sum(p['outcome_return'] is not None for p in state['forecasts'].values()),
        candidates_ready=len(state['models']),selected=state['selected'] is not None,
        last_prediction_at=max((p['issued_at'].timestamp() for p in state['forecasts'].values()),default=None),
        research_finished=state['finished'])


def _fetch_session(campaign,state,now):
    day=now.astimezone(ZoneInfo('America/New_York')).date()
    first_test=campaign.calendar.sessions[date.fromisoformat(campaign.configuration['test_days'][0])].opens
    if state['selected'] is None and now>=first_test:return None
    allowed=campaign.configuration['validation_days']+campaign.configuration['context_days']
    if state['selected'] is not None:allowed+=campaign.configuration['test_days']
    sessions=[campaign.calendar.sessions[date.fromisoformat(d)] for d in allowed
              if campaign.calendar.sessions[date.fromisoformat(d)].opens<now]
    today=campaign.calendar.sessions.get(day)
    if today in sessions and now<=today.closes+timedelta(seconds=3600):return today
    for session in sorted(sessions,key=lambda s:s.day,reverse=True):
        known={b.starts for b in state['bars'].values() if campaign.calendar.at(b.starts).day==session.day}
        if len(known)<int((session.closes-session.opens).total_seconds()//900):return session
    return None


def collect_once(campaign,*,client,clock=None):
    clock=clock or (lambda:datetime.now(timezone.utc));now=utc_time(clock())
    fallback=dict(schema=1,component='prospective_risk_research',mode='paper',research_only=True,
        execution_enabled=False,strategy_accepted=False,generated_at=now.timestamp(),status='evidence_unavailable',
        forecast_count=None,settled_count=None,last_prediction_at=None)
    try:
        state=campaign.snapshot()
        if state['finished']:return _projection(campaign,state,state['status'],now)
        day=now.astimezone(ZoneInfo('America/New_York')).date();proof=state['calendar_proofs'].get(day.isoformat())
        if proof is None:
            try:
                calendar=client.calendar(day.isoformat(),day.isoformat());now=utc_time(clock())
                verified=campaign.record_calendar(calendar,received_at=now)
            except Exception:return _projection(campaign,campaign.snapshot(),'calendar_unavailable',utc_time(clock()))
        else:verified=proof['verified'] and day.isoformat() not in state['calendar_mismatch_days']
        if not verified:return _projection(campaign,campaign.snapshot(),'calendar_changed',now)
        state=campaign.snapshot();session=_fetch_session(campaign,state,now);bars=[];data_error=False
        if session is not None:
            try:
                fetched=client.bars([campaign.configuration['symbol']],start=session.opens,end=min(now,session.closes))
                now=utc_time(clock());rows=fetched.get(campaign.configuration['symbol'],[])
                if type(rows) not in (tuple,list) or len(rows)>1000:raise ValueError('bar_count')
                # A provider may return its still-forming current bar. Its end
                # is in the future, so it cannot enter the closed-bar archive.
                bars=[b for b in rows if session.opens<=b.starts and b.ends<=session.closes and b.ends<=now]
            except Exception:data_error=True;now=utc_time(clock())
        result=campaign.observe(bars,now=now,calendar_verified=True);state=campaign.snapshot()
        return _projection(campaign,state,'data_unavailable' if data_error else result['status'],now)
    except Exception:return fallback


def main():
    parser=argparse.ArgumentParser(description='GET-only prospective paper research')
    parser.add_argument('--directory',required=True);parser.add_argument('--configuration-sha256',required=True)
    parser.add_argument('--publish');args=parser.parse_args()
    result=dict(component='prospective_risk_research',mode='paper',research_only=True,execution_enabled=False,
        strategy_accepted=False,status='collector_unavailable',generated_at=datetime.now(timezone.utc).timestamp())
    try:
        campaign=open_campaign(args.directory,args.configuration_sha256)
        key=os.getenv('ALPACA_PAPER_KEY_ID','');secret=os.getenv('ALPACA_PAPER_SECRET_KEY','')
        if not re.fullmatch('PK[A-Za-z0-9]{14,510}',key) or not re.fullmatch('[A-Za-z0-9]{16,512}',secret):
            result['status']='paper_credentials_unavailable'
        else:result=collect_once(campaign,client=AlpacaDataClient(key,secret,max_pages=3))
        if args.publish:_publish(args.publish,result)
    except Exception:pass
    print(json.dumps(result,sort_keys=True),flush=True)
    return 0 if result['status'] not in ('collector_unavailable','evidence_unavailable','paper_credentials_unavailable') else 1


if __name__=='__main__':raise SystemExit(main())
