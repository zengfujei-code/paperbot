"""Private isolated worker for replay_bridge. Standard-library driver only."""
from contextlib import closing,contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import re
from datetime import date,datetime

# Load only the trusted driver's standard-library helpers, never the current
# paperbot_us package whose risk version differs from archived source.
driver=Path(__file__).with_name('replay_bridge.py')
spec=importlib.util.spec_from_file_location('_paperbot_replay_driver',driver)
protocol=importlib.util.module_from_spec(spec);sys.modules[spec.name]=protocol
spec.loader.exec_module(protocol)

QUANT=set(('kind','identity','bar_identity','code_sha256','scan_run_id','policy','symbol','calendar',
           'bar_ends_at','features','entry_candidate','exit_candidate','final_action','reason','vetoes',
           'input_bars','generated_at','observing_started_at'))
MODEL=set(('kind','identity','attempt_identity','generated_at','code_sha256','prompt_version','source_identity',
           'source_sha256','historical','request','receipt','request_started_at','input_current_at_review','observing_started_at'))
ATTEMPT=set(('kind','identity','generated_at','code_sha256','prompt_version','source_identity','source_sha256',
             'historical','request','observing_started_at'))
BAR=set(('symbol','starts','received_at','open','high','low','close','volume','source','feed',
         'adjustment','interval_seconds','currency','volume_unit'))
FEATURE=set(('last_price_usd','sma_fast_usd','sma_slow_usd','rsi_sma','momentum_fraction',
             'volatility_15m_fraction','volume_ratio','interval_seconds','volatility_annualized','formula_version'))
POLICY=set(('spec','entry_rsi_min','entry_rsi_max','exit_rsi','min_volume_ratio','max_volatility_fraction',
            'max_position_fraction','max_participation'))


def exact(value,keys):
    if type(value) is not dict or set(value)!=keys: raise ValueError('unknown nested evidence fields')


def number(value):
    if type(value) not in (int,float) or not math.isfinite(value): raise ValueError('finite evidence number required')


def stamp(value):
    if not isinstance(value,str) or len(value)>40 or datetime.fromisoformat(value.replace('Z','+00:00')).tzinfo is None:
        raise ValueError('canonical aware timestamp required')


def symbol(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Z][A-Z0-9.\-/]{0,19}',value): raise ValueError('stock symbol required')


def record_shape(row,kind):
    # Original replayers intentionally consume only some inputs. Do not let
    # ignored metadata or unknown nested fields become an exfiltration channel
    # when exporting complete raw witnesses to the new private journal.
    for key in ('identity','code_sha256'): protocol.identity(row[key])
    for key in ('generated_at','observing_started_at'): number(row[key])
    if not 0<row['observing_started_at']<=row['generated_at']: raise ValueError('record observation time')
    if kind=='decision':
        for key in ('bar_identity','scan_run_id'): protocol.identity(row[key])
        symbol(row['symbol']);number(row['bar_ends_at'])
        if any(type(row[k]) is not bool for k in ('entry_candidate','exit_candidate')): raise ValueError('candidate flags')
        exact(row['calendar'],{'start','end','sessions'})
        calendar=row['calendar']
        for key in ('start','end'):
            if not isinstance(calendar[key],str) or date.fromisoformat(calendar[key]).isoformat()!=calendar[key]:
                raise ValueError('calendar date required')
        if type(calendar['sessions']) is not list or not 1<=len(calendar['sessions'])<=400: raise ValueError('calendar budget')
        for session in calendar['sessions']:
            exact(session,{'day','opens','closes'})
            if not isinstance(session['day'],str) or date.fromisoformat(session['day']).isoformat()!=session['day']:
                raise ValueError('session date required')
            stamp(session['opens']);stamp(session['closes'])
        exact(row['features'],FEATURE)
        for key in FEATURE-{'interval_seconds','volatility_annualized','formula_version'}: number(row['features'][key])
        if (type(row['features']['interval_seconds']) is not int or row['features']['interval_seconds']!=900
                or row['features']['volatility_annualized'] is not False or row['features']['formula_version']!='simple-window-v1'):
            raise ValueError('feature metadata')
        exact(row['policy'],POLICY);exact(row['policy']['spec'],{'fast','slow','rsi','momentum','volatility','volume'})
        for value in row['policy']['spec'].values():
            if type(value) is not int or not 1<=value<=500: raise ValueError('indicator window required')
        for key in POLICY-{'spec'}: number(row['policy'][key])
        if type(row['input_bars']) is not list or not 1<=len(row['input_bars'])<=501: raise ValueError('bar budget')
        for bar in row['input_bars']:
            exact(bar,BAR);symbol(bar['symbol']);stamp(bar['starts']);stamp(bar['received_at'])
            for key in ('open','high','low','close','volume'): number(bar[key])
            if (bar['source']!='alpaca' or bar['feed'] not in ('iex','sip') or bar['adjustment']!='raw'
                    or type(bar['interval_seconds']) is not int or bar['interval_seconds']!=900
                    or bar['currency']!='USD' or bar['volume_unit']!='shares'): raise ValueError('bar provenance')
    else:
        for key in ('source_identity','source_sha256'): protocol.identity(row[key])
        if type(row['historical']) is not bool: raise ValueError('historical flag required')
        # Full equality is rechecked against the pinned original request builder.
        exact(row['request'],{'model','messages','thinking','response_format','max_tokens','temperature','stream'})
        request=row['request']
        if type(request['messages']) is not list or len(request['messages'])!=2: raise ValueError('request messages')
        for message in request['messages']: exact(message,{'role','content'})
        exact(request['thinking'],{'type'});exact(request['response_format'],{'type'})
        if kind=='model_review':
            protocol.identity(row['attempt_identity']);number(row['request_started_at'])
            if type(row['input_current_at_review']) is not bool: raise ValueError('current flag required')
            exact(row['receipt'],{'action','reason','model','fingerprint','prompt_tokens','completion_tokens','total_tokens'})


def deny_side_effects(event,args):
    if (event.startswith('socket.') or event in ('subprocess.Popen','os.system','os.fork','os.exec',
            'os.posix_spawn','os.remove','os.rename','os.mkdir','os.rmdir','os.chmod','os.chown',
            'os.link','os.symlink','os.truncate','os.putenv')):
        raise PermissionError('offline replay side effect refused')
    if event=='open':
        mode,flags=args[1:3]
        if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (
                os.O_WRONLY|os.O_RDWR|os.O_APPEND|os.O_CREAT|os.O_TRUNC|os.O_EXCL)):
            raise PermissionError('offline replay write refused')
    if event=='sqlite3.connect':
        database=os.fsdecode(args[0]) if isinstance(args[0],(str,bytes)) else ''
        if not database.startswith('file:') or not database.endswith('?mode=ro'):
            raise PermissionError('read-only replay database required')


def authorize(action,arg1,arg2,database,source):
    if action in (sqlite3.SQLITE_READ,sqlite3.SQLITE_SELECT,sqlite3.SQLITE_TRANSACTION): return sqlite3.SQLITE_OK
    if action==sqlite3.SQLITE_PRAGMA and arg1=='journal_mode' and arg2 is None: return sqlite3.SQLITE_OK
    if action==sqlite3.SQLITE_FUNCTION and arg2 in ('count','length'): return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


@contextmanager
def archive(path,monitor):
    path=Path(path)
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or os.path.normcase(str(path.resolve()))!=os.path.normcase(str(path))):
        raise ValueError('canonical archive required')
    # Refuse WAL before SQLite may create shared-memory companions. Recheck
    # journal mode after obtaining the actual read lock as well.
    with path.open('rb') as source: header=source.read(20)
    if header[:16]!=b'SQLite format 3\0' or header[18:20]!=b'\x01\x01': raise ValueError('rollback journal required')
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=5,isolation_level=None)) as db:
        db.execute('PRAGMA query_only=ON');db.set_authorizer(authorize)
        db.execute('BEGIN');db.execute('SELECT name FROM sqlite_master LIMIT 1').fetchone()
        if db.execute('PRAGMA journal_mode').fetchone()[0]!='delete': raise ValueError('rollback journal required')
        _,count,head=monitor.ObservationArchive(path)._verify_chain(db)
        yield db,dict(sequence=count,digest=head)
        db.commit()


def select(db,identity,kind,allowed):
    matches=[]
    for sequence,payload,previous,digest in db.execute('SELECT seq,payload,previous_digest,digest FROM observations ORDER BY seq'):
        row=protocol.strict_json(payload)
        if row.get('identity')!=identity: continue
        if row.get('kind')!=kind or set(row)!=allowed: raise ValueError('unsupported record fields')
        record_shape(row,kind)
        matches.append(dict(sequence=sequence,previous_digest=previous,digest=digest,payload=payload,record=row))
    if len(matches)!=1: raise ValueError('unique archived record required')
    return matches[0]


def require_prefix(db,head):
    if head is None:return
    exact(head,{'sequence','digest'});protocol.identity(head['digest'])
    if type(head['sequence']) is not int or not 1<=head['sequence']<=100000:raise ValueError('prefix sequence')
    row=db.execute('SELECT digest FROM observations WHERE seq=?',(head['sequence'],)).fetchone()
    if row is None or row[0]!=head['digest']:raise ValueError('recorded archive prefix changed')


def main():
    raw=sys.stdin.buffer.read(32769)
    if len(raw)>32768: raise ValueError('request budget')
    value=protocol.strict_json(raw)
    if (set(value)!={'schema','kind','root','manifest','release_sha256','bridge_sha256','archive','identity','source_archive','timezone_root','timezone_sha256','expected_heads'}
            or type(value['schema']) is not int or value['schema']!=1 or value['kind'] not in ('quant','model')):
        raise ValueError('replay request contract')
    if value['identity'] is not None: protocol.identity(value['identity'])
    elif value['kind']!='model': raise ValueError('record identity required')
    if protocol.bridge_version()!=value['bridge_sha256']: raise ValueError('bridge source mismatch')
    heads=value['expected_heads']
    if type(heads) is not dict or not set(heads)<=({'quant'} if value['kind']=='quant' else {'quant','model'}):raise ValueError('expected heads contract')
    manifest=value['manifest']
    if set(manifest)!={'schema','package_sha256','files'} or manifest['schema']!=1: raise ValueError('manifest contract')
    release=protocol.PinnedRelease(value['root'],files=manifest['files'],package_sha256=manifest['package_sha256'])
    if release.verify()!=value['release_sha256']: raise ValueError('release digest mismatch')
    import zoneinfo
    timezone_root=Path(value['timezone_root'])
    if protocol.timezone_digest(timezone_root)!=value['timezone_sha256']: raise ValueError('timezone digest mismatch')
    zoneinfo.reset_tzpath([str(timezone_root)])
    sys.addaudithook(deny_side_effects)
    sys.path.insert(0,str(release.root))
    from paperbot_us import monitor,shadow
    output=dict(schema=1,kind=value['kind'],release_sha256=release.digest,timezone_sha256=value['timezone_sha256'],
        bridge_sha256=value['bridge_sha256'],
        quant=None,model=None,attempt=None,summary=None)
    if value['kind']=='quant':
        if value['source_archive'] is not None: raise ValueError('unexpected source archive')
        with archive(value['archive'],monitor) as (db,head):
            require_prefix(db,heads.get('quant'))
            quant=select(db,value['identity'],'decision',QUANT);quant['head']=head
            summary=shadow.replay_decision({**quant['record'],'audit_sha256':quant['digest']})
        output.update(quant=quant,summary=summary)
    else:
        from paperbot_us import model_runner
        with archive(value['archive'],monitor) as (db,head):
            require_prefix(db,heads.get('model'))
            selected=value['identity']
            if selected is None:
                for payload, in db.execute('SELECT payload FROM observations ORDER BY seq DESC'):
                    candidate=protocol.strict_json(payload)
                    if candidate.get('kind')=='model_review': selected=protocol.identity(candidate['identity']);break
            if selected is None: raise ValueError('archived review unavailable')
            model=select(db,selected,'model_review',MODEL);model['head']=head
            row=model['record'];attempt=select(db,row['attempt_identity'],'model_attempt',ATTEMPT);attempt['head']=head
            first=attempt['record']
            if (any(first[k]!=row[k] for k in ('code_sha256','prompt_version','source_identity','source_sha256','historical','request'))
                    or not first['generated_at']<=row['request_started_at']<=row['generated_at']
                    or attempt['sequence']>=model['sequence']): raise ValueError('attempt linkage mismatch')
            outcomes=0
            for payload, in db.execute('SELECT payload FROM observations ORDER BY seq'):
                event=protocol.strict_json(payload)
                if event.get('attempt_identity')==first['identity']: outcomes+=1
            if outcomes!=1: raise ValueError('unique model outcome required')
            with archive(value['source_archive'],monitor) as (source_db,source_head):
                require_prefix(source_db,heads.get('quant'))
                quant=select(source_db,row['source_identity'],'decision',QUANT);quant['head']=source_head
                source={**quant['record'],'audit_sha256':quant['digest']}
                summary=model_runner.replay_review({**row,'audit_sha256':model['digest']},source)
            output.update(quant=quant,model=model,attempt=attempt,summary=summary)
    if release.verify()!=value['release_sha256']: raise ValueError('release changed')
    if protocol.bridge_version()!=value['bridge_sha256']: raise ValueError('bridge changed')
    if protocol.timezone_digest(timezone_root)!=value['timezone_sha256']: raise ValueError('timezone changed')
    text=protocol.canonical(output)
    if len(text.encode())>131072: raise ValueError('response budget')
    sys.stdout.buffer.write(text.encode('utf-8')+b'\n');sys.stdout.buffer.flush()


if __name__=='__main__':
    try: main()
    except BaseException:
        print(json.dumps(dict(status='original_source_replay_unavailable')),flush=True)
        raise SystemExit(1)
