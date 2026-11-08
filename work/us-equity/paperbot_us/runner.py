"""One bounded batch of data-only scanning, with a durable resume cursor.

A cycle is a sequence of observations at different receipt times. It is not a
simultaneous full-market snapshot. Candidates require fresh revalidation before
any later research or paper execution module may use them.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
from zoneinfo import ZoneInfo

from .alpaca import AlpacaDataClient,UniverseSnapshot,Asset,ProviderError,decode_json
from .audit import ScanArchive,ArchiveError,make_scan_record
from .data import DataQualityError,utc_time
from .scanner import ScanPolicy


class RunnerError(RuntimeError):
    pass


@contextmanager
def _exclusive(path):
    """OS-held lock is released on process exit; never remove another lock file."""
    path=path.with_name(path.name+'.lock')
    if path.is_symlink(): raise RunnerError('checkpoint_link_refused')
    fd=os.open(path,os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
    locked=False
    try:
        if os.fstat(fd).st_size==0: os.write(fd,b'0')
        os.lseek(fd,0,os.SEEK_SET)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(fd,msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        locked=True
        yield
    except OSError:
        raise RunnerError('checkpoint_locked_or_unavailable') from None
    finally:
        if locked:
            if os.name=='nt':
                os.lseek(fd,0,os.SEEK_SET)
                msvcrt.locking(fd,msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(fd,fcntl.LOCK_UN)
        os.close(fd)


def _load_checkpoint(path,identity,count):
    if path.is_symlink(): raise RunnerError('checkpoint_link_refused')
    if not path.exists(): return 0
    with path.open('rb') as handle: raw=handle.read(8193)
    if len(raw)>8192: raise RunnerError('checkpoint_invalid')
    value=decode_json(raw)
    if not isinstance(value,dict) or set(value)!={'version','universe_sha256','offset'} or value['version']!=1:
        raise RunnerError('checkpoint_invalid')
    if type(value['offset']) is not int or value['offset']<0 or not isinstance(value['universe_sha256'],str):
        raise RunnerError('checkpoint_invalid')
    # A changed current universe starts a new discovery cycle. No historical
    # point-in-time or delisted coverage is implied by this current asset list.
    if value['universe_sha256']!=identity: return 0
    if value['offset']>=count: raise RunnerError('checkpoint_invalid')
    return value['offset']


def _save_json(path,value):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as handle:
            json.dump(value,handle,sort_keys=True,separators=(',',':'),allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_symlink(): raise RunnerError('checkpoint_link_refused')
        os.replace(temporary,path)
        if os.name!='nt':
            fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
    finally:
        if temporary.exists(): temporary.unlink()


def _current_universe(client,path,clock):
    if path.is_symlink(): raise RunnerError('universe_cache_link_refused')
    if path.exists():
        with path.open('rb') as handle: raw=handle.read(20*1024*1024+1)
        if len(raw)>20*1024*1024: raise RunnerError('universe_cache_invalid')
        value=decode_json(raw)
        if not isinstance(value,dict) or set(value)!={'version','sha256','universe'} or value['version']!=1:
            raise RunnerError('universe_cache_invalid')
        body=json.dumps(value['universe'],sort_keys=True,separators=(',',':'))
        if hashlib.sha256(body.encode()).hexdigest()!=value['sha256']:
            raise RunnerError('universe_cache_integrity_mismatch')
        fields=dict(value['universe'])
        fields['assets']=tuple(Asset(**a) for a in fields['assets'])
        fields['received_at']=utc_time(fields['received_at'])
        cached=UniverseSnapshot(**fields)
        age=(utc_time(clock())-cached.received_at).total_seconds()
        if age<0: raise RunnerError('future_universe_cache')
        if age<=21600:
            _validate_universe(cached,clock)
            return cached
    fresh=client.assets()
    _validate_universe(fresh,clock)
    fields=asdict(fresh)
    fields['received_at']=fresh.received_at.isoformat()
    body=json.dumps(fields,sort_keys=True,separators=(',',':'))
    _save_json(path,dict(version=1,sha256=hashlib.sha256(body.encode()).hexdigest(),universe=fields))
    return fresh


def _validate_universe(value,clock):
    if not isinstance(value,UniverseSnapshot) or (value.source,value.coverage)!=('alpaca','current_snapshot_only'):
        raise RunnerError('universe_contract_invalid')
    if not 1<=len(value.assets)<=20000 or any(not isinstance(a,Asset) for a in value.assets):
        raise RunnerError('universe_size_or_assets_invalid')
    if type(value.rejected_count) is not int or value.rejected_count<0:
        raise RunnerError('universe_rejection_count_invalid')
    if not 0<=(utc_time(clock())-utc_time(value.received_at)).total_seconds()<=21600:
        raise RunnerError('universe_age_invalid')


def collect_once(client,archive,*,checkpoint,clock=None,batch_size=100,policy=None):
    if type(batch_size) is not int or not 1<=batch_size<=100:
        raise RunnerError('invalid_batch_size')
    clock=clock or (lambda:datetime.now(timezone.utc))
    policy=policy or ScanPolicy()
    checkpoint=Path(checkpoint).absolute()
    try:
        with _exclusive(checkpoint):
            universe=_current_universe(client,checkpoint.with_name(checkpoint.name+'.universe.json'),clock)
            assets=tuple(sorted(universe.assets,key=lambda a:a.symbol))
            if not assets: raise RunnerError('empty_current_universe')
            identity=hashlib.sha256(json.dumps([asdict(a) for a in assets],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            offset=_load_checkpoint(checkpoint,identity,len(assets))
            selected=assets[offset:offset+batch_size]
            before=utc_time(clock())
            day=before.astimezone(ZoneInfo('America/New_York')).date().isoformat()
            calendar=client.calendar(day,day)
            is_open=calendar.at(before) is not None
            quotes=client.snapshots([a.symbol for a in selected]) if is_open else {}
            now=utc_time(clock())
            subset=UniverseSnapshot(selected,universe.received_at,universe.rejected_count)
            rejections=getattr(client,'snapshot_rejections',{}) if is_open else {}
            record=make_scan_record(universe=subset,snapshots=quotes,calendar=calendar,as_of=now,policy=policy,
                                    snapshot_rejections=rejections,full_universe=universe,batch_offset=offset)
            archive.append(record,full_universe=universe)
            completed=is_open and record['result']['status']=='ok'
            next_offset=(offset+len(selected))%len(assets) if completed else offset
            if completed: _save_json(checkpoint,dict(version=1,universe_sha256=identity,offset=next_offset))
            return dict(mode='data_only_no_orders',status=record['result']['status'],as_of=now.isoformat(),
                        run_id=record['run_id'],universe_count=len(assets),universe_sha256=identity,
                        checked_count=len(selected) if completed else 0,valid_snapshot_count=len(quotes),
                        candidate_count=len(record['result']['candidates']),batch_offset=offset,next_offset=next_offset,
                        cycle_finished=completed and next_offset==0,feed='iex',simultaneous_full_market_snapshot=False)
    except RunnerError:
        raise
    except (ArchiveError,ProviderError,DataQualityError,OSError,ValueError,TypeError,KeyError):
        raise RunnerError('batch_failed_progress_not_confirmed') from None


def main():
    parser=argparse.ArgumentParser(description='GET-only IEX batch scanner; no order capability')
    parser.add_argument('--archive',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--batch-size',type=int,default=100)
    args=parser.parse_args()
    try:
        result=collect_once(AlpacaDataClient.from_environment(),ScanArchive(args.archive),
                            checkpoint=args.checkpoint,batch_size=args.batch_size)
        print(json.dumps(result,allow_nan=False))
    except Exception:
        print(json.dumps(dict(status='unavailable',error='scan_batch_failed',mode='data_only_no_orders')))
        return 1
    return 0


if __name__=='__main__': sys.exit(main())
