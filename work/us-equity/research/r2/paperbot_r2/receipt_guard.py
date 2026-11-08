"""CLI guard preventing R1-reserved market prices entering the R2 raw book.

The original collector and its source-pinned book remain unchanged. Its service
must retain an enforced90s deadline; the five-minute pre-midnight lead prevents
an already-started invocation crossing into the reserved period. The extra
lead is outside regular trading hours and is an explicit coverage interruption.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zoneinfo

from paperbot_us.alpaca import decode_json
from paperbot_us.data import utc_time
from . import receipt_runner
from .receipt_book import _plan


def _need(value):
    if not value:raise ValueError('collector_guard_unavailable')


def paused(at):
    ny=zoneinfo.ZoneInfo('America/New_York')
    start=datetime(2026,11,9,23,55,tzinfo=ny)
    end=datetime(2026,11,24,0,0,tzinfo=ny)
    return start<=utc_time(at).astimezone(ny)<end


def _status(at,name):
    value=receipt_runner._unavailable(at,name)
    value.update(component='r2_raw_receipts',guard_version='r1-reserved-capture-pause-v1',
                 research_only=True,execution_enabled=False,models_fitted=0,orders_submitted=0,
                 coverage_interrupted=name=='reserved_dates_paused',book_touched=False,
                 requests_sent=0)
    return value


def main(*,clock=None,delegate=None):
    parser=argparse.ArgumentParser(description='R2 raw capture with reserved-window exclusion')
    for name in ('plan','plan-sha256','book','publish','timezone-root'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--initialize-only',action='store_true')
    args=parser.parse_args();clock=clock or (lambda:datetime.now(timezone.utc))
    result=_status(datetime.now(timezone.utc),'collector_guard_unavailable')
    delegated=False
    try:
        path=Path(args.plan)
        _need(path.is_file() and not path.is_symlink() and not path.parent.is_symlink()
              and path.stat().st_size<=16384)
        content=path.read_bytes();_need(hashlib.sha256(content).hexdigest()==args.plan_sha256)
        plan=_plan(decode_json(content))
        root=Path(args.timezone_root);tz=root/'America/New_York'
        _need(not root.is_symlink() and not (root/'America').is_symlink()
              and tz.is_file() and not tz.is_symlink() and tz.stat().st_size<=65536
              and hashlib.sha256(tz.read_bytes()).hexdigest()==plan.timezone_sha256)
        zoneinfo.reset_tzpath([str(root)]);zoneinfo.ZoneInfo.clear_cache()
        at=utc_time(clock())
        if not paused(at):
            # The original parser sees exactly the same sys.argv. No key value,
            # client or ReceiptBook is accessed by this wrapper before this gate.
            delegated=True
            return (delegate or receipt_runner.main)()
        result=_status(at,'reserved_dates_paused')
        receipt_runner._publish(args.publish,result)
        print(json.dumps(result,sort_keys=True),flush=True)
        return 0
    except Exception:
        name=('collector_guard_delegate_unavailable' if delegated else
              'publication_unavailable' if result['status']=='reserved_dates_paused' else
              'collector_guard_unavailable')
        result=_status(datetime.now(timezone.utc),name)
        if delegated:result.update(requests_sent=None,book_touched=None)
        try:receipt_runner._publish(args.publish,result)
        except Exception:pass
        print(json.dumps(result,sort_keys=True),flush=True)
        return 1


if __name__=='__main__':raise SystemExit(main())
