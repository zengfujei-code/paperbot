#!/usr/bin/env python3
"""Research-only parameter proposals; active thresholds are never auto-updated."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import time
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DB_PATH = Path(os.getenv("OPTIMIZER_DB_PATH", "/data/tradesv3.dryrun.sqlite"))
TUNING_PATH = Path(os.getenv("AI_TUNING_PATH", "/shared/tuning.json"))
INTERVAL_SECONDS = int(os.getenv("OPTIMIZER_INTERVAL_SECONDS", "3600"))
MIN_SAMPLES = 6

BASELINE: dict[str, Any] = {
    "buy_confidence": 0.68,
    "sell_confidence": 0.60,
    "min_quant_buy": 0.10,
    "last_trade_id": 0,
    "revision": 0,
}


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    temporary=Path(name)
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8') as handle:
            json.dump(value,handle,ensure_ascii=False,allow_nan=False,separators=(',',':'))
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary,path)
        _sync_directory(path.parent)
    finally:
        if temporary.exists(): temporary.unlink()


def _sync_directory(path):
    if os.name!='nt':
        descriptor=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)


def append_history(path,record):
    payload=(json.dumps(record,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode('utf-8')
    with path.open('a+b') as handle:
        size=handle.seek(0,os.SEEK_END)
        if size:
            handle.seek(max(0,size-2*1024*1024))
            tail=handle.read()
            if not tail.endswith(b'\n'): raise ValueError('incomplete parameter history; preserve for recovery')
            previous=json.loads(tail.rstrip(b'\n').split(b'\n')[-1])
            if not isinstance(previous,dict): raise ValueError('invalid parameter history')
            checksum=previous.pop('sha256',None)
            digest=hashlib.sha256(json.dumps(previous,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
            if checksum!=digest: raise ValueError('parameter history integrity mismatch')
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    _sync_directory(path.parent)


def _bounded(value,default,lower,upper):
    try:
        return float(value) if type(value) in (int,float) and math.isfinite(value) and lower<=value<=upper else default
    except OverflowError:
        return default


def _count(value):
    return value if type(value) is int and 0<=value<2**53 else 0


def load_previous(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value,dict): return dict(BASELINE)
        result=dict(BASELINE)
        for name,lower,upper in (('buy_confidence',0.64,0.82),('sell_confidence',0.50,0.70),('min_quant_buy',0.05,0.35)):
            result[name]=_bounded(value.get(name),BASELINE[name],lower,upper)
        result.update(last_trade_id=_count(value.get('last_trade_id')),revision=_count(value.get('revision')),
                      proposal_revision=_count(value.get('proposal_revision')))
        candidate=value.get('proposal')
        if isinstance(candidate,dict):
            result['proposal']={name:_bounded(candidate.get(name),result[name],lower,upper)
                               for name,lower,upper in (('buy_confidence',0.64,0.82),('sell_confidence',0.50,0.70),('min_quant_buy',0.05,0.35))}
        return result
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return dict(BASELINE)


def closed_trades(db_path: Path) -> list[tuple[int, float]]:
    if not db_path.exists():
        raise ValueError('trade database unavailable')
    try:
        with closing(sqlite3.connect(f'file:{db_path}?mode=ro',uri=True,timeout=3)) as connection:
            rows=connection.execute('SELECT id, COALESCE(close_profit_abs,realized_profit) FROM trades '
                                    'WHERE is_open=0 ORDER BY id DESC LIMIT 20').fetchall()
        if any(type(row[0]) is not int or row[0]<=0 or type(row[1]) not in (int,float)
               or not math.isfinite(row[1]) for row in rows):
            raise ValueError('invalid trade accounting')
        return [(row[0],float(row[1])) for row in rows]
    except (OSError, sqlite3.Error, TypeError, ValueError,OverflowError):
        raise ValueError('trade database unavailable') from None


def optimize(db_path: Path = DB_PATH, tuning_path: Path = TUNING_PATH, *, now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    previous = load_previous(tuning_path)
    available=True
    try: trades=closed_trades(db_path)
    except ValueError: trades,available=[],False
    profits = [profit for _, profit in trades]
    sample_size = len(profits)
    latest_trade_id = max((trade_id for trade_id, _ in trades), default=0)
    wins = [profit for profit in profits if profit > 0]
    losses = [profit for profit in profits if profit < 0]
    try:
        gross_profit=math.fsum(wins); gross_loss=abs(math.fsum(losses)); net_profit=math.fsum(profits)
        profit_factor=gross_profit/gross_loss if gross_loss else None
        if not all(math.isfinite(v) for v in (gross_profit,gross_loss,net_profit)) or (profit_factor is not None and not math.isfinite(profit_factor)):
            raise ValueError('invalid aggregate')
    except (ValueError,OverflowError):
        available=False; net_profit=profit_factor=None
    win_rate = len(wins) / sample_size if sample_size else None
    new_sample = latest_trade_id != int(previous.get("last_trade_id", 0))

    active={name:previous[name] for name in ('buy_confidence','sell_confidence','min_quant_buy')}
    buy,sell,quant=active['buy_confidence'],active['sell_confidence'],active['min_quant_buy']
    revision = int(previous.get("revision", 0))
    reason = "等待至少 6 笔已完成的模拟交易。"
    status = "collecting_data"
    proposal=previous.get('proposal')
    proposal_revision=previous.get('proposal_revision',0)

    if available and sample_size >= MIN_SAMPLES:
        status = 'awaiting_validation'
        reason = '样本只用于提出研究建议；当前生效参数保持不变。'
        if new_sample:
            proposal_revision += 1
            if net_profit < 0 or (profit_factor is not None and profit_factor < 0.9):
                buy = min(0.82, buy + 0.02)
                sell = max(0.50, sell - 0.02)
                quant = min(0.35, quant + 0.05)
                reason = "近期收益为负，已提高入场门槛并让退出更敏感。"
            elif profit_factor is not None and profit_factor >= 1.30 and win_rate >= 0.55:
                buy = max(0.64, buy - 0.01)
                quant = max(0.05, quant - 0.02)
                reason = "近期收益质量稳定，已小幅放宽入场门槛。"
            else:
                reason = "近期表现处于中性区间，参数保持不变。"
            proposal=dict(buy_confidence=round(buy,4),sell_confidence=round(sell,4),min_quant_buy=round(quant,4))
    if not available:
        status,reason='unavailable','成交证据不可用，保留当前参数，不生成新提案。'

    result = {
        "algorithm": "bounded_proposals_v2_no_auto_apply",
        "status": status,
        "reason": reason,
        **active,
        'proposal':proposal,
        'proposal_revision':proposal_revision,
        'auto_update_enabled':False,
        'evidence_available':available,
        'validation_required':['chronological_train_validation_test','transaction_costs','independent_sample','paper_forward_evidence','explicit_parameter_approval'],
        "sample_size": sample_size,
        "minimum_samples": MIN_SAMPLES,
        "win_rate": round(win_rate, 4) if available and win_rate is not None else None,
        "profit_factor": round(profit_factor, 4) if available and profit_factor is not None else None,
        "net_profit": round(net_profit, 8) if available else None,
        "last_trade_id": latest_trade_id if available else previous['last_trade_id'],
        "revision": revision,
        "updated_at": now,
        "updated_at_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
    }
    record=dict(schema=2,active_before=active,active_after=dict(active),result=result,
                closed_trade_sample=trades if available else [],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    record['sha256']=hashlib.sha256(json.dumps(record,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
    tuning_path.parent.mkdir(parents=True,exist_ok=True)
    append_history(tuning_path.with_name(tuning_path.name+'.history.jsonl'),record)
    atomic_write(tuning_path, result)
    print(json.dumps({key:result[key] for key in ('status','sample_size','revision')},separators=(',',':')),flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        started = time.monotonic()
        optimize()
        if args.once:
            return
        time.sleep(max(1.0, INTERVAL_SECONDS - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
