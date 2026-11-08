#!/usr/bin/env python3
"""Small read-only dashboard for the 100 USDT paper-trading experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import time
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DB_PATH = Path(os.getenv("DASHBOARD_DB_PATH", "/data/db/tradesv3.dryrun.sqlite"))
SIGNAL_PATH = Path(os.getenv("DASHBOARD_SIGNAL_PATH", "/data/shared/signal.json"))
EXPERIMENT_PATH = Path(os.getenv("DASHBOARD_EXPERIMENT_PATH", "/data/experiment.json"))
LOG_PATH = Path(os.getenv("DASHBOARD_LOG_PATH", "/data/logs/freqtrade.log"))
TUNING_PATH = Path(os.getenv("DASHBOARD_TUNING_PATH", "/data/shared/tuning.json"))
US_STATUS_PATH = Path(os.getenv("DASHBOARD_US_STATUS_PATH", "/data/us/status.json"))
US_PILOT_PATH = US_STATUS_PATH.with_name('pilot.json')
US_ANALYSIS_PATH = Path(os.getenv("DASHBOARD_US_ANALYSIS_PATH", "/data/us/analysis.json"))
US_MODEL_PATH = Path(os.getenv("DASHBOARD_US_MODEL_PATH", "/data/us/model.json"))


def _load_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else fallback
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return fallback


def _round(value: float) -> float:
    return round(float(value), 4)


def _number(value: Any, default: Any = None) -> Any:
    try:
        return value if type(value) in (int, float) and math.isfinite(value) else default
    except OverflowError:
        return default


def _text(value: Any, default: Any = None) -> Any:
    return value[:300] if isinstance(value, str) else default


def read_snapshot(
    db_path: Path = DB_PATH,
    signal_path: Path = SIGNAL_PATH,
    experiment_path: Path = EXPERIMENT_PATH,
    log_path: Path = LOG_PATH,
    tuning_path: Path = TUNING_PATH,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    now = time.time() if now is None else now
    experiment = _load_json(experiment_path, {"virtual_wallet_usdt": 100})
    signal = _load_json(signal_path, {})
    tuning = _load_json(tuning_path, {})
    initial = _number(experiment.get("virtual_wallet_usdt"), 100)
    if initial <= 0:
        initial = 100
    last_price = signal.get("features", {}).get("last_close") if isinstance(signal.get("features"), dict) else None
    last_price = _number(last_price)
    if last_price is not None and last_price <= 0:
        last_price = None
    signal_generated = _number(signal.get("generated_at"), 0)
    signal_expires = _number(signal.get("expires_at"), 0)
    signal_fresh = (0 < signal_generated < signal_expires
                    and signal_expires - signal_generated <= 1500
                    and -60 <= now - signal_generated <= 1500 and now < signal_expires)
    if type(signal.get('schema')) is int and signal.get('schema')==2:
        closed_at=_number(signal.get('data_closed_at'),0)
        received_at=_number(signal.get('data_received_at'),0)
        signal_fresh=signal_fresh and 0<closed_at<=received_at<=signal_generated and 0<=now-closed_at<=1500
    else:
        signal_fresh=False

    realized = 0.0
    unrealized = 0.0
    closed_count = 0
    wins = 0
    positions: list[dict[str, Any]] = []
    accounting_ok = False
    valuation_ok = True
    if db_path.exists():
        try:
            with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)) as connection:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(
                    "SELECT pair,is_open,open_rate,close_profit_abs,realized_profit,stake_amount,amount,"
                    "open_date,close_date,is_short FROM trades ORDER BY id DESC"
                ).fetchall()
            for row in rows:
                if bool(row["is_open"]):
                    open_rate = float(row["open_rate"] or 0)
                    amount = float(row["amount"] or 0)
                    stake = float(row["stake_amount"] or 0)
                    if not isinstance(row["pair"], str) or not all(math.isfinite(v) for v in (open_rate, amount, stake)) or open_rate <= 0 or amount < 0 or stake < 0:
                        raise ValueError("invalid position accounting")
                    current_profit = None
                    mark_ok = signal_fresh and last_price is not None and signal.get("pair") == row["pair"]
                    if mark_ok:
                        current_profit = amount * ((open_rate - last_price) if bool(row["is_short"]) else (last_price - open_rate))
                        if not math.isfinite(current_profit):
                            raise ValueError("position valuation overflow")
                        unrealized += current_profit
                    else:
                        valuation_ok = False
                    positions.append(
                        {
                            "pair": row["pair"],
                            "open_rate": _round(open_rate),
                            "current_rate": _round(last_price) if mark_ok else None,
                            "stake": _round(stake),
                            "profit": _round(current_profit) if current_profit is not None else None,
                            "opened_at": _text(row["open_date"]),
                        }
                    )
                else:
                    profit = row["close_profit_abs"]
                    if profit is None:
                        profit = row["realized_profit"]
                    if profit is None:
                        raise ValueError("closed trade profit unavailable")
                    profit = float(profit)
                    if not math.isfinite(profit):
                        raise ValueError("invalid realized accounting")
                    realized += profit
                    closed_count += 1
                    wins += int(profit > 0)
            accounting_ok = True
        except (OSError, sqlite3.Error, ValueError, TypeError, OverflowError):
            pass

    profit = realized + unrealized
    equity = initial + profit
    if not all(math.isfinite(v) for v in (realized, unrealized, profit, equity)):
        accounting_ok = False
    valuation_ok = valuation_ok and accounting_ok
    try:
        log_fresh = 0 <= now - log_path.stat().st_mtime <= 150
        with log_path.open("rb") as handle:
            handle.seek(max(0, log_path.stat().st_size - 8192))
            log_tail = handle.read(8192).decode("utf-8", errors="ignore")
        bot_running = log_fresh and "state='RUNNING'" in log_tail
    except OSError:
        bot_running = False
    ai_ok = signal_fresh and signal.get("status") == "ok"
    public_action=signal.get('action') if ai_ok and signal.get('action') in ('buy','sell','hold') else 'hold'
    public_reason={'buy':'量化与模型建议买入，仍需交易风控确认。',
                   'sell':'量化与模型建议退出，仍需交易风控确认。',
                   'hold':'继续观望，等待符合条件的模拟信号。'}[public_action] if ai_ok else '等待新鲜行情和有效模型结果。'
    optimizer_status=tuning.get('status')
    if optimizer_status not in ('collecting_data','stable','adapted','awaiting_validation'):
        optimizer_status='unavailable'
    if not accounting_ok:
        system_status = "成交数据不可用 · 盈亏待核验"
    elif not valuation_ok:
        system_status = "行情过期或缺失 · 估值待核验"
    elif bot_running and ai_ok:
        system_status = "正常运行"
    elif bot_running:
        system_status = "运行中 · AI 暂时观望"
    else:
        system_status = "等待服务恢复"

    return {
        "system_status": system_status,
        "bot_running": bot_running,
        "ai_ok": ai_ok,
        "accounting_ok": accounting_ok,
        "valuation_ok": valuation_ok,
        "initial_balance": _round(initial),
        "equity": _round(equity) if valuation_ok else None,
        "profit_abs": _round(profit) if valuation_ok else None,
        "return_pct": _round(profit / initial * 100 if initial else 0) if valuation_ok else None,
        "realized_profit": _round(realized) if accounting_ok else None,
        "unrealized_profit": _round(unrealized) if valuation_ok else None,
        "open_count": len(positions) if accounting_ok else None,
        "closed_count": closed_count if accounting_ok else None,
        "wins": wins if accounting_ok else None,
        "win_rate": _round(wins / closed_count * 100 if closed_count else 0) if accounting_ok else None,
        "positions": positions if accounting_ok else [],
        "last_price": _round(last_price) if last_price and signal_fresh else None,
        "signal": {
            "action": public_action,
            "confidence": max(0, min(1, _number(signal.get("confidence"), 0))) if ai_ok else 0,
            "reason": public_reason,
            "status": 'ok' if ai_ok else 'waiting',
            "generated_at": _text(signal.get("generated_at_utc")),
        },
        "optimizer": {
            "status": optimizer_status,
            "sample_size": max(0, int(_number(tuning.get("sample_size"), 0))),
            "minimum_samples": max(1, int(_number(tuning.get("minimum_samples"), 6))),
            "revision": max(0, int(_number(tuning.get("revision"), 0))),
            "buy_confidence": max(0.64, min(0.82, _number(tuning.get("buy_confidence"), 0.68))),
            "reason": '参数需要独立验证，交易样本不等于策略优势。',
        },
        "started_at": _text(experiment.get("started_at")),
        "ends_at": _text(experiment.get("ends_at")),
        "updated_at": now,
    }


def read_us_analysis(path: Path = US_ANALYSIS_PATH, *, now: float | None = None):
    now=time.time() if now is None else now
    fallback=dict(status='unavailable',generated_at=None,analysis_started_at=None,decisions=[],
                  model_status='not_configured',execution_enabled=False)
    try:
        if path.is_symlink(): return fallback
        with path.open('rb') as handle: raw=handle.read(65537)
        if len(raw)>65536: return fallback
        def pairs(items):
            result={}
            for k,v in items:
                if k in result: raise ValueError('duplicate analysis key')
                result[k]=v
            return result
        def invalid(v): raise ValueError('nonfinite analysis')
        value=json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)
        at=_number(value.get('generated_at'));started=_number(value.get('analysis_started_at'));digest=value.get('audit_sha256')
        if (value.get('schema')!=1 or value.get('mode')!='paper' or value.get('phase')!='shadow'
                or value.get('execution_enabled') is not False or value.get('model_status')!='not_configured'
                or at is None or not 0<=now-at<=75 or started is None or not 0<started<=at
                or not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest)
                or value.get('status') not in ('analyzing','market_closed')): return fallback
        rows=value.get('decisions');clean=[]
        if not isinstance(rows,list) or len(rows)>3 or (value['status']=='market_closed' and rows): return fallback
        for row in rows:
            symbol=row['symbol'];generated=_number(row.get('generated_at'));ends=_number(row.get('bar_ends_at'))
            if (not isinstance(symbol,str) or not re.fullmatch(r'[A-Z][A-Z0-9.\-/]{0,19}',symbol)
                    or generated is None or not 0<=at-generated<=1020 or ends is None or not 0<ends<=generated
                    or type(row.get('entry_candidate')) is not bool or type(row.get('exit_candidate')) is not bool
                    or row.get('final_action')!='hold' or row.get('reason') not in
                    ('rules_evaluated','insufficient_bars','input_unavailable','invalid_or_stale_bars')
                    or row.get('vetoes')!=['strategy_not_accepted','execution_not_enabled','model_not_configured']): return fallback
            features=row.get('features');output=None
            if row['reason']=='rules_evaluated':
                if not 0<=now-ends<=1020 or not 0<=generated-ends<=1020: return fallback
                if not isinstance(features,dict): return fallback
                names=('last_price_usd','sma_fast_usd','sma_slow_usd','rsi_sma','momentum_fraction','volatility_15m_fraction','volume_ratio')
                output={k:_number(features.get(k)) for k in names}
                if (any(v is None or abs(v)>1e6 for v in output.values()) or any(output[k]<=0 for k in names[:3])
                        or not 0<=output['rsi_sma']<=100 or output['momentum_fraction']<=-1
                        or output['volatility_15m_fraction']<0 or output['volume_ratio']<0
                        or features.get('interval_seconds')!=900 or features.get('volatility_annualized') is not False
                        or features.get('formula_version')!='simple-window-v1'): return fallback
            elif features is not None or row['entry_candidate'] or row['exit_candidate']: return fallback
            clean.append(dict(symbol=symbol,generated_at=generated,bar_ends_at=ends,features=output,
                entry_candidate=row['entry_candidate'],exit_candidate=row['exit_candidate'],final_action='hold',
                reason=row['reason'],vetoes=list(row['vetoes'])))
        if len({r['symbol'] for r in clean})!=len(clean): return fallback
        return dict(status=value['status'],generated_at=at,analysis_started_at=started,decisions=clean,
                    model_status='not_configured',execution_enabled=False)
    except (OSError,TypeError,ValueError,KeyError,OverflowError,AttributeError): return fallback


def read_us_model(path: Path = US_MODEL_PATH, *, now: float | None = None):
    now=time.time() if now is None else now
    fallback=dict(status='unavailable',generated_at=None,decisions=[],execution_enabled=False,
                  attempts_today=None,daily_limit=60)
    try:
        if path.is_symlink(): return fallback
        with path.open('rb') as handle: raw=handle.read(16385)
        if len(raw)>16384: return fallback
        def pairs(items):
            result={}
            for k,v in items:
                if k in result: raise ValueError('duplicate model key')
                result[k]=v
            return result
        def invalid(v): raise ValueError('invalid model number')
        value=json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)
        at=_number(value.get('generated_at'));count=value.get('attempts_today');rows=value.get('decisions');clean=[]
        statuses=('reviewed','waiting_for_new_bar','market_closed','input_unavailable','rate_limited',
                  'budget_exhausted','model_unavailable')
        if (value.get('schema')!=1 or value.get('mode')!='paper' or value.get('phase')!='model_assistance'
                or value.get('execution_enabled') is not False or at is None or not 0<=now-at<=75
                or value.get('status') not in statuses or type(count) is not int or not 0<=count<=60
                or value.get('daily_limit')!=60 or value.get('hourly_limit')!=20 or value.get('min_interval_seconds')!=60
                or not isinstance(rows,list) or len(rows)>3): return fallback
        digest=value.get('audit_sha256')
        if rows and (not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest)): return fallback
        for row in rows:
            symbol=row['symbol'];ends=_number(row.get('bar_ends_at'));reviewed=_number(row.get('reviewed_at'))
            reason=row['reason'];model=row['model'];vetoes=row['vetoes'];current=row.get('input_current_at_review')
            if (not isinstance(symbol,str) or not re.fullmatch(r'[A-Z][A-Z0-9.\-/]{0,19}',symbol)
                    or ends is None or reviewed is None or not 0<ends<=reviewed<=at or not 0<=now-reviewed<=86400
                    or row['context'] not in ('historical_review','closed_bar_review') or type(current) is not bool
                    or (current and reviewed-ends>1020) or row.get('signal_usable') is not False
                    or row.get('final_action')!='hold' or row['action'] not in ('buy','sell','hold')
                    or vetoes not in (['strategy_not_accepted','execution_not_enabled'],
                                     ['strategy_not_accepted','execution_not_enabled','model_outside_candidate'])
                    or not isinstance(reason,str) or not 1<=len(reason)<=240 or reason!=reason.strip()
                    or any(ord(c)<32 or c in '<>' for c in reason)
                    or re.search(r'(?i)sk-|https?://|bearer|password|api.?key|secret|token|[A-Za-z0-9_\-]{32,}',reason)
                    or not isinstance(model,str) or not re.fullmatch(r'(?i)deepseek[A-Za-z0-9._-]{1,80}',model)
                    or re.search(r'(?i)sk-|bearer',model)): return fallback
            clean.append(dict(symbol=symbol,bar_ends_at=ends,reviewed_at=reviewed,reason=reason,model=model,
                action=row['action'],context=row['context'],signal_usable=False,input_current_at_review=current,
                current_at_display=current and row['context']=='closed_bar_review' and 0<=now-ends<=1020,
                final_action='hold',vetoes=list(vetoes)))
        if len({r['symbol'] for r in clean})!=len(clean): return fallback
        return dict(status=value['status'],generated_at=at,decisions=clean,execution_enabled=False,
                    attempts_today=count,daily_limit=60)
    except (OSError,ValueError,TypeError,KeyError,AttributeError,OverflowError): return fallback


def _us_details(value, generated, account_at, digest, now):
    if (value.get('details_ok') is not True or account_at is None or not account_at<=generated<=now
            or now-account_at>75 or not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest)):
        return {}
    counts=[value.get(n) for n in ('positions_count','open_orders_count')]
    holdings=value.get('holdings');trades=value.get('recent_trades')
    if (any(type(n) is not int or not 0<=n<=1000 for n in counts) or not isinstance(holdings,list)
            or len(holdings)!=min(counts[0],20) or not isinstance(trades,list) or len(trades)>20
            or type(value.get('holdings_limited')) is not bool or value['holdings_limited']!=(counts[0]>20)
            or type(value.get('history_window_limited')) is not bool): return {}
    clean_holdings=[];clean_trades=[]
    try:
        for row in holdings:
            symbol=row['symbol']
            if not isinstance(symbol,str) or not re.fullmatch(r'[A-Z][A-Z0-9.\-/]{0,19}',symbol): return {}
            fields=('quantity','average_cost_usd','reference_price_usd','unrealized_usd')
            numbers={name:_number(row.get(name)) for name in fields}
            if (any(n is None or abs(n)>1e12 for n in numbers.values())
                    or any(not 0<numbers[n]<=1000000 for n in fields[:3])): return {}
            clean_holdings.append(dict(symbol=symbol,**numbers))
        if len({r['symbol'] for r in clean_holdings})!=len(clean_holdings): return {}
        for row in trades:
            symbol=row['symbol'];qty=_number(row.get('quantity'));price=_number(row.get('average_price_usd'));at=_number(row.get('reported_at'))
            if (not isinstance(symbol,str) or not re.fullmatch(r'[A-Z][A-Z0-9.\-/]{0,19}',symbol)
                    or qty is None or price is None or not 0<qty<=1000000 or not 0<price<=1000000
                    or at is None or not 0<at<=generated or row.get('side') not in ('buy','sell')
                    or row.get('status') not in ('filled','partially_filled')
                    or row.get('time_kind') not in (('filled_at','updated_at') if row['status']=='filled' else ('updated_at',))): return {}
            clean_trades.append(dict(symbol=symbol,side=row['side'],quantity=qty,average_price_usd=price,
                reported_at=at,time_kind=row['time_kind'],status=row['status']))
    except (KeyError,TypeError,ValueError): return {}
    return dict(details_ok=True,holdings=clean_holdings,recent_trades=clean_trades,
        positions_count=counts[0],open_orders_count=counts[1],holdings_limited=value['holdings_limited'],
        history_window_limited=value['history_window_limited'])


def read_us_pilot(path: Path = US_PILOT_PATH, *, now: float | None = None) -> dict[str, Any]:
    now=time.time() if now is None else now
    result=dict(mode='paper',phase='experimental_pilot',execution_enabled=False,capital_usd=10000,
        formal_strategy_accepted=False,system_status='试运行状态待核验',accounting_ok=False,
        equity_usd=None,cash_usd=None,profit_usd=None,return_pct=None,positions_count=None,open_orders_count=None,
        observed_at=None,observing_started_at=None,paper_started_at=None,details_ok=False,
        holdings=[],recent_trades=[],holdings_limited=False,history_window_limited=False,
        analysis=read_us_analysis(now=now),model=read_us_model(now=now),
        scan=dict(status='unavailable',as_of=None,pool_size=None,checked_count=None,candidate_count=None))
    def read(target):
        if target.is_symlink():raise ValueError('projection link refused')
        with target.open('rb') as handle:raw=handle.read(65537)
        if len(raw)>65536:raise ValueError('projection too large')
        def pairs(items):
            data={}
            for k,v in items:
                if k in data:raise ValueError('duplicate field')
                data[k]=v
            return data
        return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('nonfinite')))
    try:
        value=read(path);pulse=read(path.with_name('heartbeat.json'))
        generated=_number(value.get('generated_at'));beat=_number(pulse.get('generated_at'))
        if (value.get('schema')!=1 or value.get('mode')!='paper' or value.get('phase')!='experimental_pilot'
                or value.get('capital_usd')!=10000 or value.get('formal_strategy_accepted') is not False
                or value.get('live_authorized') is not False or pulse.get('mode')!='paper'
                or generated is None or not 0<=now-generated<=1000
                or beat is None or not 0<=now-beat<=75):return result
        status=value.get('status');statuses=dict(market_closed='模拟试运行｜休市待开盘',
            observing_signal='模拟试运行｜分析观望',entry_submitted='模拟入场委托已提交',
            entry_not_sent='模拟委托未发送',risk_veto='风控否决｜继续观望',waiting_for_model='等待新的模型复核',
            inputs_unavailable='行情待更新｜暂停新单',signal_already_processed='本条信号已处理',
            daily_entry_limit='达到当日入场次数限制',close_requested='模拟退出处理中',close_recovery='模拟订单恢复处理中',
            protection_repair_dispatched='模拟保护委托已发送',protection_repair_not_sent='保护委托未发送｜等待核验',
            protection_repair_unknown='保护委托结果未知｜正在核验',account_loss_limit='账户亏损限制｜暂停开仓',
            trial_complete='模拟试运行期限已结束｜继续管理已有订单',orders_halted='已暂停下单',
            reconciliation_pending='账户或订单核验中',cycle_unavailable='本轮检查失败｜等待重试',
            authorization_unavailable='试运行授权待核验',continuity_unavailable='账本连续性待核验',
            strategy_changed='策略版本待核验',whole_share_budget='整股价格超出试验额度')
        if status not in statuses:return result
        result.update(system_status=statuses[status],observed_at=generated,
            execution_enabled=value.get('execution_enabled') is True and status not in (
                'authorization_unavailable','continuity_unavailable','strategy_changed','cycle_unavailable'),
            pilot_status=status)
        started=_number(value.get('started_at'))
        if started is not None and 0<started<=generated:result.update(observing_started_at=started,paper_started_at=started)
        # Reuse the existing explicit financial/stock allowlist. This is the
        # last verified book observation, labelled with its real timestamp.
        details=_us_details({**value,'details_ok':True,'holdings_limited':value.get('positions_count',0)>20,
            'history_window_limited':False},generated,generated,'a'*64,generated)
        fields=('equity_usd','cash_usd','profit_usd','return_pct')
        numbers={n:_number(value.get(n)) for n in fields}
        good=(value.get('accounting_ok') is True and details.get('details_ok') is True
            and all(v is not None and abs(v)<=1e12 for v in numbers.values())
            and abs(numbers['equity_usd']-10000-numbers['profit_usd'])<.011
            and abs(numbers['profit_usd']/100-numbers['return_pct'])<.000001
            and abs(numbers['cash_usd']+sum(h['quantity']*h['reference_price_usd'] for h in details['holdings'])-numbers['equity_usd'])<.011)
        if good:result.update(numbers,accounting_ok=True,**details)
        result['scan']=read_us_snapshot(US_STATUS_PATH,now=now,pilot_path=None)['scan']
        return result
    except (OSError,ValueError,TypeError,KeyError,OverflowError):return result


def read_us_snapshot(path: Path = US_STATUS_PATH, *, now: float | None = None, pilot_path: Path | None = US_PILOT_PATH) -> dict[str, Any]:
    now=time.time() if now is None else now
    if pilot_path is not None and (pilot_path.exists() or pilot_path.is_symlink()):return read_us_pilot(pilot_path,now=now)
    result=dict(mode='paper',phase='observing',execution_enabled=False,capital_usd=10000,
        system_status='状态暂时不可用',accounting_ok=False,equity_usd=None,cash_usd=None,
        profit_usd=None,return_pct=None,positions_count=None,open_orders_count=None,
        observed_at=None,observing_started_at=None,paper_started_at=None,
        details_ok=False,holdings=[],recent_trades=[],holdings_limited=False,history_window_limited=False,
        analysis=read_us_analysis(now=now),model=read_us_model(now=now),
        scan=dict(status='unavailable',as_of=None,pool_size=None,checked_count=None,candidate_count=None))
    try:
        if path.is_symlink(): return result
        with path.open('rb') as handle: raw=handle.read(65537)
        if len(raw)>65536: return result
        def pairs(items):
            value={}
            for key,item in items:
                if key in value: raise ValueError('duplicate projection key')
                value[key]=item
            return value
        def constant(value): raise ValueError('nonfinite projection')
        value=json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)
        if (not isinstance(value,dict) or type(value.get('schema')) is not int or value['schema']!=1
                or value.get('mode')!='paper' or value.get('currency')!='USD'
                or type(value.get('capital_usd')) is not int or value['capital_usd']!=10000
                or value.get('phase')!='observing' or value.get('execution_enabled') is not False
                or value.get('paper_started_at') is not None): return result
        generated=_number(value.get('generated_at'));account_at=_number(value.get('account_received_at'))
        if generated is None or not 0<=now-generated<=75: return result
        result['observed_at']=generated
        started=_number(value.get('observing_started_at'))
        if started is not None and 0<started<=generated: result['observing_started_at']=started
        statuses=dict(observing='观察运行中',account_unreconciled='账户待核验',
                      account_unavailable='账户连接待恢复',audit_unavailable='记录异常，盈亏待核验')
        result['system_status']=statuses.get(value.get('status'),'状态暂时不可用')
        fields=('equity_usd','cash_usd','profit_usd','return_pct','positions_count','open_orders_count')
        numbers={name:_number(value.get(name)) for name in fields}
        digest=value.get('audit_sha256')
        details=_us_details(value,generated,account_at,digest,now)
        healthy=(value.get('accounting_ok') is True and value.get('status')=='observing'
            and details.get('details_ok') is True and not details['recent_trades'] and not details['history_window_limited']
            and account_at is not None and account_at<=generated and 0<=now-account_at<=75
            and isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest)
            and all(number is not None for number in numbers.values())
            and numbers['equity_usd']==numbers['cash_usd']==10000
            and numbers['profit_usd']==numbers['return_pct']==numbers['positions_count']==numbers['open_orders_count']==0)
        if healthy: result.update(numbers,accounting_ok=True)
        else: result['system_status']='盈亏待核验' if value.get('status')=='observing' else result['system_status']
        result.update(details)
        scan=value.get('scan')
        if isinstance(scan,dict):
            at=_number(scan.get('as_of'));pool=scan.get('pool_size');checked=scan.get('checked_count');candidates=scan.get('candidate_count')
            if (scan.get('status') in ('ok','market_closed') and at is not None and at<=generated and 0<=now-at<=180
                    and all(type(n) is int for n in (pool,checked,candidates))
                    and 0<=candidates<=checked<=min(100,pool) and 0<pool<=20000
                    and (scan['status']!='market_closed' or checked==candidates==0)):
                result['scan']=dict(status=scan['status'],as_of=at,pool_size=pool,checked_count=checked,candidate_count=candidates)
        return result
    except (OSError,ValueError,TypeError,OverflowError): return result


US_PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Paperbot 美股模拟实验</title>
<style>
:root{color-scheme:dark;--bg:#101419;--surface:#181e26;--ink:#e5e9ee;--muted:#a7b0bd;--accent:#d6b36a;--line:#303944}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:"Microsoft YaHei UI","PingFang SC",sans-serif;font-size:15px;line-height:1.65}main{max-width:960px;margin:auto;padding:44px 24px 30px}header{display:flex;justify-content:space-between;align-items:start;gap:20px}h1{font-size:26px;font-weight:600;margin:0 0 4px}p{margin:0}a{color:var(--muted);text-underline-offset:4px}a:focus-visible{outline:2px solid var(--accent);outline-offset:5px}.muted{color:var(--muted)}.state{color:var(--accent);border:1px solid var(--line);padding:5px 12px;border-radius:20px;white-space:nowrap}.money{padding:35px 0 28px;border-bottom:1px solid var(--line)}.equity{font-size:58px;line-height:1.3;font-weight:500;letter-spacing:-1px;font-variant-numeric:tabular-nums}.currency{font-size:18px;color:var(--muted);margin-left:8px}.results{display:flex;gap:34px;margin-top:16px}.results strong{font-weight:500;font-size:23px;font-variant-numeric:tabular-nums}.results span{display:block;color:var(--muted);font-size:13px}.positive{color:#8dc6ae}.negative{color:#eea0a0}.details{display:grid;grid-template-columns:1fr 1fr;gap:32px;margin-top:30px}section{min-width:0}h2{font-size:17px;margin:0 0 16px;font-weight:600}.row{display:flex;justify-content:space-between;gap:20px;padding:9px 0;border-bottom:1px solid #252e38}.row span:first-child{color:var(--muted)}.row strong{font-weight:500;text-align:right;font-variant-numeric:tabular-nums}.activity{background:var(--surface);padding:22px;border-radius:12px}.activity p{margin:7px 0}.notice{margin-top:18px;font-size:13px;color:var(--muted);max-width:64ch}footer{margin-top:30px;display:flex;justify-content:space-between;gap:18px;color:var(--muted);font-size:12px}.unavailable{color:var(--muted)}@media(max-width:620px){main{padding:26px 18px}header{display:block}.state{display:inline-block;margin-top:14px}.equity{font-size:44px}.details{grid-template-columns:1fr;gap:24px}.money{padding-top:26px}.results{gap:28px}footer{display:block}footer a{display:inline-block;margin-top:8px}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
.stock-section{margin-top:32px;padding-top:22px;border-top:1px solid var(--line)}.stock-list{white-space:pre-line;line-height:1.9;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}.stock-section .muted{font-size:13px;margin-top:12px}
</style></head><body><main>
<header><div><h1>Paperbot 美股模拟实验</h1><p class="muted">虚拟本金 10,000 美元</p></div><span id="state" class="state" role="status">正在读取状态</span></header>
<div class="money"><p class="muted">模拟账户总资产</p><div><span id="equity" class="equity">—</span><span class="currency">美元</span></div><div class="results"><div><strong id="profit">—</strong><span>累计盈亏 · 美元</span></div><div><strong id="return">—</strong><span>累计收益率</span></div></div></div>
<div class="details"><section><h2>账户概况</h2><div class="row"><span>可用现金</span><strong id="cash">—</strong></div><div class="row"><span>当前持仓</span><strong id="positions">—</strong></div><div class="row"><span>待成交订单</span><strong id="orders">—</strong></div><div class="row"><span>观察开始时间</span><strong id="started">—</strong></div></section>
<section class="activity"><h2>正在做什么</h2><p id="activity">正在读取行情和账户状态。</p><div class="row"><span>当前股票池</span><strong id="pool">—</strong></div><div class="row"><span>本批通过筛选</span><strong id="candidates">—</strong></div><p class="notice">通过筛选不等于买入。实际动作以独立风控和模拟订单记录为准；试运行不代表策略通过正式验收。</p></section></div>
<section class="stock-section"><h2>持续分析</h2><p id="engineStatus" role="status">正在读取分析状态。</p><div id="analysisList" class="stock-list">分析记录待更新。</div><p id="analysisUpdated" class="muted">等待首次分析记录。</p><p class="muted">每轮检查最新扫描批次中最多 3 个候选，用已收盘的 15 分钟行情计算指标。DeepSeek 的独立复核列在下方；量化建议与实际成交分别记录。</p></section>
<section class="stock-section"><h2>DeepSeek 复核</h2><p id="modelStatus" role="status">正在读取模型状态。</p><div id="modelList" class="stock-list">模型记录待更新。</div><p id="modelUpdated" class="muted">等待模型记录。</p><p class="muted">有新的有效行情时进行限额分析；休市停止定期调用。分析意见不代表预测或成交，风控保留否决权。</p></section>
<section class="stock-section"><h2>当前持仓</h2><div id="holdingList" class="stock-list">正在读取持仓。</div><p id="holdingNote" class="muted">成本、参考现价及浮动盈亏来自模拟券商，供查看。</p></section>
<section class="stock-section"><h2>最近买卖</h2><div id="tradeList" class="stock-list">正在读取成交记录。</div><p class="muted">显示最近 50 笔委托中的成交订单，最多 20 条；部分成交显示最近更新时间，不是逐笔撮合明细。</p></section>
<p id="quality" class="notice">盈亏仅在账户记录通过核验后显示。</p><footer><span id="updated">等待首次更新</span><a href="/">查看 BTC 100 USDT 实验</a></footer></main>
<script>
const $=id=>document.getElementById(id),money=n=>typeof n==='number'&&Number.isFinite(n)?n.toLocaleString('zh-CN',{minimumFractionDigits:2,maximumFractionDigits:2}):'—',date=n=>typeof n==='number'?new Date(n*1000).toLocaleString('zh-CN'):'—';
function clearMetrics(){for(const id of ['equity','profit','return','cash','positions','orders'])$(id).textContent='—';$('profit').className='unavailable';$('return').className='unavailable';$('holdingList').textContent='持仓状态待更新。';$('tradeList').textContent='成交记录待更新。';$('analysisList').textContent='分析记录待更新。';$('engineStatus').textContent='分析状态待更新。';$('analysisUpdated').textContent='等待连接恢复。';$('modelList').textContent='模型记录待更新。';$('modelStatus').textContent='模型状态待更新。';$('modelUpdated').textContent='等待连接恢复。'}
function paintAnalysis(a){if(!a||a.status==='unavailable')return;$('engineStatus').textContent=a.status==='market_closed'?'美股休市，服务保持在线。':'量化分析服务运行中，实际执行另行核验。';const reasons={insufficient_bars:'完整行情数量不足，等待指标预热。',input_unavailable:'本轮行情读取失败，等待重试。',invalid_or_stale_bars:'行情缺失或过期，已阻止使用。'};$('analysisList').textContent=(a.decisions||[]).map(d=>{if(!d.features)return d.symbol+'  ·  观望\n'+(reasons[d.reason]||'输入待核验。');const f=d.features,signal=d.entry_candidate?'满足入场规则':d.exit_candidate?'满足退出规则（需有持仓）':'继续观望';return d.symbol+'  ·  '+signal+'\nSMA '+money(f.sma_fast_usd)+' / '+money(f.sma_slow_usd)+'  ·  RSI '+money(f.rsi_sma)+'\n动量 '+money(f.momentum_fraction*100)+'%  ·  15 分钟波动率 '+money(f.volatility_15m_fraction*100)+'%\n量化候选仅供参考；实际动作见运行状态。\n分析时间 '+date(d.generated_at)}).join('\n\n')||(a.status==='market_closed'?'等待下一个正常交易时段。':'本批暂无可进入指标分析的候选。');$('analysisUpdated').textContent='服务最近检查：'+date(a.generated_at)+' · 观察规则不会冒充已成交订单。'}
function paintModel(m){if(!m||m.status==='unavailable')return;const states={reviewed:'已收到并留档 DeepSeek 分析。',waiting_for_new_bar:'等待新的有效行情，避免重复调用。',market_closed:'美股休市，定期模型调用已暂停。',input_unavailable:'行情不足或过期，暂不请求模型。',rate_limited:'已达到本时段调用限制，稍后继续。',budget_exhausted:'今日调用额度已用完，下一天恢复。',model_unavailable:'本次模型响应未通过核验，保持观望。'};$('modelStatus').textContent=states[m.status]||'模型状态待更新。';$('modelList').textContent=(m.decisions||[]).map(r=>r.symbol+'  ·  '+(r.current_at_display?'收盘行情复核':'历史复核')+'  ·  '+({buy:'偏向买入',sell:'偏向卖出',hold:'观望'}[r.action]||'观望')+'\n'+r.reason+'\n行情截至 '+date(r.bar_ends_at)+'  ·  复核时间 '+date(r.reviewed_at)+'\n最终观望：'+(r.vetoes.includes('model_outside_candidate')?'模型意见与量化候选不一致。':'历史辅助意见，实际动作以风控及执行记录为准。')).join('\n\n')||'暂无通过核验的模型复核记录。';$('modelUpdated').textContent='今日已占用 '+m.attempts_today+' / '+m.daily_limit+' 次调用额度 · 最近检查 '+date(m.generated_at)}
function paintStocks(d){if(!d.details_ok)return;$('positions').textContent=d.positions_count+' 只';$('orders').textContent=d.open_orders_count+' 笔';$('holdingList').textContent=d.holdings.length?d.holdings.map(h=>h.symbol+'  ·  '+h.quantity+' 股\n成本 '+money(h.average_cost_usd)+'  /  参考现价 '+money(h.reference_price_usd)+' 美元\n参考浮动盈亏 '+(h.unrealized_usd>0?'+':'')+money(h.unrealized_usd)+' 美元').join('\n\n'):'暂无持仓。';$('holdingNote').textContent=(d.holdings_limited?'仅显示前 20 只持仓。':'')+'成本、参考现价及浮动盈亏来自模拟券商，不作为下单报价。';$('tradeList').textContent=d.recent_trades.length?d.recent_trades.map(t=>(t.side==='buy'?'买入 ':'卖出 ')+t.symbol+'  ·  已成交 '+t.quantity+' 股\n均价 '+money(t.average_price_usd)+' 美元  ·  '+(t.status==='filled'?'全部成交':'部分成交')+'\n'+(t.time_kind==='filled_at'?'成交时间 ':'更新时间 ')+date(t.reported_at)).join('\n\n'):'最近 50 笔委托中暂无成交记录。'}
function paint(d){clearMetrics();$('state').textContent=d.system_status||'状态暂时不可用';if(d.accounting_ok){$('equity').textContent=money(d.equity_usd);$('cash').textContent=money(d.cash_usd)+' 美元';$('profit').textContent=(d.profit_usd>0?'+':'')+money(d.profit_usd);$('return').textContent=(d.return_pct>0?'+':'')+money(d.return_pct)+'%';for(const [id,n] of [['profit',d.profit_usd],['return',d.return_pct]])$(id).className=n>0?'positive':n<0?'negative':'';$('positions').textContent=d.positions_count+' 只';$('orders').textContent=d.open_orders_count+' 笔'}paintStocks(d);paintAnalysis(d.analysis);paintModel(d.model);const s=d.scan||{};$('pool').textContent=Number.isInteger(s.pool_size)?s.pool_size.toLocaleString('zh-CN')+' 只':'—';$('candidates').textContent=Number.isInteger(s.candidate_count)?s.candidate_count+' 只':'—';$('activity').textContent=s.status==='ok'?'正在分批扫描行情，记录筛选结果。':s.status==='market_closed'?'美股当前休市，继续观察账户和运行状态。':'行情待更新，暂不生成交易信号。';$('started').textContent=date(d.observing_started_at);$('quality').textContent=d.accounting_ok?'账户已核验。当前为虚拟资金，尚未产生本项目的美股成交。':'账户或记录暂未通过核验，盈亏待确认。';$('updated').textContent='最近记录：'+date(d.observed_at)+' · 每 10 秒刷新'}
function paintPilot(d){if(d.phase!=='experimental_pilot')return;$('started').textContent=date(d.paper_started_at);$('activity').textContent=d.system_status||'试运行状态待更新';$('quality').textContent=d.accounting_ok?'纯模拟试运行。上方为最近一次已对账的资金和持仓；策略绩效尚未通过正式验收。':'试运行中的账户或订单待对账，暂不显示盈亏。';$('updated').textContent='最近执行检查：'+date(d.observed_at)+' · 网页每 10 秒刷新';}
async function refresh(){try{const r=await fetch('/api/us/status',{cache:'no-store',signal:AbortSignal.timeout(5000)});if(!r.ok)throw Error('unavailable');const d=await r.json();paint(d);paintPilot(d)}catch(e){clearMetrics();$('state').textContent='连接中断，状态待更新';$('quality').textContent='暂时无法取得新记录，已隐藏旧的资金数字。';$('pool').textContent='—';$('candidates').textContent='—';$('activity').textContent='等待连接恢复。'}}refresh();setInterval(refresh,10000);
</script></body></html>"""


PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI 模拟盘状态</title><style>
:root{color-scheme:dark;--bg:#0b0d10;--card:#15191e;--line:#262c33;--text:#f5f7f8;--muted:#9199a4;--green:#56d69a;--red:#ff7777;--gold:#e6bd68}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif}.wrap{width:min(1040px,calc(100% - 32px));margin:0 auto;padding:46px 0 64px}
header{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;margin-bottom:30px}.eyebrow{font-size:12px;letter-spacing:.16em;color:var(--gold);text-transform:uppercase}.title{font-size:clamp(28px,5vw,48px);font-weight:650;letter-spacing:-.04em;margin:7px 0 5px}.sub{color:var(--muted);font-size:14px}.live{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--muted);white-space:nowrap;margin-top:9px}.dot{width:9px;height:9px;border-radius:50%;background:var(--green);box-shadow:0 0 0 5px #56d69a18}.dot.bad{background:var(--red);box-shadow:0 0 0 5px #ff777718}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:22px}.metric{grid-column:span 3}.wide{grid-column:span 8}.side{grid-column:span 4}.label{color:var(--muted);font-size:13px;margin-bottom:13px}.value{font-size:30px;font-weight:650;letter-spacing:-.035em}.unit{font-size:15px;color:var(--muted);margin-left:5px}.positive{color:var(--green)}.negative{color:var(--red)}.neutral{color:var(--text)}
.row{display:flex;justify-content:space-between;gap:18px;padding:12px 0;border-top:1px solid var(--line);font-size:14px}.row:first-of-type{border-top:0}.muted{color:var(--muted)}.signal{font-size:20px;font-weight:600;margin:3px 0 10px}.reason{color:#c2c8cf;line-height:1.75;font-size:14px;min-height:49px}.pill{display:inline-flex;padding:6px 10px;border:1px solid var(--line);border-radius:999px;font-size:12px;color:var(--muted);margin-top:16px}.progress{height:7px;background:#252a31;border-radius:99px;overflow:hidden;margin:18px 0 10px}.bar{height:100%;width:0;background:var(--gold);border-radius:inherit;transition:width .5s}.foot{margin-top:18px;text-align:right;color:#656d77;font-size:12px}
@media(max-width:760px){.wrap{padding-top:28px}.metric{grid-column:span 6}.wide,.side{grid-column:1/-1}header{display:block}.live{margin-top:18px}}@media(max-width:430px){.metric{grid-column:1/-1}.grid{gap:10px}.card{padding:19px}}
</style></head><body><main class="wrap"><header><div><div class="eyebrow">DeepSeek × Freqtrade</div><h1 class="title">100 USDT 模拟盘</h1><div class="sub">30 天实验 · BTC/USDT 现货 · 只读状态</div></div><div class="live"><span id="dot" class="dot"></span><span id="status">正在读取</span></div></header>
<section class="grid"><article class="card metric"><div class="label">当前总资产</div><div class="value"><span id="equity">—</span><span class="unit">USDT</span></div></article><article class="card metric"><div class="label">累计盈亏</div><div id="profit" class="value">—</div></article><article class="card metric"><div class="label">累计收益率</div><div id="return" class="value">—</div></article><article class="card metric"><div class="label">当前持仓</div><div class="value"><span id="openCount">—</span><span class="unit">笔</span></div></article>
<article class="card wide"><div class="label">最新 AI 判断</div><div id="signal" class="signal">—</div><div id="reason" class="reason">—</div><div id="confidence" class="pill">模型自评分 —</div></article><article class="card side"><div class="label">交易表现</div><div class="row"><span class="muted">已完成交易</span><strong id="closed">—</strong></div><div class="row"><span class="muted">胜率</span><strong id="winRate">—</strong></div><div class="row"><span class="muted">已实现盈亏</span><strong id="realized">—</strong></div><div class="row"><span class="muted">浮动盈亏</span><strong id="unrealized">—</strong></div><div class="row"><span class="muted">参数验证</span><strong id="optimizer">—</strong></div></article>
<article class="card wide"><div class="label">当前仓位</div><div id="position"><div class="muted">暂无持仓，等待符合条件的模拟信号。</div></div></article><article class="card side"><div class="label">实验进度</div><div class="progress"><div id="bar" class="bar"></div></div><div class="row"><span class="muted">开始</span><span id="started">—</span></div><div class="row"><span class="muted">结束</span><span id="ends">—</span></div></article></section><div id="updated" class="foot">—</div></main>
<script>
const $=id=>document.getElementById(id), money=n=>n==null?'—':Number(n).toFixed(2), pct=n=>n==null?'—':(n>=0?'+':'')+Number(n).toFixed(2)+'%', tone=(el,n)=>{el.className='value '+(n>0?'positive':n<0?'negative':'neutral')}, dt=v=>v?new Date(v).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'}):'—';
function paintPilot(d){if(d.phase!=='experimental_pilot')return;$('started').textContent=date(d.paper_started_at);$('activity').textContent=d.system_status||'试运行状态待更新';$('quality').textContent=d.accounting_ok?'纯模拟试运行。上方为最近一次已对账的资金和持仓；策略绩效尚未通过正式验收。':'试运行中的账户或订单待对账，暂不显示盈亏。';$('updated').textContent='最近执行检查：'+date(d.observed_at)+' · 网页每 10 秒刷新';}
async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'});const d=await r.json();$('status').textContent=d.system_status;$('dot').className='dot'+(d.bot_running&&d.accounting_ok&&d.valuation_ok?'':' bad');$('equity').textContent=money(d.equity);$('profit').textContent=d.profit_abs==null?'—':(d.profit_abs>=0?'+':'')+money(d.profit_abs)+' USDT';tone($('profit'),d.profit_abs);$('return').textContent=pct(d.return_pct);tone($('return'),d.return_pct);$('openCount').textContent=d.open_count??'—';const map={buy:'建议买入',hold:'继续观望',sell:'建议卖出'};$('signal').textContent=map[d.signal.action]||'继续观望';$('reason').textContent=d.signal.reason;$('confidence').textContent='模型自评分（非盈利概率） '+Math.round(Number(d.signal.confidence)*100)+'% · '+dt(d.signal.generated_at);$('closed').textContent=d.closed_count==null?'—':d.closed_count+' 笔';$('winRate').textContent=d.win_rate==null?'—':Number(d.win_rate).toFixed(1)+'%';$('realized').textContent=d.realized_profit==null?'—':(d.realized_profit>=0?'+':'')+money(d.realized_profit)+' U';$('unrealized').textContent=d.unrealized_profit==null?'—':(d.unrealized_profit>=0?'+':'')+money(d.unrealized_profit)+' U';const o=d.optimizer;$('optimizer').textContent=o.sample_size<o.minimum_samples?'收集中 '+o.sample_size+'/'+o.minimum_samples:'提案待验证 · '+o.sample_size+' 笔样本';$('optimizer').title=o.reason;$('position').innerHTML='';if(!d.positions.length){const x=document.createElement('div');x.className='muted';x.textContent=d.accounting_ok?'暂无持仓，等待符合条件的模拟信号。':'成交数据不可用，持仓待核验。';$('position').appendChild(x)}else d.positions.forEach(p=>{const x=document.createElement('div');x.className='row';const name=document.createElement('span');name.textContent=p.pair+' · '+money(p.stake)+' USDT';const amount=document.createElement('strong');amount.className=p.profit==null?'neutral':p.profit>=0?'positive':'negative';amount.textContent=p.profit==null?'—':(p.profit>=0?'+':'')+money(p.profit)+' U';x.append(name,amount);$('position').appendChild(x)});$('started').textContent=dt(d.started_at);$('ends').textContent=dt(d.ends_at);const a=new Date(d.started_at),b=new Date(d.ends_at),n=new Date();$('bar').style.width=Math.max(0,Math.min(100,(n-a)/(b-a)*100))+'%';$('updated').textContent='自动刷新 · '+new Date(d.updated_at*1000).toLocaleTimeString('zh-CN')}catch(e){$('status').textContent='状态页暂时无法更新';$('dot').className='dot bad'}}refresh();setInterval(refresh,10000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ('/us','/us/'):
            self._send(200,"text/html; charset=utf-8",US_PAGE.encode('utf-8'))
        elif self.path=='/api/us/status':
            body=json.dumps(read_us_snapshot(),ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
            self._send(200,"application/json; charset=utf-8",body)
        elif self.path == "/" or self.path.startswith("/?"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif self.path == "/api/status":
            try:
                body = json.dumps(read_snapshot(), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
            except (ValueError, TypeError, OverflowError):
                self._send(503, "application/json; charset=utf-8", b'{"accounting_ok":false,"valuation_ok":false,"system_status":"unavailable"}')
            else:
                self._send(200, "application/json; charset=utf-8", body)
        elif self.path == "/health":
            self._send(200, "text/plain; charset=utf-8", b"ok\n")
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found\n")

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
