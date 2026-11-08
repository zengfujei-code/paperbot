#!/usr/bin/env python3
"""Generate a fail-closed paper-trading signal from Gate candles and DeepSeek."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PAIR = os.getenv("PAIR", "BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "15m")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
SIGNAL_PATH = Path(os.getenv("AI_SIGNAL_PATH", "/shared/signal.json"))
DECISION_LOG = Path(os.getenv("DECISION_LOG_PATH", "/shared/decisions.jsonl"))
TUNING_PATH = Path(os.getenv("AI_TUNING_PATH", "/shared/tuning.json"))
INTERVAL_SECONDS = int(os.getenv("ADVISOR_INTERVAL_SECONDS", "900"))
SIGNAL_TTL_SECONDS = int(os.getenv("SIGNAL_TTL_SECONDS", "1200"))


def _number(value: Any, *, strings: bool = False) -> float:
    if type(value) not in (int, float) and not (strings and isinstance(value, str)):
        raise ValueError('invalid numeric type')
    try:
        number = float(value)
        if not math.isfinite(number): raise ValueError('nonfinite number')
        return number
    except (ValueError, TypeError, OverflowError):
        raise ValueError('invalid finite number') from None


def _strict_json(raw: Any) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('duplicate JSON field')
            result[key] = value
        return result
    def invalid(value): raise ValueError('nonstandard JSON number')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def validate_candles(candles: list[list[float]]) -> None:
    if not isinstance(candles, list) or not 50 <= len(candles) <= 1000:
        raise ValueError('at least 50 bounded candles are required')
    previous = None
    for candle in candles:
        if not isinstance(candle, (list, tuple)) or len(candle) != 6:
            raise ValueError('invalid candle fields')
        stamp, opened, high, low, close, volume = [_number(v) for v in candle]
        if not (0 <= stamp <= 253402299900000 and stamp % 900000 == 0
                and 0 < low <= min(opened, close) <= max(opened, close) <= high and volume >= 0):
            raise ValueError('invalid candle chronology or OHLCV')
        if previous is not None and stamp - previous != 900000:
            raise ValueError('duplicate, missing or unordered candle')
        previous = stamp


def closed_candles(candles: list[list[float]], *, now: float) -> list[list[float]]:
    validate_candles(candles)
    now = _number(now)
    if candles[-1][0] / 1000 > math.floor(now / 900) * 900:
        raise ValueError('future candle')
    closed = [c for c in candles if c[0] / 1000 + 900 <= now]
    validate_candles(closed)
    if not 0 <= now - (closed[-1][0] / 1000 + 900) <= 1020:
        raise ValueError('stale candle source')
    return closed


def _pct(current: float, previous: float) -> float:
    return (current / previous - 1.0) * 100.0 if previous else 0.0


def compute_features(candles: list[list[float]]) -> dict[str, float | int]:
    """Compute compact, deterministic features from OHLCV candles."""
    validate_candles(candles)
    closes = [float(candle[4]) for candle in candles]
    volumes = [float(candle[5]) for candle in candles]
    deltas = [b - a for a, b in zip(closes[-15:-1], closes[-14:])]
    gains = [max(delta, 0.0) for delta in deltas]
    losses = [max(-delta, 0.0) for delta in deltas]
    avg_gain = sum(gains) / 14.0
    avg_loss = sum(losses) / 14.0
    if avg_gain == 0 and avg_loss == 0:
        rsi = 50.0
    elif avg_loss == 0:
        rsi = 100.0
    else:
        rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    returns = [_pct(b, a) for a, b in zip(closes[-21:-1], closes[-20:])]
    result: dict[str, float | int] = {
        "candles": len(candles),
        "last_close": closes[-1],
        "return_15m_pct": _pct(closes[-1], closes[-2]),
        "return_1h_pct": _pct(closes[-1], closes[-5]),
        "return_3h_pct": _pct(closes[-1], closes[-13]),
        "sma20": sum(closes[-20:]) / 20.0,
        "sma50": sum(closes[-50:]) / 50.0,
        "rsi14": rsi,
        "volatility20_pct": statistics.pstdev(returns),
        "volume_ratio20": volumes[-1] / (sum(volumes[-20:]) / 20.0) if sum(volumes[-20:]) else 0.0,
    }
    if not all(math.isfinite(float(value)) for value in result.values()):
        raise ValueError("computed feature is not finite")
    return {key: round(value, 8) if isinstance(value, float) else value for key, value in result.items()}


def compute_quant_score(features: dict[str, float | int]) -> float:
    """Blend several independent indicators into a bounded directional score."""
    sma20 = float(features["sma20"])
    sma50 = float(features["sma50"])
    trend = math.tanh(((sma20 / sma50) - 1.0) * 18.0) if sma50 else 0.0
    momentum_3h = math.tanh(float(features["return_3h_pct"]) / 2.0)
    momentum_1h = math.tanh(float(features["return_1h_pct"]) / 1.0)
    rsi = math.tanh((float(features["rsi14"]) - 50.0) / 15.0)
    volume = math.tanh((float(features["volume_ratio20"]) - 1.0) / 0.8)
    raw = 0.32 * trend + 0.34 * momentum_3h + 0.16 * momentum_1h + 0.12 * rsi + 0.06 * volume
    volatility = max(0.0, float(features["volatility20_pct"]))
    dampener = 1.0 / (1.0 + max(0.0, volatility - 1.0) * 0.2)
    return round(max(-1.0, min(1.0, raw * dampener)), 6)


def load_tuning() -> dict[str, Any]:
    try:
        value = _strict_json(TUNING_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def apply_quant_guard(decision: dict[str, Any], features: dict[str, float | int], *, evidence=None,minimum=None) -> dict[str, Any]:
    """Require independent quantitative confirmation before opening a position."""
    score = float(features.get("quant_score", 0.0))
    tuning = load_tuning() if minimum is None else {'min_quant_buy':minimum}
    try:
        minimum = max(0.05, min(0.35, _number(tuning.get("min_quant_buy", 0.10))))
    except ValueError:
        minimum = 0.10
    if evidence is not None:
        evidence.update(min_quant_buy=minimum,quant_score=score,raw_decision=dict(decision),
                        entry_veto=decision['action']=='buy' and score<minimum,
                        scope='signal_entry_guard_only_not_account_risk')
    if decision["action"] == "buy" and score < minimum:
        return {
            "action": "hold",
            "confidence": min(float(decision["confidence"]), 0.5),
            "reason": f"量化评分 {score:.2f} 未达到入场门槛 {minimum:.2f}，继续观望。",
        }
    return decision


def parse_model_reply(text: str) -> dict[str, Any]:
    """Parse and validate the model's deliberately tiny response schema."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
    value = _strict_json(stripped)
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    action = value.get("action")
    if action not in {"buy", "hold", "sell"}:
        raise ValueError("action must be buy, hold, or sell")
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number")
    confidence = _number(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    return {"action": action, "confidence": confidence, "reason": reason.strip()[:300]}


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
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


def append_decision(value: dict[str, Any]) -> None:
    DECISION_LOG.parent.mkdir(parents=True, exist_ok=True)
    payload=(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode('utf-8')
    if len(payload)>2*1024*1024: raise ValueError('decision record too large')
    with DECISION_LOG.open('a+b') as handle:
        size=handle.seek(0,os.SEEK_END)
        if size:
            handle.seek(max(0,size-2*1024*1024))
            tail=handle.read()
            if not tail.endswith(b'\n'): raise ValueError('journal incomplete; preserve for recovery')
            previous=_strict_json(tail.rstrip(b'\n').split(b'\n')[-1])
            if not isinstance(previous,dict): raise ValueError('invalid journal tail')
            if previous.get('schema')==2:
                checksum=previous.pop('sha256',None)
                digest=hashlib.sha256(json.dumps(previous,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()
                if checksum!=digest: raise ValueError('journal tail integrity mismatch')
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(DECISION_LOG.parent)


def normalize_gate_candles(raw: list[list[Any]]) -> list[list[float]]:
    """Map Gate's documented array fields to timestamp/open/high/low/close/volume."""
    candles = []
    if not isinstance(raw, list) or not 1 <= len(raw) <= 1000:
        raise ValueError('invalid Gate data shape')
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 7:
            raise ValueError("Gate returned an incomplete candle")
        candles.append(
            [
                _number(item[0], strings=True) * 1000,
                _number(item[5], strings=True),
                _number(item[3], strings=True),
                _number(item[4], strings=True),
                _number(item[2], strings=True),
                _number(item[6], strings=True),
            ]
        )
    candles.sort(key=lambda candle: candle[0])
    return candles


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('redirect refused')


def _response_json(request, *, timeout):
    opener = urllib.request.build_opener(_NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024: raise ValueError('response too large')
    return _strict_json(raw)


def fetch_input() -> tuple[dict[str, Any], dict[str, float | int]]:
    if PAIR != 'BTC/USDT' or TIMEFRAME != '15m':
        raise ValueError('unsupported experiment identity')
    pair = PAIR.replace("/", "_")
    url = f"https://api.gateio.ws/api/v4/spot/candlesticks?currency_pair={pair}&interval={TIMEFRAME}&limit=80"
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "deepseek-papertrade/1.0"})
    raw = _response_json(request, timeout=30)
    if not isinstance(raw, list):
        raise ValueError("Gate returned a non-list response")
    received = time.time()
    candles = closed_candles(normalize_gate_candles(raw), now=received)
    features = compute_features(candles)
    features.update(data_closed_at=candles[-1][0] / 1000 + 900, data_received_at=received)
    evidence = dict(source='gate_spot', pair=PAIR, timeframe=TIMEFRAME, interval_seconds=900,
                    price_unit='USDT_per_BTC', volume_unit='BTC', received_at=received,
                    latest_closed_at=features['data_closed_at'], candles=candles,
                    return_unit='percent', volatility='population_std_of_20_simple_returns_percent_not_annualized',
                    rsi='14_change_simple_window', volume_ratio='current_volume_over_current_inclusive_20_bar_mean')
    return evidence, features


def fetch_features() -> dict[str, float | int]:
    return fetch_input()[1]


def replay_decision(record):
    """Replay saved math/model reply/guard; never make a model or market call.

    Error outcomes can be verified as recorded errors, not independently
    reproduced network failures. Hashes do not resist a privileged rewrite.
    """
    content=dict(record)
    checksum=content.pop('sha256',None)
    digest=hashlib.sha256(json.dumps(content,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()
    if content.get('schema')!=2 or digest!=checksum:
        raise ValueError('decision integrity mismatch')
    if content['source_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('matching archived source version required')
    signal=content['signal']
    if signal['status']=='degraded':
        if signal['action']!='hold' or signal['confidence']!=0:
            raise ValueError('invalid recorded fallback')
        return dict(signal)
    candles=content['market_input']['candles']
    expected=compute_features(candles)
    expected['quant_score']=compute_quant_score(expected)
    expected.update(data_closed_at=signal['data_closed_at'],data_received_at=signal['data_received_at'])
    if signal['data_closed_at']!=candles[-1][0]/1000+900 or expected!=signal['features']:
        raise ValueError('recorded indicator mismatch')
    decision=parse_model_reply(content['model']['raw_reply'])
    if decision!=content['model']['parsed_decision'] or decision!=content['guard']['raw_decision']:
        raise ValueError('recorded model parse mismatch')
    final=apply_quant_guard(decision,expected,minimum=content['guard']['min_quant_buy'])
    if any(final[key]!=signal[key] for key in ('action','confidence','reason')):
        raise ValueError('recorded guard result mismatch')
    return dict(signal)


def ask_deepseek(features: dict[str, float | int], *, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    if API_URL != 'https://api.deepseek.com/chat/completions':
        raise ValueError('model endpoint refused')
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing")
    system = (
        "You are a conservative signal adviser for a 100 USDT spot paper-trading experiment. "
        "No leverage, derivatives, real orders, or exchange credentials exist. "
        "Return exactly one JSON object with action buy|hold|sell, confidence 0..1, and a short reason. "
        "Write the reason in Simplified Chinese using at most 30 Chinese characters. "
        "Prefer hold when evidence is mixed. Do not use markdown or add text outside the JSON object."
    )
    user = "Assess BTC/USDT 15-minute features: " + json.dumps(features, separators=(",", ":"))
    request_body = {
            "model": MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.1,
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
    payload = json.dumps(request_body, allow_nan=False).encode('utf-8')
    if evidence is not None:
        evidence.update(request=request_body, prompt_version='btc-conservative-v1', requested_model=MODEL)
    request = urllib.request.Request(
        API_URL,
        data=payload,
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        body = _response_json(request, timeout=60)
    except urllib.error.HTTPError as exc:
        raise RuntimeError('model_http_error') from None
    reply = body['choices'][0]['message']['content']
    decision = parse_model_reply(reply)
    if evidence is not None:
        evidence.update(returned_model=str(body.get('model', MODEL))[:100], raw_reply=reply,
                        parsed_decision=decision)
    return decision


def generate_signal() -> dict[str, Any]:
    features: dict[str, float | int] = {}
    market_input: dict[str, Any] = {}
    model_evidence: dict[str, Any] = {}
    guard_evidence: dict[str, Any] = {}
    error_code = None
    try:
        market_input, features = fetch_input()
        features["quant_score"] = compute_quant_score(features)
        decision = apply_quant_guard(ask_deepseek(features, evidence=model_evidence), features,evidence=guard_evidence)
        if not 0 <= time.time() - _number(features['data_closed_at']) <= 1020:
            raise ValueError('source aged during model request')
        status = "ok"
    except Exception:
        decision = {"action": "hold", "confidence": 0.0, "reason": "数据或模型暂不可用，暂停生成交易建议。"}
        error_code = 'data_or_model_unavailable'
        status = "degraded"
    generated_at = time.time()
    expires_at = generated_at + min(1200, max(1, SIGNAL_TTL_SECONDS))
    if status == 'ok': expires_at = min(expires_at, features['data_closed_at'] + 1500)
    signal = {
        **decision,
        "pair": PAIR,
        "timeframe": TIMEFRAME,
        "model": MODEL,
        "status": status,
        "generated_at": generated_at,
        "generated_at_utc": datetime.fromtimestamp(generated_at, timezone.utc).isoformat(),
        "expires_at": expires_at,
        "data_closed_at": features.get('data_closed_at'),
        "data_received_at": features.get('data_received_at'),
        "schema": 2,
        "features": features,
    }
    record = dict(schema=2, market_input=market_input, model=model_evidence,guard=guard_evidence,
                  signal=dict(signal), error_code=error_code,
                  runtime_python=sys.version.split()[0],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    record['sha256'] = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False,
                                                allow_nan=False, separators=(',', ':')).encode()).hexdigest()
    try:
        append_decision(record)
        signal['decision_sha256']=record['sha256']
    except Exception:
        signal.update(action='hold', confidence=0.0, status='degraded',
                      reason='决策记录暂不可写，暂停生成交易建议。')
    atomic_write_json(SIGNAL_PATH, signal)
    print(json.dumps({key: signal[key] for key in ("action", "status", "generated_at_utc")}, ensure_ascii=False), flush=True)
    return signal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        started = time.monotonic()
        generate_signal()
        if args.once:
            return
        time.sleep(max(1.0, INTERVAL_SECONDS - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
