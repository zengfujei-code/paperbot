"""Freqtrade spot dry-run strategy driven by a validated local signal file."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

from pandas import DataFrame

from freqtrade.strategy import IStrategy


class DeepSeekPaperStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 1
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {"0": 0.04, "240": 0.02, "720": 0.0}
    stoploss = -0.05

    buy_threshold = 0.68
    sell_threshold = 0.60

    @property
    def signal_path(self) -> Path:
        return Path(os.getenv("AI_SIGNAL_PATH", "/shared/signal.json"))

    @property
    def tuning_path(self) -> Path:
        return Path(os.getenv("AI_TUNING_PATH", "/shared/tuning.json"))

    @staticmethod
    def _parse_json(raw):
        def pairs(items):
            result={}
            for key,value in items:
                if key in result: raise ValueError('duplicate JSON field')
                result[key]=value
            return result
        def constant(value): raise ValueError('invalid JSON number')
        return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)

    def _threshold(self, name: str, default: float, lower: float, upper: float) -> float:
        try:
            value = self._parse_json(self.tuning_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                return default
            number = value.get(name, default)
            if type(number) not in (int, float) or not math.isfinite(number):
                return default
            return max(lower, min(upper, float(number)))
        except (OSError, ValueError, TypeError, OverflowError, json.JSONDecodeError):
            return default

    def _read_signal(self, expected_pair: str | None = None) -> dict[str, Any] | None:
        try:
            value = self._parse_json(self.signal_path.read_text(encoding="utf-8"))
            now = time.time()
            if not isinstance(value, dict):
                return None
            if type(value.get('schema')) is not int or value.get('schema')!=2:
                return None
            audit=value.get('decision_sha256')
            if not isinstance(audit,str) or len(audit)!=64 or any(c not in '0123456789abcdef' for c in audit):
                return None
            if not expected_pair or value.get("pair") != expected_pair:
                return None
            if value.get("timeframe") != self.timeframe or value.get("status") != "ok":
                return None
            if value.get("action") not in {"buy", "hold", "sell"}:
                return None
            for field in ("confidence", "generated_at", "expires_at",'data_closed_at','data_received_at'):
                number = value[field]
                if type(number) not in (int, float) or not math.isfinite(number):
                    return None
            confidence = float(value["confidence"])
            generated_at = float(value["generated_at"])
            expires_at = float(value["expires_at"])
            closed_at=float(value['data_closed_at'])
            received_at=float(value['data_received_at'])
            if not 0<closed_at<=received_at<=generated_at or not 0<=now-closed_at<=1500 or expires_at>closed_at+1500:
                return None
            if not 0.0 <= confidence <= 1.0:
                return None
            if not 0 < generated_at < expires_at or expires_at - generated_at > 1500:
                return None
            if generated_at > now + 60 or now >= expires_at or now - generated_at > 1500:
                return None
            value["confidence"] = confidence
            return value
        except (OSError, ValueError, TypeError, KeyError, OverflowError, json.JSONDecodeError):
            return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        signal = self._read_signal(metadata.get("pair"))
        threshold = self._threshold("buy_confidence", self.buy_threshold, 0.64, 0.82)
        if signal and signal["action"] == "buy" and signal["confidence"] >= threshold and not dataframe.empty:
            dataframe.loc[dataframe.index[-1], "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        signal = self._read_signal(metadata.get("pair"))
        threshold = self._threshold("sell_confidence", self.sell_threshold, 0.50, 0.70)
        if signal and signal["action"] == "sell" and signal["confidence"] >= threshold and not dataframe.empty:
            dataframe.loc[dataframe.index[-1], "exit_long"] = 1
        return dataframe
