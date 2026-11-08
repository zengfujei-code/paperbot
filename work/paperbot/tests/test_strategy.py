import importlib.util
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import pandas as pd


def load_strategy_module():
    path = Path("/app/DeepSeekPaperStrategy.py")
    spec = importlib.util.spec_from_file_location("deepseek_strategy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.signal_path = Path(self.tmp.name) / "signal.json"
        os.environ["AI_SIGNAL_PATH"] = str(self.signal_path)
        self.module = load_strategy_module()
        self.strategy = self.module.DeepSeekPaperStrategy({"stake_currency": "USDT"})
        self.frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC"),
                "open": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [99, 100, 101],
                "close": [100.5, 101.5, 102.5],
                "volume": [10, 11, 12],
            }
        )

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("AI_SIGNAL_PATH", None)
        os.environ.pop("AI_TUNING_PATH", None)

    def write_signal(self, action, confidence, age=0):
        stamp=time.time()-age
        self.signal_path.write_text(
            json.dumps(
                {
                    "action": action,
                    "confidence": confidence,
                    "generated_at": stamp,
                    "expires_at": stamp + 1200,
                    "schema": 2,
                    "data_closed_at": stamp-120,
                    "data_received_at": stamp-1,
                    "decision_sha256": 'a'*64,
                    "pair": "BTC/USDT",
                    "timeframe": "15m",
                    "status": "ok",
                }
            )
        )

    def test_fresh_high_confidence_buy_enters_only_latest_candle(self):
        self.write_signal("buy", 0.72)

        result = self.strategy.populate_entry_trend(self.frame.copy(), {"pair": "BTC/USDT"})

        self.assertEqual(result["enter_long"].tolist(), [0, 0, 1])

    def test_stale_buy_signal_does_not_enter(self):
        self.write_signal("buy", 0.90, age=1800)

        result = self.strategy.populate_entry_trend(self.frame.copy(), {"pair": "BTC/USDT"})

        self.assertEqual(result["enter_long"].tolist(), [0, 0, 0])

    def test_fresh_sell_signal_exits_only_latest_candle(self):
        self.write_signal("sell", 0.65)

        result = self.strategy.populate_exit_trend(self.frame.copy(), {"pair": "BTC/USDT"})

        self.assertEqual(result["exit_long"].tolist(), [0, 0, 1])

    def test_malformed_signal_fails_closed(self):
        self.signal_path.write_text("not-json")

        result = self.strategy.populate_entry_trend(self.frame.copy(), {"pair": "BTC/USDT"})

        self.assertEqual(result["enter_long"].tolist(), [0, 0, 0])

    def test_adaptive_threshold_can_require_more_confidence(self):
        tuning_path = Path(self.tmp.name) / "tuning.json"
        tuning_path.write_text(json.dumps({"buy_confidence": 0.8, "sell_confidence": 0.6}))
        os.environ["AI_TUNING_PATH"] = str(tuning_path)
        self.write_signal("buy", 0.72)

        result = self.strategy.populate_entry_trend(self.frame.copy(), {"pair": "BTC/USDT"})

        self.assertEqual(result["enter_long"].tolist(), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
