import json
import math
import os
import tempfile
import unittest
from pathlib import Path


class AdvisorTests(unittest.TestCase):
    def test_parse_model_reply_accepts_json_code_fence(self):
        from advisor import parse_model_reply

        result = parse_model_reply(
            '```json\n{"action":"buy","confidence":0.72,"reason":"trend"}\n```'
        )

        self.assertEqual(result["action"], "buy")
        self.assertEqual(result["confidence"], 0.72)
        self.assertEqual(result["reason"], "trend")

    def test_parse_model_reply_rejects_unknown_action(self):
        from advisor import parse_model_reply

        with self.assertRaisesRegex(ValueError, "action"):
            parse_model_reply('{"action":"moon","confidence":0.9,"reason":"bad"}')

    def test_parse_model_reply_rejects_out_of_range_confidence(self):
        from advisor import parse_model_reply

        with self.assertRaisesRegex(ValueError, "confidence"):
            parse_model_reply('{"action":"buy","confidence":1.5,"reason":"bad"}')

    def test_parse_model_reply_rejects_json_followed_by_explanation(self):
        from advisor import parse_model_reply

        with self.assertRaises(ValueError):
            parse_model_reply('{"action":"hold","confidence":0.4,"reason":"观望"}\nextra')

    def test_compute_features_has_finite_bounded_values(self):
        from advisor import compute_features

        candles = []
        for i in range(80):
            close = 100 + i * 0.25 + math.sin(i / 3)
            candles.append([i * 900_000, close - 0.2, close + 0.5, close - 0.5, close, 10 + i / 10])

        result = compute_features(candles)

        self.assertEqual(result["candles"], 80)
        self.assertGreaterEqual(result["rsi14"], 0)
        self.assertLessEqual(result["rsi14"], 100)
        self.assertTrue(all(math.isfinite(v) for k, v in result.items() if k != "candles"))

    def test_normalize_gate_candles_maps_documented_fields(self):
        from advisor import normalize_gate_candles

        raw = [["1700000000", "1234.5", "101.5", "103", "99", "100", "12.25", "true"]]

        self.assertEqual(normalize_gate_candles(raw), [[1700000000000, 100.0, 103.0, 99.0, 101.5, 12.25]])

    def test_quant_score_is_bounded_and_rewards_positive_trend(self):
        from advisor import compute_quant_score

        positive = {
            "last_close": 110,
            "sma20": 108,
            "sma50": 100,
            "return_15m_pct": 0.2,
            "return_1h_pct": 1.0,
            "return_3h_pct": 2.5,
            "rsi14": 62,
            "volatility20_pct": 0.5,
            "volume_ratio20": 1.3,
        }
        negative = {**positive, "sma20": 96, "sma50": 104, "return_1h_pct": -1.0, "return_3h_pct": -2.5, "rsi14": 38}

        self.assertGreater(compute_quant_score(positive), 0)
        self.assertLess(compute_quant_score(negative), 0)
        self.assertLessEqual(abs(compute_quant_score(positive)), 1)

    def test_atomic_write_replaces_complete_json(self):
        from advisor import atomic_write_json

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "signal.json"
            atomic_write_json(target, {"action": "hold", "confidence": 0.0})
            atomic_write_json(target, {"action": "buy", "confidence": 0.7})

            self.assertEqual(json.loads(target.read_text()), {"action": "buy", "confidence": 0.7})
            self.assertFalse((Path(tmp) / "signal.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
