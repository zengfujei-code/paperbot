import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / "trades.sqlite"
        self.signal = self.root / "signal.json"
        self.experiment = self.root / "experiment.json"
        self.log = self.root / "freqtrade.log"
        self.tuning = self.root / "tuning.json"
        self.now = time.time()
        self.experiment.write_text(
            json.dumps({"started_at": "2026-01-01T00:00:00Z", "ends_at": "2026-01-31T00:00:00Z", "virtual_wallet_usdt": 100})
        )
        self.signal.write_text(
            json.dumps(
                {
                    "action": "hold",
                    "confidence": 0.62,
                    "reason": "等待更清晰的趋势",
                    "status": "ok",
                    "model": "deepseek-flash",
                    "pair": "BTC/USDT",
                    "generated_at": self.now,
                    "generated_at_utc": "2026-01-01T00:00:00+00:00",
                    "expires_at": self.now + 1200,
                    "features": {"last_close": 110},
                    "schema": 2,
                    "data_closed_at": self.now-120,
                    "data_received_at": self.now-1,
                }
            )
        )
        self.log.write_text("Bot heartbeat state='RUNNING'\n")
        os.utime(self.log, (self.now, self.now))
        self.tuning.write_text(json.dumps({"status": "collecting_data", "sample_size": 2, "minimum_samples": 6, "revision": 0}))
        connection = sqlite3.connect(self.db)
        connection.execute(
            "CREATE TABLE trades (id INTEGER PRIMARY KEY, pair TEXT, is_open INTEGER, open_rate REAL, close_profit_abs REAL, "
            "realized_profit REAL, stake_amount REAL, amount REAL, open_date TEXT, close_date TEXT, is_short INTEGER)"
        )
        connection.commit()
        connection.close()

    def tearDown(self):
        self.tmp.cleanup()

    def snapshot(self):
        from dashboard import read_snapshot

        return read_snapshot(self.db, self.signal, self.experiment, self.log, self.tuning, now=self.now)

    def test_source_age_and_public_projection_are_independent_of_new_signal_time(self):
        signal=json.loads(self.signal.read_text())
        signal.update(schema=2,data_closed_at=self.now-2400,data_received_at=self.now,
                      status='degraded',reason='PRIVATE_EXCEPTION_MARKER',model='PRIVATE_MODEL_MARKER')
        self.signal.write_text(json.dumps(signal))
        self.tuning.write_text(json.dumps({'reason':'PRIVATE_TUNING_MARKER','status':'PRIVATE_STATUS_MARKER'}))
        result=self.snapshot()
        self.assertFalse(result['ai_ok'])
        self.assertNotIn('PRIVATE_',json.dumps(result))
        self.assertIsNone(result['last_price'])

    def test_legacy_schema_does_not_bypass_source_age_validation(self):
        signal=json.loads(self.signal.read_text())
        signal.update(schema=1,data_closed_at=self.now-86400)
        self.signal.write_text(json.dumps(signal))
        result=self.snapshot()
        self.assertFalse(result['ai_ok'])
        self.assertIsNone(result['last_price'])

    def test_empty_database_shows_initial_balance(self):
        result = self.snapshot()

        self.assertEqual(result["equity"], 100)
        self.assertEqual(result["profit_abs"], 0)
        self.assertEqual(result["return_pct"], 0)
        self.assertEqual(result["open_count"], 0)
        self.assertTrue(result["bot_running"])
        self.assertEqual(result["optimizer"]["sample_size"], 2)

    def test_missing_database_is_not_reported_as_zero_profit(self):
        self.db.unlink()
        result = self.snapshot()
        self.assertFalse(result["accounting_ok"])
        self.assertIsNone(result["equity"])
        self.assertIsNone(result["profit_abs"])
        self.assertIsNone(result["closed_count"])
        self.assertNotEqual(result["system_status"], "正常运行")

    def test_invalid_database_is_not_reported_as_zero_profit(self):
        self.db.write_bytes(b'not a sqlite database')
        result = self.snapshot()
        self.assertFalse(result["accounting_ok"])
        self.assertIsNone(result["equity"])
        self.assertIsNone(result["return_pct"])

    def test_stale_quote_does_not_produce_an_account_equity(self):
        connection = sqlite3.connect(self.db)
        connection.execute("INSERT INTO trades VALUES (1,'BTC/USDT',1,100,NULL,0,50,0.5,'2026-01-01',NULL,0)")
        connection.commit()
        connection.close()
        signal = json.loads(self.signal.read_text())
        signal["generated_at"] = self.now - 1800
        signal["expires_at"] = self.now - 600
        self.signal.write_text(json.dumps(signal))
        result = self.snapshot()
        self.assertTrue(result["accounting_ok"])
        self.assertFalse(result["valuation_ok"])
        self.assertIsNone(result["equity"])
        self.assertEqual(result["open_count"], 1)

    def test_malformed_signal_numbers_do_not_crash_or_emit_nonfinite_json(self):
        signal = json.loads(self.signal.read_text())
        signal.update(generated_at="bad", expires_at=float('nan'), confidence=float('inf'))
        signal["features"]["last_close"] = float('nan')
        self.signal.write_text(json.dumps(signal))
        self.tuning.write_text(json.dumps({"sample_size":[],"buy_confidence":float('nan')}))
        result = self.snapshot()
        self.assertFalse(result["ai_ok"])
        json.dumps(result, allow_nan=False)

    def test_unavailable_numbers_are_not_formatted_as_zero_in_page(self):
        from dashboard import PAGE
        if not shutil.which('node'):
            self.skipTest('Node is required only for the browser formatter contract')
        script = re.search(r'<script>(.*?)</script>', PAGE, re.S).group(1)
        functions = script.split('async function refresh')[0]
        result = subprocess.run(['node','-e',functions+';console.log(JSON.stringify([money(null),pct(null),money(0)]));'],
                                check=True,capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(json.loads(result.stdout), ['—','—','0.00'])

    def test_huge_numbers_and_untyped_public_fields_are_sanitized(self):
        signal = json.loads(self.signal.read_text())
        signal.update(confidence=10**400, reason=float('nan'), status=float('inf'), generated_at_utc=float('nan'))
        self.signal.write_text(json.dumps(signal))
        self.tuning.write_text(json.dumps({'buy_confidence':10**400,'reason':float('inf')}))
        self.experiment.write_text(json.dumps({'virtual_wallet_usdt':10**400,'started_at':float('nan')}))
        result = self.snapshot()
        json.dumps(result, allow_nan=False)
        self.assertFalse(result['ai_ok'])

    def test_finite_operands_overflow_cannot_be_a_valid_valuation(self):
        connection = sqlite3.connect(self.db)
        connection.execute("INSERT INTO trades VALUES (1,'BTC/USDT',1,100,NULL,0,50,1e308,'2026-01-01',NULL,0)")
        connection.commit()
        connection.close()
        signal = json.loads(self.signal.read_text())
        signal['features']['last_close'] = 1e308
        self.signal.write_text(json.dumps(signal))
        result = self.snapshot()
        self.assertFalse(result['valuation_ok'])
        self.assertIsNone(result['equity'])
        json.dumps(result, allow_nan=False)

    def test_closed_trade_without_profit_is_not_a_breakeven_trade(self):
        connection = sqlite3.connect(self.db)
        connection.execute("INSERT INTO trades VALUES (1,'BTC/USDT',0,100,NULL,NULL,50,0.5,'2026-01-01','2026-01-02',0)")
        connection.commit()
        connection.close()
        result = self.snapshot()
        self.assertFalse(result['accounting_ok'])
        self.assertIsNone(result['realized_profit'])

    def test_endpoint_does_not_send_nonstandard_json(self):
        from dashboard import Handler
        handler = object.__new__(Handler)
        handler.path = '/api/status'
        sent = []
        handler._send = lambda *args: sent.append(args)
        with patch('dashboard.read_snapshot',return_value={'equity':float('inf')}):
            handler.do_GET()
        self.assertEqual(sent[0][0],503)
        self.assertNotIn('Infinity',sent[0][2].decode())

    def test_closed_and_open_profit_are_included(self):
        connection = sqlite3.connect(self.db)
        connection.execute(
            "INSERT INTO trades (pair,is_open,open_rate,close_profit_abs,realized_profit,stake_amount,amount,open_date,close_date,is_short) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("BTC/USDT", 0, 100, 2, 2, 50, 0.5, "2026-01-01", "2026-01-02", 0),
        )
        connection.execute(
            "INSERT INTO trades (pair,is_open,open_rate,close_profit_abs,realized_profit,stake_amount,amount,open_date,close_date,is_short) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("BTC/USDT", 1, 100, None, 0, 50, 0.5, "2026-01-03", None, 0),
        )
        connection.commit()
        connection.close()

        result = self.snapshot()

        self.assertEqual(result["realized_profit"], 2)
        self.assertEqual(result["unrealized_profit"], 5)
        self.assertEqual(result["equity"], 107)
        self.assertEqual(result["return_pct"], 7)
        self.assertEqual(result["closed_count"], 1)
        self.assertEqual(result["open_count"], 1)


if __name__ == "__main__":
    unittest.main()
