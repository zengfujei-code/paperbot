import sqlite3
import tempfile
import unittest
from pathlib import Path


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / "trades.sqlite"
        self.tuning = self.root / "tuning.json"
        connection = sqlite3.connect(self.db)
        connection.execute(
            "CREATE TABLE trades (id INTEGER PRIMARY KEY, is_open INTEGER, close_profit_abs REAL, "
            "realized_profit REAL, close_date TEXT)"
        )
        connection.commit()
        connection.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_waits_for_enough_closed_trades(self):
        from optimizer import optimize

        result = optimize(self.db, self.tuning, now=1000)

        self.assertEqual(result["status"], "collecting_data")
        self.assertEqual(result["sample_size"], 0)
        self.assertEqual(result["buy_confidence"], 0.68)

    def test_losing_sample_only_proposes_change_and_preserves_active_thresholds(self):
        from optimizer import optimize

        connection = sqlite3.connect(self.db)
        profits = [-1.0, -0.8, 0.3, -0.7, 0.2, -0.5]
        for index, profit in enumerate(profits, 1):
            connection.execute(
                "INSERT INTO trades VALUES (?,?,?,?,?)",
                (index, 0, profit, profit, f"2026-01-{index:02d}"),
            )
        connection.commit()
        connection.close()

        first = optimize(self.db, self.tuning, now=1000)
        second = optimize(self.db, self.tuning, now=2000)

        self.assertEqual(first["status"], "awaiting_validation")
        self.assertEqual(first["buy_confidence"], 0.68)
        self.assertEqual(first["min_quant_buy"], 0.10)
        self.assertEqual(first['proposal']['buy_confidence'],0.7)
        self.assertEqual(second["buy_confidence"], 0.68)
        self.assertEqual(second["revision"], first["revision"])

    def test_missing_database_is_unavailable_rather_than_zero_performance(self):
        from optimizer import optimize
        self.db.unlink()
        result=optimize(self.db,self.tuning,now=1000)
        self.assertEqual(result['status'],'unavailable')
        self.assertIsNone(result['net_profit'])

    def test_bad_previous_values_cannot_change_active_thresholds(self):
        import json
        from optimizer import optimize
        for value in (True,float('nan'),'0.05',{},10**400):
            self.tuning.write_text(json.dumps({'buy_confidence':value,'revision':value}),encoding='utf-8')
            result=optimize(self.db,self.tuning,now=1000)
            self.assertEqual((result['buy_confidence'],result['revision']),(0.68,0))

    def test_unknown_profit_is_not_treated_as_a_breakeven_trade(self):
        from optimizer import optimize
        connection=sqlite3.connect(self.db)
        connection.execute('INSERT INTO trades VALUES (1,0,NULL,NULL,NULL)')
        connection.commit(); connection.close()
        result=optimize(self.db,self.tuning,now=1000)
        self.assertEqual(result['status'],'unavailable')
        self.assertIsNone(result['net_profit'])

    def test_journal_records_active_values_and_evidence_before_publication(self):
        import json
        from optimizer import optimize
        result=optimize(self.db,self.tuning,now=1000)
        rows=(self.root/'tuning.json.history.jsonl').read_text(encoding='utf-8').splitlines()
        record=json.loads(rows[-1])
        self.assertEqual(record['active_before'],record['active_after'])
        self.assertEqual(record['active_after']['buy_confidence'],result['buy_confidence'])
        self.assertEqual(record['schema'],2)

    def test_incomplete_history_blocks_publication_without_overwriting_evidence(self):
        from optimizer import optimize
        journal=self.root/'tuning.json.history.jsonl'
        journal.write_text('{"partial":',encoding='utf-8')
        with self.assertRaises(ValueError): optimize(self.db,self.tuning,now=1000)
        self.assertFalse(self.tuning.exists())
        self.assertEqual(journal.read_text(encoding='utf-8'),'{"partial":')


if __name__ == "__main__":
    unittest.main()
