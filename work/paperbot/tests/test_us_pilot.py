import json,tempfile,time,unittest
from pathlib import Path


class PilotDashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'pilot.json';self.now=time.time()
        self.value=dict(schema=1,mode='paper',phase='experimental_pilot',capital_usd=10000,
            execution_enabled=True,formal_strategy_accepted=False,live_authorized=False,
            generated_at=self.now-10,status='observing_signal',action='hold',symbol='AAPL',reasons=[],
            accounting_ok=True,cash_usd=9600,equity_usd=10004,profit_usd=4,return_pct=.04,
            positions_count=1,open_orders_count=1,holdings=[dict(symbol='AAPL',quantity=2,
                average_cost_usd=200,reference_price_usd=202,unrealized_usd=4)],recent_trades=[],
            started_at=self.now-100,entries_paused=False,orders_halted=False,account_id='PRIVATE_ACCOUNT')
        self.heartbeat=self.path.with_name('heartbeat.json')
        self.heartbeat.write_text(json.dumps(dict(mode='paper',generated_at=self.now-1)))

    def read(self):
        from dashboard import read_us_pilot
        self.path.write_text(json.dumps(self.value));return read_us_pilot(self.path,now=self.now)

    def test_actual_pilot_pnl_holdings_and_status_are_public_without_account_id(self):
        r=self.read();self.assertTrue(r['accounting_ok']);self.assertEqual(r['profit_usd'],4)
        self.assertTrue(r['execution_enabled']);self.assertEqual(r['holdings'][0]['symbol'],'AAPL')
        self.assertFalse(r['formal_strategy_accepted']);self.assertNotIn('PRIVATE',json.dumps(r))

    def test_stale_dead_worker_or_wrong_profit_never_show_verified_pnl(self):
        for change in (dict(generated_at=self.now-1001),dict(profit_usd=5),dict(live_authorized=True),dict(mode='live')):
            original=dict(self.value);self.value.update(change)
            with self.subTest(change=change):self.assertFalse(self.read()['accounting_ok'])
            self.value=original
        self.heartbeat.write_text(json.dumps(dict(mode='paper',generated_at=self.now-76)))
        self.assertFalse(self.read()['execution_enabled'])

    def test_invalid_pilot_file_does_not_fall_back_to_pristine_observer(self):
        from dashboard import read_us_snapshot
        self.path.write_text('{}')
        r=read_us_snapshot(self.path.with_name('status.json'),pilot_path=self.path,now=self.now)
        self.assertFalse(r['accounting_ok']);self.assertIsNone(r['profit_usd'])


if __name__=='__main__':unittest.main()
