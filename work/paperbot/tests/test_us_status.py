import json
from pathlib import Path
import tempfile
import time
import unittest
import re
import shutil
import subprocess


class USStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'status.json';self.now=time.time()
        self.value=dict(schema=1,mode='paper',currency='USD',capital_usd=10000,phase='observing',execution_enabled=False,
            generated_at=self.now-1,account_received_at=self.now-2,observing_started_at=self.now-60,paper_started_at=None,
            status='observing',accounting_ok=True,equity_usd=10000,cash_usd=10000,profit_usd=0,return_pct=0,
            positions_count=0,open_orders_count=0,audit_sha256='a'*64,
            details_ok=True,holdings=[],recent_trades=[],holdings_limited=False,history_window_limited=False,
            scan=dict(status='ok',as_of=self.now-10,pool_size=13200,checked_count=100,candidate_count=8),
            id='PRIVATE_ACCOUNT_FIXTURE',reason='PRIVATE_MODEL_FIXTURE')

    def snapshot(self):
        from dashboard import read_us_snapshot
        self.path.write_text(json.dumps(self.value),encoding='utf-8')
        return read_us_snapshot(self.path,now=self.now)

    def test_current_verified_baseline_has_real_values_and_no_private_metadata(self):
        value=self.snapshot()
        self.assertTrue(value['accounting_ok']);self.assertEqual(value['equity_usd'],10000)
        self.assertEqual(value['scan']['candidate_count'],8)
        self.assertFalse(value['execution_enabled']);self.assertIsNone(value['paper_started_at'])
        self.assertNotIn('PRIVATE',json.dumps(value))
        self.assertTrue(value['details_ok']);self.assertEqual(value['holdings'],[])

    def test_broker_holdings_and_trade_rows_are_visible_even_when_account_pnl_is_unreconciled(self):
        self.value.update(accounting_ok=False,status='account_unreconciled',positions_count=1,
            holdings=[dict(symbol='AAPL',quantity=2,average_cost_usd=200,reference_price_usd=202,unrealized_usd=4,
                           account_id='PRIVATE_ACCOUNT')],
            recent_trades=[dict(symbol='AAPL',side='buy',quantity=2,average_price_usd=200,reported_at=self.now-20,
                                time_kind='filled_at',status='filled',order_id='PRIVATE_ORDER')])
        value=self.snapshot();self.assertFalse(value['accounting_ok']);self.assertIsNone(value['profit_usd'])
        self.assertTrue(value['details_ok']);self.assertEqual(value['holdings'][0]['symbol'],'AAPL')
        self.assertEqual(value['recent_trades'][0]['quantity'],2);self.assertNotIn('PRIVATE',json.dumps(value))

    def test_stale_or_invalid_holdings_do_not_become_an_empty_verified_list(self):
        self.value.update(positions_count=1,accounting_ok=False,status='account_unreconciled',
            holdings=[dict(symbol='<script>',quantity=2,average_cost_usd=200,reference_price_usd=202,unrealized_usd=4)])
        value=self.snapshot();self.assertFalse(value['details_ok']);self.assertEqual(value['holdings'],[])
        self.value['holdings'][0]['symbol']='AAPL';self.value['account_received_at']=self.now-76
        self.assertFalse(self.snapshot()['details_ok'])

    def test_financial_baseline_requires_verified_empty_complete_details(self):
        for change in (dict(details_ok=False),dict(history_window_limited=True),dict(recent_trades=[
                dict(symbol='AAPL',side='buy',quantity=1,average_price_usd=200,reported_at=self.now-20,
                     time_kind='filled_at',status='filled')])):
            original=dict(self.value);self.value.update(change)
            with self.subTest(change=change):
                value=self.snapshot();self.assertFalse(value['accounting_ok']);self.assertIsNone(value['profit_usd'])
            self.value=original

    def test_old_account_or_worker_clears_financial_values(self):
        for field in ('generated_at','account_received_at'):
            self.value[field]=self.now-76
            with self.subTest(field=field):
                value=self.snapshot();self.assertFalse(value['accounting_ok'])
                self.assertIsNone(value['equity_usd']);self.assertIsNone(value['profit_usd'])

    def test_unknown_paper_phase_or_unreconciled_account_is_not_healthy(self):
        for changes in (dict(phase='live'),dict(execution_enabled=True),dict(accounting_ok=False),dict(equity_usd=10001),
                        dict(cash_usd=True),dict(profit_usd=float('nan')),dict(generated_at=self.now+1),
                        dict(account_received_at=self.now+1),dict(audit_sha256=None)):
            original=dict(self.value);self.value.update(changes)
            with self.subTest(changes=changes):
                value=self.snapshot();self.assertFalse(value['accounting_ok']);self.assertIsNone(value['equity_usd'])
                self.assertNotIn('PRIVATE',json.dumps(value))
            self.value=original

    def test_missing_corrupt_large_or_duplicate_json_never_means_zero_profit(self):
        from dashboard import read_us_snapshot
        for raw in (None,b'broken',b'x'*65537,b'{"schema":1,"schema":1}'):
            if raw is not None: self.path.write_bytes(raw)
            with self.subTest(raw=type(raw)):
                value=read_us_snapshot(self.path,now=self.now)
                self.assertIsNone(value['profit_usd']);self.assertFalse(value['accounting_ok'])

    def test_scan_health_does_not_override_account_health_or_claim_quote_validity(self):
        self.value['scan']['as_of']=self.now-181
        value=self.snapshot();self.assertTrue(value['accounting_ok']);self.assertEqual(value['scan']['status'],'unavailable')
        self.value['accounting_ok']=False
        value=self.snapshot();self.assertFalse(value['accounting_ok']);self.assertIsNone(value['profit_usd'])

    def test_continuous_analysis_is_current_allowlisted_and_distinct_from_execution(self):
        from dashboard import read_us_analysis
        value=dict(schema=1,mode='paper',phase='shadow',execution_enabled=False,model_status='not_configured',
            generated_at=self.now-1,analysis_started_at=self.now-60,status='analyzing',audit_sha256='b'*64,
            decisions=[dict(symbol='AAPL',generated_at=self.now-2,bar_ends_at=self.now-30,features=dict(
                last_price_usd=200,sma_fast_usd=201,sma_slow_usd=199,rsi_sma=65,momentum_fraction=.01,
                volatility_15m_fraction=.002,volume_ratio=1.2,interval_seconds=900,volatility_annualized=False,formula_version='simple-window-v1'),
                entry_candidate=True,exit_candidate=False,final_action='hold',reason='rules_evaluated',
                vetoes=['strategy_not_accepted','execution_not_enabled','model_not_configured'],identity='c'*64,
                prompt='PRIVATE_PROMPT')])
        self.path.write_text(json.dumps(value),encoding='utf-8')
        result=read_us_analysis(self.path,now=self.now)
        self.assertEqual(result['status'],'analyzing');self.assertTrue(result['decisions'][0]['entry_candidate'])
        self.assertEqual(result['decisions'][0]['final_action'],'hold');self.assertNotIn('PRIVATE',json.dumps(result))
        for change in (dict(generated_at=self.now-76),dict(execution_enabled=True)):
            self.path.write_text(json.dumps({**value,**change}),encoding='utf-8')
            self.assertEqual(read_us_analysis(self.path,now=self.now)['status'],'unavailable')
        value['decisions'][0]['bar_ends_at']=self.now-86400
        self.path.write_text(json.dumps(value),encoding='utf-8')
        self.assertEqual(read_us_analysis(self.path,now=self.now)['status'],'unavailable')

    def model_value(self):
        return dict(schema=1,mode='paper',phase='model_assistance',execution_enabled=False,
            generated_at=self.now-1,status='reviewed',attempts_today=1,daily_limit=60,hourly_limit=20,
            min_interval_seconds=60,audit_sha256='a'*64,PRIVATE='do not publish',decisions=[dict(
                symbol='AAPL',bar_ends_at=self.now-7200,reviewed_at=self.now-2,context='historical_review',
                input_current_at_review=False,signal_usable=False,action='hold',reason='量价尚未确认，继续观望。',
                model='deepseek-flash',final_action='hold',vetoes=['strategy_not_accepted','execution_not_enabled'])])

    def test_model_projection_displays_validated_reason_and_historical_context_only(self):
        import dashboard
        self.assertTrue(hasattr(dashboard,'read_us_model'),'model projection reader required')
        self.path.write_text(json.dumps(self.model_value()),encoding='utf-8')
        value=dashboard.read_us_model(self.path,now=self.now)
        self.assertEqual(value['status'],'reviewed');self.assertEqual(value['decisions'][0]['symbol'],'AAPL')
        self.assertFalse(value['decisions'][0]['signal_usable']);self.assertNotIn('PRIVATE',json.dumps(value))

    def test_model_projection_rejects_secrets_staleness_controls_and_invalid_json(self):
        import dashboard
        self.assertTrue(hasattr(dashboard,'read_us_model'),'model projection reader required')
        for change in (dict(generated_at=self.now-76),dict(execution_enabled=True),dict(attempts_today=61)):
            self.path.write_text(json.dumps({**self.model_value(),**change}),encoding='utf-8')
            self.assertEqual(dashboard.read_us_model(self.path,now=self.now)['decisions'],[])
        for field,value in (('reason','sk-'+'a'*32),('model','sk-'+'a'*32),('signal_usable',True),('final_action','buy')):
            row=self.model_value();row['decisions'][0][field]=value
            self.path.write_text(json.dumps(row),encoding='utf-8')
            self.assertEqual(dashboard.read_us_model(self.path,now=self.now)['status'],'unavailable')
        self.path.write_text('{"schema":1,"schema":1}',encoding='utf-8')
        self.assertEqual(dashboard.read_us_model(self.path,now=self.now)['status'],'unavailable')

    @unittest.skipUnless(shutil.which('node'),'Node is only required for browser-script verification')
    def test_browser_shows_model_reason_without_turning_it_into_a_fill_and_clears_on_failure(self):
        from dashboard import US_PAGE
        script=re.search(r'<script>(.*?)</script>',US_PAGE,re.S).group(1)
        harness="""
const vm=require('node:vm'),assert=require('node:assert/strict'),nodes=new Map();
const node=id=>{if(!nodes.has(id))nodes.set(id,{textContent:'',className:''});return nodes.get(id)};
const c={document:{getElementById:node},fetch:async()=>{throw Error('offline')},AbortSignal:{timeout:()=>({})},setInterval:()=>{},console};
vm.createContext(c);vm.runInContext(SOURCE,c);
setTimeout(async()=>{vm.runInContext('paintModel('+JSON.stringify(VALUE)+')',c);
assert.match(node('modelList').textContent,/AAPL/);assert.match(node('modelList').textContent,/历史复核/);
assert.match(node('modelList').textContent,/量价尚未确认/);assert.doesNotMatch(node('tradeList').textContent,/AAPL/);
await vm.runInContext('refresh()',c);assert.doesNotMatch(node('modelList').textContent,/AAPL/);
console.log('MODEL_DISPLAY_AND_CLEAR_PASSED')},10);
""".replace('SOURCE',json.dumps(script)).replace('VALUE',json.dumps(self.model_value(),ensure_ascii=True))
        result=subprocess.run(['node','-e',harness],capture_output=True,text=True,encoding='utf-8',timeout=10)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('MODEL_DISPLAY_AND_CLEAR_PASSED',result.stdout)

    @unittest.skipUnless(shutil.which('node'),'Node is only required for browser-script verification')
    def test_browser_network_failure_clears_previously_displayed_money(self):
        from dashboard import US_PAGE
        script=re.search(r'<script>(.*?)</script>',US_PAGE,re.S).group(1)
        harness="""
const vm=require('node:vm'),assert=require('node:assert/strict');
const nodes=new Map();const node=id=>{if(!nodes.has(id))nodes.set(id,{textContent:'10000',className:''});return nodes.get(id)};
const context={document:{getElementById:node},fetch:async()=>{throw Error('fixture offline')},
  AbortSignal:{timeout:()=>({})},setInterval:()=>{},console};
vm.createContext(context);vm.runInContext(SOURCE,context);
setTimeout(()=>{for(const id of ['equity','profit','return','cash','positions','orders'])assert.equal(node(id).textContent,'—');
assert.match(node('state').textContent,/连接中断/);console.log('OFFLINE_CLEARS_VALUES')},10);
""".replace('SOURCE',json.dumps(script))
        result=subprocess.run(['node','-e',harness],capture_output=True,text=True,encoding='utf-8',timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('OFFLINE_CLEARS_VALUES',result.stdout)

    @unittest.skipUnless(shutil.which('node'),'Node is only required for browser-script verification')
    def test_browser_displays_actual_stock_rows_and_clears_them_after_failure(self):
        from dashboard import US_PAGE
        script=re.search(r'<script>(.*?)</script>',US_PAGE,re.S).group(1)
        harness="""
const vm=require('node:vm'),assert=require('node:assert/strict');const nodes=new Map();
const node=id=>{if(!nodes.has(id))nodes.set(id,{textContent:'',className:''});return nodes.get(id)};
const context={document:{getElementById:node},fetch:async()=>{throw Error('offline')},
 AbortSignal:{timeout:()=>({})},setInterval:()=>{},console};vm.createContext(context);vm.runInContext(SOURCE,context);
setTimeout(async()=>{vm.runInContext(`paint({details_ok:true,positions_count:1,open_orders_count:0,
 holdings:[{symbol:'AAPL',quantity:2,average_cost_usd:200,reference_price_usd:202,unrealized_usd:4}],
 recent_trades:[{symbol:'AAPL',side:'buy',quantity:2,average_price_usd:200,reported_at:1700000000,time_kind:'filled_at',status:'filled'}]})`,context);
 assert.match(node('holdingList').textContent,/AAPL/);assert.match(node('holdingList').textContent,/200.00/);
 assert.match(node('tradeList').textContent,/买入/);await vm.runInContext('refresh()',context);
 assert.doesNotMatch(node('holdingList').textContent,/AAPL/);assert.doesNotMatch(node('tradeList').textContent,/AAPL/);
 console.log('STOCK_ROWS_AND_FAILURE_PASSED')},10);
""".replace('SOURCE',json.dumps(script))
        result=subprocess.run(['node','-e',harness],capture_output=True,text=True,encoding='utf-8',timeout=10)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('STOCK_ROWS_AND_FAILURE_PASSED',result.stdout)


if __name__=='__main__': unittest.main()
