import copy
from contextlib import nullcontext
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import advisor


class AdvisorSafetyTests(unittest.TestCase):
    def setUp(self):
        self.now=900*1000+120
        self.candles=[[900000*(920+i),100,102,99,101,20] for i in range(80)]
        self.features=advisor.compute_features(self.candles)
        self.features.update(data_closed_at=900*1000,data_received_at=self.now)

    def test_candle_faults_are_rejected_instead_of_scored(self):
        variants=[]
        for field,value in ((1,True),(2,float('nan')),(3,105),(4,0),(5,-1),(0,1)):
            candles=copy.deepcopy(self.candles); candles[-1][field]=value; variants.append(candles)
        variants.extend((self.candles[:-1]+[self.candles[-2]],self.candles[:40]+self.candles[41:]))
        for candles in variants:
            with self.subTest(last=candles[-1]),self.assertRaises(ValueError): advisor.compute_features(candles)

    def test_current_forming_candle_is_excluded_and_old_history_rejected(self):
        closed=advisor.closed_candles(self.candles+[[900000*1000,100,102,99,101,20]],now=self.now)
        self.assertEqual(closed,self.candles)
        with self.assertRaises(ValueError): advisor.closed_candles(self.candles,now=self.now+1800)

    def test_strict_model_json_rejects_duplicates_nonfinite_and_trailing_text(self):
        for text in ('{"action":"buy","action":"hold","confidence":0.8,"reason":"ok"}',
                     '{"action":"buy","confidence":NaN,"reason":"ok"}',
                     '{"action":"buy","confidence":0.8,"reason":"ok"} extra'):
            with self.subTest(text=text),self.assertRaises(ValueError): advisor.parse_model_reply(text)

    def test_tuning_wrong_types_never_loosen_quant_guard(self):
        for value in (True,float('nan'),'0.05',{},10**400):
            with patch('advisor.load_tuning',return_value={'min_quant_buy':value}):
                result=advisor.apply_quant_guard({'action':'buy','confidence':0.9,'reason':'ok'},
                                                {**self.features,'quant_score':0.09})
                self.assertEqual(result['action'],'hold')

    def generate(self,root,fetch=None,model=None,append=None,score=0.5):
        paths=(patch.object(advisor,'SIGNAL_PATH',root/'signal.json'),
               patch.object(advisor,'DECISION_LOG',root/'decisions.jsonl'))
        with paths[0],paths[1],patch('advisor.time.time',return_value=self.now), \
             patch('advisor.fetch_input',side_effect=fetch or (lambda:({'candles':self.candles},dict(self.features)))), \
             patch('advisor.ask_deepseek',side_effect=model or (lambda f,**kw:{'action':'buy','confidence':0.9,'reason':'ok'})), \
             patch('advisor.load_tuning',return_value={}), \
             (patch('advisor.compute_quant_score',return_value=score) if score is not None else nullcontext()):
            if append:
                with patch('advisor.append_decision',side_effect=append): return advisor.generate_signal()
            return advisor.generate_signal()

    def test_provider_exception_details_never_enter_signal_or_printed_summary(self):
        marker='PRIVATE_EXCEPTION_MARKER'
        def fail(): raise RuntimeError(marker)
        with tempfile.TemporaryDirectory() as tmp,patch('builtins.print') as printer:
            root=Path(tmp); signal=self.generate(root,fetch=fail)
            self.assertEqual((signal['action'],signal['status']),('hold','degraded'))
            self.assertNotIn(marker,json.dumps(signal)+str(printer.call_args_list)+(root/'decisions.jsonl').read_text(encoding='utf-8'))

    def test_audit_failure_vetoes_buy_before_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            result=self.generate(root,append=OSError('disk full'))
            self.assertEqual((result['action'],result['status']),('hold','degraded'))
            self.assertEqual(json.loads((root/'signal.json').read_text(encoding='utf-8'))['action'],'hold')

    def test_truncated_existing_journal_cannot_publish_an_unauditable_buy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            target=root/'decisions.jsonl'
            target.write_text('{"schema":2,"partial":',encoding='utf-8')
            result=self.generate(root)
            self.assertEqual((result['action'],result['status']),('hold','degraded'))
            self.assertEqual(target.read_text(encoding='utf-8'),'{"schema":2,"partial":')

    def test_source_age_is_checked_after_slow_model_and_audit_is_replayable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); signal=self.generate(root)
            record=json.loads((root/'decisions.jsonl').read_text(encoding='utf-8').splitlines()[0])
            self.assertEqual(record['schema'],2)
            self.assertEqual(record['market_input']['candles'],self.candles)
            self.assertEqual(record['signal']['action'],'buy')
            self.assertIn('sha256',record)
            self.assertLessEqual(signal['expires_at'],signal['data_closed_at']+1500)
        def slow_model(features,**kwargs):
            features['data_closed_at']=self.now-1800
            return {'action':'buy','confidence':0.9,'reason':'ok'}
        with tempfile.TemporaryDirectory() as tmp:
            signal=self.generate(Path(tmp),model=slow_model)
            self.assertEqual((signal['action'],signal['status']),('hold','degraded'))

    def test_record_replays_with_saved_model_reply_and_guard_not_current_tuning(self):
        def fake_model(features,**kwargs):
            result={'action':'buy','confidence':0.9,'reason':'ok'}
            kwargs['evidence'].update(raw_reply=json.dumps(result),parsed_decision=dict(result))
            return result
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); signal=self.generate(root,model=fake_model,score=None)
            record=json.loads((root/'decisions.jsonl').read_text(encoding='utf-8'))
            with patch('advisor.load_tuning',side_effect=AssertionError('replay must not use live settings')):
                self.assertEqual(advisor.replay_decision(record)['action'],signal['action'])


if __name__=='__main__': unittest.main()
