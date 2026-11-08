import copy
import importlib
import importlib.util
import json
import unittest
from unittest.mock import patch


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.llm'), 'bounded model adapter must exist')
        self.m=importlib.import_module('paperbot_us.llm')
        self.key='sk-'+'x'*32
        self.decision=dict(symbol='AAPL',bar_ends_at=1792782000,entry_candidate=True,exit_candidate=False,
            features=dict(last_price_usd=101,sma_fast_usd=100.9,sma_slow_usd=100,rsi_sma=60,
                momentum_fraction=.01,volatility_15m_fraction=.002,volume_ratio=1.5,
                interval_seconds=900,volatility_annualized=False,formula_version='simple-window-v1'),
            PRIVATE_ACCOUNT='must never leave process')

    def response(self):
        return dict(model='deepseek-flash',system_fingerprint='fp_123',choices=[dict(index=0,finish_reason='stop',
            message=dict(role='assistant',content='{"action":"hold","reason":"趋势偏强，等待价量确认。"}'))],
            usage=dict(prompt_tokens=250,completion_tokens=40,total_tokens=290))

    def test_request_is_allowlisted_bounded_and_explicitly_historical(self):
        body=self.m.request_for(self.decision,historical=True)
        self.assertEqual(body['model'],'deepseek-flash');self.assertEqual(body['max_tokens'],512)
        self.assertEqual(body['thinking'],{'type':'disabled'});self.assertFalse(body['stream'])
        self.assertNotIn('PRIVATE',json.dumps(body));self.assertNotIn('tools',body)
        data=json.loads(body['messages'][1]['content']);self.assertEqual(data['context'],'historical_review')
        self.assertLess(len(json.dumps(body).encode()),8192)

    def test_receipt_has_only_validated_output_model_version_and_usage(self):
        seen=[]
        def transport(body,key): seen.append((body,key));return self.response()
        client=self.m.DeepSeekClient(self.key,transport=transport)
        result=client.assess(self.m.request_for(self.decision,historical=False))
        self.assertEqual(result['action'],'hold');self.assertEqual(result['total_tokens'],290)
        self.assertEqual(len(seen),1);self.assertNotIn(self.key,repr(client)+json.dumps(result))
        self.assertNotIn('choices',result)

    def test_optional_backend_fingerprint_is_opaque_and_has_no_required_vendor_prefix(self):
        for fingerprint in ('','1234567890abcdef','backend-v4.1_1017'):
            row=self.response();row['system_fingerprint']=fingerprint
            try: result=self.m.DeepSeekClient(self.key,transport=lambda *args:row).assess(self.m.request_for(self.decision,historical=False))
            except self.m.ModelError: self.fail('valid opaque backend metadata must be accepted')
            self.assertEqual(result['fingerprint'],fingerprint)
        for fingerprint in ('sk-'+'x'*32,'fp_'+self.key,'x'*97,'bad\nmetadata'):
            row=self.response();row['system_fingerprint']=fingerprint
            with self.assertRaises(self.m.ModelError):
                self.m.DeepSeekClient(self.key,transport=lambda *args:row).assess(self.m.request_for(self.decision,historical=False))

    def test_invalid_or_truncated_output_never_becomes_a_decision(self):
        cases=[]
        for content in ('{"action":"buy","action":"sell","reason":"x"}',
                        '{"action":"buy","reason":"x","confidence":.9}',
                        '{"action":"sell","reason":"'+self.key+'"}',
                        '{"action":"hold","reason":"<script>bad</script>"}',
                        '{"action":"hold","reason":""}',
                        '{"action":"buy","reason":"x","price":NaN}'):
            row=self.response();row['choices'][0]['message']['content']=content;cases.append(row)
        for status in ('length','tool_calls','content_filter'):
            row=self.response();row['choices'][0]['finish_reason']=status;cases.append(row)
        row=self.response();row['usage']['completion_tokens']=513;cases.append(row)
        for field in ('model','system_fingerprint'):
            row=self.response();row[field]=self.key;cases.append(row)
        for row in cases:
            with self.subTest(row=cases.index(row)):
                with self.assertRaisesRegex(self.m.ModelError,'model_response_invalid'):
                    self.m.DeepSeekClient(self.key,transport=lambda *a:row).assess(self.m.request_for(self.decision,historical=False))

    def test_transport_failure_is_constant_and_never_retried(self):
        seen=[]
        def transport(*args): seen.append(1);raise OSError(self.key+' PRIVATE_PROVIDER_BODY')
        with self.assertRaises(self.m.ModelError) as error:
            self.m.DeepSeekClient(self.key,transport=transport).assess(self.m.request_for(self.decision,historical=False))
        self.assertEqual(str(error.exception),'model_request_failed');self.assertEqual(seen,[1])

    def test_features_and_credentials_are_validated_before_request(self):
        for key in ('','bad','sk-\nbad'):
            with self.assertRaises(self.m.ModelError): self.m.DeepSeekClient(key)
        for name,value in (('last_price_usd',float('nan')),('rsi_sma',101),('interval_seconds',60)):
            row=copy.deepcopy(self.decision);row['features'][name]=value
            with self.assertRaises(self.m.ModelError): self.m.request_for(row,historical=False)

    def test_http_uses_fixed_tls_host_no_redirect_or_proxy_and_bounds_body(self):
        class Response:
            status=302
            def getheader(self,*args): return 'identity'
            def read1(self,*args): raise AssertionError('redirect body must not be read')
        class Connection:
            def __init__(self,*args,**kwargs): self.args=args;self.kwargs=kwargs;self.closed=False
            def request(self,method,path,body,headers):
                self.sent=(method,path,body,headers)
            def getresponse(self): return Response()
            def close(self): self.closed=True
        connection=Connection()
        with patch.object(self.m.http.client,'HTTPSConnection',return_value=connection) as constructor:
            with self.assertRaises(self.m.ModelError): self.m._http(self.m.request_for(self.decision,historical=False),self.key)
        self.assertEqual(constructor.call_args.args,('api.deepseek.com',))
        self.assertEqual(connection.sent[:2],('POST','/chat/completions'));self.assertTrue(connection.closed)


if __name__=='__main__': unittest.main()
