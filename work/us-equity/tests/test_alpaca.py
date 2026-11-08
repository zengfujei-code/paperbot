import importlib
import importlib.util
import io
import gzip
import http.client
import os
import unittest
import urllib.error
from datetime import datetime, timezone
from urllib.parse import urlsplit, parse_qs
from unittest.mock import patch


class AlpacaReadOnlyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.alpaca'), 'read-only provider is not implemented')
        self.m = importlib.import_module('paperbot_us.alpaca')
        self.now = datetime(2026,10,21,15,0,tzinfo=timezone.utc)
        self.calls = []
        self.responses = []
        self.elapsed = 0.0
        self.sleeps = []

    def client(self, **kwargs):
        def transport(url,headers):
            self.calls.append((url,headers))
            value = self.responses.pop(0)
            if isinstance(value,Exception):raise value
            return value
        def sleep(seconds):
            self.sleeps.append(seconds)
            self.elapsed += seconds
        return self.m.AlpacaDataClient('PK_TEST_ONLY_NOT_A_REAL_KEY','TEST_ONLY_NOT_A_REAL_SECRET',
            transport=transport,clock=lambda:self.now,monotonic=lambda:self.elapsed,sleep=sleep,**kwargs)

    def row(self, stamp='2026-10-21T14:00:00Z'):
        return dict(t=stamp,o=100,h=102,l=99,c=101,v=500)

    def test_credentials_are_required_and_not_in_repr(self):
        with patch.dict(os.environ,{},clear=True),self.assertRaisesRegex(self.m.ProviderError,'credentials_missing'):
            self.m.AlpacaDataClient.from_environment()
        client = self.client()
        self.assertNotIn('TEST_ONLY',repr(client))

    def test_bar_pages_are_complete_and_always_explicit_iex_raw_usd(self):
        self.responses = [{'bars':{'AAPL':[self.row()]},'next_page_token':'next'},
                          {'bars':{'MSFT':[self.row()]},'next_page_token':None}]
        result = self.client().bars(['AAPL','MSFT'],start=self.now.replace(hour=13),end=self.now)
        self.assertEqual(set(result),{'AAPL','MSFT'})
        self.assertEqual(len(self.calls),2)
        for url,headers in self.calls:
            parts = urlsplit(url)
            self.assertEqual(parts.netloc,'data.alpaca.markets')
            self.assertEqual(parts.path,'/v2/stocks/bars')
            query = parse_qs(parts.query)
            self.assertEqual((query['feed'],query['adjustment'],query['currency']),(['iex'],['raw'],['USD']))
            self.assertEqual(query['timeframe'],['15Min'])
            self.assertIn('APCA-API-KEY-ID',headers)
        self.assertEqual(parse_qs(urlsplit(self.calls[1][0]).query)['page_token'],['next'])
        self.assertTrue(self.sleeps)

    def test_pagination_loop_and_missing_structure_fail_without_partial_results(self):
        self.responses = [{'bars':{'AAPL':[self.row()]},'next_page_token':'same'},
                          {'bars':{},'next_page_token':'same'}]
        with self.assertRaisesRegex(self.m.ProviderError,'pagination_loop'):
            self.client().bars(['AAPL'],start=self.now.replace(hour=13),end=self.now)
        self.responses = [{}]
        with self.assertRaises(self.m.ProviderError):
            self.client().bars(['AAPL'],start=self.now.replace(hour=13),end=self.now)

    def test_unrequested_symbols_and_duplicate_bars_are_rejected(self):
        for bars in ({'OTHER':[self.row()]},{'AAPL':[self.row(),self.row()]}):
            self.responses = [{'bars':bars,'next_page_token':None}]
            with self.subTest(bars=bars),self.assertRaises(self.m.ProviderError):
                self.client().bars(['AAPL'],start=self.now.replace(hour=13),end=self.now)

    def test_request_scope_cannot_be_a_live_broker_or_order_operation(self):
        client = self.client()
        for host,path in (('https://api.alpaca.markets','/v2/account'),
                          ('https://paper-api.alpaca.markets','/v2/orders'),
                          ('https://example.invalid','/v2/assets')):
            with self.subTest(host=host,path=path),self.assertRaises(self.m.ProviderError):
                client._get(host,path,{})
        self.assertEqual(self.calls,[])

    def test_http_error_bodies_and_credentials_are_not_exposed(self):
        self.responses = [urllib.error.HTTPError('https://paper-api.alpaca.markets/v2/assets',401,
            'TEST_ONLY_SECRET_IN_ERROR',{},io.BytesIO(b'TEST_ONLY_SECRET_IN_BODY'))]
        with self.assertRaises(self.m.ProviderError) as caught:
            self.client().assets()
        self.assertEqual(str(caught.exception),'HTTP_401')
        self.assertNotIn('SECRET',str(caught.exception))
        self.assertEqual(len(self.calls),1)

    def test_retry_is_bounded_and_does_not_retry_authentication(self):
        self.responses = [urllib.error.HTTPError('https://data.alpaca.markets',429,'rate',{},None) for _ in range(3)]
        with self.assertRaisesRegex(self.m.ProviderError,'HTTP_429'):
            self.client().snapshots(['AAPL'])
        self.assertEqual(len(self.calls),3)

    def test_calendar_has_explicit_coverage_and_paper_only_endpoint(self):
        self.responses = [[{'date':'2026-10-21','open':'09:30','close':'16:00'}]]
        calendar = self.client().calendar('2026-10-21','2026-10-22')
        self.assertIsNotNone(calendar.at(self.now))
        self.assertEqual(urlsplit(self.calls[0][0]).netloc,'paper-api.alpaca.markets')

    def test_asset_snapshot_filters_without_claiming_historical_membership(self):
        base = dict(id='security-id-test',symbol='AAPL',status='active',**{'class':'us_equity'},
                    exchange='NASDAQ',tradable=True,fractionable=True)
        self.responses = [[base,{**base,'symbol':'OTC','exchange':'OTC'},
                           {**base,'symbol':'HALT','tradable':False},
                           {**base,'symbol':'BAD','tradable':'true'}]]
        result = self.client().assets()
        self.assertEqual([a.symbol for a in result.assets],['AAPL'])
        self.assertEqual(result.rejected_count,3)
        self.assertEqual(result.coverage,'current_snapshot_only')

    def test_asset_response_uses_class_not_the_query_parameter_name(self):
        valid=dict(id='public-instrument',symbol='AAPL',status='active',exchange='NASDAQ',tradable=True,
                   fractionable=True,**{'class':'us_equity'})
        self.responses=[[valid,{**valid,'symbol':'CRYPTO','class':'crypto','asset_class':'us_equity'}]]
        result=self.client().assets()
        self.assertEqual([a.symbol for a in result.assets],['AAPL'])
        self.assertEqual(result.rejected_count,1)

    def test_missing_snapshot_is_not_a_zero_price(self):
        self.responses = [{'AAPL':{}}]
        result = self.client().snapshots(['AAPL','MSFT'])
        self.assertEqual(set(result),set())
        self.assertNotIn('MSFT',result)

    def snapshot(self):
        return dict(latestTrade={'t':'2026-10-21T14:59:50Z','p':101},
                    latestQuote={'t':'2026-10-21T14:59:55Z','bp':100.9,'ap':101.1,'bs':50,'as':70},
                    dailyBar={'t':'2026-10-21T04:00:00Z','v':100000})

    def test_snapshots_have_validated_prices_provenance_and_receipt_time(self):
        self.responses = [{'AAPL':self.snapshot()}]
        result = self.client().snapshots(['AAPL'])
        self.assertEqual((result['AAPL'].feed,result['AAPL'].currency,result['AAPL'].volume_unit),('iex','USD','shares'))
        self.assertEqual(result['AAPL'].received_at,self.now)

    def test_bad_snapshots_are_quarantined_not_scannable(self):
        import copy
        for field,value in (('p',float('inf')),('p',True),('p',-1),('t','2099-01-01T00:00:00Z'),('t','2026-10-20T10:00:00Z')):
            snapshot=copy.deepcopy(self.snapshot())
            snapshot['latestTrade'][field]=value
            self.responses=[{'AAPL':snapshot}]
            client=self.client()
            with self.subTest(field=field,value=value):
                self.assertEqual(client.snapshots(['AAPL']),{})
                self.assertIn('AAPL',client.snapshot_rejections)

    def test_protocol_errors_never_expose_response_fragments(self):
        for error in (http.client.BadStatusLine('TEST_ONLY_PRIVATE_RESPONSE_LINE'),http.client.IncompleteRead(b'TEST_ONLY_PRIVATE_RESPONSE')):
            self.responses=[error]
            with self.subTest(error=type(error).__name__),self.assertRaises(self.m.ProviderError) as caught:
                self.client().assets()
            self.assertEqual(str(caught.exception),'http_protocol_error')

    def test_strict_json_rejects_overflow_duplicate_fields_and_constants(self):
        for raw in (b'{"p":1e400}',b'{"p":NaN}',b'{"p":1,"p":2}',b'{"p":'+b'1'+b'0'*400+b'}'):
            with self.subTest(raw_length=len(raw)),self.assertRaises(self.m.ProviderError):
                self.m.decode_json(raw)

    def test_cross_host_redirect_is_refused_before_headers_can_be_forwarded(self):
        with self.assertRaisesRegex(self.m.ProviderError,'redirect_refused'):
            self.m._NoRedirect().redirect_request(None,None,302,'redirect',{},'https://example.invalid')

    def test_page_budget_applies_to_entire_batched_call(self):
        symbols=['A'+str(i) for i in range(101)]
        self.responses=[{'bars':{},'next_page_token':None}]*2
        with self.assertRaisesRegex(self.m.ProviderError,'pagination_budget_exceeded'):
            self.client(max_pages=1).bars(symbols,start=self.now.replace(hour=13),end=self.now)
        self.assertEqual(len(self.calls),1)

    def test_slow_chunked_body_has_a_total_response_deadline(self):
        elapsed=[0.0]
        class SlowResponse:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self,size): return b'{}'
            def read1(self,size):
                elapsed[0]+=10
                return b' '
        class Opener:
            def open(self,*args,**kwargs): return SlowResponse()
        with patch.object(self.m.urllib.request,'build_opener',return_value=Opener()), \
             patch.object(self.m.time,'monotonic',side_effect=lambda:elapsed[0]), \
             self.assertRaisesRegex(self.m.ProviderError,'response_deadline_exceeded'):
            self.m.AlpacaDataClient._http_json('https://data.alpaca.markets/v2/stocks/snapshots',{})
        self.assertLessEqual(elapsed[0],40)

    def test_finite_chunked_body_is_decoded_normally(self):
        class Opener:
            def open(self,*args,**kwargs): return io.BytesIO(b'{"okay":true}')
        with patch.object(self.m.urllib.request,'build_opener',return_value=Opener()):
            self.assertEqual(self.m.AlpacaDataClient._http_json('https://data.alpaca.markets/v2/stocks/snapshots',{}),{'okay':True})

    def test_gzip_response_is_requested_and_decoded_with_output_bound(self):
        class Response(io.BytesIO):
            headers={'Content-Encoding':'gzip'}
        class Opener:
            def open(self,request,**kwargs):
                self.encoding=request.get_header('Accept-encoding')
                return Response(gzip.compress(b'{"okay":true}'))
        opener=Opener()
        with patch.object(self.m.urllib.request,'build_opener',return_value=opener):
            self.assertEqual(self.m.AlpacaDataClient._http_json('https://paper-api.alpaca.markets/v2/assets',{}),{'okay':True})
        self.assertEqual(opener.encoding,'gzip')
        class HugeOpener:
            def open(self,*args,**kwargs):
                return Response(gzip.compress(b' '*(20*1024*1024+1)))
        with patch.object(self.m.urllib.request,'build_opener',return_value=HugeOpener()), \
             self.assertRaisesRegex(self.m.ProviderError,'response_too_large'):
            self.m.AlpacaDataClient._http_json('https://paper-api.alpaca.markets/v2/assets',{})

    def test_infrequent_asset_refresh_has_separate_bounded_budget(self):
        elapsed=[0.0]
        class Response:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read1(self,size): elapsed[0]+=20; return b' '
        class Opener:
            def open(self,*args,**kwargs): return Response()
        with patch.object(self.m.urllib.request,'build_opener',return_value=Opener()), \
             patch.object(self.m.time,'monotonic',side_effect=lambda:elapsed[0]), \
             self.assertRaisesRegex(self.m.ProviderError,'response_deadline_exceeded'):
            self.m.AlpacaDataClient._http_json('https://paper-api.alpaca.markets/v2/assets?status=active',{})
        self.assertGreaterEqual(elapsed[0],120)
        self.assertLessEqual(elapsed[0],140)


if __name__ == '__main__':
    unittest.main()
