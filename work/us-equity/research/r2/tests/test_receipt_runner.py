from datetime import timedelta
from dataclasses import asdict,replace
from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit,parse_qs

sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_receipt_book as fixtures
from paperbot_us.alpaca import AlpacaDataClient
from paperbot_r2.receipt_runner import collect_once,main


class ReceiptRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixtures.ReceiptBookTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):fixtures.ReceiptBookTests.tearDownClass.__func__(cls)

    def setUp(self):
        fixtures.ReceiptBookTests.setUp(self)
        self.book.initialize(self.plan,now=self.now)
        self.at = self.now;self.calls=[];self.rows={};self.provider_fails=False

    def calendar(self,session=None):return fixtures.ReceiptBookTests.calendar(self,session)
    def bar(self,**values):return fixtures.ReceiptBookTests.bar(self,**values)

    def client(self):
        def transport(url,headers):
            parsed=urlsplit(url);base=parsed.scheme+'://'+parsed.netloc;path=parsed.path
            params={k:v[0] for k,v in parse_qs(parsed.query).items()}
            self.calls.append((base,path,dict(params)))
            if self.provider_fails:raise RuntimeError('SECRET_MUST_NOT_APPEAR')
            if path == '/v2/calendar':return [dict(date='2020-03-09',open='09:30',close='16:00')]
            if path == '/v2/stocks/bars':
                return {'bars':{s:[dict(t=b.starts.isoformat(),o=b.open,h=b.high,l=b.low,c=b.close,v=b.volume) for b in bars]
                               for s,bars in self.rows.items()},'next_page_token':None}
            raise AssertionError('unapproved_endpoint')
        return AlpacaDataClient('PK_NOT_A_REAL_KEY','NOT_A_REAL_SECRET',transport=transport,
                                clock=lambda:self.at,sleep=lambda _:None,max_pages=1)

    def collect(self):return collect_once(self.book,client=self.client(),clock=lambda:self.at)

    def test_closed_calendar_is_cached_and_never_fetches_bars_before_first_close(self):
        first=self.collect();self.assertEqual(first['status'],'market_closed')
        self.assertEqual(first['slot_count'],78)
        self.at+=timedelta(seconds=30);self.collect()
        self.assertEqual([c[1] for c in self.calls],['/v2/calendar'])

    def test_one_fixed_batch_get_records_real_receipts_and_no_orders(self):
        self.at=self.session.opens+timedelta(seconds=902)
        self.rows={'AAPL':[self.bar()]}
        result=self.collect()
        self.assertEqual(result['raw_first_count'],1);self.assertEqual(result['on_time_count'],1)
        self.assertEqual(result['orders_submitted'],0);self.assertEqual(result['models_fitted'],0)
        self.assertEqual([c[1] for c in self.calls],['/v2/calendar','/v2/stocks/bars'])
        self.assertEqual(self.calls[-1][2]['symbols'],'AAPL,MSFT,SPY')

    def test_provider_failure_is_sanitized_and_still_records_due_missing_slots(self):
        self.collect();self.at=self.session.opens+timedelta(seconds=1030);self.provider_fails=True
        result=self.collect()
        self.assertIn(result['status'],('calendar_unavailable','data_unavailable'))
        self.assertEqual(result['missed_count'],3)
        self.assertNotIn('SECRET_MUST_NOT_APPEAR',json.dumps(result))

    def test_recovery_fetch_continues_after_close_then_stops(self):
        self.at=self.session.closes+timedelta(seconds=30)
        self.rows={'AAPL':[self.bar(offset=25,delay=1)]}
        result=self.collect();self.assertEqual(result['raw_first_count'],1)
        before=len([c for c in self.calls if c[1]=='/v2/stocks/bars'])
        self.at=self.session.closes+timedelta(seconds=3601);result=self.collect()
        self.assertEqual(result['status'],'market_closed')
        self.assertEqual(len([c for c in self.calls if c[1]=='/v2/stocks/bars']),before)

    def test_capture_ignores_scanner_rotation_and_keeps_first_after_restart(self):
        self.at=self.session.opens+timedelta(seconds=902);self.rows={'AAPL':[self.bar()]}
        self.collect();self.at+=timedelta(seconds=30)
        result=self.collect();self.assertEqual(result['raw_first_count'],1)
        self.assertEqual(result['revision_count'],0)
        self.assertEqual(self.calls[-1][2]['symbols'],'AAPL,MSFT,SPY')

    def test_outside_plan_dates_do_not_call_provider(self):
        self.at=self.session.opens+timedelta(days=1)
        result=self.collect();self.assertEqual(result['status'],'collection_closed')
        self.assertEqual(self.calls,[])

    def test_misaligned_provider_data_still_records_deadline_misses(self):
        self.collect();self.at=self.session.opens+timedelta(seconds=1030)
        bar=self.bar();self.rows={'AAPL':[replace(bar,starts=bar.starts+timedelta(seconds=60))]}
        result=self.collect()
        self.assertEqual(result['status'],'data_unavailable')
        self.assertEqual(result['missed_count'],3)
        self.assertEqual(sum(slot['missed'] for slot in self.book.snapshot()['slots'].values()),3)

    def test_main_publication_failure_is_nonzero_and_sanitized(self):
        root=Path(self.temp.name);plan_path=root/'plan.json'
        raw=json.dumps(asdict(self.plan)).encode();plan_path.write_bytes(raw)
        args=['receipt_runner','--plan',str(plan_path),'--plan-sha256',sha256(raw).hexdigest(),
              '--book',str(root/'publication.sqlite'),'--publish',str(root/'missing-parent/status.json'),
              '--timezone-root',str(fixtures.ROOT/'outputs/zoneinfo'),'--initialize-only']
        output=io.StringIO()
        with patch.object(sys,'argv',args),redirect_stdout(output):code=main()
        result=json.loads(output.getvalue())
        self.assertEqual(code,1)
        self.assertEqual(result['status'],'publication_unavailable')


if __name__=='__main__':unittest.main()
