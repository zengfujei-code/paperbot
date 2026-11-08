from contextlib import redirect_stdout
from datetime import datetime
from dataclasses import asdict
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zoneinfo
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(Path(__file__).resolve().parents[3]),str(Path(__file__).resolve().parents[1])]
from paperbot_r2 import receipt_guard
from paperbot_r2.receipt_book import ReceiptPlan,ReceiptBook
from paperbot_us.alpaca import AlpacaDataClient


class ReceiptGuardTests(unittest.TestCase):
    def setUp(self):
        old=zoneinfo.TZPATH
        self.addCleanup(lambda:(zoneinfo.reset_tzpath(old),zoneinfo.ZoneInfo.clear_cache()))
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        tzroot=ROOT/'outputs/zoneinfo';tzsha=sha256((tzroot/'America/New_York').read_bytes()).hexdigest()
        plan=ReceiptPlan(('AAPL','MSFT','SPY'),'2026-10-27','2026-11-25',tzsha)
        blob=json.dumps(asdict(plan),sort_keys=True).encode();path=self.root/'plan.json';path.write_bytes(blob)
        self.book=self.root/'raw.sqlite';self.publish=self.root/'status.json'
        self.argv=['guard','--plan',str(path),'--plan-sha256',sha256(blob).hexdigest(),
                   '--book',str(self.book),'--publish',str(self.publish),'--timezone-root',str(tzroot)]

    def run_at(self,stamp,delegate):
        stream=io.StringIO()
        with patch.object(sys,'argv',self.argv),redirect_stdout(stream):
            status=receipt_guard.main(clock=lambda:datetime.fromisoformat(stamp),delegate=delegate)
        return status,stream.getvalue()

    def test_reserved_dates_and_lead_buffer_never_open_book_or_read_key(self):
        for stamp in ('2026-11-10T04:55:00+00:00','2026-11-10T15:00:00+00:00',
                      '2026-11-24T04:59:59+00:00'):
            with self.subTest(stamp=stamp):
                with patch.object(ReceiptBook,'snapshot',side_effect=AssertionError('book read')) as book:
                    with patch.object(os,'getenv',side_effect=AssertionError('key read')) as key:
                        with patch.object(AlpacaDataClient,'__init__',side_effect=AssertionError('client')) as api:
                            delegate=unittest.mock.Mock(side_effect=AssertionError('delegate'))
                            code,text=self.run_at(stamp,delegate)
                value=json.loads(text);self.assertEqual(code,0)
                self.assertEqual(value['status'],'reserved_dates_paused')
                self.assertIsNone(value['raw_first_count']);self.assertTrue(value['coverage_interrupted'])
                self.assertEqual((book.call_count,key.call_count,api.call_count,delegate.call_count),(0,0,0,0))
                self.assertFalse(self.book.exists())

    def test_before_buffer_and_after_pause_delegate_exact_cli(self):
        for stamp in ('2026-11-10T04:54:59+00:00','2026-11-24T05:00:00+00:00',
                      '2026-10-29T14:00:00+00:00'):
            seen=[]
            def delegate():seen.append(list(sys.argv));return 7
            code,text=self.run_at(stamp,delegate)
            self.assertEqual(code,7);self.assertEqual(seen,[self.argv]);self.assertEqual(text,'')

    def test_invalid_timezone_refuses_before_delegate_or_book(self):
        self.argv[-1]=str(self.root/'missing_timezone')
        delegate=unittest.mock.Mock()
        code,text=self.run_at('2026-11-10T15:00:00+00:00',delegate)
        self.assertEqual(code,1);self.assertEqual(json.loads(text)['status'],'collector_guard_unavailable')
        self.assertEqual(delegate.call_count,0);self.assertFalse(self.book.exists())

    def test_paused_initialize_flag_cannot_create_or_reset_state(self):
        self.argv.append('--initialize-only');self.book.write_bytes(b'existing evidence')
        delegate=unittest.mock.Mock()
        code,_=self.run_at('2026-11-10T15:00:00+00:00',delegate)
        self.assertEqual(code,0);self.assertEqual(delegate.call_count,0)
        self.assertEqual(self.book.read_bytes(),b'existing evidence')

    def test_publication_failure_is_not_reported_as_a_successful_pause(self):
        with patch.object(receipt_guard.receipt_runner,'_publish',side_effect=OSError('fixture')):
            code,text=self.run_at('2026-11-10T15:00:00+00:00',unittest.mock.Mock())
        self.assertEqual(code,1)
        self.assertEqual(json.loads(text)['status'],'publication_unavailable')

    def test_unexpected_delegate_failure_does_not_claim_zero_requests(self):
        delegate=unittest.mock.Mock(side_effect=RuntimeError('fixture'))
        code,text=self.run_at('2026-10-29T14:00:00+00:00',delegate)
        value=json.loads(text);self.assertEqual(code,1)
        self.assertEqual(value['status'],'collector_guard_delegate_unavailable')
        self.assertIsNone(value['requests_sent']);self.assertIsNone(value['book_touched'])


if __name__=='__main__':unittest.main()
