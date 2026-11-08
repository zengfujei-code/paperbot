from datetime import datetime,timezone
import re
import unittest
from unittest.mock import patch
from paperbot_us.data import utc_time,DataQualityError


class TimeParseTests(unittest.TestCase):
    def test_vendor_nanoseconds_are_explicitly_floored_to_microseconds_on_legacy_runtime(self):
        class LegacyDatetime(datetime):
            @classmethod
            def fromisoformat(cls,value):
                fraction=re.search(r'\.(\d+)',value)
                if fraction and len(fraction.group(1)) not in (3,6): raise ValueError('legacy fraction precision')
                return super().fromisoformat(value)
        with patch('paperbot_us.data.datetime',LegacyDatetime):
            value=utc_time('2026-10-23T17:00:00.123456789Z')
            self.assertEqual(value.microsecond,123456)
            self.assertEqual(value.utcoffset().total_seconds(),0)

    def test_fraction_lengths_and_offsets_have_one_cross_runtime_policy(self):
        for digits,expected in (('1',100000),('1234',123400),('1234567',123456),('999999999',999999)):
            with self.subTest(digits=digits):
                value=utc_time('2026-10-23T13:00:00.'+digits+'-04:00')
                self.assertEqual((value.hour,value.microsecond),(17,expected))

    def test_non_rfc3339_and_unbounded_fractional_precision_are_rejected(self):
        for value in ('2026-10-23T17:00:00.1234567890Z','2026-10-23X17:00:00Z',
                      '2026-10-23T17:00:00+00:99','2026-10-23T17:00:00',
                      '2026-10-23T17:00:60Z'):
            with self.subTest(value=value),self.assertRaises(DataQualityError): utc_time(value)


if __name__=='__main__': unittest.main()
