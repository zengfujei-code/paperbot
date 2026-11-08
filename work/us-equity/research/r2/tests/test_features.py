from dataclasses import asdict, replace
from datetime import date, timedelta
from hashlib import sha256
import math
from pathlib import Path
import sys
import unittest
import zoneinfo

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[1])]
from paperbot_us.data import Bar, Calendar, Session
from paperbot_r2.contracts import ChallengerError, digest
from paperbot_r2.features import build_features, label_example


class FeatureAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_tzpath = zoneinfo.TZPATH
        zoneinfo.reset_tzpath([str(ROOT / 'outputs/zoneinfo')])
        zoneinfo.ZoneInfo.clear_cache()

    @classmethod
    def tearDownClass(cls):
        zoneinfo.reset_tzpath(cls.old_tzpath)
        zoneinfo.ZoneInfo.clear_cache()

    def calendar(self, day='2020-01-02', close='16:00'):
        session = Session.from_alpaca(dict(date=day, open='09:30', close=close))
        return Calendar([session], start=day, end=day)

    def window(self, calendar=None, start=None):
        calendar = calendar or self.calendar()
        session = next(iter(calendar.sessions.values()))
        start = start or session.opens
        result = []
        for i, (price, volume) in enumerate(zip((100., 110., 99.), (100., 200., 300.))):
            at = start + timedelta(seconds=900 * i)
            result.append(Bar('AAPL', at, at + timedelta(seconds=902), price, price, price, price, volume))
        return tuple(result)

    def bundle(self, bars=None, calendar=None, **changes):
        calendar = calendar or self.calendar()
        bars = bars or self.window(calendar)
        args = dict(calendar=calendar, issued_at=bars[-1].ends + timedelta(seconds=3),
                    lookback_returns=2, availability_policy='synthetic_fixture', record_sha256=digest('source-record'))
        args.update(changes)
        return build_features(bars, **args)

    def target(self, bundle, price=100.5):
        return Bar('AAPL', bundle.target_start, bundle.target_end + timedelta(seconds=2),
                   100., max(100., price), min(100., price), price, 100.)

    def example(self, bundle, target=None, **changes):
        target = target or self.target(bundle)
        args = dict(record_sha256=digest('outcome-record'), threshold_fraction=.005,
                    observed_at=target.received_at + timedelta(seconds=1), availability_policy='synthetic_fixture')
        args.update(changes)
        return label_example(bundle, target, **args)

    def test_independent_formula_oracle_and_conservative_times(self):
        bars = self.window()
        bundle = self.bundle(bars)
        self.assertEqual(bundle.vector.feature_names, ('trend_fraction', 'volatility_fraction', 'iex_volume_ratio'))
        for actual, expected in zip(bundle.vector.values, (-.01, .1, 1.)):
            self.assertAlmostEqual(actual, expected, places=12)
        self.assertEqual(bundle.vector.event_at, bars[-1].ends)
        self.assertEqual(bundle.input_available_at, bars[-1].received_at)
        self.assertEqual(bundle.vector.available_at, bundle.issued_at)
        self.assertEqual(bundle.target_start, bars[-1].ends + timedelta(seconds=900))
        self.assertEqual(bundle.target_end, bars[-1].ends + timedelta(seconds=1800))

    def test_label_boundary_and_future_prices_do_not_change_features(self):
        bundle = self.bundle()
        before = asdict(bundle)
        self.assertEqual(self.example(bundle).label, 1)
        self.assertEqual(self.example(bundle, self.target(bundle, 100.4)).label, 0)
        self.assertEqual(self.example(bundle, self.target(bundle, 99.5)).label, 1)
        self.assertEqual(asdict(bundle), before)
        self.assertEqual(self.example(bundle).features, bundle.vector)

    def test_late_or_unclosed_input_and_assumed_history_are_rejected(self):
        bars = self.window()
        issued = bars[-1].ends + timedelta(seconds=3)
        for changed in (bars[:-1] + (replace(bars[-1], received_at=issued + timedelta(seconds=1)),),
                        bars[:-1] + (replace(bars[-1], received_at=bars[-1].ends - timedelta(seconds=1)),)):
            with self.assertRaises(ChallengerError):
                self.bundle(changed)
        with self.assertRaises(ChallengerError):
            self.bundle(availability_policy='vendor_history_assumed_at_close')
        with self.assertRaises(ChallengerError):
            self.bundle(issued_at=bars[-1].ends - timedelta(seconds=1))

    def test_gaps_duplicates_order_mixed_identity_and_zero_volume_fail(self):
        bars = self.window()
        cases = [tuple(reversed(bars)), bars[:-1] + (bars[-2],),
                 bars[:-1] + (replace(bars[-1], starts=bars[-1].starts + timedelta(seconds=1)),),
                 bars[:-1] + (replace(bars[-1], symbol='MSFT'),),
                 bars[:-1] + (replace(bars[-1], feed='sip'),),
                 bars[:-1] + (replace(bars[-1], volume=0),)]
        for values in cases:
            with self.subTest(last=values[-1].starts), self.assertRaises(ChallengerError):
                self.bundle(values)

    def test_freshness_boundary_and_regular_session_target(self):
        bars = self.window()
        self.bundle(bars, issued_at=bars[-1].ends + timedelta(seconds=120))
        with self.assertRaises(ChallengerError):
            self.bundle(bars, issued_at=bars[-1].ends + timedelta(seconds=121))
        early = self.calendar(close='13:00')
        session = next(iter(early.sessions.values()))
        bars = self.window(early, start=session.closes - timedelta(seconds=3600))
        with self.assertRaises(ChallengerError):
            self.bundle(bars, calendar=early)

    def test_dst_calendar_and_calendar_coverage(self):
        bundles = []
        for day, expected_hour in [('2020-03-06', 14), ('2020-03-09', 13)]:
            calendar = self.calendar(day)
            self.assertEqual(next(iter(calendar.sessions.values())).opens.hour, expected_hour)
            bundles.append(self.bundle(calendar=calendar))
        self.assertEqual(bundles[0].vector.stream_id, bundles[1].vector.stream_id)
        self.assertNotEqual(bundles[0].calendar_sha256, bundles[1].calendar_sha256)
        with self.assertRaises(ChallengerError):
            self.bundle(calendar=self.calendar('2020-01-03'), bars=self.window())

    def test_wrong_outcome_symbol_horizon_provenance_or_availability_rejects(self):
        bundle = self.bundle()
        target = self.target(bundle)
        for value in (replace(target, symbol='MSFT'), replace(target, feed='sip'),
                      replace(target, starts=target.starts + timedelta(seconds=900)),
                      replace(target, volume=0), replace(target, received_at=target.ends - timedelta(seconds=1))):
            with self.assertRaises(ChallengerError):
                self.example(bundle, value)
        with self.assertRaises(ChallengerError):
            self.example(bundle, observed_at=target.ends)
        with self.assertRaises(ChallengerError):
            self.example(bundle, availability_policy='archived_receipts')

    def test_source_calendar_policy_and_receipt_commitments(self):
        bundle = self.bundle()
        self.assertNotEqual(bundle.vector.source_id, self.bundle(record_sha256=digest('other-source')).vector.source_id)
        changed = self.window()
        changed = (replace(changed[0], received_at=changed[0].received_at + timedelta(seconds=1)),) + changed[1:]
        self.assertNotEqual(bundle.vector.source_id, self.bundle(changed).vector.source_id)
        archive = self.bundle(availability_policy='archived_receipts')
        self.assertNotEqual(bundle.vector.stream_id, archive.vector.stream_id)
        self.assertNotEqual(self.example(bundle).label_schema_id,
                            self.example(bundle, threshold_fraction=.006).label_schema_id)

    def test_seal_restore_and_source_hash_dependencies(self):
        bundle = self.bundle()
        with self.assertRaises(ChallengerError):
            replace(bundle, target_start=bundle.target_start + timedelta(seconds=1))
        restored = replace(bundle, issued_at=bundle.issued_at.isoformat(),
            input_available_at=bundle.input_available_at.isoformat(), target_start=bundle.target_start.isoformat(),
            target_end=bundle.target_end.isoformat())
        self.assertEqual(bundle, restored)
        import paperbot_r2.features as module
        import paperbot_r2.classifier as classifier
        import paperbot_r2.contracts as contracts
        from paperbot_us import data, regime_baselines
        paths = [Path(m.__file__) for m in (module, classifier, contracts, data, regime_baselines)]
        self.assertEqual(bundle.source_sha256, digest({p.name:sha256(p.read_bytes()).hexdigest() for p in paths}))

    def test_bounded_parameters_and_no_silent_window_selection(self):
        for count in (True, 1, 21):
            with self.assertRaises(ChallengerError):
                self.bundle(lookback_returns=count)
        with self.assertRaises(ChallengerError):
            self.bundle(self.window() + (self.window()[-1],))
        for threshold in (True, 0., .21, math.inf, math.nan):
            with self.assertRaises(ChallengerError):
                self.example(self.bundle(), threshold_fraction=threshold)

    def test_flat_finite_prices_never_cross_a_positive_movement_threshold(self):
        bundle = self.bundle()
        for price in (5e-324, 1e308):
            flat = replace(self.target(bundle), open=price, high=price, low=price, close=price)
            for threshold in (.005, 5e-324):
                with self.subTest(price=price, threshold=threshold):
                    self.assertEqual(self.example(bundle, flat, threshold_fraction=threshold).label, 0)

    def test_positive_finite_volume_extremes_preserve_equal_volume_ratio(self):
        for volume in (5e-324, 1.7e308):
            bars = tuple(replace(bar, volume=volume) for bar in self.window())
            with self.subTest(volume=volume):
                self.assertEqual(self.bundle(bars).vector.values[2], 0.)


if __name__ == '__main__':
    unittest.main()
