"""Formula oracles on captured synthetic inputs, never prospective market results."""
from dataclasses import replace
from datetime import timedelta
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zoneinfo

sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_training_data import ROOT,TZ,Tick
from issue_fixture import IssueWorld
from paperbot_r2 import forecast_inputs as inputs,candidate_forecasts as forecasts
from paperbot_r2 import model_training


class CandidateForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_tz=zoneinfo.TZPATH
        zoneinfo.reset_tzpath([str(ROOT/'outputs/zoneinfo')]);zoneinfo.ZoneInfo.clear_cache()
        cls.addClassCleanup(zoneinfo.reset_tzpath,cls.old_tz)
        cls.addClassCleanup(zoneinfo.ZoneInfo.clear_cache)
        cls.temp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        with patch.dict(os.environ,{},clear=True):
            cls.w=IssueWorld(Path(cls.temp.name));cls.key=cls.w.receive_until('2026-11-24',9)
            cls.value=inputs.capture_issue_input(cls.w.training_receipt,cls.w.registration,cls.w.source_pairs(),cls.key,
                timezone_file=TZ,clock=Tick(cls.w.at))

    def calculate(self,value=None,at=None):
        return forecasts.compute_candidates(self.w.training_receipt,value or self.value,
            clock=Tick(at or self.value.available_at+timedelta(seconds=1)))

    def test_five_rows_failed_fit_retained_and_prevalence_exact(self):
        panel=self.calculate();rows={r.family:r for r in panel.rows}
        self.assertEqual(tuple(r.family for r in panel.rows),model_training.FAMILIES)
        self.assertEqual(rows['prevalence'].probability,2./3.)
        self.assertEqual(rows['gaussian_nb'].status,'failed')
        self.assertIsNone(rows['gaussian_nb'].probability)
        self.assertTrue(panel.math_only)
        self.assertGreater(panel.completed_at,self.value.available_at)
        self.assertEqual(panel.input_sha256,self.value.seal)

    def test_ewma_garch_recurrence_and_two_step_variance_match_independent_oracles(self):
        panel=self.calculate();rows={r.family:r for r in panel.rows}
        attempts={a.family:a for a in self.w.training_receipt.fits.attempts if a.symbol=='AAPL'}
        variance=attempts['ewma'].payload.state.variance
        for r in self.value.returns:variance=.94*variance+.06*r.return_fraction**2
        self.assertAlmostEqual(rows['ewma'].gap_variance,variance,places=15)
        self.assertEqual(rows['ewma'].target_variance,rows['ewma'].gap_variance)
        model=attempts['garch'].payload.model;variance=attempts['garch'].payload.state.variance
        for r in self.value.returns:variance=model.omega+.1*(r.return_fraction-model.mean)**2+.85*variance
        self.assertAlmostEqual(rows['garch'].gap_variance,variance,places=15)
        self.assertAlmostEqual(rows['garch'].target_variance,model.omega+.95*variance,places=15)

    def test_hmm_target_weights_advance_twice_from_captured_posterior(self):
        panel=self.calculate();row=next(r for r in panel.rows if r.family=='hmm')
        self.assertEqual(row.status,'computed')
        from paperbot_r2 import hmm
        a=next(a for a in self.w.training_receipt.fits.attempts if a.symbol=='AAPL' and a.family=='hmm')
        posterior=a.payload.state.log_posterior
        for obs in self.value.returns:posterior,_=hmm.filter_hmm(a.payload.model.parameters,posterior,obs.return_fraction)
        weights=tuple(math.exp(v) for v in posterior)
        transition=a.payload.model.parameters.transition
        for _ in range(2):weights=tuple(sum(weights[i]*transition[i][j] for i in range(2)) for j in range(2))
        for actual,expected in zip(row.target_weights,weights):self.assertAlmostEqual(actual,expected,places=11)

    def test_later_live_gap_and_revision_cannot_change_frozen_input_math(self):
        before=self.calculate()
        # A future raw append happens after capture; the numerical helper has
        # no reference to any source book and must not reread it.
        self.w.receive_until('2026-11-24',10,derive=False)
        after=self.calculate(at=self.w.at)
        self.assertEqual([(r.probability,r.state_sha256) for r in before.rows],
                         [(r.probability,r.state_sha256) for r in after.rows])

    def test_symbol_binding_and_exact_float_zero_one_logs(self):
        with self.assertRaises(inputs.ForecastInputError):replace(self.value,symbol='MSFT')
        self.assertEqual(forecasts.probability_logs(0.),(-math.inf,0.))
        self.assertEqual(forecasts.probability_logs(1.),(0.,-math.inf))
        panel=self.calculate()
        for row in panel.rows:
            if row.status=='computed':
                self.assertEqual((row.log_event_probability,row.log_no_event_probability),
                                 forecasts.probability_logs(row.probability))


if __name__=='__main__':unittest.main()
