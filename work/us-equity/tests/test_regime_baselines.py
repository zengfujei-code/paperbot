from dataclasses import replace
from datetime import datetime,timedelta,timezone
import importlib,math,unittest


class RegimeBaselineTests(unittest.TestCase):
    def setUp(self):
        self.m=importlib.import_module('paperbot_us.regime_baselines')
        self.t=datetime(2026,10,23,15,tzinfo=timezone.utc)

    def observations(self,values,offset=0):
        return tuple(self.m.ReturnObservation(self.t+timedelta(minutes=15*(i+offset)),
            self.t+timedelta(minutes=15*(i+offset),seconds=1),v) for i,v in enumerate(values))

    def fit(self,values=(-.01,.01),**kwargs):
        rows=self.observations(values);self.cutoff=rows[-1].available_at
        return self.m.fit_baseline(rows,as_of=self.cutoff,decay=kwargs.get('decay',.5),
            cusum_k=kwargs.get('k',.5),cusum_h=kwargs.get('h',3))

    def test_hand_oracle_predicts_before_using_the_current_return(self):
        model=self.fit();rows=self.observations([.02,0],offset=2)
        result=self.m.evaluate_online(model,rows,as_of=rows[-1].available_at)
        self.assertAlmostEqual(result[0].variance_forecast,.0001)
        self.assertAlmostEqual(result[0].next_variance_forecast,.00025)
        self.assertAlmostEqual(result[1].variance_forecast,.00025)
        self.assertAlmostEqual(result[1].next_variance_forecast,.000125)
        self.assertEqual(result[0].variance_unit,'fraction_squared_per_15m')
        self.assertFalse(result[0].annualized)

    def test_adding_or_changing_a_future_suffix_cannot_change_past_forecasts(self):
        model=self.fit();rows=self.observations([.02,-.01,0,.4],offset=2)
        full=self.m.evaluate_online(model,rows,as_of=rows[-1].available_at)
        prefix=self.m.evaluate_online(model,rows[:2],as_of=rows[1].available_at)
        changed=self.m.evaluate_online(model,rows[:2]+self.observations([-.5,2],offset=4),as_of=rows[-1].available_at)
        self.assertEqual(full[:2],prefix);self.assertEqual(changed[:2],prefix)

    def test_training_fit_has_stable_identity_and_keeps_parameters_frozen(self):
        a=self.fit();b=self.fit()
        self.assertEqual(a,b);self.assertEqual(a.training_count,2)
        self.assertNotEqual(a.training_sha256,self.fit(values=(-.02,.02)).training_sha256)
        self.assertEqual(a.decay,.5)

    def test_training_and_evaluation_require_actual_availability_before_cutoff(self):
        rows=self.observations([-.01,.01])
        with self.assertRaises(self.m.BaselineError):self.m.fit_baseline(rows,
            as_of=rows[-1].ends_at,decay=.5,cusum_k=.5,cusum_h=3)
        model=self.fit();future=self.observations([.01],offset=2)
        with self.assertRaises(self.m.BaselineError):self.m.evaluate_online(model,future,as_of=future[0].ends_at)
        with self.assertRaises(self.m.BaselineError):self.m.evaluate_online(model,rows,as_of=self.cutoff)

    def test_cusum_detects_up_and_down_shifts_and_resets_after_each_alarm(self):
        model=self.fit();rows=self.observations([.03,.03,-.03,-.03],offset=2)
        r=self.m.evaluate_online(model,rows,as_of=rows[-1].available_at)
        self.assertEqual(r[1].alarms,('up',));self.assertEqual(r[3].alarms,('down',))
        self.assertAlmostEqual(r[2].cusum_down,2.5)

    def test_stable_control_and_constant_baseline_are_not_fake_regime_probabilities(self):
        model=self.fit();rows=self.observations([-.01,.01]*20,offset=2)
        r=self.m.evaluate_online(model,rows,as_of=rows[-1].available_at)
        self.assertFalse(any(p.alarms for p in r))
        constant=self.fit(values=(0,0));shock=self.observations([.01],offset=2)
        out=self.m.evaluate_online(constant,shock,as_of=shock[-1].available_at)[0]
        self.assertEqual(out.detector_status,'constant_training_baseline')
        self.assertIsNone(out.cusum_up);self.assertGreater(out.next_variance_forecast,0)

    def test_nonfinite_bool_wrong_frequency_and_impossible_returns_are_rejected(self):
        for value in (True,float('nan'),float('inf'),-1,-2):
            with self.subTest(value=value),self.assertRaises(self.m.BaselineError):self.observations([value])
        good=self.observations([.01])[0]
        with self.assertRaises(self.m.BaselineError):replace(good,interval_seconds=60)
        with self.assertRaises(self.m.BaselineError):replace(good,available_at=good.ends_at-timedelta(seconds=1))

    def test_duplicate_times_and_invalid_parameters_are_rejected(self):
        rows=self.observations([-.01,.01]);at=rows[-1].available_at
        for changes in (dict(decay=1),dict(decay=-.1),dict(cusum_k=-1),dict(cusum_h=0),dict(decay=True)):
            with self.subTest(changes=changes),self.assertRaises(self.m.BaselineError):
                self.m.fit_baseline(rows,**{**dict(as_of=at,decay=.5,cusum_k=.5,cusum_h=3),**changes})
        with self.assertRaises(self.m.BaselineError):self.m.fit_baseline((rows[0],rows[0]),as_of=at,decay=.5,cusum_k=.5,cusum_h=3)

    def test_extreme_integer_and_model_from_the_future_fail_with_contract_errors(self):
        with self.assertRaises(self.m.BaselineError):self.observations([10**1000])
        model=self.fit()
        with self.assertRaises(self.m.BaselineError):self.m.evaluate_online(model,(),as_of=self.t)

    def test_declared_15m_return_windows_cannot_overlap(self):
        first=self.observations([-.01])[0]
        second=replace(first,ends_at=first.ends_at+timedelta(minutes=1),
            available_at=first.available_at+timedelta(minutes=1),return_fraction=.01)
        with self.assertRaises(self.m.BaselineError):self.m.fit_baseline((first,second),
            as_of=second.available_at,decay=.5,cusum_k=.5,cusum_h=3)
        model=self.fit();near=self.observations([.01],offset=2)[0]
        near=replace(near,ends_at=model.training_ends_at+timedelta(minutes=1),
            available_at=model.fitted_at+timedelta(minutes=1))
        with self.assertRaises(self.m.BaselineError):self.m.evaluate_online(model,(near,),as_of=near.available_at)

    def test_receipt_delay_has_an_explicit_shorter_future_horizon(self):
        model=self.fit();row=self.observations([.01],offset=2)[0]
        point=self.m.evaluate_online(model,(row,),as_of=row.available_at)[0]
        self.assertEqual(point.forecast_horizon_seconds,899)
        self.assertEqual(point.target_elapsed_seconds_at_forecast,1)
        self.assertEqual(point.target_starts_at,model.training_ends_at)
        self.assertEqual(point.forecast_scope,'conditional_15m_return_not_remaining_time_variance')

    def test_underflow_never_turns_nonzero_returns_or_variance_into_zero_risk(self):
        with self.assertRaises(self.m.BaselineError):
            self.fit(values=(-2e-162,2e-162))
        model=self.fit(values=(-1e-100,1e-100),decay=1e-200)
        rows=self.observations([0],offset=2)
        with self.assertRaises(self.m.BaselineError):self.m.evaluate_online(model,rows,as_of=rows[-1].available_at)

    def test_observed_state_can_consume_late_backfill_without_inventing_forecasts(self):
        model=self.fit();rows=self.observations([.02,0],offset=2)
        receipt=rows[-1].available_at+timedelta(hours=1)
        state=self.m.initial_state(model)
        for row in rows:
            state=self.m.advance_observed(model,state,replace(row,available_at=receipt),observed_at=receipt)
        self.assertAlmostEqual(state.variance,.000125)
        self.assertEqual(state.updated_at,receipt);self.assertEqual(state.observations,2)
        self.assertFalse(hasattr(state,'forecast_at'))
        with self.assertRaises(self.m.BaselineError):
            self.m.advance_observed(model,state,rows[-1],observed_at=receipt)

    def test_observed_state_is_bound_to_the_frozen_model_and_actual_receipt(self):
        model=self.fit();row=self.observations([.02],offset=2)[0];state=self.m.initial_state(model)
        other=self.fit(values=(-.02,.02))
        with self.assertRaises(self.m.BaselineError):self.m.advance_observed(other,state,row,observed_at=row.available_at)
        with self.assertRaises(self.m.BaselineError):self.m.advance_observed(model,state,row,observed_at=row.ends_at)


if __name__=='__main__':unittest.main()
