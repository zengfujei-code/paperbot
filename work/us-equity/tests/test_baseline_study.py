from dataclasses import replace
from datetime import datetime,timedelta,timezone
import importlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from paperbot_us.data import Bar,Calendar,Session
from paperbot_us.research import ResearchDataset,time_split


class BaselineStudyTests(unittest.TestCase):
    def setUp(self):
        self.m=importlib.import_module('paperbot_us.baseline_study')
        from paperbot_us.experiments import ExperimentRegistry
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.registry=ExperimentRegistry(self.root/'registry.sqlite');self.registry.initialize()
        day=datetime(2026,9,1,tzinfo=timezone.utc)
        sessions=[Session.from_alpaca(dict(date=(day+timedelta(days=i)).date().isoformat(),open='09:30',close='16:00')) for i in range(10)]
        downloaded=day+timedelta(days=11);calendar=Calendar(sessions,start=sessions[0].day.isoformat(),end=sessions[-1].day.isoformat())
        bars=[]
        for d,s in enumerate(sessions):
            for i in range(26):
                close=100+i*.2+(.4 if i%3==0 else 0)+d*.1
                bars.append(Bar('AAPL',s.opens+timedelta(minutes=15*i),downloaded,close,close+1,close-1,close,100000))
        self.dataset=ResearchDataset.from_bars(bars,calendar=calendar,downloaded_at=downloaded,availability_policy='vendor_history_assumed_at_close')
        self.split=time_split(tuple(s.day for s in sessions),train_sessions=3,validation_sessions=2,test_sessions=3)
        self.candidates=(self.m.BaselineCandidate(.5,.5,6),self.m.BaselineCandidate(.94,.5,6))

    def run_case(self,**changes):
        kwargs=dict(split=self.split,candidates=self.candidates,registry=self.registry,directory=self.root/'study')
        kwargs.update(changes);return self.m.run_baseline_study(self.dataset,**kwargs)

    def test_real_attempts_are_registered_before_fit_and_selection_before_test_access(self):
        from paperbot_us.experiments import digest
        fit=self.m.fit_baseline;returns=self.m._returns
        with patch.object(self.registry,'register',wraps=self.registry.register) as registered:
            def checked_fit(rows,**kwargs):
                self.assertEqual(len(registered.call_args_list),1)
                state=self.registry.load(digest(registered.call_args.args[0]))
                self.assertIsNone(state['selected'])
                self.assertLessEqual(rows[-1].ends_at,self.dataset.calendar.sessions[self.split.train[-1]].closes)
                return fit(rows,**kwargs)
            def checked_returns(dataset):
                if dataset.sessions[-1].day>=self.split.test[0]:
                    state=self.registry.load(digest(registered.call_args.args[0]));self.assertIsNotNone(state['selected'])
                return returns(dataset)
            with patch.object(self.m,'fit_baseline',side_effect=checked_fit),patch.object(self.m,'_returns',side_effect=checked_returns):
                report=self.run_case()
        self.assertFalse(report['strategy_accepted']);self.assertEqual(report['status'],'completed')
        state=self.registry.load(report['experiment_id']);self.assertTrue(state['completed'])
        self.assertEqual(len(state['trials']),5)

    def test_final_test_prices_cannot_change_validation_metrics_or_selected_parameters(self):
        first=self.run_case()
        from paperbot_us.experiments import ExperimentRegistry
        other=ExperimentRegistry(self.root/'other.sqlite');other.initialize()
        starts=self.dataset.calendar.sessions[self.split.test[0]].opens
        changed=[replace(b,open=200,high=220,low=180,close=210 if i%2 else 190) if b.starts>=starts else b
                 for i,b in enumerate(self.dataset.bars)]
        self.dataset=ResearchDataset.from_bars(changed,calendar=self.dataset.calendar,downloaded_at=self.dataset.downloaded_at,
            availability_policy=self.dataset.availability_policy)
        second=self.run_case(registry=other,directory=self.root/'other')
        self.assertEqual(first['selected_candidate_id'],second['selected_candidate_id'])
        self.assertEqual(first['validation'],second['validation'])
        self.assertNotEqual(first['test']['variance_mse'],second['test']['variance_mse'])

    def test_completed_report_replays_without_any_fit_or_new_evaluation(self):
        first=self.run_case()
        with (patch.object(self.m,'fit_baseline',side_effect=AssertionError('refit')),
              patch.object(self.m,'evaluate_online',side_effect=AssertionError('reevaluation'))):
            second=self.run_case()
        self.assertEqual(first,second)

    def test_failed_candidate_is_retained_without_raw_error_or_retry(self):
        fit=self.m.fit_baseline
        def fail_one(rows,**kwargs):
            if kwargs['decay']==.5:raise self.m.BaselineError('SENSITIVE_TEST_MARKER')
            return fit(rows,**kwargs)
        with patch.object(self.m,'fit_baseline',side_effect=fail_one):report=self.run_case()
        state=self.registry.load(report['experiment_id'])
        self.assertEqual(sum(v['status']=='failed' for v in state['trials'].values()),1)
        self.assertNotIn('SENSITIVE_TEST_MARKER',json.dumps(report)+json.dumps(state))
        self.assertEqual(report['status'],'completed')

    def test_archived_receipts_are_not_replaced_with_assumed_close_times(self):
        self.dataset=replace(self.dataset,availability_policy='archived_receipts')
        report=self.run_case()
        self.assertEqual(report['status'],'failed_no_selection');self.assertIsNone(report['test'])
        self.assertFalse(self.registry.load(report['experiment_id'])['completed'])

    def test_within_session_returns_have_explicit_assumptions_and_no_overnight_gap(self):
        rows=self.m._returns(self.dataset)
        self.assertEqual(len(rows),250)
        self.assertTrue(all(row.available_at==row.ends_at for row in rows))
        report=self.run_case()
        self.assertEqual(report['availability_policy'],'vendor_history_assumed_at_close')
        self.assertEqual(report['cost_scope'],'risk_forecast_only_no_trade_returns')
        self.assertEqual(report['forecast_unit'],'fraction_squared_per_15m')

    def test_invalid_candidate_budget_and_missing_embargo_are_rejected(self):
        for candidates in ((self.candidates[0],self.candidates[0]),(),(object(),)):
            with self.assertRaises(self.m.BaselineStudyError):self.run_case(candidates=candidates)
        split=time_split(tuple(s.day for s in self.dataset.sessions),train_sessions=4,validation_sessions=3,test_sessions=3,embargo_sessions=0)
        with self.assertRaises(self.m.BaselineStudyError):self.run_case(split=split)

    def test_synthetic_fixture_is_labelled_and_separate_from_market_exposure(self):
        self.registry.import_exposure(domain='us_equity',days=[str(self.split.test[0])],source_sha256='f'*64)
        report=self.run_case(domain='synthetic')
        self.assertEqual(report['configuration']['data_origin'],'synthetic_fixture')
        self.assertEqual(self.registry.load(report['experiment_id'])['plan']['domain'],'synthetic')
        with self.assertRaises(self.m.BaselineStudyError):self.run_case(directory=self.root/'market')


if __name__=='__main__':unittest.main()
