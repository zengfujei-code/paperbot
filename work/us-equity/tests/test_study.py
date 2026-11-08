from datetime import datetime,timedelta,timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from paperbot_us.data import Bar,Calendar,Session
from paperbot_us.research import ResearchDataset,time_split
from paperbot_us.backtest import ReferencePolicy,CostModel


class StudyTests(unittest.TestCase):
    def setUp(self):
        from paperbot_us.study import run_study,StudyError
        self.run_study,self.Error=run_study,StudyError
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        start=datetime(2026,10,1,13,30,tzinfo=timezone.utc)
        downloaded=start+timedelta(days=11)
        sessions=[Session.from_alpaca({'date':(start+timedelta(days=d)).date().isoformat(),'open':'09:30','close':'16:00'}) for d in range(10)]
        calendar=Calendar(sessions,start=sessions[0].day.isoformat(),end=sessions[-1].day.isoformat())
        bars=[Bar('AAPL',s.opens+timedelta(minutes=15*i),downloaded,100+i,102+i,99+i,101+i,100000)
              for s in sessions for i in range(26)]
        self.dataset=ResearchDataset.from_bars(bars,calendar=calendar,downloaded_at=downloaded,
                                               availability_policy='vendor_history_assumed_at_close')
        self.split=time_split(tuple(s.day for s in sessions),train_sessions=3,validation_sessions=2,test_sessions=3)
        self.policies=(ReferencePolicy(),ReferencePolicy(entry_rsi_max=100,exit_rsi=100))
        self.costs=CostModel(commission_per_share_usd=0.01,slippage_bps=10)
    def run_case(self,policies=None):
        return self.run_study(self.dataset,split=self.split,policies=policies or self.policies,
                              costs=self.costs,directory=self.root)

    def test_precommit_exists_before_validation_or_test_is_evaluated(self):
        from paperbot_us.backtest import backtest as original
        def guarded(dataset,**kwargs):
            if dataset.sessions[0].day>=self.split.validation[0]:
                self.assertTrue(list(self.root.glob('*.locked.json')))
            return original(dataset,**kwargs)
        with patch('paperbot_us.study.backtest',side_effect=guarded): report=self.run_case()
        self.assertTrue(report['selection_locked_before_holdout'])
        self.assertEqual(report['test']['closed_trades'],3)
        self.assertFalse(report['strategy_accepted'])

    def test_identical_study_is_idempotent(self):
        first=self.run_case()
        with patch('paperbot_us.study.backtest',side_effect=AssertionError('unexpected repeated evaluation')):
            second=self.run_case()
        self.assertEqual(first,second)

    def test_changed_parameter_search_cannot_reuse_exposed_holdout_as_new(self):
        self.run_case()
        with self.assertRaises(self.Error): self.run_case((ReferencePolicy(entry_rsi_max=95),))

    def test_changed_split_cannot_reuse_overlapping_exposed_days(self):
        self.run_case()
        changed=time_split(tuple(s.day for s in self.dataset.sessions),train_sessions=3,validation_sessions=3,test_sessions=2)
        with self.assertRaises(self.Error):
            self.run_study(self.dataset,split=changed,policies=(ReferencePolicy(entry_rsi_max=95),),
                           costs=self.costs,directory=self.root)

    def test_rule_subclasses_cannot_share_a_frozen_base_policy_identity(self):
        class DifferentRule(ReferencePolicy):
            def action(self,features,position_open): return 'hold' if position_open else 'buy'
        with self.assertRaises(self.Error): self.run_case((DifferentRule(),))

    def test_study_requires_a_complete_embargo_session_at_both_boundaries(self):
        split=time_split(tuple(s.day for s in self.dataset.sessions),train_sessions=4,
                         validation_sessions=3,test_sessions=3,embargo_sessions=0)
        with self.assertRaises(self.Error):
            self.run_study(self.dataset,split=split,policies=self.policies,costs=self.costs,directory=self.root)


if __name__=='__main__': unittest.main()
