"""Synthetic future policy is committed through real registry BEFORE issuance."""
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from issue_fixture import IssueWorld,Tick,TZ
from test_canonical_history import HANDOFF
from paperbot_r2 import common_registration as common,validation_issue_journal as issuer
from paperbot_r2 import validation_scores,validation_settlements


class ComparisonWorld:
    def __init__(self,directory):
        self.directory=Path(directory);self.policy=validation_scores.make_policy('f'*64)
        original=common.build_common_plan
        def configured(*args,**kwargs):
            kwargs['configuration_sha256']=self.policy.configuration_sha256
            return original(*args,**kwargs)
        with patch.object(common,'build_common_plan',side_effect=configured):
            self.w=IssueWorld(self.directory,issue_source_sha256=issuer.source_sha256())
        self.issues=issuer.ValidationIssueJournal(self.directory/('r2-validation-'+self.w.identity+'.sqlite'))
        self.issues.initialize(self.w.registry,self.w.identity,self.w.training,self.w.source_pairs(),
            timezone_file=TZ,clock=Tick(HANDOFF+timedelta(minutes=1)))
        self.settlements=validation_settlements.ValidationSettlementJournal(self.directory/('r2-validation-settlements-'+self.w.identity+'.sqlite'))
        self.settlements.initialize(self.issues,self.policy,timezone_file=TZ,clock=Tick(HANDOFF+timedelta(minutes=2)))
        self.key=None;self.clock=None

    def issue(self,day='2026-11-24',count=9):
        self.key=self.w.receive_until(day,count);self.clock=Tick(self.w.at)
        self.issues.issue_once(self.key,registry=self.w.registry,training_journal=self.w.training,pairs=self.w.source_pairs(),
            timezone_file=TZ,clock=self.clock)
        return self.key

    def outcome(self,day='2026-11-24',count=11):
        self.w.receive_until(day,count,derive=False);self.clock=Tick(self.w.at)

    def settle(self):return self.settlements.settle_once(self.key,timezone_file=TZ,clock=self.clock)
