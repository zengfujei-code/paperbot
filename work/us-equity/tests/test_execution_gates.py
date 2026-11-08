from copy import deepcopy
from datetime import timedelta
import hashlib,importlib,importlib.util,json,tempfile,unittest
from pathlib import Path

from paperbot_us.execution_policy import ExecutionPolicy,PolicySnapshot
from paperbot_us.orders import _hash,_json,_base
from paperbot_us.risk import RiskLimits
import test_orders


class AcceptanceGateTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.execution_gates'),
            'execution needs verified paper-only acceptance content, not a boolean reference')
        self.m=importlib.import_module('paperbot_us.execution_gates')
        self.f=test_orders.OrderLedgerTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.root=self.f.path.parent/'acceptance';self.root.mkdir();self.now=self.f.now
        self.limits=RiskLimits();self.strategy='1'*64;self.dataset='2'*64
        self.metrics={
            'security':dict(secret_findings=0,unauthed_write_paths=0),
            'data':dict(observations=1000,invalid_timestamps=0,future_rows=0,unresolved_actions=0,
                missing_calendar_rows=0,unlicensed_rows=0,survivorship_unresolved=0),
            'quant':dict(cases=20,failures=0,units_reviewed=True),
            'backtest':dict(leaked_rows=0,unpriced_costs=0,omitted_fill_constraints=0,calibrated_costs=True),
            'decision':dict(cases=20,untraced_decisions=0,unsafe_fallbacks=0),
            'risk':dict(cases=20,failed_limits=0,duplicate_mutations=0,kill_switch_failures=0),
            'paper_lifecycle':dict(provider_mode='paper',real_provider=True,completed_order_cycles=1,
                unexplained_cash_cents=0,unexplained_shares=0,live_requests=0,unresolved_orders=0),
            'evaluation':dict(train_end=(self.now-timedelta(days=30)).isoformat(),
                validation_start=(self.now-timedelta(days=29)).isoformat(),validation_end=(self.now-timedelta(days=20)).isoformat(),
                test_start=(self.now-timedelta(days=19)).isoformat(),test_end=(self.now-timedelta(days=1)).isoformat(),
                test_closed_trades=40,walk_forward_folds=4,net_return_fraction=.01,maximum_drawdown_fraction=.01,
                daily_volatility_fraction=.01,turnover_fraction=2,total_cost_usd=5,multiple_testing_corrected=True),
            'reproducibility':dict(cases=20,failures=0,missing_inputs=0)}
        self.value=dict(schema=1,kind='paper_strategy_acceptance',mode='paper',capital_usd=10000,
            approved_at=self.now.isoformat(),expires_at=(self.now+timedelta(days=2)).isoformat(),
            strategy_sha256=self.strategy,dataset_sha256=self.dataset,risk_limits_sha256=_hash(_json(_base(self.limits,RiskLimits))),
            symbols=['SPY'],criteria=dict(registered_at=(self.now-timedelta(days=40)).isoformat(),
                min_test_closed_trades=30,min_walk_forward_folds=3,min_net_return_fraction=0,max_drawdown_fraction=.05),
            evidence={})
        self.save()

    def save(self):
        self.value['evidence']={}
        for name,metrics in self.metrics.items():
            report=dict(schema=1,kind=name,mode='paper',strategy_sha256=self.strategy,dataset_sha256=self.dataset,metrics=metrics)
            raw=_json(report).encode();path=self.root/(name+'.json');path.write_bytes(raw)
            self.value['evidence'][name]=dict(path=path.name,sha256=hashlib.sha256(raw).hexdigest())
        self.path=self.root/'acceptance.json';raw=_json(self.value).encode();self.path.write_bytes(raw)
        self.digest=hashlib.sha256(raw).hexdigest()
        self.policy=PolicySnapshot(ExecutionPolicy(1,10000,True,self.strategy,self.digest,
            self.value['risk_limits_sha256'],5,True),'3'*64)
        self.gate=self.m.PaperAcceptanceFile(self.path,self.digest)

    def verify(self,**changes):
        return self.gate.verify(**{**dict(policy=self.policy,limits=self.limits,strategy_sha256=self.strategy,now=self.now),**changes})

    def test_complete_pinned_paper_reports_pass_and_disclose_only_safe_scope(self):
        r=self.verify();self.assertEqual(r.strategy_sha256,self.strategy);self.assertEqual(r.symbols,('SPY',))
        self.assertFalse(r.public_summary()['live_authorized'])

    def test_reference_boolean_without_nine_reports_never_passes(self):
        self.value=dict(strategy_accepted=True,acceptance_status='engineering_reference_only')
        raw=_json(self.value).encode();self.path.write_bytes(raw)
        self.gate=self.m.PaperAcceptanceFile(self.path,hashlib.sha256(raw).hexdigest())
        with self.assertRaises(self.m.GateError):self.verify()

    def test_changed_report_or_manifest_is_rejected(self):
        (self.root/'risk.json').write_text('{}')
        with self.assertRaises(self.m.GateError):self.verify()

    def test_live_scope_wrong_strategy_or_limits_are_rejected(self):
        with self.assertRaises(self.m.GateError):self.verify(strategy_sha256='f'*64)
        self.value['mode']='live';self.save()
        with self.assertRaises(self.m.GateError):self.verify()

    def test_expired_or_not_yet_approved_report_cannot_enable_entries(self):
        for delta in (-1,3*86400):
            with self.subTest(delta=delta),self.assertRaises(self.m.GateError):
                self.verify(now=self.now+timedelta(seconds=delta))

    def test_zero_trades_or_unmet_folds_fail_predeclared_performance_criteria(self):
        self.metrics['evaluation']['test_closed_trades']=0;self.save()
        with self.assertRaises(self.m.GateError):self.verify()

    def test_training_overlap_or_posthoc_criteria_fail(self):
        self.metrics['evaluation']['train_end']=self.now.isoformat();self.save()
        with self.assertRaises(self.m.GateError):self.verify()

    def test_unlicensed_data_and_uncalibrated_costs_fail(self):
        self.metrics['data']['unlicensed_rows']=1;self.save()
        with self.assertRaises(self.m.GateError):self.verify()

    def test_simulated_provider_report_is_not_actual_paper_lifecycle(self):
        self.metrics['paper_lifecycle']['real_provider']=False;self.save()
        with self.assertRaises(self.m.GateError):self.verify()

    def test_path_traversal_is_rejected_before_reading_other_files(self):
        self.value['evidence']['risk']['path']='../outside.json'
        raw=_json(self.value).encode();self.path.write_bytes(raw);self.digest=hashlib.sha256(raw).hexdigest()
        self.gate=self.m.PaperAcceptanceFile(self.path,self.digest)
        with self.assertRaises(self.m.GateError):self.verify()


if __name__=='__main__':unittest.main()


class ContinuityGateTests(unittest.TestCase):
    def setUp(self):
        import test_book_continuity
        from paperbot_us import execution_gates as m
        self.assertTrue(hasattr(m,'ExecutionStateGate'),'the coordinator must replay pinned original account continuity')
        self.m=m;self.c=test_book_continuity.BookContinuityTests();self.c.setUp();self.addCleanup(self.c.doCleanups)
        self.proof=self.c.carry();self.book=self.c.new_book;self.ledger=self.c.new_ledger
        from paperbot_us.native_reconciler import NativePaperReconciler
        from paperbot_us.risk_baselines import RiskBaselineStore
        from paperbot_us.decision_journal import DecisionJournal
        from paperbot_us.data import Calendar
        from paperbot_us.broker import HttpReply
        from urllib.parse import urlsplit
        original=self.c.f.broker._transport
        def transport(method,url,headers,body):
            if urlsplit(url).path=='/v2/orders':return HttpReply(200,[])
            return original(method,url,headers,body)
        self.c.f.broker._transport=transport
        self.calendar=Calendar([],start='2026-10-24',end='2026-10-24')
        self.reconciler=NativePaperReconciler(broker=self.c.f.broker,ledger=self.ledger,book=self.book,
            journal=self.book.path.with_name('current-reconciliation.sqlite'),calendar=self.calendar,clock=lambda:self.c.f.now)
        cycle=self.reconciler.refresh()
        self.baselines=RiskBaselineStore(self.book.path.with_name('risk.sqlite'))
        self.baselines.initialize(book=self.book,cycle=cycle,now=self.c.f.now);self.seed=self.baselines.replay()
        self.decisions=DecisionJournal(self.book.path.with_name('decisions.sqlite'),clock=lambda:self.c.f.now);self.decisions.initialize()
        self.paths=dict(book=self.book.path,ledger=self.ledger.path,baselines=self.baselines.path,decisions=self.decisions.path)
        self.gate=m.ExecutionStateGate(replay=self.c.replay,continuity_journal=self.c.journal,
            expected_proof_sha256=self.proof.proof_sha256,expected_completion_sha256=self.proof.audit_sha256,
            baseline_seed_sha256=self.seed.audit_sha256,paths=self.paths)

    def verify(self,**changes):
        return self.gate.verify(**{**dict(book=self.book,ledger=self.ledger,baselines=self.baselines,
            decisions=self.decisions,now=self.c.f.now),**changes})

    def test_original_continuity_and_baseline_seed_are_replayed_without_network_calls(self):
        before=len(self.c.f.calls);r=self.verify()
        self.assertEqual(r.proof_sha256,self.proof.proof_sha256);self.assertEqual(len(self.c.f.calls),before)

    def test_other_book_or_order_path_cannot_substitute_for_the_registered_state(self):
        from paperbot_us.orders import OrderLedger
        other=OrderLedger(self.book.path.with_name('different-orders.sqlite'));other.initialize()
        with self.assertRaises(self.m.GateError):self.verify(ledger=other)

    def test_changed_original_source_fails_instead_of_resetting_any_state(self):
        p=self.c.release/'paperbot_us'/'accountbook.py';p.write_bytes(p.read_bytes()+b'\n')
        with self.assertRaises(self.m.GateError):self.verify()

    def test_wrong_baseline_seed_cannot_be_used_to_reset_the_high_water_history(self):
        self.gate=self.m.ExecutionStateGate(replay=self.c.replay,continuity_journal=self.c.journal,
            expected_proof_sha256=self.proof.proof_sha256,expected_completion_sha256=self.proof.audit_sha256,
            baseline_seed_sha256='f'*64,paths=self.paths)
        with self.assertRaises(self.m.GateError):self.verify()
