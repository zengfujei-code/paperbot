from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
import hashlib,importlib,json,sqlite3,tempfile,unittest
from pathlib import Path
from unittest.mock import patch


class ExperimentRegistryTests(unittest.TestCase):
    def setUp(self):
        self.m=importlib.import_module('paperbot_us.experiments')
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.registry=self.m.ExperimentRegistry(self.root/'experiments.sqlite')
        self.registry.initialize();self.a='1'*64;self.b='2'*64

    def plan(self,**changes):
        value=dict(schema=1,domain='us_equity',family='ewma_cusum',data_sha256='a'*64,
            code_sha256='b'*64,configuration_sha256='c'*64,
            split=dict(train=['2026-10-01'],validation=['2026-10-03'],test=['2026-10-05'],
                       context=['2026-10-02','2026-10-04']),
            candidate_ids=[self.a,self.b],trial_budget=2,selection_stage='validation',objective='variance_mse_min')
        value.update(changes);return value

    def trial(self,identity,candidate,phase,score=1.,failure=None):
        self.registry.start_trial(identity,candidate,phase)
        self.registry.finish_trial(identity,candidate,phase,
            metrics={} if failure else dict(variance_mse=score),failure=failure)

    def ready(self):
        identity=self.registry.register(self.plan())
        for c,score in ((self.a,2.),(self.b,1.)):
            self.trial(identity,c,'train');self.trial(identity,c,'validation',score)
        return identity

    def test_reservation_is_durable_idempotent_and_hashes_the_full_plan(self):
        plan=self.plan();identity=self.registry.register(plan)
        expected=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.assertEqual(identity,expected);self.assertEqual(identity,self.registry.register(deepcopy(plan)))
        other=self.m.ExperimentRegistry(self.registry.path)
        self.assertEqual(other.load(identity)['plan'],plan)

    def test_other_family_or_revision_cannot_reclaim_previously_reserved_days(self):
        self.registry.register(self.plan())
        for changes in (dict(family='hmm'),dict(data_sha256='d'*64),dict(code_sha256='e'*64)):
            with self.subTest(changes=changes),self.assertRaises(self.m.ExperimentError):
                self.registry.register(self.plan(**changes))

    def test_old_test_can_be_future_training_but_not_a_new_holdout(self):
        self.registry.register(self.plan())
        new=self.plan(split=dict(train=['2026-10-05'],validation=['2026-10-07'],test=['2026-10-09'],context=[]))
        self.assertTrue(self.registry.register(new))

    def test_failed_and_interrupted_attempts_cannot_be_retried(self):
        identity=self.registry.register(self.plan())
        self.trial(identity,self.a,'train',failure='invalid_input')
        self.registry.start_trial(identity,self.b,'train')
        for c in (self.a,self.b):
            with self.assertRaises(self.m.ExperimentError):self.registry.start_trial(identity,c,'train')
        state=self.registry.load(identity)
        self.assertEqual(state['trials'][self.a+':train']['status'],'failed')
        self.assertEqual(state['trials'][self.b+':train']['status'],'started')

    def test_phase_order_and_selection_reject_test_and_wrong_winner(self):
        identity=self.registry.register(self.plan())
        with self.assertRaises(self.m.ExperimentError):self.registry.start_trial(identity,self.a,'test')
        with self.assertRaises(self.m.ExperimentError):self.registry.start_trial(identity,self.a,'validation')
        for c,score in ((self.a,2.),(self.b,1.)):
            self.trial(identity,c,'train');self.trial(identity,c,'validation',score)
        with self.assertRaises(self.m.ExperimentError):self.registry.select(identity,self.a)
        self.registry.select(identity,self.b)
        with self.assertRaises(self.m.ExperimentError):self.registry.start_trial(identity,self.a,'test')
        self.trial(identity,self.b,'test',3.)
        self.registry.finish(identity,report_sha256='f'*64,completed=True)
        self.assertEqual(self.registry.load(identity)['report_sha256'],'f'*64)

    def test_selection_waits_for_all_candidates_and_accounts_for_failed_training(self):
        identity=self.registry.register(self.plan());self.trial(identity,self.a,'train')
        self.trial(identity,self.a,'validation',3.)
        with self.assertRaises(self.m.ExperimentError):self.registry.select(identity,self.a)
        self.trial(identity,self.b,'train',failure='numeric_model_failure')
        self.registry.select(identity,self.a)

    def test_legacy_exposure_and_late_discovery_remain_binding(self):
        self.registry.import_exposure(domain='us_equity',days=['2026-10-03'],source_sha256='d'*64)
        with self.assertRaises(self.m.ExperimentError):self.registry.register(self.plan())
        identity=self.registry.register(self.plan(domain='synthetic'))
        self.registry.import_exposure(domain='synthetic',days=['2026-10-05'],source_sha256='e'*64)
        self.assertTrue(self.registry.load(identity)['invalidated'])
        with self.assertRaises(self.m.ExperimentError):self.registry.start_trial(identity,self.a,'train')

    def test_invalid_or_unbounded_metadata_and_secret_like_errors_are_rejected(self):
        for plan in (self.plan(trial_budget=1),self.plan(candidate_ids=[self.a,self.a]),
                     self.plan(family='unknown'),self.plan(objective='profit_probability'),
                     self.plan(split=dict(train=['2026-10-05'],validation=['2026-10-03'],test=['2026-10-01'],context=[]))):
            with self.assertRaises(self.m.ExperimentError):self.registry.register(plan)
        identity=self.registry.register(self.plan());self.registry.start_trial(identity,self.a,'train')
        for kwargs in (dict(metrics={'variance_mse':float('nan')}),dict(metrics={},failure='raw arbitrary error text')):
            with self.assertRaises(self.m.ExperimentError):self.registry.finish_trial(identity,self.a,'train',**kwargs)

    def test_corrupt_or_truncated_chain_is_rejected(self):
        identity=self.ready()
        with closing(sqlite3.connect(self.registry.path)) as db:
            with db:db.execute("UPDATE events SET digest=? WHERE seq=2",('0'*64,))
        with self.assertRaises(self.m.ExperimentError):self.registry.load(identity)

    def test_deleted_tail_cannot_disagree_with_retained_head(self):
        identity=self.ready()
        with closing(sqlite3.connect(self.registry.path)) as db:
            with db:db.execute('DELETE FROM events WHERE seq=(SELECT MAX(seq) FROM events)')
        with self.assertRaises(self.m.ExperimentError):self.registry.load(identity)

    def test_transaction_serializes_conflicting_reservations(self):
        def reserve(family):
            try:return self.registry.register(self.plan(family=family))
            except self.m.ExperimentError:return None
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(reserve,('hmm','garch')))
        self.assertEqual(sum(value is not None for value in results),1)

    def test_read_uses_one_snapshot_when_an_unrelated_commit_precedes_head_query(self):
        identity=self.registry.register(self.plan())
        with closing(sqlite3.connect(self.registry.path)) as db:db.execute('PRAGMA journal_mode=WAL')
        registry=self.registry;original=sqlite3.connect
        class Interleaved(sqlite3.Connection):
            def execute(self,sql,*args,**kwargs):
                if sql=='SELECT seq,digest FROM head' and not getattr(self,'injected',False):
                    self.injected=True
                    registry.import_exposure(domain='synthetic',days=['2026-10-03'],source_sha256='e'*64)
                return super().execute(sql,*args,**kwargs)
        def connect(path,*args,**kwargs):
            if 'mode=ro' in str(path):kwargs['factory']=Interleaved
            return original(path,*args,**kwargs)
        with patch('paperbot_us.experiments.sqlite3.connect',side_effect=connect):
            self.assertEqual(self.registry.load(identity)['plan'],self.plan())


if __name__=='__main__':unittest.main()
