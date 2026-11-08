from datetime import datetime,timedelta,timezone
import importlib,unittest


class TemporalValidationTests(unittest.TestCase):
    def setUp(self):
        self.m=importlib.import_module('paperbot_us.temporal_validation')
        self.t=datetime(2026,10,23,15,tzinfo=timezone.utc)

    def label(self,n,start,end,available=None):
        return self.m.LabelInterval(f'{n:064x}',self.t+timedelta(minutes=start),
            self.t+timedelta(minutes=end),self.t+timedelta(minutes=end if available is None else available))

    def test_overlapping_label_boundary_and_embargo_are_purged(self):
        labels=(self.label(0,0,5),self.label(1,5,10),self.label(2,10,20),
            self.label(3,21,25),self.label(4,26,30),self.label(5,31,35))
        r=self.m.purged_training(labels,(0,1,3,4,5),(2,),cutoff=self.t+timedelta(hours=1),
            embargo=timedelta(minutes=10))
        self.assertEqual(r,(0,5))

    def test_late_maturing_label_cannot_enter_training_despite_old_feature_date(self):
        labels=(self.label(0,0,1,available=30),self.label(1,2,3),self.label(2,10,15))
        r=self.m.purged_training(labels,(0,1),(2,),cutoff=self.t+timedelta(minutes=5),embargo=timedelta(0))
        self.assertEqual(r,(1,))

    def test_multiple_overlapping_test_horizons_are_merged_without_losing_end_boundary(self):
        labels=(self.label(0,0,1),self.label(1,2,5),self.label(2,4,10),self.label(3,12,14),self.label(4,15,16))
        r=self.m.purged_training(labels,(0,3,4),(1,2),cutoff=self.t+timedelta(hours=1),embargo=timedelta(minutes=2))
        self.assertEqual(r,(0,4))

    def test_indices_and_identity_contract_rejects_duplicates_and_overlap(self):
        labels=(self.label(0,0,1),self.label(1,5,6))
        for candidates,test in (((0,0),(1,)),((0,),(0,)),((True,),(0,)),((8,),(1,))):
            with self.subTest(candidates=candidates),self.assertRaises(self.m.ValidationError):
                self.m.purged_training(labels,candidates,test,cutoff=self.t+timedelta(hours=1),embargo=timedelta(0))
        with self.assertRaises(self.m.ValidationError):self.label(0,3,2)
        with self.assertRaises(self.m.ValidationError):self.label(0,0,3,available=2)

    def test_walk_forward_fixed_windows_have_two_explicit_gaps_and_disjoint_tests(self):
        folds=self.m.walk_forward(16,train=6,validation=2,test=2,gap=1,step=2)
        self.assertEqual(len(folds),3)
        self.assertEqual(folds[0].train,tuple(range(6)))
        self.assertEqual(folds[0].validation,(7,8));self.assertEqual(folds[0].test,(10,11))
        self.assertEqual(folds[0].excluded,(6,9));self.assertEqual(folds[1].train,tuple(range(2,8)))
        all_tests=[i for f in folds for i in f.test];self.assertEqual(len(all_tests),len(set(all_tests)))

    def test_expanding_windows_keep_initial_training_prefix_without_looking_forward(self):
        folds=self.m.walk_forward(16,train=6,validation=2,test=2,gap=1,step=2,expanding=True)
        self.assertEqual(folds[1].train,tuple(range(8)))
        for f in folds:
            self.assertLess(max(f.train),min(f.validation));self.assertLess(max(f.validation),min(f.test))

    def test_fold_budget_and_overlapping_test_windows_are_rejected(self):
        for changes in (dict(step=1),dict(train=0),dict(length=True),dict(gap=-1)):
            with self.subTest(changes=changes),self.assertRaises(self.m.ValidationError):
                self.m.walk_forward(**{**dict(length=16,train=6,validation=2,test=2,gap=1,step=2),**changes})
        with self.assertRaises(self.m.ValidationError):self.m.walk_forward(100000,train=2,validation=1,test=1,gap=0,step=1)

    def test_timestamp_offsets_are_normalized_without_confusing_dst_repeated_hour(self):
        a=self.m.LabelInterval('a'*64,'2026-11-01T01:00:00-04:00','2026-11-01T01:30:00-04:00','2026-11-01T01:30:01-04:00')
        b=self.m.LabelInterval('b'*64,'2026-11-01T01:00:00-05:00','2026-11-01T01:30:00-05:00','2026-11-01T01:30:01-05:00')
        self.assertLess(a.ends_at,b.starts_at)
        r=self.m.purged_training((a,b),(0,),(1,),cutoff=b.available_at,embargo=timedelta(0))
        self.assertEqual(r,(0,))


if __name__=='__main__':unittest.main()
