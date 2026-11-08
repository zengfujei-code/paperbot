from contextlib import closing
from datetime import timedelta
import sqlite3
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_common_outcome_journal as fixtures
from paperbot_r2.source_views import VerifiedFeatureView, SourceViewError


class FeatureViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixtures.OutcomeJournalTests.setUpClass.__func__(cls)
    @classmethod
    def tearDownClass(cls):fixtures.OutcomeJournalTests.tearDownClass.__func__(cls)
    def setUp(self):fixtures.OutcomeJournalTests.setUp(self)
    bar=fixtures.OutcomeJournalTests.bar
    receive=fixtures.OutcomeJournalTests.receive
    ready_feature=fixtures.OutcomeJournalTests.ready_feature

    def test_verified_snapshot_is_frozen_and_reused_only_while_source_unchanged(self):
        self.ready_feature()
        original=self.features.snapshot
        with patch.object(self.features,'snapshot',wraps=original) as read:
            with VerifiedFeatureView(self.features,read_guard=lambda:None) as view:
                first=view.snapshot(deep=True)
                second=view.snapshot(deep=True)
                self.assertIs(first,second)
                self.assertEqual(read.call_count,1)
                with self.assertRaises(TypeError):first['entries']['unexpected']={}

    def test_old_prefix_update_invalidates_cached_snapshot_even_with_same_head(self):
        self.ready_feature()
        with VerifiedFeatureView(self.features,read_guard=lambda:None) as view:
            with closing(sqlite3.connect(self.features.path)) as db:
                db.execute('UPDATE observations SET digest=? WHERE seq=1',('0'*64,));db.commit()
            with self.assertRaises(SourceViewError):view.snapshot()

    def test_same_inode_rollback_is_not_silently_used(self):
        original=self.features.path.read_bytes()
        self.ready_feature()
        with VerifiedFeatureView(self.features,read_guard=lambda:None) as view:
            self.features.path.write_bytes(original)
            with self.assertRaises(SourceViewError):view.snapshot()

    def test_guard_refusal_happens_before_connection_or_deep_read(self):
        def refuse():raise ValueError('reserved date')
        with patch.object(self.features,'snapshot',side_effect=AssertionError('read')) as read:
            with self.assertRaises(SourceViewError):VerifiedFeatureView(self.features,read_guard=refuse)
            self.assertEqual(read.call_count,0)

    def test_closed_view_cannot_supply_old_snapshot(self):
        view=VerifiedFeatureView(self.features,read_guard=lambda:None)
        view.close()
        with self.assertRaises(SourceViewError):view.snapshot()


if __name__=='__main__':unittest.main()
