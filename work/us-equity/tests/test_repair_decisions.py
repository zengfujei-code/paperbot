from dataclasses import replace
from datetime import timedelta
import json,unittest
from unittest.mock import patch

import test_repair_execution
from paperbot_us.orders import OrderLedger,LedgerError


class RepairDecisionTests(unittest.TestCase):
    def setUp(self):
        self.f=test_repair_execution.RepairExecutionTests();self.f.setUp();self.addCleanup(self.f.doCleanups)

    def test_replay_requires_the_original_committed_ledger_prefix(self):
        receipt=self.f.record()
        other=OrderLedger(self.f.n.f.f.path.with_name('unrelated-empty-ledger.sqlite'));other.initialize()
        with self.assertRaises(self.f.d.RepairDecisionError):
            self.f.journal.replay(receipt.sequence,ledger=other)

    def test_changed_receipt_cannot_replace_a_committed_decision(self):
        receipt=self.f.record();raw=receipt.read();raw['inputs']['body']['qty']='4'
        forged=replace(receipt,payload=json.dumps(raw))
        with self.assertRaises(LedgerError):self.f.r.m.reserve_repair(ledger=self.f.ledger,plan=self.f.plan,
            book=self.f.f.book,baselines=self.f.f.f.baselines,calendar=self.f.f.f.calendar,limits=self.f.f.limits,
            now=self.f.n.now,decision=forged,journal=self.f.journal)
        self.assertEqual(self.f.r.m.repair_rows(self.f.ledger.load(self.f.r.family)),())

    def test_source_change_preserves_replay_but_cannot_authorize_admission(self):
        receipt=self.f.record();self.f.n.now+=timedelta(seconds=1);self.f.n.reconciler.refresh()
        self.assertTrue(self.f.journal.replay(receipt.sequence,ledger=self.f.ledger).read()['outcome']['allowed'])
        with self.assertRaises(LedgerError):self.f.r.m.reserve_repair(ledger=self.f.ledger,plan=self.f.plan,
            book=self.f.f.book,baselines=self.f.f.f.baselines,calendar=self.f.f.f.calendar,limits=self.f.f.limits,
            now=self.f.n.now,decision=receipt,journal=self.f.journal)

    def test_clock_rollback_records_a_veto_instead_of_an_order_permission(self):
        self.f.n.now-=timedelta(seconds=1);receipt=self.f.record();record=receipt.read()
        self.assertFalse(record['outcome']['allowed'])
        self.assertIn('repair_clock_regression',record['outcome']['vetoes'])
        self.assertEqual(self.f.journal.replay(receipt.sequence,ledger=self.f.ledger).read(),record)

    def test_failed_decision_write_does_not_allocate_any_shares(self):
        with patch.object(self.f.journal.archive,'append',side_effect=OSError('PRIVATE_DISK')):
            with self.assertRaises(self.f.d.RepairDecisionError) as caught:self.f.stage()
        self.assertNotIn('PRIVATE',str(caught.exception));self.assertEqual(self.f.posts,[])
        self.assertEqual(self.f.r.m.repair_rows(self.f.ledger.load(self.f.r.family)),())

    def test_public_decision_summary_contains_no_provider_or_client_identifiers(self):
        receipt=self.f.record();summary=json.dumps(receipt.public_summary())
        for private in ('pb3r-','00000000-0000','PKTT','account_sha256'):
            self.assertNotIn(private,summary)
        self.assertFalse(receipt.public_summary()['execution_enabled'])


if __name__=='__main__':unittest.main()
