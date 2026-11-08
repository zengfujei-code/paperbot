from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
import importlib
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import test_accounting


class AccountBookTests(unittest.TestCase):
    def setUp(self):
        self.m=importlib.import_module('paperbot_us.accountbook')
        self.fixture=test_accounting.AccountingTests();self.fixture.setUp();self.args=self.fixture.inputs()
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'book.sqlite';self.book=self.m.AccountBook(self.path)
        self.book.initialize(self.args['anchor'])

    def refresh(self,**changes):
        return self.book.refresh(**{k:v for k,v in {**self.args,**changes}.items() if k!='anchor'})

    def test_commit_precedes_receipt_and_restart_replays_the_same_profit(self):
        receipt=self.refresh();self.assertTrue(receipt.result.reconciled)
        fresh=self.m.AccountBook(self.path)
        self.assertEqual(fresh.replay(receipt.sequence),receipt.result)
        self.assertEqual(fresh.verify()['snapshots'],1);self.assertEqual(fresh.verify()['activities'],3)

    def test_repeated_input_is_idempotent_and_anchor_cannot_be_reset(self):
        first=self.refresh();self.assertEqual(self.refresh(),first)
        self.assertEqual(self.book.verify()['snapshots'],1)
        with self.assertRaises(self.m.BookError): self.book.initialize(replace(self.args['anchor'],evidence_sha256='9'*64))

    def test_existing_anchor_is_read_from_verified_history_without_creating_a_database(self):
        self.assertTrue(hasattr(self.book,'read_anchor'),'verified runtime anchor read required')
        self.assertEqual(self.book.read_anchor(),self.args['anchor'])
        self.assertEqual(self.book.verify()['snapshots'],0)
        with self.assertRaises(self.m.BookError): self.m.AccountBook(self.path.with_name('absent.sqlite')).read_anchor()

    def test_late_fill_recovers_but_preserves_the_old_unreconciled_snapshot(self):
        window=self.args['window'];partial=replace(window,activities=tuple(self.fixture.fills[1:]+[self.fixture.fee]))
        first=self.refresh(window=partial);self.assertFalse(first.result.reconciled)
        second=self.refresh(now=self.args['now']+timedelta(seconds=1))
        self.assertTrue(second.result.reconciled);self.assertIsNone(self.book.replay(first.sequence).profit_usd)
        self.assertEqual(self.book.replay(second.sequence).profit_usd,self.args['account'].equity_usd-10000)

    def test_missing_or_conflicting_previously_seen_event_fails_without_rewriting_history(self):
        self.refresh();window=self.args['window']
        for activities in (window.activities[1:],(replace(window.activities[0],price_usd='99'),*window.activities[1:])):
            with self.subTest(activities=activities),self.assertRaises(self.m.BookError): self.refresh(window=replace(window,activities=activities))
        self.assertEqual(self.book.verify()['snapshots'],1)

    def test_disk_failure_rolls_back_activity_inserts_before_any_receipt(self):
        with patch.object(self.book,'_append_snapshot',side_effect=OSError('PRIVATE_DISK')):
            with self.assertRaises(self.m.BookError) as caught: self.refresh()
        self.assertNotIn('PRIVATE',str(caught.exception));self.assertEqual(self.book.verify()['activities'],0)
        self.assertEqual(self.book.verify()['snapshots'],0);self.assertTrue(self.refresh().result.reconciled)

    def test_corruption_or_missing_database_is_not_normal_empty_accounting(self):
        self.refresh()
        with closing(sqlite3.connect(self.path)) as db: db.execute("UPDATE activities SET payload='{}' WHERE sequence=1");db.commit()
        with self.assertRaises(self.m.BookError): self.book.verify()
        absent=Path(self.temp.name)/'missing.sqlite'
        with self.assertRaises(self.m.BookError): self.m.AccountBook(absent).verify()
        self.assertFalse(absent.exists())

    def test_valid_exponential_decimal_initial_cash_is_replayable_without_changing_its_value(self):
        anchor=replace(self.args['anchor'],cash_usd=Decimal('1E+4'))
        book=self.m.AccountBook(self.path.with_name('exponent.sqlite'));book.initialize(anchor)
        self.assertEqual(book.read_anchor(),anchor)

    def test_incomplete_window_and_source_regression_never_replace_last_good_snapshot(self):
        self.refresh();window=self.args['window']
        for changed in (replace(window,complete=False),replace(window,until=window.until-timedelta(seconds=1))):
            with self.subTest(changed=changed),self.assertRaises(self.m.BookError): self.refresh(window=changed)
        self.assertEqual(self.book.verify()['snapshots'],1)

    def test_replay_requires_the_recorded_code_version(self):
        receipt=self.refresh()
        with patch.object(self.m,'code_hash',return_value='1'*64):
            with self.assertRaises(self.m.BookError): self.book.replay(receipt.sequence)

    def test_digest_lookup_returns_the_exact_historical_receipt_and_refuses_missing_identity(self):
        self.assertTrue(hasattr(self.book,'snapshot_by_digest'),'decision/account continuity requires exact digest lookup')
        first=self.refresh()
        self.refresh(now=self.args['now']+timedelta(seconds=1))
        snapshot=self.book.snapshot_by_digest(first.digest)
        self.assertEqual(snapshot.receipt,first)
        self.assertEqual(snapshot.anchor,self.args['anchor'])
        with self.assertRaises(self.m.BookError):self.book.snapshot_by_digest('f'*64)


if __name__=='__main__': unittest.main()
