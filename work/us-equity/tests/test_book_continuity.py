from contextlib import closing
from datetime import timedelta
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

import test_bootstrap
from paperbot_us.accountbook import AccountBook
from paperbot_us.accounting import AccountSnapshot, ActivityWindow
from paperbot_us.monitor import ObservationArchive
from paperbot_us.orders import OrderLedger
from paperbot_us.replay_bridge import PinnedRelease


class BookContinuityTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.book_continuity'),
                             'explicit original-book continuity is required')
        self.m = importlib.import_module('paperbot_us.book_continuity')
        self.r = importlib.import_module('paperbot_us.book_replay')
        self.f = test_bootstrap.BootstrapTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.f.run_bootstrap(); self.anchor = self.f.book.read_anchor()
        for _ in range(2):
            self.f.now += timedelta(seconds=1)
            self.f.book.refresh(window=ActivityWindow(self.anchor.as_of, self.f.now, self.f.now, ()),
                orders=(), account=AccountSnapshot(self.f.now, 10000, 10000, False, (), self.anchor.account_sha256),
                now=self.f.now)
        self.release = self.f.root / 'original'; package = self.release / 'paperbot_us'
        package.mkdir(parents=True); files = {}
        for path in Path(self.m.__file__).parent.glob('*.py'):
            raw = path.read_bytes(); (package / path.name).write_bytes(raw)
            files['paperbot_us/' + path.name] = hashlib.sha256(raw).hexdigest()
        self.pinned = PinnedRelease(self.release, files=files, package_sha256='a' * 64)
        self.replay = self.r.BookReplay(self.pinned, book=self.f.book.path,
            baseline=self.f.journal, ledger=self.f.ledger.path,
            timezone_root=os.environ.get('PYTHONTZPATH') or None)
        self.new_book = AccountBook(self.f.root / 'next/book.sqlite')
        self.new_ledger = OrderLedger(self.f.root / 'next/orders.sqlite'); self.new_ledger.initialize()
        self.journal = self.f.root / 'next/continuity.sqlite'
        self.f.calls.clear(); self.f.account_reads = self.f.position_reads = self.f.history_reads = 0
        self.f.now += timedelta(seconds=2)

    def carry(self):
        return self.m.carry_forward_empty_book(replay=self.replay, broker=self.f.broker,
            ledger=self.new_ledger, book=self.new_book, journal=self.journal, clock=lambda: self.f.now)

    def test_original_source_replays_all_empty_snapshots_even_if_current_version_differs(self):
        from paperbot_us import accountbook
        paths = [self.f.book.path, self.f.journal, self.f.ledger.path, *sorted((self.release/'paperbot_us').glob('*.py'))]
        before = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        with patch.object(accountbook, 'code_hash', return_value='f'*64):
            with self.assertRaises(Exception): self.f.book.snapshot()
            proof = self.replay.replay().read()
        self.assertEqual(proof['kind'], 'pristine_book')
        self.assertEqual(proof['book']['snapshots'], 2)
        self.assertEqual(proof['anchor']['as_of'], self.anchor.as_of.isoformat())
        self.assertEqual(proof['book']['activities'], 0)
        self.assertEqual(proof['summary']['high_water_usd'], '10000')
        self.assertEqual(before, {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        self.assertEqual(list(self.release.rglob('__pycache__')), [])

    def test_continuity_keeps_the_original_anchor_and_saves_witness_before_new_book(self):
        initialize = self.new_book.initialize
        def checked(anchor):
            self.assertEqual(anchor, self.anchor)
            self.assertEqual(ObservationArchive(self.journal).verify()['records'], 1)
            initialize(anchor)
        with patch.object(self.new_book, 'initialize', side_effect=checked): receipt = self.carry()
        self.assertEqual(self.new_book.read_anchor(), self.anchor)
        self.assertEqual(self.new_book.snapshot().activity_until.date(), self.f.now.date())
        self.assertTrue(receipt.book_receipt.result.reconciled)
        self.assertEqual(receipt.book_receipt.result.profit_usd, 0)
        self.assertFalse(receipt.public_summary()['execution_enabled'])
        self.assertIsNone(receipt.public_summary()['paper_started_at'])
        self.assertEqual(ObservationArchive(self.journal).verify()['records'], 2)
        self.assertEqual(self.f.account_reads, 2)
        self.assertEqual(self.f.history_reads, 2)
        self.assertEqual(self.f.position_reads, 2)
        self.assertNotIn(self.f.account['id'], json.dumps(receipt.public_summary()))
        self.assertNotIn(self.anchor.account_sha256, json.dumps(receipt.public_summary()))

    def test_existing_or_partial_targets_refuse_without_any_new_provider_read(self):
        self.carry(); original = self.new_book.path.read_bytes(); self.f.calls.clear()
        with self.assertRaises(self.m.ContinuityError): self.carry()
        self.assertEqual(self.f.calls, [])
        self.assertEqual(self.new_book.path.read_bytes(), original)

    def test_old_source_corruption_wal_unknown_fields_or_forged_original_proof_fail_closed(self):
        for target, statement in (
            (self.f.book.path, "UPDATE snapshots SET digest='bad' WHERE sequence=1"),
            (self.f.journal, "UPDATE observations SET digest='bad' WHERE seq=1"),
            (self.f.book.path, 'PRAGMA journal_mode=WAL'),
        ):
            with self.subTest(statement=statement):
                raw = target.read_bytes()
                with closing(sqlite3.connect(target)) as db: db.execute(statement); db.commit()
                with self.assertRaises(self.r.BookReplayError): self.replay.replay()
                target.write_bytes(raw)
        path = self.release/'paperbot_us/accounting.py'; path.write_bytes(path.read_bytes()+b'\n# changed\n')
        with self.assertRaises(self.m.ContinuityError): self.carry()
        self.assertEqual(self.f.calls, [])

    def test_nonempty_original_ledger_is_never_discarded_even_if_book_is_empty(self):
        from paperbot_us import orders
        with closing(sqlite3.connect(self.f.ledger.path)) as db:
            db.execute("INSERT INTO orders VALUES('x','x','reserved','2026-10-24T00:00:00+00:00','{}','x')"); db.commit()
        with self.assertRaises(self.r.BookReplayError): self.replay.replay()

    def test_changed_identity_nonempty_current_account_or_any_prior_order_refuses(self):
        for change in ({'cash':'9999'}, {'equity':'10001'}, {'trading_blocked':True},
                       {'id':'00000000-0000-4000-8000-000000000099'}):
            with self.subTest(change=change):
                self.f.account_after=change; self.f.account_reads=0
                with self.assertRaises(self.m.ContinuityError): self.carry()
                self.assertFalse(self.new_book.path.exists()); self.assertFalse(self.journal.exists())
        self.f.account_after={}; self.f.history_after=True; self.f.history_reads=0
        with self.assertRaises(self.m.ContinuityError): self.carry()
        self.assertFalse(self.new_book.path.exists())

    def test_activity_since_original_anchor_and_failed_reads_refuse_without_new_book(self):
        self.f.activity=[dict(id='20261024020000000::00000000-0000-4000-8000-000000000020', activity_type='CSD')]
        with self.assertRaises(self.m.ContinuityError): self.carry()
        self.assertFalse(self.journal.exists())
        self.f.activity=[]; self.f.failure='/v2/account/activities'
        with self.assertRaises(self.m.ContinuityError) as caught: self.carry()
        self.assertNotIn('PRIVATE',str(caught.exception)); self.assertFalse(self.new_book.path.exists())

    def test_old_evidence_changes_during_reads_refuse_before_persisting(self):
        original=self.f.broker._transport
        def changed(method,url,headers,body):
            result=original(method,url,headers,body)
            if self.f.account_reads==2:
                with closing(sqlite3.connect(self.f.book.path)) as db:
                    db.execute("UPDATE snapshots SET digest='bad' WHERE sequence=1"); db.commit()
            return result
        self.f.broker._transport=changed
        with self.assertRaises(self.m.ContinuityError): self.carry()
        self.assertFalse(self.journal.exists())

    def test_failed_proof_or_book_creation_preserves_partial_state_and_never_retries(self):
        with patch.object(ObservationArchive,'append',side_effect=OSError('PRIVATE_DISK')):
            with self.assertRaises(self.m.ContinuityError): self.carry()
        self.assertFalse(self.new_book.path.exists())
        with patch.object(self.new_book,'initialize',side_effect=OSError('PRIVATE_DISK')):
            with self.assertRaises(self.m.ContinuityError): self.carry()
        self.assertTrue(self.journal.exists()); self.f.calls.clear()
        with self.assertRaises(self.m.ContinuityError): self.carry()
        self.assertEqual(self.f.calls, [])

    def test_clock_rollback_or_stale_commit_cannot_complete_continuity(self):
        append=ObservationArchive.append
        def delayed(archive,report):
            result=append(archive,report); self.f.now+=timedelta(seconds=31); return result
        with patch.object(ObservationArchive,'append',delayed):
            with self.assertRaises(self.m.ContinuityError): self.carry()
        self.assertFalse(self.new_book.path.exists()); self.assertTrue(self.journal.exists())

    def test_no_worker_inherits_provider_credentials_or_can_create_a_file(self):
        path=self.release/'paperbot_us/__init__.py'
        path.write_text('import os\nassert "ALPACA_PAPER_SECRET_KEY" in os.environ\n')
        files={str(p.relative_to(self.release)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (self.release/'paperbot_us').glob('*.py')}
        replay=self.r.BookReplay(PinnedRelease(self.release,files=files,package_sha256='a'*64),
            book=self.f.book.path,baseline=self.f.journal,ledger=self.f.ledger.path,
            timezone_root=os.environ.get('PYTHONTZPATH') or None)
        with patch.dict(os.environ,{'ALPACA_PAPER_SECRET_KEY':'PRIVATE_MARKER'}):
            with self.assertRaises(self.r.BookReplayError): replay.replay()

    def test_completed_continuity_replays_exact_original_and_new_book_evidence_without_network(self):
        self.assertTrue(hasattr(self.m,'replay_continuity'),'completed continuity needs independent replay')
        receipt=self.carry(); self.f.calls.clear()
        restored=self.m.replay_continuity(replay=self.replay,book=self.new_book,journal=self.journal)
        self.assertEqual(restored,receipt)
        self.assertEqual(self.f.calls,[])
        with patch.object(self.m,'code_hash',return_value='f'*64):
            with self.assertRaises(self.m.ContinuityError):
                self.m.replay_continuity(replay=self.replay,book=self.new_book,journal=self.journal)

    def test_unknown_rehashed_proof_fields_cannot_enter_a_continuity_replay(self):
        self.assertTrue(hasattr(self.m,'replay_continuity'),'strict continuity replay required')
        self.carry()
        with closing(sqlite3.connect(self.journal)) as db:
            rows=db.execute('SELECT payload FROM observations ORDER BY seq').fetchall()
        target=self.f.root/'extra.sqlite'; archive=ObservationArchive(target)
        first=json.loads(rows[0][0]);first['accounts'][0]['secret']='PRIVATE_MARKER'
        first=archive.append(first)
        done=json.loads(rows[1][0]);done['proof_sha256']=first['audit_sha256'];archive.append(done)
        with self.assertRaises(self.m.ContinuityError) as caught:
            self.m.replay_continuity(replay=self.replay,book=self.new_book,journal=target)
        self.assertNotIn('PRIVATE',str(caught.exception))

    def test_complete_current_history_is_requested_from_the_original_anchor(self):
        from urllib.parse import urlsplit,parse_qs
        original=self.f.broker._transport; queries=[]
        def captured(method,url,headers,body):
            if urlsplit(url).path=='/v2/account/activities': queries.append(parse_qs(urlsplit(url).query))
            return original(method,url,headers,body)
        self.f.broker._transport=captured; self.carry()
        self.assertEqual(len(queries),1)
        self.assertEqual(queries[0]['after'],[self.anchor.as_of.isoformat(timespec='microseconds')])


if __name__=='__main__': unittest.main()
