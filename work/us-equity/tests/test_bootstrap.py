from contextlib import closing
from datetime import datetime, timedelta, timezone
import importlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from paperbot_us.accountbook import AccountBook
from paperbot_us.broker import PaperBroker, HttpReply
from paperbot_us.orders import OrderLedger


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.bootstrap'),
                             'observed empty-account bootstrap is required')
        self.m = importlib.import_module('paperbot_us.bootstrap')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.book = AccountBook(self.root / 'book.sqlite')
        self.ledger = OrderLedger(self.root / 'orders.sqlite')
        self.ledger.initialize()
        self.journal = self.root / 'baseline.sqlite'
        self.now = datetime(2026, 10, 24, 2, tzinfo=timezone.utc)
        self.calls = []
        self.account_reads = 0
        self.position_reads = 0
        self.history_reads = 0
        self.account = dict(id='00000000-0000-4000-8000-000000000050', status='ACTIVE',
                            currency='USD', cash='10000', equity='10000',
                            trading_blocked=False, account_blocked=False,
                            trade_suspended_by_user=False)
        self.activity = []
        self.position_after = False
        self.history_after = False
        self.account_after = {}
        self.failure = None
        self.broker = PaperBroker('PK' + 'T' * 20, 'T' * 32,
                                  transport=self.transport, clock=lambda: self.now)

    def transport(self, method, url, headers, body):
        self.assertEqual(method, 'GET', 'bootstrap must never place/cancel orders')
        self.assertIsNone(body)
        self.now += timedelta(milliseconds=100)
        path = urlsplit(url).path
        self.calls.append(path)
        if path == self.failure:
            raise OSError('PRIVATE_PROVIDER_ERROR')
        if path == '/v2/account':
            self.account_reads += 1
            return HttpReply(200, {**self.account, **(self.account_after if self.account_reads > 1 else {})})
        if path == '/v2/positions':
            self.position_reads += 1
            return HttpReply(200, [dict(asset_class='us_equity', symbol='AAPL', side='long',
                                       qty='1', qty_available='1')]
                             if self.position_after and self.position_reads > 1 else [])
        if path == '/v2/orders':
            self.history_reads += 1
            self.assertEqual(parse_qs(urlsplit(url).query),
                             dict(status=['all'], limit=['1'], direction=['desc'], nested=['false']))
            return HttpReply(200, [dict(status='filled')]
                             if self.history_after and self.history_reads > 1 else [])
        if path == '/v2/account/activities':
            return HttpReply(200, self.activity)
        raise AssertionError('unexpected API path')

    def run_bootstrap(self):
        return self.m.bootstrap_empty_account(broker=self.broker, ledger=self.ledger,
                                             book=self.book, journal=self.journal,
                                             clock=lambda: self.now)

    def test_genuine_empty_observations_are_persisted_before_a_bound_anchor_is_created(self):
        original = self.book.initialize
        def initialize(anchor):
            self.assertTrue(self.journal.exists(), 'proof must precede anchor')
            from paperbot_us.monitor import ObservationArchive
            self.assertEqual(ObservationArchive(self.journal).verify()['records'], 1)
            original(anchor)
        with patch.object(self.book, 'initialize', side_effect=initialize):
            receipt = self.run_bootstrap()
        self.assertEqual(self.account_reads, 2)
        self.assertEqual(self.position_reads, 2)
        self.assertEqual(self.history_reads, 2)
        anchor = self.book.read_anchor()
        self.assertEqual(anchor.cash_usd, 10000)
        self.assertEqual(anchor.evidence_sha256, receipt.audit_sha256)
        self.assertEqual(self.book.verify()['snapshots'], 0)
        self.assertFalse(receipt.public_summary()['execution_enabled'])
        self.assertIsNone(receipt.public_summary()['paper_started_at'])
        self.assertNotIn(self.account['id'], json.dumps(receipt.public_summary()))
        self.assertNotIn(anchor.account_sha256, json.dumps(receipt.public_summary()))
        with closing(sqlite3.connect(self.journal)) as db:
            payload = db.execute('SELECT payload FROM observations').fetchone()[0]
        self.assertNotIn(self.account['id'], payload)
        self.assertNotIn('T' * 32, payload)

    def test_existing_book_or_proof_is_never_reinitialized_and_causes_no_network_request(self):
        self.run_bootstrap()
        old = self.book.path.read_bytes()
        self.calls.clear()
        with self.assertRaises(self.m.BootstrapError):
            self.run_bootstrap()
        self.assertEqual(self.calls, [])
        self.assertEqual(old, self.book.path.read_bytes())

    def test_nonempty_or_changed_account_or_blocked_flag_never_creates_a_baseline(self):
        changes = [dict(cash='9999'), dict(equity='10001'), dict(trading_blocked=True),
                   dict(trade_suspended_by_user=True),
                   dict(id='00000000-0000-4000-8000-000000000099')]
        for values in changes:
            with self.subTest(values=values):
                self.account_reads = self.position_reads = self.history_reads = 0
                self.account_after = values
                with self.assertRaises(self.m.BootstrapError):
                    self.run_bootstrap()
                self.assertFalse(self.book.path.exists())
                self.assertFalse(self.journal.exists())

    def test_positions_or_any_order_history_even_closed_orders_block_initialization(self):
        self.position_after = True
        with self.assertRaises(self.m.BootstrapError):
            self.run_bootstrap()
        self.assertFalse(self.book.path.exists())
        self.position_after = False
        self.history_after = True
        self.account_reads = self.position_reads = self.history_reads = 0
        with self.assertRaises(self.m.BootstrapError):
            self.run_bootstrap()
        self.assertFalse(self.journal.exists())

    def test_any_activity_or_failed_provider_read_is_not_an_empty_account(self):
        self.activity = [dict(id='20261024020000000::00000000-0000-4000-8000-000000000020',
                              activity_type='CSD')]
        with self.assertRaises(self.m.BootstrapError):
            self.run_bootstrap()
        self.assertFalse(self.book.path.exists())
        self.activity = []
        self.failure = '/v2/account/activities'
        with self.assertRaises(self.m.BootstrapError) as caught:
            self.run_bootstrap()
        self.assertNotIn('PRIVATE', str(caught.exception))
        self.assertFalse(self.journal.exists())

    def test_proof_write_failure_never_creates_a_book_and_book_failure_preserves_proof(self):
        with patch.object(self.m.ObservationArchive, 'append', side_effect=OSError('PRIVATE_DISK')):
            with self.assertRaises(self.m.BootstrapError):
                self.run_bootstrap()
        self.assertFalse(self.book.path.exists())
        with patch.object(self.book, 'initialize', side_effect=OSError('PRIVATE_DISK')):
            with self.assertRaises(self.m.BootstrapError):
                self.run_bootstrap()
        self.assertTrue(self.journal.exists())
        self.calls.clear()
        with self.assertRaises(self.m.BootstrapError):
            self.run_bootstrap()
        self.assertEqual(self.calls, [], 'partial bootstrap needs explicit recovery')

    def test_invalid_history_responses_do_not_establish_absence_of_prior_orders(self):
        original = self.transport
        for payload in (None, {}, [{}], [dict(status='filled'), dict(status='filled')]):
            with self.subTest(payload=payload):
                def invalid(method, url, headers, body):
                    if urlsplit(url).path == '/v2/orders':
                        return HttpReply(200, payload)
                    return original(method, url, headers, body)
                self.broker._transport = invalid
                with self.assertRaises(self.m.BootstrapError):
                    self.run_bootstrap()
                self.assertFalse(self.book.path.exists())
                self.assertFalse(self.journal.exists())

    def test_existing_ledger_intents_or_conflicting_paths_fail_before_provider_access(self):
        with patch.object(self.ledger, 'all_orders', return_value=({'state': 'filled'},)):
            with self.assertRaises(self.m.BootstrapError):
                self.run_bootstrap()
        self.assertEqual(self.calls, [])
        self.journal = self.book.path
        with self.assertRaises(self.m.BootstrapError):
            self.run_bootstrap()
        self.assertEqual(self.calls, [])

    def test_slow_proof_commit_is_preserved_but_cannot_initialize_an_old_baseline(self):
        from paperbot_us.monitor import ObservationArchive
        append = ObservationArchive.append
        def delayed(archive, report):
            result = append(archive, report)
            self.now += timedelta(seconds=31)
            return result
        with patch.object(ObservationArchive, 'append', delayed):
            with self.assertRaises(self.m.BootstrapError):
                self.run_bootstrap()
        self.assertTrue(self.journal.exists())
        self.assertFalse(self.book.path.exists())

    def test_slow_reads_or_clock_regression_never_publish_a_usable_baseline(self):
        original = self.broker._transport
        def delayed(method, url, headers, body):
            reply = original(method, url, headers, body)
            if urlsplit(url).path == '/v2/account/activities':
                self.now += timedelta(seconds=31)
            return reply
        self.broker._transport = delayed
        with self.assertRaises(self.m.BootstrapError):
            self.run_bootstrap()
        self.assertFalse(self.book.path.exists())
        self.broker._transport = original
        self.account_reads = self.position_reads = self.history_reads = 0
        original_init = self.book.initialize
        def rewind(anchor):
            original_init(anchor)
            self.now -= timedelta(seconds=1)
        with patch.object(self.book, 'initialize', side_effect=rewind):
            with self.assertRaises(self.m.BootstrapError):
                self.run_bootstrap()


if __name__ == '__main__':
    unittest.main()
