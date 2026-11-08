import tempfile
from pathlib import Path
import unittest
from datetime import timedelta
import test_accounting

from paperbot_us.accountbook import AccountBook, BookError


class BookSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_accounting.AccountingTests()
        self.fixture.setUp()
        self.args = self.fixture.inputs()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.book = AccountBook(Path(self.temp.name) / 'book.sqlite')
        self.book.initialize(self.args['anchor'])

    def test_read_returns_a_replayed_persisted_receipt_and_source_times(self):
        self.assertTrue(hasattr(self.book, 'snapshot'), 'verified account snapshot read required')
        receipt = self.book.refresh(**{k: v for k, v in self.args.items() if k != 'anchor'})
        saved = self.book.snapshot()
        self.assertEqual(saved.receipt, receipt)
        self.assertEqual(saved.anchor, self.args['anchor'])
        self.assertEqual(saved.account, self.args['account'])
        self.assertEqual(saved.activity_until, self.args['window'].until)
        self.assertEqual(saved.activity_count, 3)
        self.assertEqual(self.book.snapshot(receipt.sequence), saved)
        self.assertEqual(self.book.verify()['snapshots'], 1)

    def test_empty_corrupt_or_unknown_sequence_is_not_a_verified_snapshot(self):
        self.assertTrue(hasattr(self.book, 'snapshot'), 'verified account snapshot read required')
        with self.assertRaises(BookError):
            self.book.snapshot()
        for sequence in (True, 0, -1, 2):
            with self.subTest(sequence=sequence), self.assertRaises(BookError):
                self.book.snapshot(sequence)

    def test_bounded_history_read_preserves_every_intermediate_snapshot(self):
        self.assertTrue(hasattr(self.book, 'snapshots_after'), 'risk catch-up needs verified accounting history')
        args = {k: v for k, v in self.args.items() if k != 'anchor'}
        first = self.book.refresh(**args)
        second = self.book.refresh(**{**args, 'now': args['now'] + timedelta(seconds=1)})
        history = self.book.snapshots_after(0)
        self.assertEqual([s.receipt for s in history], [first, second])
        self.assertEqual(self.book.snapshots_after(1), (history[1],))
        with self.assertRaises(BookError):
            self.book.snapshots_after(0, limit=1)
        self.assertEqual(self.book.snapshots_after(2), ())


if __name__ == '__main__':
    unittest.main()
