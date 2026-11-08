"""Explicit first-time empty paper-account baseline. GET-only; no CLI or orders.

This records observed initial state, not account history completeness or trading
readiness. Nonempty or partially initialized state requires explicit recovery;
normal operation cannot reset a book. Subsequent complete-window reconciliation
must still detect delayed activities or changes racing these independent reads.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from .accountbook import AccountBook, _json
from .accounting import AccountAnchor
from .broker import PaperBroker
from .data import utc_time
from .monitor import ObservationArchive
from .orders import OrderLedger
from .runner import _exclusive


class BootstrapError(RuntimeError):
    pass


def _code_hash():
    digest = hashlib.sha256()
    for name in ('bootstrap.py', 'broker.py', 'accountbook.py', 'accounting.py',
                 'orders.py', 'risk.py', 'data.py', 'monitor.py', 'runner.py'):
        digest.update(name.encode() + b'\0' + Path(__file__).with_name(name).read_bytes() + b'\0')
    return digest.hexdigest()


@dataclass(frozen=True)
class BootstrapReceipt:
    as_of: datetime
    audit_sha256: str
    anchor: AccountAnchor = field(repr=False)

    def public_summary(self):
        return dict(mode='paper', phase='baseline_created', status='baseline_created',
                    generated_at=self.as_of.timestamp(), execution_enabled=False,
                    ready_for_risk=False, paper_started_at=None)


def bootstrap_empty_account(*, broker, ledger, book, journal, clock=None, monotonic=None):
    """Require two unchanged empty USD 10k observations, then commit proof first.

    The order ledger must already exist and be verified empty. Only a new book
    and a new proof archive may be created. Failure after proof or book creation
    deliberately leaves evidence in place and refuses an automatic second run.
    An outer process deadline must bound DNS and the complete network sequence.
    """
    try:
        if type(broker) is not PaperBroker or type(ledger) is not OrderLedger or type(book) is not AccountBook:
            raise ValueError('exact bootstrap dependencies required')
        clock = clock or (lambda: datetime.now(timezone.utc))
        monotonic = monotonic or time.monotonic
        archive = ObservationArchive(journal)
        if len({archive.path, book.path, ledger.path}) != 3:
            raise ValueError('separate state paths required')
        started = last = utc_time(clock())
        start_mono = last_mono = monotonic()
        receipts = [started]

        def check():
            nonlocal last, last_mono
            now, mono = utc_time(clock()), monotonic()
            if (now < last or mono < last_mono or mono - start_mono >= 25
                    or not 0 <= (now - started).total_seconds() <= 30):
                raise ValueError('bootstrap clock or deadline')
            last, last_mono = now, mono
            receipts.append(now)
            return now

        def new_paths():
            if any(p.exists() or p.is_symlink() for p in (archive.path, book.path)):
                raise ValueError('new bootstrap state required')

        new_paths()
        with _exclusive(archive.path):
            new_paths()
            if ledger.all_orders():
                raise ValueError('empty verified ledger required')
            check()
            accounts = []
            position_receipts, history_receipts = [], []

            def read_account():
                account = broker.account()
                check()
                if (account.blocked or account.cash_usd != 10000 or account.equity_usd != 10000
                        or not started <= account.received_at <= last
                        or (accounts and account.account_sha256 != accounts[0].account_sha256)):
                    raise ValueError('unchanged empty paper account required')
                receipts.append(account.received_at)
                accounts.append(account)

            read_account()
            for pass_number in range(2):
                if pass_number:
                    check()
                    window = broker.activities(after=accounts[0].received_at, until=check(), max_pages=1)
                    check()
                    if (not window.complete or window.activities or window.after != accounts[0].received_at
                            or window.received_at > last):
                        raise ValueError('empty complete initial activity window required')
                    receipts.append(window.received_at)
                check()
                positions = broker.positions()
                position_receipts.append(check())
                if positions:
                    raise ValueError('initial positions found')
                history = broker.has_order_history()
                history_receipts.append(check())
                if history:
                    raise ValueError('initial order history found')
            read_account()
            if ledger.all_orders():
                raise ValueError('ledger changed during bootstrap')
            new_paths()
            now = check()
            if now < max(receipts):
                raise ValueError('future bootstrap observation')
            proof = dict(kind='bootstrap_proof', schema=1, generated_at=now.timestamp(),
                         code_sha256=_code_hash(), mode='paper', capital_usd=10000,
                         anchor_as_of=accounts[0].received_at.isoformat(),
                         accounts=json.loads(_json(accounts)),
                         empty_positions_received_at=[t.isoformat() for t in position_receipts],
                         empty_history_received_at=[t.isoformat() for t in history_receipts],
                         window=json.loads(_json(window)), ledger_verified_empty=True,
                         paper_started_at=None, execution_enabled=False)
            saved = archive.append(proof)
            check()
            anchor = AccountAnchor(accounts[0].received_at, accounts[0].cash_usd,
                                   saved['audit_sha256'], accounts[0].account_sha256)
            book.initialize(anchor)
            if book.read_anchor() != anchor:
                raise ValueError('bootstrap anchor verification failed')
            finished = check()
            return BootstrapReceipt(finished, saved['audit_sha256'], anchor)
    except Exception:
        raise BootstrapError('paper_bootstrap_unavailable') from None
