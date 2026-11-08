"""Explicit pristine-book continuity. Never resets/migrates old state or trades."""
from dataclasses import dataclass,field
from contextlib import closing
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from .accountbook import AccountBook,BookReceipt,_json
from .accounting import AccountAnchor,AccountSnapshot,ActivityWindow,reconcile
from .book_replay import BookReplay,replay_version
from .broker import PaperBroker
from .data import utc_time
from .monitor import ObservationArchive
from .orders import OrderLedger
from .runner import _exclusive
from .replay_bridge import strict_json


class ContinuityError(RuntimeError): pass


def code_hash():
    digest=hashlib.sha256()
    for name in ('book_continuity.py','bootstrap.py','accountbook.py','accounting.py','orders.py','broker.py','data.py'):
        digest.update(name.encode()+b'\0'+Path(__file__).with_name(name).read_bytes()+b'\0')
    digest.update(replay_version().encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class ContinuityReceipt:
    as_of: datetime
    proof_sha256: str
    audit_sha256: str
    anchor: AccountAnchor=field(repr=False)
    book_receipt: BookReceipt=field(repr=False)

    def public_summary(self):
        return dict(mode='paper',phase='book_continuity',status='verified_empty_continuity',
                    generated_at=self.as_of.timestamp(),execution_enabled=False,
                    ready_for_risk=False,paper_started_at=None,orders_submitted=0)


def carry_forward_empty_book(*,replay,broker,ledger,book,journal,clock=None,monotonic=None):
    """Keep the original anchor after original replay and fresh GET-only checks.

    New separate state only; any partial state is preserved and refuses rerun.
    The caller must pin/administer the original source and hold the future global
    coordinator lock before execution. This diagnostic has no order authority.
    """
    try:
        if (type(replay) is not BookReplay or type(broker) is not PaperBroker
                or type(ledger) is not OrderLedger or type(book) is not AccountBook):
            raise ValueError('exact continuity dependencies required')
        archive=ObservationArchive(journal)
        targets=(book.path,archive.path,ledger.path)
        if len(set(targets+replay.paths))!=6: raise ValueError('separate original/new state required')
        if any(str(p.resolve())!=str(p) or p.is_symlink() for p in targets): raise ValueError('canonical targets required')
        clock=clock or (lambda:datetime.now(timezone.utc)); monotonic=monotonic or time.monotonic
        started=last=utc_time(clock()); start_mono=last_mono=monotonic()
        def check():
            nonlocal last,last_mono
            now,mono=utc_time(clock()),monotonic()
            if now<last or mono<last_mono or mono-start_mono>=25 or not 0<=(now-started).total_seconds()<=30:
                raise ValueError('continuity deadline or clock regression')
            last,last_mono=now,mono
            return now
        def new_paths():
            if any(p.exists() or p.is_symlink() for p in (book.path,archive.path)):
                raise ValueError('new continuity targets required')
        new_paths()
        with _exclusive(archive.path):
            new_paths()
            if ledger.all_orders(): raise ValueError('new empty verified ledger required')
            original=replay.replay(); check(); old=original.read()
            anchor=AccountAnchor(**old['anchor'])
            if anchor.as_of>started or utc_time(old['book']['last_as_of'])>started: raise ValueError('future original evidence')
            accounts=[]; positions=[]; histories=[]
            def read_account():
                value=broker.account(); now=check()
                if (value.blocked or value.cash_usd!=10000 or value.equity_usd!=10000
                        or value.account_sha256!=anchor.account_sha256 or not started<=value.received_at<=now):
                    raise ValueError('unchanged pristine account required')
                accounts.append(value)
            read_account()
            for _ in range(2):
                if broker.positions(): raise ValueError('positions exist')
                positions.append(check())
                if broker.has_order_history(): raise ValueError('prior order exists')
                histories.append(check())
            window=broker.activities(after=anchor.as_of,until=check(),max_pages=1); check()
            if (not window.complete or window.activities or window.after!=anchor.as_of
                    or window.until<utc_time(old['book']['last_as_of'])): raise ValueError('empty complete original history required')
            read_account()
            result=reconcile(anchor=anchor,window=window,orders=(),account=AccountSnapshot(
                accounts[-1].received_at,accounts[-1].cash_usd,accounts[-1].equity_usd,False,(),anchor.account_sha256),now=check())
            if not result.reconciled or result.profit_usd!=0: raise ValueError('new account does not reconcile')
            if replay.replay().digest!=original.digest: raise ValueError('original evidence changed')
            if ledger.all_orders(): raise ValueError('ledger changed')
            now=check(); new_paths()
            proof=dict(kind='book_continuity_proof',schema=1,generated_at=now.timestamp(),code_sha256=code_hash(),
                original_sha256=original.digest,original=old,anchor=json.loads(_json(anchor)),
                accounts=json.loads(_json(accounts)),window=json.loads(_json(window)),
                empty_positions_received_at=[v.isoformat() for v in positions],
                empty_history_received_at=[v.isoformat() for v in histories],
                ledger_verified_empty=True,execution_enabled=False,paper_started_at=None)
            saved=archive.append(proof); check()
            book.initialize(anchor); check()
            receipt=book.refresh(window=window,orders=(),account=AccountSnapshot(accounts[-1].received_at,
                10000,10000,False,(),anchor.account_sha256),now=check())
            if not receipt.result.reconciled or book.read_anchor()!=anchor: raise ValueError('continuity book mismatch')
            check()
            done=archive.append(dict(kind='book_continuity_complete',schema=1,generated_at=last.timestamp(),
                code_sha256=code_hash(),proof_sha256=saved['audit_sha256'],book_sha256=receipt.digest,
                book_sequence=receipt.sequence,execution_enabled=False,paper_started_at=None))
            finished=check()
            return ContinuityReceipt(datetime.fromtimestamp(done['generated_at'],timezone.utc),
                                     saved['audit_sha256'],done['audit_sha256'],anchor,receipt)
    except Exception: raise ContinuityError('pristine_book_continuity_unavailable') from None


def replay_continuity(*,replay,book,journal):
    """Historical verification only. No initialization, provider calls or orders."""
    try:
        if type(replay) is not BookReplay or type(book) is not AccountBook:
            raise ValueError('exact continuity dependencies required')
        path=Path(journal).absolute()
        if (path.is_symlink() or not path.is_file() or str(path.resolve())!=str(path)
                or path.stat().st_size>64*1024*1024): raise ValueError('bounded canonical journal required')
        with path.open('rb') as handle: header=handle.read(20)
        if header[:16]!=b'SQLite format 3\0' or header[18:20]!=b'\x01\x01': raise ValueError('DELETE journal required')
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=5,isolation_level=None)) as db:
            db.execute('PRAGMA query_only=ON'); db.execute('BEGIN')
            if db.execute('PRAGMA journal_mode').fetchone()[0]!='delete': raise ValueError('DELETE journal required')
            _,count,_=ObservationArchive(path)._verify_chain(db)
            if count!=2: raise ValueError('completed continuity required')
            rows=db.execute('SELECT payload,digest FROM observations ORDER BY seq').fetchall()
        proof,done=[strict_json(row[0]) for row in rows]
        def exact(value,keys):
            if type(value) is not dict or set(value)!=set(keys.split()): raise ValueError('unknown continuity fields')
        exact(proof,'kind schema generated_at code_sha256 original_sha256 original anchor accounts window '
                    'empty_positions_received_at empty_history_received_at ledger_verified_empty '
                    'execution_enabled paper_started_at observing_started_at')
        exact(done,'kind schema generated_at code_sha256 proof_sha256 book_sha256 book_sequence '
                   'execution_enabled paper_started_at observing_started_at')
        if proof['kind']!='book_continuity_proof' or done['kind']!='book_continuity_complete': raise ValueError('continuity kinds')
        for value in (proof,done):
            if (type(value['schema']) is not int or value['schema']!=1 or value['code_sha256']!=code_hash()
                    or value['execution_enabled'] is not False or value['paper_started_at'] is not None
                    or type(value['generated_at']) not in (int,float)):
                raise ValueError('continuity version or authority mismatch')
        if (proof['observing_started_at']!=proof['generated_at'] or done['observing_started_at']!=proof['generated_at']
                or not proof['generated_at']<=done['generated_at']): raise ValueError('continuity times')
        original=replay.replay()
        if original.digest!=proof['original_sha256'] or original.read()!=proof['original']:
            raise ValueError('original continuity witness changed')
        if proof['anchor']!=proof['original']['anchor'] or proof['ledger_verified_empty'] is not True:
            raise ValueError('original continuity anchor changed')
        anchor=AccountAnchor(**proof['anchor'])
        if type(proof['accounts']) is not list or len(proof['accounts'])!=2: raise ValueError('two account reads required')
        accounts=[]
        for raw in proof['accounts']:
            exact(raw,'received_at cash_usd equity_usd blocked account_sha256')
            account=AccountSnapshot(**{**raw,'positions':()})
            if (account.blocked or account.account_sha256!=anchor.account_sha256
                    or account.cash_usd!=10000 or account.equity_usd!=10000): raise ValueError('empty account mismatch')
            accounts.append(account)
        now=datetime.fromtimestamp(proof['generated_at'],timezone.utc)
        completed=datetime.fromtimestamp(done['generated_at'],timezone.utc)
        if not utc_time(proof['original']['book']['last_as_of'])<=accounts[0].received_at<=accounts[1].received_at<=now<=completed:
            raise ValueError('continuity observation chronology')
        if not 0<=(completed-accounts[0].received_at).total_seconds()<=30: raise ValueError('continuity completion age')
        for key in ('empty_positions_received_at','empty_history_received_at'):
            values=proof[key]
            if (type(values) is not list or len(values)!=2
                    or not accounts[0].received_at<=utc_time(values[0])<=utc_time(values[1])<=accounts[-1].received_at):
                raise ValueError('continuity empty receipts')
        exact(proof['window'],'after until received_at activities complete')
        if proof['window']['activities']!=[]: raise ValueError('continuity activities found')
        window=ActivityWindow(**{**proof['window'],'activities':()})
        if window.until<utc_time(proof['original']['book']['last_as_of']): raise ValueError('continuity window gap')
        result=reconcile(anchor=anchor,window=window,orders=(),account=accounts[-1],now=completed)
        if not result.reconciled or result.profit_usd!=0: raise ValueError('continuity calculation mismatch')
        if (done['proof_sha256']!=rows[0][1] or type(done['book_sequence']) is not int or done['book_sequence']!=1):
            raise ValueError('continuity completion linkage')
        snapshot=book.snapshot_by_digest(done['book_sha256'])
        if (snapshot.receipt.sequence!=1 or snapshot.anchor!=anchor or snapshot.account!=accounts[-1]
                or snapshot.activity_count!=0 or snapshot.activity_until!=window.until
                or snapshot.receipt.result!=result or not now<=snapshot.receipt.as_of<=completed):
            raise ValueError('continuity book linkage')
        return ContinuityReceipt(completed,rows[0][1],rows[1][1],anchor,snapshot.receipt)
    except Exception: raise ContinuityError('pristine_book_continuity_replay_unavailable') from None
