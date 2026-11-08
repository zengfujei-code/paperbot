"""GET-only paper reconciliation coordinator. No admission, POST or DELETE path.

A verified result is eligible for independent risk evaluation, not permission to
trade. Strategy acceptance, protective execution, quotes and risk baselines are
separate gates. Each REST observation can race another; disagreement vetoes.
"""
from dataclasses import dataclass,field
from datetime import datetime,timezone
from decimal import Decimal
import hashlib
from pathlib import Path
import time

from .accountbook import AccountBook,BookReceipt
from .accounting import AccountSnapshot,OrderEvidence
from .broker import PaperBroker
from .data import utc_time
from .monitor import ObservationArchive
from .orders import OrderLedger,ACTIVE,_json
from .runner import _exclusive


def _digest(records): return hashlib.sha256(_json(records).encode()).hexdigest()


def _code_hash():
    digest=hashlib.sha256()
    for name in ('reconciler.py','accountbook.py','accounting.py','orders.py','broker.py','risk.py','data.py'):
        digest.update(name.encode()+b'\0'+Path(__file__).with_name(name).read_bytes()+b'\0')
    return digest.hexdigest()


def _matches(record,order):
    intent=record['intent']
    return (order.owned and order.client_order_id==record['client_order_id']
        and (order.symbol,order.side,order.quantity,order.limit_price_usd)==
            (intent['symbol'],intent['side'],intent['quantity'],Decimal(intent['limit_price_usd']))
        and record['broker_order_id'] in (None,order.broker_order_id)
        and (record.get('broker_created_at') is None or utc_time(record['broker_created_at'])==order.created_at)
        and order.created_at>=utc_time(record['submission_claimed_at'] or record['created_at']))


def _evidence(records):
    return tuple(OrderEvidence(r['intent_id'],r['broker_order_id'],r['intent']['symbol'],r['intent']['side'],
        r['intent']['quantity'],r['intent']['limit_price_usd'],r['filled_quantity'],r['filled_notional_usd'],
        r['broker_created_at'],r['broker_updated_at']) for r in records if r['broker_order_id'] is not None)


@dataclass(frozen=True)
class ReconciliationCycle:
    as_of: datetime
    status: str
    ready_for_risk: bool
    findings: tuple
    book_receipt: BookReceipt|None=field(repr=False)
    orders_sha256: str|None
    audit_sha256: str|None
    orders_count: int|None
    positions_count: int|None
    positions_available: tuple=field(default=(),repr=False)
    native_states: tuple=field(default=(),repr=False)

    def public_summary(self):
        receipt=self.book_receipt
        good=receipt is not None and receipt.result.reconciled and 'stale_reconciliation' not in self.findings
        result=receipt.result if good else None
        return dict(mode='paper',phase='reconciling',execution_enabled=False,status=self.status,
            generated_at=self.as_of.timestamp(),ready_for_risk=self.ready_for_risk,findings=list(self.findings),
            accounting_ok=good,equity_usd=float(result.equity_usd) if good else None,
            cash_usd=float(result.expected_cash_usd) if good else None,profit_usd=float(result.profit_usd) if good else None,
            orders_count=self.orders_count,positions_count=self.positions_count)


class PaperReconciler:
    def __init__(self,*,broker,ledger,book,journal,clock=None,monotonic=None):
        if type(broker) is not PaperBroker or type(ledger) is not OrderLedger or type(book) is not AccountBook:
            raise ValueError('exact reconciliation dependencies required')
        self.broker,self.ledger,self.book=broker,ledger,book
        self.journal=ObservationArchive(journal)
        self._clock=clock or (lambda:datetime.now(timezone.utc));self._monotonic=monotonic or time.monotonic

    def refresh(self):
        now=utc_time(self._clock());deadline=self._monotonic()+25
        def check():
            if self._monotonic()>=deadline: raise ValueError('bounded reconciliation deadline')
        try:
            with _exclusive(self.journal.path):
                if self.journal.path.exists(): self.journal.verify()
                records=self.ledger.all_orders();anchor=self.book.read_anchor()
                receipts=[now,*(utc_time(r['updated_at']) for r in records)]
                active=[r for r in records if r['state'] in ACTIVE]
                if len(active)>20: raise ValueError('active lookup budget')
                check();window=self.broker.activities(after=anchor.as_of,until=now)
                receipts.extend((window.received_at,utc_time(self._clock())))
                check();visible=self.broker.open_orders();known={r['client_order_id']:r for r in records};findings=set()
                receipts.extend((utc_time(self._clock()),*(o.received_at for o in visible)))
                for order in visible:
                    record=known.get(order.client_order_id)
                    if record is None or not order.owned: findings.add('unrecognized_broker_order')
                    elif not _matches(record,order): findings.add('broker_order_mismatch')
                    elif record['state'] not in ACTIVE: findings.add('terminal_order_reported_open')
                    elif record['state']=='reserved': findings.add('unclaimed_order_reported_at_broker')
                for record in active:
                    if record['state']=='reserved': continue
                    check();order=self.broker.lookup(record['client_order_id'])
                    receipts.append(utc_time(self._clock()))
                    if order is None:
                        findings.add('unconfirmed_submission' if record['state']=='submission_unknown' else 'missing_known_order')
                        continue
                    receipts.append(order.received_at)
                    if not _matches(record,order): findings.add('broker_order_mismatch');continue
                    self.ledger.observe(record['intent_id'],status=order.status,filled_quantity=order.filled_quantity,
                        average_fill_price_usd=order.average_fill_price_usd,broker_order_id=order.broker_order_id,
                        broker_created_at=order.created_at,broker_event_at=order.updated_at,received_at=order.received_at)
                records=self.ledger.all_orders();known={r['client_order_id']:r for r in records}
                visible_ids={o.client_order_id for o in visible if o.owned}
                for record in records:
                    if record['state']=='submission_unknown': findings.add('unconfirmed_submission')
                    if record['state'] in ('open','partially_filled') and record['client_order_id'] not in visible_ids:
                        findings.add('missing_known_order')
                for order in visible:
                    record=known.get(order.client_order_id)
                    if record is not None and (record['state']!=order.status or record['filled_quantity']!=order.filled_quantity
                            or not _matches(record,order) or record['broker_updated_at'] is None
                            or utc_time(record['broker_updated_at'])!=order.updated_at
                            or abs(Decimal(record['filled_notional_usd'])-(order.average_fill_price_usd or Decimal(0))*order.filled_quantity)
                                >Decimal('0.00000001')*max(1,order.filled_quantity)):
                        findings.add('non_atomic_order_snapshot')
                check();available=self.broker.positions()
                receipts.append(utc_time(self._clock()))
                check();marks=self.broker.account_marks()
                receipts.append(utc_time(self._clock()))
                check();account=self.broker.account();at=utc_time(self._clock());check()
                if {p.symbol:p.quantity for p in available}!={p.symbol:p.quantity for p in marks}:
                    findings.add('position_snapshot_mismatch')
                reserved_sells={}
                for order in visible:
                    if order.side=='sell': reserved_sells[order.symbol]=reserved_sells.get(order.symbol,0)+order.quantity-order.filled_quantity
                for position in available:
                    if position.available_quantity!=position.quantity-reserved_sells.get(position.symbol,0):
                        findings.add('available_share_mismatch')
                snapshot=AccountSnapshot(account.received_at,account.cash_usd,account.equity_usd,account.blocked,marks,account.account_sha256)
                receipt=self.book.refresh(window=window,orders=_evidence(records),account=snapshot,now=at)
                findings.update(receipt.result.violations)
                digest=_digest(records)
                if _digest(self.ledger.all_orders())!=digest: findings.add('order_snapshot_changed')
                floor=max(*receipts,now,window.received_at,account.received_at,receipt.as_of,
                    *(p.received_at for p in marks),*(p.received_at for p in available))
                at=utc_time(self._clock())
                if at<floor or not 0<=(at-window.until).total_seconds()<=30: findings.add('stale_reconciliation')
                floor=max(floor,at)
                good=receipt.result.reconciled and not findings
                report=self.journal.append(dict(kind='reconciliation',generated_at=at.timestamp(),code_sha256=_code_hash(),
                    status='reconciled' if good else 'vetoed',ready_for_risk=good,findings=sorted(findings),
                    orders_sha256=digest,orders_count=len(records),positions_count=len(marks),
                    book_sequence=receipt.sequence,book_sha256=receipt.digest,activity_until=window.until.timestamp()))
                # Admission must revalidate this receipt too. A slow final fsync
                # cannot make a previously fresh observation usable indefinitely.
                at=utc_time(self._clock())
                if at<floor or not 0<=(at-window.until).total_seconds()<=30: findings.add('stale_reconciliation');good=False
                return ReconciliationCycle(at,'reconciled' if good else 'vetoed',good,tuple(sorted(findings)),receipt,
                    digest,report['audit_sha256'],len(records),len(marks),available)
        except Exception:
            return ReconciliationCycle(utc_time(self._clock()),'unavailable',False,('reconciliation_unavailable',),None,None,None,None,None)
