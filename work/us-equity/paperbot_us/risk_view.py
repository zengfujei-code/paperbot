"""Prepare bound accounting/quote/risk inputs. No broker mutations or model calls.

The coordinator must hold its own global execution lock while reconciling,
preparing, journalling and admitting an intent. These bindings are evidence,
not an unforgeable capability. Final durable admission and pre-network checks
must still revalidate age/reservations. Only the risk baseline journal is written.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

from .accountbook import AccountBook
from .broker import BrokerPosition
from .data import Calendar, utc_time
from .orders import OrderLedger, ACTIVE, _json, _base
from .reconciler import ReconciliationCycle
from .risk import Portfolio, Position, PendingOrder, Quote, RiskState, _identity, _symbol
from .risk_baselines import RiskBaselineStore


class RiskViewError(RuntimeError):
    pass


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _version():
    digest = hashlib.sha256()
    for name in ('risk_view.py', 'risk_baselines.py', 'accountbook.py', 'accounting.py',
                 'reconciler.py', 'orders.py', 'risk.py', 'data.py', 'broker.py',
                 'native_risk_view.py', 'native_reconciler.py', 'native_orders.py', 'native_broker.py', 'protection.py',
                 'native_cleanup.py', 'native_close.py', 'cleanup_recovery.py','repair_orders.py','repair_decisions.py','repair_execution.py','compensation.py','native_admission.py','native_execution.py'):
        digest.update(name.encode() + b'\0' + Path(__file__).with_name(name).read_bytes() + b'\0')
    return digest.hexdigest()


@dataclass(frozen=True)
class PreparedRiskView:
    portfolio: Portfolio
    quote: Quote
    state: RiskState
    findings: tuple
    book_digest: str
    baseline_digest: str
    orders_digest: str
    quotes_digest: str
    code_sha256: str

    def public_summary(self):
        return dict(mode='paper', phase='risk_prepared', execution_enabled=False,
                    generated_at=self.portfolio.as_of.timestamp(), entries_paused=self.state.pause_entries,
                    orders_halted=self.state.halt_orders, findings=list(self.findings))


def prepare_risk_view(*, book, ledger, baselines, cycle, quotes, symbol, operator_state, calendar, now):
    return _prepare_risk_view(book=book, ledger=ledger, baselines=baselines, cycle=cycle,
        quotes=quotes, symbol=symbol, operator_state=operator_state, calendar=calendar, now=now,
        reservation_builder=None)


def _prepare_risk_view(*, book, ledger, baselines, cycle, quotes, symbol, operator_state, calendar, now,
                       reservation_builder, accounting_validator=None):
    try:
        now = utc_time(now)
        for value, exact in ((book, AccountBook), (ledger, OrderLedger), (baselines, RiskBaselineStore),
                             (cycle, ReconciliationCycle), (operator_state, RiskState), (calendar, Calendar)):
            if type(value) is not exact:
                raise ValueError('exact risk view dependencies required')
        _symbol(symbol)
        if (type(cycle.positions_count) is not int or type(cycle.orders_count) is not int
                or accounting_validator is None and
                (cycle.ready_for_risk is not True or cycle.status != 'reconciled' or cycle.findings)):
            raise ValueError('ready complete reconciliation required')
        records = ledger.all_orders()
        orders_digest = _digest(records)
        if orders_digest != cycle.orders_sha256 or len(records) != cycle.orders_count:
            raise ValueError('ledger changed after reconciliation')
        if accounting_validator is not None:
            accounting_validator(book, records, cycle, calendar, now)
        if reservation_builder is None:
            if any(r.get('native_stop') is not None for r in records):
                raise ValueError('native risk preparation required')
            stop_reservations, group_times = (), ()
        else:
            stop_reservations, group_times = reservation_builder(records, cycle, calendar, now)
        snapshot = book.snapshot()
        if snapshot.receipt != cycle.book_receipt or not snapshot.receipt.result.reconciled:
            raise ValueError('accounting receipt changed')
        if type(quotes) is not tuple or len(quotes) > 2000 or any(type(q) is not Quote for q in quotes):
            raise ValueError('bounded exact quote batch required')
        quote_by_symbol = {q.symbol: q for q in quotes}
        if len(quote_by_symbol) != len(quotes):
            raise ValueError('duplicate quote symbol')
        required = {symbol} | {p.symbol for p in snapshot.account.positions} | {
            r['intent']['symbol'] for r in records if r['state'] in ACTIVE and r['intent']['side'] == 'buy'}
        if not required <= set(quote_by_symbol):
            raise ValueError('complete portfolio quotes required')
        if any(not (q.as_of <= q.received_at <= now and 0 <= (now - q.as_of).total_seconds() <= 30) for q in quotes):
            raise ValueError('portfolio quote freshness required')
        pending = tuple(PendingOrder(r['intent_id'], r['intent']['symbol'], r['intent']['side'],
                                    r['intent']['quantity'] - r['filled_quantity'], r['intent']['limit_price_usd'])
                        for r in records if r['state'] in ('open', 'partially_filled'))
        # Reserved intents are not broker orders. OrderLedger merges and checks
        # their reservations under its own transaction at admission/submission.
        available = cycle.positions_available
        if (type(available) is not tuple or any(type(p) is not BrokerPosition for p in available)
                or len({p.symbol for p in available}) != len(available)):
            raise ValueError('verified available positions required')
        held = {p.symbol: p.quantity for p in snapshot.account.positions}
        if len(held) != cycle.positions_count or {p.symbol: p.quantity for p in available} != held:
            raise ValueError('position source mismatch')
        for p in available:
            if (type(p.quantity) is not int or type(p.available_quantity) is not int
                    or not 0 <= p.available_quantity <= p.quantity
                    or not 0 <= (now - p.received_at).total_seconds() <= 30):
                raise ValueError('available position contract invalid')
            reserved = sum(o.remaining_quantity for o in pending if o.symbol == p.symbol and o.side == 'sell')
            reserved += sum(o.remaining_quantity for o in stop_reservations if o.symbol == p.symbol)
            if p.available_quantity != p.quantity - reserved:
                raise ValueError('available position reservation mismatch')
        baseline = (baselines.observe(book=book, cycle=cycle, calendar=calendar, now=now)
                    if accounting_validator is None else
                    baselines.observe_accounting(book=book, ledger=ledger, cycle=cycle, calendar=calendar, now=now))
        if baseline.book_digest != snapshot.receipt.digest or baseline.book_sequence != snapshot.receipt.sequence:
            raise ValueError('risk baseline account binding mismatch')
        source_time = min(snapshot.activity_until, snapshot.account.received_at, cycle.as_of,
                          baseline.source_as_of, *(p.received_at for p in snapshot.account.positions),
                          *(p.received_at for p in available), *group_times)
        if not 0 <= (now - source_time).total_seconds() <= 30:
            raise ValueError('stale accounting source')
        positions = tuple(Position(p.symbol, p.quantity, p.reference_price_usd, p.received_at)
                          for p in snapshot.account.positions)
        portfolio = Portfolio(source_time, snapshot.account.cash_usd, snapshot.account.equity_usd,
                              baseline.day_start_equity_usd, baseline.high_water_equity_usd,
                              positions=positions, pending_orders=pending, reconciled=True,
                              stop_reservations=stop_reservations,
                              valuation_quotes=tuple(sorted(quotes, key=lambda q: q.symbol)))
        state = RiskState(pause_entries=operator_state.pause_entries or not baseline.entries_ready or accounting_validator is not None,
                          halt_orders=operator_state.halt_orders,
                          seen_intent_ids=operator_state.seen_intent_ids | frozenset(r['intent_id'] for r in records))
        if (book.snapshot().receipt != snapshot.receipt or _digest(ledger.all_orders()) != orders_digest
                or baselines.replay().audit_sha256 != baseline.audit_sha256):
            raise ValueError('risk sources changed during preparation')
        _identity(baseline.audit_sha256)
        return PreparedRiskView(portfolio, quote_by_symbol[symbol], state, baseline.findings,
                                snapshot.receipt.digest, baseline.audit_sha256, orders_digest,
                                _digest([_base(q, Quote) for q in portfolio.valuation_quotes]), _version())
    except Exception:
        raise RiskViewError('risk_view_unavailable') from None
