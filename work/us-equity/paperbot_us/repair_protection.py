"""Bound proposal for a standalone protective stop after a failed close.

No broker I/O, order reservation, sender, CLI or service. Preparation can append
the existing accounting risk baseline. A request body neither owns shares nor
proves protection: a future coordinator needs a durable family allocation,
decision/control checks, one-shot dispatch and independently reconciled fills.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN, localcontext
import hashlib
from pathlib import Path

from .accountbook import AccountBook
from .broker import _tick
from .data import Calendar, Session, utc_time
from .native_cleanup import PreparedCleanupView, prepare_cleanup_risk_view, project_groups
from .orders import OrderLedger, _base, _hash, _json
from .risk import RiskLimits, _identity, _price
from .risk_baselines import RiskBaselineStore
from .risk_view import PreparedRiskView, _version as view_version


class RepairError(RuntimeError):
    pass


def _version():
    digest = hashlib.sha256()
    for name in ('repair_protection.py', 'native_cleanup.py', 'cleanup_recovery.py',
                 'risk_view.py', 'risk_baselines.py', 'orders.py', 'risk.py', 'data.py',
                 'broker.py', 'native_reconciler.py', 'native_orders.py',
                 'accountbook.py', 'accounting.py', 'protection.py', 'repair_orders.py','repair_decisions.py','repair_execution.py','compensation.py','native_admission.py','native_execution.py'):
        digest.update(name.encode() + b'\0' + Path(__file__).with_name(name).read_bytes() + b'\0')
    return digest.hexdigest()


def _calendar(calendar):
    if type(calendar) is not Calendar:
        raise ValueError('exact repair calendar required')
    return _json(dict(start=calendar.start, end=calendar.end,
        sessions=[_base(s, Session) for _, s in sorted(calendar.sessions.items())]))


def _limits(limits):
    if type(limits) is not RiskLimits:
        raise ValueError('exact repair limits required')
    return _hash(_json(_base(limits, RiskLimits)))


def _facts(prepared, records, *, limits, calendar, now):
    if type(prepared) is not PreparedCleanupView or type(prepared.risk) is not PreparedRiskView:
        raise ValueError('exact repair view required')
    _identity(prepared.family_id)
    view = prepared.risk
    group = project_groups(records)[prepared.family_id]
    family = next(r for r in records if r['intent_id'] == prepared.family_id)
    session = calendar.at(now)
    age = min(30, limits.max_age_seconds)
    quote = view.quote
    for record in records:
        owner = record.get('native_cleanup')
        if (owner is not None and owner['family_intent_id'] == prepared.family_id
                and record['submission_claimed_at'] is not None):
            if (record.get('cleanup_lookup_state') != 'observed'
                    or record.get('cleanup_observed_at') is None
                    or not 0 <= (now - utc_time(record['cleanup_observed_at'])).total_seconds() <= age):
                raise ValueError('fresh successful terminal cleanup recovery required')
    if (view.code_sha256 != view_version() or view.orders_digest != _hash(_json(records))
            or not group.parent_terminal or not group.child_terminal or group.remaining_quantity <= 0
            or group.available_quantity != group.remaining_quantity
            or prepared.available_quantity != group.available_quantity
            or prepared.remaining_quantity != group.remaining_quantity
            or group.observed_at is None or not 0 <= (now - group.observed_at).total_seconds() <= age
            or not view.portfolio.reconciled or not view.state.pause_entries or view.state.halt_orders
            or not 0 <= (now - view.portfolio.as_of).total_seconds() <= age
            or session is None or quote.symbol != group.symbol
            or not session.opens <= quote.as_of < session.closes
            or not quote.as_of <= quote.received_at <= now
            or not 0 <= (now - quote.as_of).total_seconds() <= age):
        raise ValueError('fresh unreserved remaining family shares required')
    with localcontext() as ctx:
        ctx.prec = 40
        if ((quote.ask - quote.bid) * 10000 / quote.bid > limits.max_spread_bps
                or quote.ask * group.remaining_quantity > min(Decimal('1000'), limits.max_order_notional_usd)):
            raise ValueError('repair outside fixed price/liquidity envelope')
        original = _price(family['native_stop']['plan']['stop_price_usd'])
        candidate = min(original, quote.bid - Decimal('.01'))
        tick = Decimal('.01') if candidate >= 1 else Decimal('.0001')
        trigger = candidate.quantize(tick, rounding=ROUND_DOWN)
        if trigger <= 0 or quote.bid - trigger < Decimal('.01'):
            raise ValueError('legal positive repair trigger unavailable')
        _tick(trigger)
    return group, trigger


@dataclass(frozen=True)
class ProtectionRepairPlan:
    prepared: PreparedCleanupView = field(repr=False)
    family_id: str = field(repr=False)
    quantity: int
    stop_price_usd: Decimal
    prepared_at: datetime
    expires_at: datetime
    limits_sha256: str = field(repr=False)
    calendar_json: str = field(repr=False)
    code_sha256: str = field(repr=False)

    def public_summary(self):
        return dict(mode='paper', phase='protection_repair_prepared', quantity=self.quantity,
            preview_stop_price_usd=float(self.stop_price_usd), orders_submitted=0,
            protection_active=False, ready_for_execution=False, execution_enabled=False)

    def request_body(self, *, book, ledger, baselines, calendar, limits, now):
        try:
            now = utc_time(now)
            for value, exact in ((book, AccountBook), (ledger, OrderLedger), (baselines, RiskBaselineStore)):
                if type(value) is not exact:
                    raise ValueError('exact repair accounting sources required')
            if (self.limits_sha256 != _limits(limits) or self.calendar_json != _calendar(calendar)
                    or self.code_sha256 != _version() or self.family_id != self.prepared.family_id
                    or self.expires_at != self.prepared_at + timedelta(seconds=20)
                    or not self.prepared_at <= now < self.expires_at):
                raise ValueError('repair plan expired or changed')
            records = ledger.all_orders()
            group, trigger = _facts(self.prepared, records, limits=limits, calendar=calendar, now=now)
            snapshot = book.snapshot()
            baseline = baselines.replay()
            view = self.prepared.risk
            if (type(self.quantity) is not int or self.quantity != group.remaining_quantity
                    or self.stop_price_usd != trigger or snapshot.receipt.digest != view.book_digest
                    or not snapshot.receipt.result.reconciled or baseline.audit_sha256 != view.baseline_digest
                    or baseline.book_digest != snapshot.receipt.digest
                    or baseline.book_sequence != snapshot.receipt.sequence
                    or not 0 <= (now - baseline.source_as_of).total_seconds() <= min(30, limits.max_age_seconds)):
                raise ValueError('repair accounting or allocation proposal changed')
            identity = _hash(_json(['protection-repair-v1', self.family_id, view.orders_digest,
                view.book_digest, view.baseline_digest, self.quantity, trigger, self.prepared_at]))
            return dict(symbol=group.symbol, side='sell', type='stop', qty=str(self.quantity),
                stop_price=_tick(trigger), time_in_force='gtc', extended_hours=False,
                order_class='simple', client_order_id='pb3r-' + identity[:40])
        except Exception:
            raise RepairError('protection_repair_request_unavailable') from None


def prepare_repair(*, book, ledger, baselines, cycle, quotes, family_id,
                   operator_state, calendar, limits, now):
    try:
        now = utc_time(now)
        limits_sha = _limits(limits)
        calendar_json = _calendar(calendar)
        prepared = prepare_cleanup_risk_view(book=book, ledger=ledger, baselines=baselines,
            cycle=cycle, quotes=quotes, family_id=family_id, operator_state=operator_state,
            calendar=calendar, now=now)
        group, trigger = _facts(prepared, ledger.all_orders(), limits=limits, calendar=calendar, now=now)
        return ProtectionRepairPlan(prepared, family_id, group.remaining_quantity, trigger, now,
            now + timedelta(seconds=20), limits_sha, calendar_json, _version())
    except Exception:
        raise RepairError('protection_repair_preparation_unavailable') from None
