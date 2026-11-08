"""Native OTO stop request/observation contracts. No HTTP or order submission.

A plan is not evidence of active protection. Partial parent fills, held/missing
children, pending cancellation and closed sessions remain explicit unsafe states.
Stop-market fills may be worse than the trigger. Group accounting and recovery
must be integrated before activating this path; the simple broker adapter does
not silently accept these orders.
"""
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from .broker import _tick, _shares, _uuid
from .data import Calendar, utc_time
from .risk import OrderIntent, Quote, _identity, _price, _symbol

STATUSES = frozenset(('held', 'new', 'accepted', 'pending_new', 'accepted_for_bidding', 'partially_filled',
                      'filled', 'pending_cancel', 'canceled', 'expired', 'rejected', 'replaced',
                      'pending_replace', 'done_for_day', 'suspended', 'calculated', 'stopped'))
ACTIVE_STOP = frozenset(('new', 'accepted', 'partially_filled'))


class ProtectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class NativeStopPlan:
    intent_id: str
    symbol: str
    quantity: int
    entry_limit_usd: Decimal
    stop_price_usd: Decimal
    expires_at: datetime
    time_in_force: str = 'day'

    def __post_init__(self):
        _identity(self.intent_id)
        _symbol(self.symbol)
        if type(self.quantity) is not int or not 1 <= self.quantity <= 1000000:
            raise ProtectionError('whole_native_stop_quantity_required')
        object.__setattr__(self, 'entry_limit_usd', _price(self.entry_limit_usd))
        object.__setattr__(self, 'stop_price_usd', _price(self.stop_price_usd))
        object.__setattr__(self, 'expires_at', utc_time(self.expires_at))
        _tick(self.entry_limit_usd)
        _tick(self.stop_price_usd)
        if (self.entry_limit_usd - self.stop_price_usd < Decimal('.01')
                or self.quantity * self.entry_limit_usd > 1000 or self.time_in_force not in ('day', 'gtc')):
            raise ProtectionError('native_stop_plan_outside_envelope')

    @classmethod
    def from_intent(cls, intent, *, time_in_force='day'):
        if type(intent) is not OrderIntent or intent.side != 'buy' or intent.stop_price_usd is None:
            raise ProtectionError('long_entry_intent_required')
        return cls(intent.intent_id, intent.symbol, intent.quantity, intent.limit_price_usd,
                   intent.stop_price_usd, intent.expires_at, time_in_force)

    def request_body(self, *, quote, market_price_usd, market_at, calendar, now):
        try:
            now, market_at = utc_time(now), utc_time(market_at)
            if type(quote) is not Quote or type(calendar) is not Calendar:
                raise ValueError('exact native stop market inputs required')
            session=calendar.at(now)
            if (quote.symbol != self.symbol or session is None or not now < self.expires_at
                    or (self.expires_at - now).total_seconds() > 60
                    or not quote.as_of <= quote.received_at <= now
                    or not 0 <= (now - quote.as_of).total_seconds() <= 30
                    or not 0 <= (now - market_at).total_seconds() <= 120
                    or not session.opens<=quote.as_of<session.closes or not session.opens<=market_at<session.closes
                    or min(self.entry_limit_usd, quote.bid, _price(market_price_usd)) - self.stop_price_usd < Decimal('.01')):
                raise ValueError('native stop input unavailable')
            return dict(symbol=self.symbol, side='buy', qty=str(self.quantity), type='limit',
                        limit_price=_tick(self.entry_limit_usd), time_in_force=self.time_in_force,
                        extended_hours=False, order_class='oto', client_order_id='pb3-' + self.intent_id[:40],
                        stop_loss=dict(stop_price=_tick(self.stop_price_usd)))
        except Exception:
            raise ProtectionError('native_stop_request_invalid') from None


@dataclass(frozen=True)
class GroupLeg:
    broker_order_id: str = field(repr=False)
    group_id: str | None = field(repr=False)
    status: str
    quantity: int
    filled_quantity: int
    average_fill_price_usd: Decimal | None
    created_at: datetime
    updated_at: datetime
    filled_at: datetime | None


@dataclass(frozen=True)
class OTOGroup:
    plan: NativeStopPlan
    parent: GroupLeg
    stop: GroupLeg | None
    received_at: datetime


def _leg(row, plan, received, parent):
    if not isinstance(row, dict):
        raise ValueError('group leg object required')
    quantity, filled = _shares(row['qty']), _shares(row['filled_qty'], zero=True)
    status = row['status']
    average = _price(row['filled_avg_price']) if filled else None
    created, updated = utc_time(row['created_at']), utc_time(row['updated_at'])
    completed = utc_time(row['filled_at']) if row.get('filled_at') is not None else None
    group = row.get('group_id')
    if group is not None:
        group = _uuid(group)
    if (row['asset_class'] != 'us_equity' or row['symbol'] != plan.symbol
            or quantity != plan.quantity or filled > quantity or status not in STATUSES
            or not created <= updated <= received or row['time_in_force'] != plan.time_in_force
            or row['extended_hours'] is not False
            or (status == 'filled' and filled != quantity)
            or (status == 'partially_filled' and not 0 < filled < quantity)
            or (status in ('held', 'new', 'accepted', 'pending_new', 'accepted_for_bidding', 'rejected') and filled != 0)
            or (completed is not None and (status != 'filled' or not created <= completed <= updated))
            or (not filled and row['filled_avg_price'] is not None and Decimal(str(row['filled_avg_price'])) != 0)):
        raise ValueError('group leg semantics mismatch')
    if parent:
        if (row['side'] != 'buy' or row['type'] != 'limit' or row['order_class'] != 'oto'
                or row['client_order_id'] != 'pb3-' + plan.intent_id[:40]
                or _price(row['limit_price']) != plan.entry_limit_usd
                or row.get('group_role') not in (None, 'entry') or (filled and average > plan.entry_limit_usd)):
            raise ValueError('entry group contract mismatch')
    elif (row['side'] != 'sell' or row['type'] != 'stop' or row.get('limit_price') is not None
          or row.get('order_class') not in ('oto', 'simple', '')
          or _price(row['stop_price']) != plan.stop_price_usd
          or row.get('group_role') not in (None, 'stop_loss') or row.get('legs') not in (None, [])):
        raise ValueError('stop group contract mismatch')
    return GroupLeg(_uuid(row['id']), group, status, quantity, filled, average, created, updated, completed)


def parse_oto_group(row, *, plan, received_at):
    try:
        if type(plan) is not NativeStopPlan:
            raise ValueError('exact native plan required')
        received = utc_time(received_at)
        parent = _leg(row, plan, received, True)
        legs = row.get('legs')
        if legs is None:
            legs = []
        if type(legs) is not list or len(legs) > 1:
            raise ValueError('single stop leg required')
        stop = _leg(legs[0], plan, received, False) if legs else None
        if stop is not None and (stop.broker_order_id == parent.broker_order_id
                or stop.created_at < parent.created_at or stop.updated_at < parent.created_at
                or (parent.group_id is not None and stop.group_id is not None and parent.group_id != stop.group_id)):
            raise ValueError('group identity mismatch')
        return OTOGroup(plan, parent, stop, received)
    except Exception:
        raise ProtectionError('native_stop_group_invalid') from None


@dataclass(frozen=True)
class ProtectionState:
    status: str
    remaining_quantity: int
    native_stop_active: bool
    pause_entries: bool
    required_actions: tuple


def assess_protection(group, *, calendar, now):
    try:
        now = utc_time(now)
        if type(group) is not OTOGroup or type(calendar) is not Calendar or not 0 <= (now - group.received_at).total_seconds() <= 30:
            raise ValueError('fresh exact group evidence required')
        parent, stop = group.parent, group.stop
        remaining = parent.filled_quantity - (stop.filled_quantity if stop is not None else 0)
        def state(status, active=False, pause=True, actions=('pause_entries', 'reconcile_group')):
            return ProtectionState(status, max(0, remaining), active, pause, actions)
        if remaining < 0 or (stop is not None and stop.filled_quantity and parent.filled_quantity != group.plan.quantity):
            return state('inconsistent_group')
        if stop is None:
            return state('protection_evidence_missing')
        if parent.filled_quantity == group.plan.quantity:
            if parent.status != 'filled':
                return state('inconsistent_group')
            if parent.filled_at is None:
                return state('entry_fill_time_unavailable')
            if ((stop.status in ACTIVE_STOP or stop.filled_quantity) and stop.updated_at < parent.filled_at
                    or (stop.filled_at is not None and stop.filled_at < parent.filled_at)):
                return state('inconsistent_group')
        if remaining == 0 and parent.filled_quantity == group.plan.quantity:
            if stop.filled_at is None:
                return state('exit_fill_time_unavailable')
            return state('closed', pause=False, actions=())
        if parent.filled_quantity == 0:
            if stop.status in ACTIVE_STOP or stop.status == 'pending_cancel':
                return state('inconsistent_group')
            if (parent.status in ('canceled', 'expired', 'rejected') and stop.filled_quantity == 0
                    and stop.status in ('canceled', 'expired', 'rejected')):
                return state('entry_not_filled', pause=False, actions=())
            if calendar.at(now) is None or now >= group.plan.expires_at:
                return state('entry_requires_cancellation', actions=('pause_entries', 'cancel_unfilled_entry', 'reconcile_group'))
            if parent.status in ('new', 'accepted', 'pending_new', 'accepted_for_bidding') and stop.status == 'held':
                return state('awaiting_entry', pause=False, actions=())
            return state('entry_state_unresolved')
        if calendar.at(now) is None:
            return state('out_of_session_exposure')
        if parent.filled_quantity < group.plan.quantity:
            if stop.status in ACTIVE_STOP or stop.status == 'pending_cancel':
                return state('inconsistent_group')
            actions = ('pause_entries', 'cancel_unfilled_entry', 'reconcile_group') if parent.status == 'partially_filled' else (
                'pause_entries', 'reconcile_group', 'review_unprotected_position')
            return state('unprotected_partial_entry', actions=actions)
        if parent.status != 'filled':
            return state('inconsistent_group')
        if stop.status in ACTIVE_STOP:
            return state('protected', active=True, pause=False, actions=())
        if stop.status in ('held', 'pending_new', 'accepted_for_bidding'):
            return state('protection_pending')
        if stop.status == 'pending_cancel':
            return state('protection_cancel_pending')
        return state('unprotected_position', actions=('pause_entries', 'reconcile_group', 'review_unprotected_position'))
    except Exception:
        raise ProtectionError('native_stop_assessment_unavailable') from None
