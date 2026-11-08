"""Group-aware paper GET observations and durable recovery. No POST or DELETE.

Client lookup establishes the parent ID; a separate nested GET reads its family.
404 never proves a timed-out submission was unsent. Open orders are flat,
bounded references for cross-checking, not standalone proof of active protection.
"""
from dataclasses import dataclass,field
from datetime import datetime
from decimal import Decimal
import re
import time

from .broker import PaperBroker,BrokerError,_uuid,_shares
from .data import utc_time
from .native_orders import _leg_record,_check_leg
from .orders import OrderLedger
from .protection import NativeStopPlan,STATUSES,parse_oto_group
from .risk import _price,_symbol


@dataclass(frozen=True)
class OpenOrderReference:
    broker_order_id: str=field(repr=False)
    client_order_id: str|None=field(repr=False)
    symbol: str
    side: str
    order_type: str
    order_class: str
    quantity: int
    filled_quantity: int
    average_fill_price_usd: Decimal|None
    limit_price_usd: Decimal|None
    stop_price_usd: Decimal|None
    status: str
    time_in_force: str
    created_at: datetime
    updated_at: datetime
    received_at: datetime
    filled_at: datetime|None=None
    mode: str=field(default='paper',init=False)


def _reference(row,received):
    symbol=row['symbol'];_symbol(symbol)
    qty,filled=_shares(row['qty']),_shares(row['filled_qty'],zero=True)
    created,updated=utc_time(row['created_at']),utc_time(row['updated_at'])
    status,kind,side=row['status'],row['type'],row['side']
    client=row.get('client_order_id')
    average=_price(row['filled_avg_price']) if filled else None
    if (row['asset_class']!='us_equity' or side not in ('buy','sell') or kind not in ('limit','stop')
            or row.get('order_class') not in ('simple','oto','') or row['time_in_force'] not in ('day','gtc')
            or row['extended_hours'] is not False or row.get('legs') not in (None,[])
            or status not in STATUSES or filled>qty or not created<=updated<=received
            or (client is not None and (not isinstance(client,str) or not re.fullmatch(r'[A-Za-z0-9._-]{1,128}',client)))
            or (not filled and row['filled_avg_price'] is not None and Decimal(str(row['filled_avg_price']))!=0)
            or (status=='filled' and filled!=qty) or (status=='partially_filled' and not 0<filled<qty)
            or (status in ('held','new','accepted','pending_new','accepted_for_bidding','rejected') and filled!=0)):
        raise ValueError('open order reference mismatch')
    completed=utc_time(row['filled_at']) if row.get('filled_at') is not None else None
    if completed is not None and (filled!=qty or not created<=completed<=updated):
        raise ValueError('invalid complete fill timestamp')
    limit=_price(row['limit_price']) if kind=='limit' else None
    stop=_price(row['stop_price']) if kind=='stop' else None
    if kind=='stop' and (side!='sell' or row.get('limit_price') is not None):
        raise ValueError('only protective sell stop-market supported')
    if kind=='limit' and row.get('stop_price') is not None: raise ValueError('unexpected stop price')
    return OpenOrderReference(_uuid(row['id']),client,symbol,side,kind,row['order_class'],qty,filled,
        average,limit,stop,status,row['time_in_force'],created,updated,received,completed)


class NativePaperReader:
    def __init__(self,broker,*,monotonic=None):
        if type(broker) is not PaperBroker: raise BrokerError('exact_paper_broker_required')
        self.broker=broker
        self._monotonic=monotonic or time.monotonic

    def _lookup(self,plan):
        if type(plan) is not NativeStopPlan: raise BrokerError('exact_native_plan_required')
        deadline=self._monotonic()+25
        def check():
            if self._monotonic()>=deadline: raise ValueError('native read deadline')
        client='pb3-'+plan.intent_id[:40]
        reply=self.broker._request('GET','/v2/orders:by_client_order_id',params=dict(client_order_id=client))
        check()
        if reply.status==404: return None
        if reply.status!=200: raise ValueError('client lookup status')
        first=parse_oto_group(reply.payload,plan=plan,received_at=utc_time(self.broker._clock()))
        reply=self.broker._request('GET','/v2/orders/'+first.parent.broker_order_id,params=dict(nested='true'))
        check()
        if reply.status!=200: raise ValueError('nested lookup unavailable')
        group=parse_oto_group(reply.payload,plan=plan,received_at=utc_time(self.broker._clock()))
        if group.received_at<first.received_at: raise ValueError('native receipt regression')
        _check_leg(_leg_record(first.parent),_leg_record(group.parent),parent=True,limit=plan.entry_limit_usd)
        if first.stop is not None:
            if group.stop is None: raise ValueError('first observed child disappeared')
            _check_leg(_leg_record(first.stop),_leg_record(group.stop),parent=False,limit=None)
        return group,reply.payload

    def lookup(self,plan):
        try:
            result=self._lookup(plan)
            return result[0] if result is not None else None
        except Exception: raise BrokerError('native_paper_lookup_unavailable') from None

    def recover(self,ledger,identity):
        try:
            if type(ledger) is not OrderLedger: raise ValueError('exact ledger required')
            record=ledger.load(identity);native=record.get('native_stop')
            if native is None or record['submission_claimed_at'] is None:
                raise ValueError('claimed native plan required')
            result=self._lookup(NativeStopPlan(**native['plan']))
            if result is None: return None
            group,row=result
            return ledger.observe_native_group(identity,row=row,received_at=group.received_at)
        except Exception: raise BrokerError('native_paper_recovery_unavailable') from None

    def open_orders(self):
        try:
            reply=self.broker._request('GET','/v2/orders',params=dict(status='open',limit=500,nested='false'))
            if reply.status!=200 or type(reply.payload) is not list or len(reply.payload)>=500:
                raise ValueError('complete bounded flat orders required')
            received=utc_time(self.broker._clock())
            result=tuple(_reference(row,received) for row in reply.payload)
            if len({o.broker_order_id for o in result})!=len(result): raise ValueError('duplicate open order')
            return result
        except Exception: raise BrokerError('native_paper_open_orders_unavailable') from None
