"""Private paper-only broker adapter. No CLI, scheduler or live fallback.

Public submit/cancel go through the durable ledger. Transport helpers are
private; only the coordinator may supply already normalized risk observations.
No account identifiers, raw HTTP errors or provider bodies are logged/retained.
Simple limit orders do NOT provide broker protective stops. Activation requires
the separate reconciliation, protection and strategy acceptance gates.
"""
from dataclasses import dataclass,field,replace
from contextlib import ExitStack
from datetime import date,datetime,timedelta,timezone
from decimal import Decimal
import json
import hashlib
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from .alpaca import decode_json
from .data import utc_time,DataQualityError
from .risk import OrderIntent,_money,_price,_symbol,_identity

PAPER='https://paper-api.alpaca.markets'


class BrokerError(RuntimeError): pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): raise BrokerError('paper_redirect_refused')


@dataclass(frozen=True)
class HttpReply:
    status: int
    payload: object=None


@dataclass(frozen=True)
class AccountView:
    received_at: datetime
    cash_usd: Decimal
    equity_usd: Decimal
    blocked: bool
    account_sha256: str=field(repr=False)
    mode: str=field(default='paper',init=False)


@dataclass(frozen=True)
class BrokerPosition:
    symbol: str
    quantity: int
    available_quantity: int
    received_at: datetime


@dataclass(frozen=True)
class BrokerOrder:
    broker_order_id: str=field(repr=False)
    client_order_id: str|None=field(repr=False)
    owned: bool
    symbol: str
    side: str
    quantity: int
    limit_price_usd: Decimal
    status: str
    filled_quantity: int
    average_fill_price_usd: Decimal|None
    created_at: datetime
    updated_at: datetime
    received_at: datetime


def _uuid(value):
    if not isinstance(value,str) or str(uuid.UUID(value))!=value.lower(): raise ValueError('order identity')
    return value.lower()


def _account_fingerprint(value):
    return hashlib.sha256(b'paperbot-paper-account-v1\0'+_uuid(value).encode()).hexdigest()


def _client_id(value):
    if not isinstance(value,str) or not re.fullmatch(r'pb3-[0-9a-f]{40}',value): raise BrokerError('invalid_client_order_id')
    return value


def _shares(value,*,zero=False):
    amount=_money(value,maximum=Decimal('1000000'))
    if amount!=amount.to_integral_value() or (not zero and amount==0): raise ValueError('whole shares')
    return int(amount)


def _tick(value):
    value=_price(value)
    decimals=2 if value>=1 else 4
    if value!=value.quantize(Decimal(1).scaleb(-decimals)): raise BrokerError('invalid_limit_tick')
    return format(value,f'.{decimals}f')


def _order(row,received):
    try:
        if not isinstance(row,dict): raise ValueError('order object')
        identity=_uuid(row['id']);client=row['client_order_id'];owned=isinstance(client,str) and bool(re.fullmatch(r'pb3-[0-9a-f]{40}',client))
        _symbol(row['symbol'])
        qty,filled=_shares(row['qty']),_shares(row['filled_qty'],zero=True)
        limit=_price(row['limit_price']);_tick(limit)
        average=_price(row['filled_avg_price']) if filled else None
        created,updated=utc_time(row['created_at']),utc_time(row['updated_at'])
        statuses=dict(new='open',accepted='open',pending_new='open',accepted_for_bidding='open',
            partially_filled='partially_filled',filled='filled',canceled='canceled',expired='expired',rejected='rejected')
        status=statuses[row['status']]
        if (row['asset_class']!='us_equity' or row['side'] not in ('buy','sell') or row['type']!='limit'
                or row['order_class'] not in ('','simple') or row['time_in_force']!='day' or row['extended_hours'] is not False
                or not created<=updated<=received or filled>qty
                or (status=='open' and filled!=0) or (status=='partially_filled' and not 0<filled<qty)
                or (status=='filled' and filled!=qty) or (status=='rejected' and filled!=0)
                or (not filled and row['filled_avg_price'] is not None and _money(row['filled_avg_price'])!=0)
                or (filled and ((row['side']=='buy' and average>limit) or (row['side']=='sell' and average<limit)))):
            raise ValueError('order semantics')
        return BrokerOrder(identity,client if owned else None,owned,row['symbol'],row['side'],qty,limit,
            status,filled,average,created,updated,received)
    except Exception: raise BrokerError('unsupported_or_invalid_order') from None


def _activity(row,received):
    from .accounting import Activity,signed_money
    if not isinstance(row,dict): raise ValueError('activity object')
    raw_id=row['id']
    if not isinstance(raw_id,str) or not re.fullmatch(r'[0-9]{17}::[0-9a-fA-F-]{36}',raw_id):
        raise ValueError('activity identity')
    canonical=raw_id[:19]+_uuid(raw_id[19:]);identity=hashlib.sha256(canonical.encode()).hexdigest()
    kind=row['activity_type']
    if kind=='FILL':
        at=utc_time(row['transaction_time'])
        if at>received or row['type'] not in ('fill','partial_fill'): raise ValueError('fill time/type')
        return Activity(identity,'FILL',symbol=row['symbol'],side=row['side'],quantity=_shares(row['qty']),
            price_usd=_price(row['price']),cumulative_quantity=_shares(row['cum_qty'],zero=True),
            leaves_quantity=_shares(row['leaves_qty'],zero=True),broker_order_id=_uuid(row['order_id']),occurred_at=at)
    if kind=='FEE':
        if row.get('currency','USD')!='USD': raise ValueError('fee currency')
        raw_day=row['date']
        at=None if isinstance(raw_day,str) and len(raw_day)==10 else utc_time(raw_day)
        if at is not None and at>received: raise ValueError('future fee source timestamp')
        day=date.fromisoformat(raw_day) if at is None else at.date()
        return Activity(identity,'FEE',net_usd=signed_money(row['net_amount']),source_day=day.isoformat(),source_time=at)
    return Activity(identity,'UNSUPPORTED',provider_kind=kind)


class PaperBroker:
    def __init__(self,key_id,secret,*,transport=None,clock=None):
        if (not isinstance(key_id,str) or not re.fullmatch(r'PK[A-Za-z0-9]{14,510}',key_id)
                or not isinstance(secret,str) or not re.fullmatch(r'[A-Za-z0-9]{16,512}',secret)):
            raise BrokerError('paper_credentials_required')
        self._key,self._secret=key_id,secret
        self._transport=transport or self._http
        self._clock=clock or (lambda:datetime.now(timezone.utc))

    def __repr__(self): return '<PaperBroker paper-only>'

    @classmethod
    def from_environment(cls):
        return cls(os.environ.get('ALPACA_PAPER_KEY_ID'),os.environ.get('ALPACA_PAPER_SECRET_KEY'))

    @staticmethod
    def _http(method,url,headers,body):
        parts=urllib.parse.urlsplit(url)
        if (parts.scheme!='https' or parts.netloc!='paper-api.alpaca.markets' or parts.fragment
                or method not in ('GET','POST','DELETE')): raise BrokerError('paper_scope_refused')
        raw=None if body is None else json.dumps(body,allow_nan=False,separators=(',',':')).encode()
        request=urllib.request.Request(url,data=raw,headers=headers,method=method)
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),_NoRedirect())
        try:
            with opener.open(request,timeout=12) as response:
                status=response.status
                if status==204: return HttpReply(status)
                if response.headers.get('Content-Encoding','identity').lower()!='identity': raise BrokerError('paper_encoding_refused')
                chunks=[];total=0;limit=2*1024*1024;deadline=time.monotonic()+20
                while True:
                    if time.monotonic()>=deadline: raise BrokerError('paper_response_deadline')
                    chunk=response.read1(min(65536,limit-total+1))
                    if time.monotonic()>=deadline: raise BrokerError('paper_response_deadline')
                    if not chunk: break
                    total+=len(chunk)
                    if total>limit: raise BrokerError('paper_response_too_large')
                    chunks.append(chunk)
                return HttpReply(status,decode_json(b''.join(chunks)))
        except urllib.error.HTTPError as error:
            status=error.code;error.close()
            return HttpReply(status)
        except Exception: raise BrokerError('paper_transport_failed') from None

    def _request(self,method,path,*,params=None,body=None):
        exact={("GET",'/v2/account'),("GET",'/v2/positions'),("GET",'/v2/orders'),("GET",'/v2/account/activities'),
               ("GET",'/v2/orders:by_client_order_id'),("POST",'/v2/orders')}
        if (method,path) not in exact and not (method in ('GET','DELETE') and re.fullmatch(r'/v2/orders/[0-9a-f-]{36}',path)):
            raise BrokerError('paper_scope_refused')
        if path.startswith('/v2/orders/'):
            try: _uuid(path.rsplit('/',1)[1])
            except Exception: raise BrokerError('paper_scope_refused') from None
        url=PAPER+path+('?' +urllib.parse.urlencode(params) if params else '')
        headers={'APCA-API-KEY-ID':self._key,'APCA-API-SECRET-KEY':self._secret,
            'Accept':'application/json','Accept-Encoding':'identity','Content-Type':'application/json','User-Agent':'paperbot-us-paper/0.1'}
        try:
            reply=self._transport(method,url,headers,body)
            if type(reply) is not HttpReply or type(reply.status) is not int: raise ValueError('reply contract')
            return reply
        except Exception:
            raise BrokerError('submission_outcome_unknown' if method=='POST' else 'cancel_outcome_unknown' if method=='DELETE' else 'paper_read_failed') from None

    def account(self):
        try:
            reply=self._request('GET','/v2/account');row=reply.payload
            if (reply.status!=200 or not isinstance(row,dict) or row['status']!='ACTIVE' or row['currency']!='USD'
                    or type(row['trading_blocked']) is not bool or type(row['account_blocked']) is not bool): raise ValueError('account contract')
            suspended=row.get('trade_suspended_by_user',False)
            if type(suspended) is not bool: raise ValueError('account suspension flag')
            return AccountView(utc_time(self._clock()),_money(row['cash']),_money(row['equity']),
                row['trading_blocked'] or row['account_blocked'] or suspended,_account_fingerprint(row['id']))
        except Exception: raise BrokerError('paper_account_unavailable') from None

    def positions(self):
        try:
            reply=self._request('GET','/v2/positions');rows=reply.payload
            if reply.status!=200 or not isinstance(rows,list) or len(rows)>1000: raise ValueError('positions budget')
            result=[];now=utc_time(self._clock())
            for row in rows:
                _symbol(row['symbol']);qty=_shares(row['qty']);available=_shares(row['qty_available'],zero=True)
                if row['asset_class']!='us_equity' or row['side']!='long' or available>qty: raise ValueError('position semantics')
                result.append(BrokerPosition(row['symbol'],qty,available,now))
            if len({p.symbol for p in result})!=len(result): raise ValueError('duplicate position')
            return tuple(result)
        except Exception: raise BrokerError('paper_positions_unavailable') from None

    def open_orders(self):
        try:
            reply=self._request('GET','/v2/orders',params=dict(status='open',limit=500,direction='asc',nested='false'))
            if reply.status!=200 or not isinstance(reply.payload,list) or len(reply.payload)>=500: raise ValueError('possibly truncated book')
            now=utc_time(self._clock());orders=tuple(_order(row,now) for row in reply.payload)
            if (len({o.broker_order_id for o in orders})!=len(orders)
                    or len({o.client_order_id for o in orders if o.owned})!=sum(o.owned for o in orders)
                    or any(o.status not in ('open','partially_filled') for o in orders)): raise ValueError('book semantics')
            return orders
        except Exception: raise BrokerError('paper_order_book_unavailable') from None

    def has_order_history(self):
        """Conservative bootstrap probe; all statuses, not just open orders.

        One returned object is sufficient to refuse an empty-account baseline.
        This is neither a complete historical archive nor a fill reconciliation.
        """
        try:
            reply=self._request('GET','/v2/orders',params=dict(status='all',limit=1,direction='desc',nested='false'))
            if (reply.status!=200 or not isinstance(reply.payload,list) or len(reply.payload)>1
                    or any(not isinstance(row,dict) or not row for row in reply.payload)):
                raise ValueError('order history probe contract')
            return bool(reply.payload)
        except Exception: raise BrokerError('paper_order_history_unavailable') from None

    def account_marks(self):
        """Broker accounting marks; receipt time does not establish quote freshness."""
        from .accounting import AccountMark
        try:
            reply=self._request('GET','/v2/positions');rows=reply.payload
            if reply.status!=200 or not isinstance(rows,list) or len(rows)>1000: raise ValueError('account marks budget')
            now=utc_time(self._clock());result=[]
            for row in rows:
                if row['asset_class']!='us_equity' or row['side']!='long': raise ValueError('account mark semantics')
                result.append(AccountMark(row['symbol'],_shares(row['qty']),row['current_price'],row['market_value'],now))
            if len({p.symbol for p in result})!=len(result): raise ValueError('duplicate account mark')
            return tuple(result)
        except Exception: raise BrokerError('paper_account_marks_unavailable') from None

    def activities(self,*,after,until=None,max_pages=20):
        """Full anchor window, all types. Partial pagination never returns success."""
        from .accounting import ActivityWindow
        try:
            before=utc_time(self._clock());after=utc_time(after);until=utc_time(until) if until is not None else before
            if not after<until<=before or type(max_pages) is not int or not 1<=max_pages<=100: raise ValueError('activity range/budget')
            token=None;seen=set();result=[];deadline=time.monotonic()+90
            for _ in range(max_pages):
                if time.monotonic()>=deadline: raise ValueError('activity deadline')
                params=dict(after=after.isoformat(),until=until.isoformat(),direction='asc',page_size=100)
                if token is not None: params['page_token']=token
                reply=self._request('GET','/v2/account/activities',params=params)
                if time.monotonic()>=deadline or reply.status!=200 or not isinstance(reply.payload,list) or len(reply.payload)>100:
                    raise ValueError('activity response')
                received=utc_time(self._clock())
                for row in reply.payload:
                    item=_activity(row,received)
                    if item.identity in seen: raise ValueError('repeated activity')
                    seen.add(item.identity);result.append(item)
                if len(reply.payload)<100:
                    return ActivityWindow(after,until,received,tuple(result))
                token=reply.payload[-1]['id']
            raise ValueError('activity pagination incomplete')
        except Exception: raise BrokerError('paper_activity_window_unavailable') from None

    def lookup(self,client_order_id):
        _client_id(client_order_id)
        try:
            reply=self._request('GET','/v2/orders:by_client_order_id',params=dict(client_order_id=client_order_id))
            if reply.status==404: return None
            if reply.status!=200: raise ValueError('lookup status')
            order=_order(reply.payload,utc_time(self._clock()))
            if order.client_order_id!=client_order_id or not order.owned: raise ValueError('lookup identity')
            return order
        except Exception: raise BrokerError('paper_order_lookup_failed') from None

    def _submit_limit(self,intent,*,before_send=None):
        if type(intent) is not OrderIntent: raise BrokerError('exact_order_intent_required')
        now=utc_time(self._clock())
        if before_send is None and not now<intent.expires_at<=now+timedelta(seconds=60): raise BrokerError('intent_not_fresh')
        price=_tick(intent.limit_price_usd)
        if intent.quantity*intent.limit_price_usd>1000: raise BrokerError('initial_order_budget_exceeded')
        body=dict(symbol=intent.symbol,qty=str(intent.quantity),side=intent.side,type='limit',time_in_force='day',
            limit_price=price,client_order_id='pb3-'+intent.intent_id[:40],extended_hours=False,order_class='simple')
        if before_send is not None and not before_send(utc_time(self._clock())): return None
        try:
            reply=self._request('POST','/v2/orders',body=body)
            if reply.status not in (200,201): raise ValueError('submission status')
            order=_order(reply.payload,utc_time(self._clock()))
            if (not order.owned or order.client_order_id!=body['client_order_id'] or order.symbol!=intent.symbol
                    or order.side!=intent.side or order.quantity!=intent.quantity or order.limit_price_usd!=intent.limit_price_usd):
                raise ValueError('submission response mismatch')
            return order
        except Exception: raise BrokerError('submission_outcome_unknown') from None

    def submit(self,ledger,identity,*,controls=None,control_snapshot=None,**risk_inputs):
        from .orders import OrderLedger,recheck_ticket
        from .controls import ControlJournal,ControlSnapshot,ControlError
        if type(ledger) is not OrderLedger: raise BrokerError('durable_ledger_required')
        if type(controls) is not ControlJournal or type(control_snapshot) is not ControlSnapshot:
            raise BrokerError('durable_operator_controls_required')
        _identity(identity)
        if ledger.load(identity).get('native_stop') is not None:
            raise BrokerError('native_group_gateway_required')
        ticket=ledger.prepare_submission(identity,**{**risk_inputs,'now':utc_time(self._clock()),'gateway':'simple_limit'})
        if ticket is None: return None
        guard_at=utc_time(self._clock())
        def final_check(at,current):
            # The control read lease spans this callback and HTTP dispatch.
            # Recheck time AFTER acquisition, including any slow journal read.
            checked_at=utc_time(self._clock())
            decision=recheck_ticket(ticket,now=checked_at)
            reasons=list(decision.violations)
            floor=max(at,guard_at,utc_time(ticket['updated_at']))
            if checked_at<floor:
                reasons.append('submission_clock_regression')
            if not current: reasons.append('operator_control_changed')
            elif control_snapshot.halt_orders: reasons.append('all_orders_halted')
            elif control_snapshot.pause_entries and ticket['intent']['side']=='buy': reasons.append('entries_paused')
            if reasons: decision=replace(decision,allowed=False,violations=tuple(dict.fromkeys(reasons)))
            if not decision.allowed:
                ledger.reject_unsubmitted(ticket,decision,at=max(checked_at,floor),observed_at=checked_at)
            return decision.allowed
        with ExitStack() as stack:
            try: current=stack.enter_context(controls.dispatch_lease(control_snapshot,now=guard_at))
            except ControlError:
                final_check(guard_at,False)
                return None
            order=self._submit_limit(OrderIntent(**ticket['intent']),before_send=lambda at:final_check(at,current is True))
        if order is None: return None
        return ledger.observe(identity,status=order.status,filled_quantity=order.filled_quantity,
            average_fill_price_usd=order.average_fill_price_usd,broker_order_id=order.broker_order_id,
            broker_created_at=order.created_at,broker_event_at=order.updated_at,received_at=order.received_at)

    def _cancel(self,broker_order_id):
        try:
            identity=_uuid(broker_order_id)
            reply=self._request('DELETE','/v2/orders/'+identity)
            if reply.status!=204: raise ValueError('cancellation not acknowledged')
            return 'cancel_requested'
        except Exception: raise BrokerError('cancel_outcome_unknown') from None

    def cancel(self,ledger,identity):
        from .orders import OrderLedger
        if type(ledger) is not OrderLedger: raise BrokerError('durable_ledger_required')
        ticket=ledger.prepare_cancellation(identity,at=utc_time(self._clock()))
        return self._cancel(ticket['broker_order_id']) if ticket else None
