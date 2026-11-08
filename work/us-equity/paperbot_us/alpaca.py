"""Fixed-host, GET-only IEX data adapter. No order or live-account methods."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import gzip
import io
import http.client
import math
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from .data import Bar, Calendar, DataQualityError, Session, Snapshot, utc_time

DATA = 'https://data.alpaca.markets'
PAPER = 'https://paper-api.alpaca.markets'
_ALLOWED = {(DATA,'/v2/stocks/bars'), (DATA,'/v2/stocks/snapshots'),
            (PAPER,'/v2/assets'), (PAPER,'/v2/calendar'), (PAPER,'/v2/clock')}


class ProviderError(RuntimeError):
    """Only constant error codes; never HTTP bodies, credentials or URLs."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProviderError('redirect_refused')


def decode_json(raw):
    def parse_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('nonfinite JSON')
        return number
    def parse_int(value):
        number = int(value)
        if not -(2**63) <= number < 2**63:
            raise ValueError('integer outside market-data range')
        return number
    def constant(value):
        raise ValueError('nonstandard JSON')
    def unique_pairs(pairs):
        result = {}
        for key,value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    try:
        return json.loads(raw,parse_float=parse_float,parse_int=parse_int,
                          parse_constant=constant,object_pairs_hook=unique_pairs)
    except (ValueError,TypeError,UnicodeError,OverflowError):
        raise ProviderError('invalid_response') from None


@dataclass(frozen=True)
class Asset:
    security_id: str
    symbol: str
    exchange: str
    fractionable: bool

    def __post_init__(self):
        if not isinstance(self.security_id,str) or not self.security_id.strip():
            raise DataQualityError('missing security identity')
        if not isinstance(self.symbol,str) or not re.fullmatch(r'[A-Z][A-Z0-9.\-/]{0,19}',self.symbol):
            raise DataQualityError('invalid asset symbol')
        if self.exchange not in ('NYSE','NASDAQ','ARCA','NYSEARCA','BATS','AMEX') or type(self.fractionable) is not bool:
            raise DataQualityError('invalid asset eligibility')


@dataclass(frozen=True)
class UniverseSnapshot:
    assets: tuple
    received_at: datetime
    rejected_count: int
    source: str = 'alpaca'
    coverage: str = 'current_snapshot_only'


def _symbols(symbols):
    if not isinstance(symbols, (list, tuple)) or not 1 <= len(symbols) <= 20000:
        raise ProviderError('invalid_symbols')
    if any(not isinstance(s, str) or not re.fullmatch(r'[A-Z][A-Z0-9.\-/]{0,19}',s) for s in symbols):
        raise ProviderError('invalid_symbols')
    if len(set(symbols)) != len(symbols):
        raise ProviderError('duplicate_symbols')
    return list(symbols)


class AlpacaDataClient:
    def __init__(self, key_id, secret, *, transport=None, clock=None, monotonic=None, sleep=None, max_pages=100):
        if not isinstance(key_id,str) or not key_id or not isinstance(secret,str) or not secret:
            raise ProviderError('credentials_missing')
        if type(max_pages) is not int or not 1 <= max_pages <= 100:
            raise ProviderError('invalid_page_budget')
        self._key_id, self._secret = key_id, secret
        self._transport = transport or self._http_json
        self._clock = clock or (lambda:datetime.now(timezone.utc))
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleep or time.sleep
        self._last_request = None
        self._lock = threading.Lock()
        self.max_pages = max_pages
        self.snapshot_rejections = {}

    def __repr__(self):
        return '<AlpacaDataClient paper-only IEX read-only>'

    @classmethod
    def from_environment(cls, **kwargs):
        return cls(os.getenv('ALPACA_PAPER_KEY_ID'),os.getenv('ALPACA_PAPER_SECRET_KEY'),**kwargs)

    @staticmethod
    def _http_json(url, headers):
        request = urllib.request.Request(url,headers={**headers,'Accept-Encoding':'gzip'},method='GET')
        opener = urllib.request.build_opener(_NoRedirect())
        parsed = urllib.parse.urlsplit(url)
        # A current asset directory is large and refreshes infrequently. It is
        # not a price input; give that exact GET a separate bounded budget.
        budget = 120 if parsed.netloc=='paper-api.alpaca.markets' and parsed.path=='/v2/assets' else 30
        deadline = time.monotonic()+budget
        chunks,total = [],0
        limit = 20*1024*1024
        with opener.open(request,timeout=12) as response:
            encoding = getattr(response,'headers',{}).get('Content-Encoding','identity').strip().lower()
            if encoding not in ('identity','gzip'):
                raise ProviderError('unsupported_content_encoding')
            # read1 returns an available chunk rather than waiting to fill the
            # entire 20 MiB bound. A slow trickle cannot reset the total budget.
            # One socket operation may additionally consume its 12s timeout;
            # the service's process deadline also bounds DNS and all retries.
            while True:
                if time.monotonic()>=deadline:
                    raise ProviderError('response_deadline_exceeded')
                chunk = response.read1(min(65536,limit-total+1))
                if time.monotonic()>=deadline:
                    raise ProviderError('response_deadline_exceeded')
                if not chunk: break
                total += len(chunk)
                if total>limit: raise ProviderError('response_too_large')
                chunks.append(chunk)
        raw = b''.join(chunks)
        if encoding=='gzip':
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(raw)) as handle:
                    raw = handle.read(limit+1)
            except (OSError,EOFError):
                raise ProviderError('invalid_compressed_response') from None
            if len(raw)>limit: raise ProviderError('response_too_large')
        return decode_json(raw)

    def _rate_gate(self):
        # Single shared client: < 172 requests/minute, below Basic's 200/min.
        with self._lock:
            now = self._monotonic()
            if self._last_request is not None:
                self._sleep(max(0.0,0.35-(now-self._last_request)))
            self._last_request = self._monotonic()

    def _get(self, host, path, params):
        if (host,path) not in _ALLOWED:
            raise ProviderError('request_scope_refused')
        url = host+path+'?'+urllib.parse.urlencode(params)
        headers = {'APCA-API-KEY-ID':self._key_id,'APCA-API-SECRET-KEY':self._secret,
                   'Accept':'application/json','User-Agent':'paperbot-us/0.1.0'}
        for attempt in range(3):
            self._rate_gate()
            try:
                payload = self._transport(url,headers)
                if not isinstance(payload,(list,dict)):
                    raise ProviderError('invalid_response')
                return payload
            except urllib.error.HTTPError as exc:
                code = exc.code
                exc.close()
                if code in (429,500,502,503,504) and attempt < 2:
                    self._sleep(2**attempt)
                    continue
                raise ProviderError('HTTP_'+str(code)) from None
            except ProviderError:
                raise
            except (ValueError,TypeError,UnicodeError):
                raise ProviderError('invalid_response') from None
            except http.client.HTTPException:
                raise ProviderError('http_protocol_error') from None
            except (OSError,urllib.error.URLError):
                raise ProviderError('network_unavailable') from None
        raise ProviderError('request_failed')

    def assets(self):
        rows = self._get(PAPER,'/v2/assets',{'status':'active','asset_class':'us_equity'})
        if not isinstance(rows,list):
            raise ProviderError('invalid_assets')
        accepted, rejected, seen = [],0,set()
        for row in rows:
            if not isinstance(row,dict):
                rejected += 1
                continue
            symbol = row.get('symbol')
            # The HTTP query is asset_class; the asset object's response field
            # is class. Do not accept a query-shaped fixture as a real asset.
            eligible = (row.get('status') == 'active' and row.get('class') == 'us_equity'
                        and row.get('tradable') is True
                        and row.get('exchange') in ('NYSE','NASDAQ','ARCA','NYSEARCA','BATS','AMEX')
                        and isinstance(row.get('id'),str) and bool(row.get('id'))
                        and isinstance(symbol,str) and re.fullmatch(r'[A-Z][A-Z0-9.\-/]{0,19}',symbol)
                        and type(row.get('fractionable',False)) is bool)
            if not eligible:
                rejected += 1
                continue
            if symbol in seen:
                raise ProviderError('duplicate_asset_symbol')
            seen.add(symbol)
            accepted.append(Asset(row['id'],symbol,row['exchange'],row.get('fractionable',False)))
        return UniverseSnapshot(tuple(sorted(accepted,key=lambda a:a.symbol)),utc_time(self._clock()),rejected)

    def calendar(self, start, end):
        # Parse coverage before a request; missing dates inside this range mean closed.
        Calendar([],start=start,end=end)
        rows = self._get(PAPER,'/v2/calendar',{'start':start,'end':end,'date_type':'TRADING'})
        if not isinstance(rows,list):
            raise ProviderError('invalid_calendar')
        try:
            return Calendar([Session.from_alpaca(r) for r in rows],start=start,end=end)
        except DataQualityError:
            raise ProviderError('invalid_calendar') from None

    def snapshots(self, symbols):
        symbols = _symbols(symbols)
        result = {}
        self.snapshot_rejections = {}
        if (len(symbols)+99)//100 > self.max_pages:
            raise ProviderError('pagination_budget_exceeded')
        for offset in range(0,len(symbols),100):
            batch = symbols[offset:offset+100]
            response = self._get(DATA,'/v2/stocks/snapshots',{'symbols':','.join(batch),'feed':'iex','currency':'USD'})
            if not isinstance(response,dict) or not set(response).issubset(batch):
                raise ProviderError('invalid_snapshots')
            if any(not isinstance(value,dict) for value in response.values()):
                raise ProviderError('invalid_snapshots')
            received = utc_time(self._clock())
            for symbol in batch:
                if symbol not in response:
                    self.snapshot_rejections[symbol] = 'missing_snapshot'
                    continue
                try:
                    result[symbol] = Snapshot.from_alpaca(symbol,response[symbol],received_at=received)
                except DataQualityError:
                    self.snapshot_rejections[symbol] = 'invalid_snapshot'
        return result

    def bars(self, symbols, *, start, end):
        symbols = _symbols(symbols)
        try:
            start, end = utc_time(start), utc_time(end)
        except DataQualityError:
            raise ProviderError('invalid_date_range') from None
        if not start < end:
            raise ProviderError('invalid_date_range')
        result = {symbol:[] for symbol in symbols}
        total_pages = 0
        for offset in range(0,len(symbols),100):
            batch = symbols[offset:offset+100]
            token, seen_tokens, seen_bars = None,set(),set()
            for page in range(self.max_pages):
                if total_pages >= self.max_pages:
                    raise ProviderError('pagination_budget_exceeded')
                total_pages += 1
                params = dict(symbols=','.join(batch),timeframe='15Min',start=start.isoformat(),end=end.isoformat(),
                              limit=10000,feed='iex',adjustment='raw',currency='USD',sort='asc')
                if token:
                    params['page_token'] = token
                response = self._get(DATA,'/v2/stocks/bars',params)
                received = utc_time(self._clock())
                if not isinstance(response,dict) or not isinstance(response.get('bars'),dict):
                    raise ProviderError('invalid_bars')
                if not set(response['bars']).issubset(batch):
                    raise ProviderError('unrequested_symbol')
                for symbol,rows in response['bars'].items():
                    if not isinstance(rows,list):
                        raise ProviderError('invalid_bars')
                    for row in rows:
                        try:
                            bar = Bar.from_alpaca(symbol,row,received_at=received,feed='iex')
                        except (DataQualityError,TypeError):
                            raise ProviderError('invalid_bar') from None
                        identity = (symbol,bar.starts)
                        if identity in seen_bars:
                            raise ProviderError('duplicate_bar')
                        seen_bars.add(identity)
                        if not start <= bar.starts <= end or bar.starts > received:
                            raise ProviderError('bar_outside_request')
                        # The API may include the current unfinished aggregate.
                        if bar.ends > min(end,received):
                            continue
                        if result[symbol] and bar.starts <= result[symbol][-1].starts:
                            raise ProviderError('unordered_bars')
                        result[symbol].append(bar)
                token = response.get('next_page_token')
                if token is None:
                    break
                if not isinstance(token,str) or not token or token in seen_tokens:
                    raise ProviderError('pagination_loop')
                seen_tokens.add(token)
            else:
                raise ProviderError('pagination_budget_exceeded')
        return result
