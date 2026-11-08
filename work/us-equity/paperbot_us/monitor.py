"""GET-only paper-account observation. No execution or model calls."""
import argparse
from contextlib import closing
from datetime import datetime,timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import uuid

from .alpaca import AlpacaDataClient,decode_json
from .data import utc_time
from .risk import _money,_symbol


class MonitorError(RuntimeError): pass


def _json(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def _hash(value): return hashlib.sha256(value.encode()).hexdigest()


def _signed(value):
    if not isinstance(value,str): raise ValueError('reported decimal required')
    negative=value.startswith('-');magnitude=_money(value[1:] if negative else value)
    return -magnitude if negative else magnitude


def _holdings(rows):
    result=[]
    for row in rows:
        _symbol(row['symbol'])
        if row['asset_class']!='us_equity' or row['side']!='long': raise ValueError('long US equities required')
        result.append(dict(symbol=row['symbol'],quantity=float(_money(row['qty'],positive=True,maximum=Decimal('1000000'))),
            average_cost_usd=float(_money(row['avg_entry_price'],positive=True,maximum=Decimal('1000000'))),
            reference_price_usd=float(_money(row['current_price'],positive=True,maximum=Decimal('1000000'))),
            unrealized_usd=float(_signed(row['unrealized_pl']))))
    if len({r['symbol'] for r in result})!=len(result): raise ValueError('duplicate holding')
    return sorted(result,key=lambda r:r['symbol'])[:20]


def _recent_trades(rows,now):
    result=[];seen=set()
    statuses={'new','accepted','pending_new','accepted_for_bidding','partially_filled','filled','done_for_day',
        'canceled','expired','replaced','pending_cancel','pending_replace','stopped','rejected','suspended','calculated'}
    for row in rows:
        identity=row['id'];_symbol(row['symbol'])
        if (not isinstance(identity,str) or not 1<=len(identity)<=128 or identity in seen
                or row['asset_class']!='us_equity' or row['side'] not in ('buy','sell') or row['status'] not in statuses):
            raise ValueError('order summary contract')
        seen.add(identity)
        quantity=_money(row['qty'],positive=True,maximum=Decimal('1000000'))
        filled=_money(row['filled_qty'],maximum=Decimal('1000000'))
        submitted=utc_time(row['submitted_at']).timestamp()
        if not submitted<=now or filled>quantity or (row['status']=='filled' and filled!=quantity):
            raise ValueError('fill summary bounds')
        if not filled: continue
        kind='filled_at' if row['status']=='filled' else 'updated_at'
        at=utc_time(row[kind]).timestamp()
        if not submitted<=at<=now: raise ValueError('fill summary time')
        result.append(dict(symbol=row['symbol'],side=row['side'],quantity=float(filled),
            average_price_usd=float(_money(row['filled_avg_price'],positive=True,maximum=Decimal('1000000'))),
            reported_at=at,time_kind=kind,status='filled' if row['status']=='filled' else 'partially_filled'))
    return result[:20]


def _source_hash():
    digest=hashlib.sha256()
    for name in ('monitor.py','risk.py','data.py','alpaca.py','audit.py','scanner.py','runner.py'):
        digest.update(name.encode()+b'\0'+Path(__file__).with_name(name).read_bytes()+b'\0')
    return digest.hexdigest()


class AccountReader:
    def __init__(self,key,secret,*,transport=None,clock=None):
        if (not isinstance(key,str) or not re.fullmatch(r'PK[A-Za-z0-9]{14,510}',key)
                or not isinstance(secret,str) or not re.fullmatch(r'[A-Za-z0-9]{16,512}',secret)):
            raise MonitorError('paper_credentials_required')
        self._key,self._secret=key,secret
        self._transport=transport or AlpacaDataClient._http_json
        self._clock=clock or (lambda:datetime.now(timezone.utc))

    def __repr__(self): return '<AccountReader paper-only GET-only>'

    @classmethod
    def from_environment(cls):
        return cls(os.environ.get('ALPACA_PAPER_KEY_ID'),os.environ.get('ALPACA_PAPER_SECRET_KEY'))

    def read(self):
        try:
            headers={'APCA-API-KEY-ID':self._key,'APCA-API-SECRET-KEY':self._secret,'Accept':'application/json'}
            account=self._transport('https://paper-api.alpaca.markets/v2/account',headers)
            received=utc_time(self._clock()).timestamp()
            positions=self._transport('https://paper-api.alpaca.markets/v2/positions',headers)
            orders=self._transport('https://paper-api.alpaca.markets/v2/orders?status=open&limit=500',headers)
            history=self._transport('https://paper-api.alpaca.markets/v2/orders?status=all&limit=50&direction=desc&nested=false',headers)
            if (not isinstance(account,dict) or account['currency']!='USD' or account['status']!='ACTIVE'
                    or type(account['trading_blocked']) is not bool or type(account['account_blocked']) is not bool
                    or not isinstance(positions,list) or len(positions)>1000 or not isinstance(orders,list) or len(orders)>=500
                    or not isinstance(history,list) or len(history)>50):
                raise ValueError('account contract')
            extra=account.get('trade_suspended_by_user',False)
            if type(extra) is not bool: raise ValueError('account flag')
            return dict(cash_usd=_money(account['cash']),equity_usd=_money(account['equity']),
                positions_count=len(positions),orders_count=len(orders),received_at=received,
                holdings=_holdings(positions),recent_trades=_recent_trades(history,utc_time(self._clock()).timestamp()),
                holdings_limited=len(positions)>20,history_window_limited=len(history)==50,
                blocked=account['trading_blocked'] or account['account_blocked'] or extra)
        except Exception: raise MonitorError('paper_account_read_failed') from None


class ObservationArchive:
    def __init__(self,path): self.path=Path(path).absolute()

    def _head(self,db):
        meta=db.execute('SELECT version,started_at,head_seq,head_digest FROM metadata').fetchall()
        if len(meta)!=1 or meta[0][0]!=1: raise MonitorError('observation_schema_mismatch')
        version,started,sequence,digest=meta[0]
        last=db.execute('SELECT seq,payload,previous_digest,digest FROM observations ORDER BY seq DESC LIMIT 1').fetchone()
        if sequence==0:
            if last is not None or digest!='0'*64: raise MonitorError('observation_head_mismatch')
        elif (last is None or last[0]!=sequence or last[3]!=digest or _hash(last[2]+'\0'+last[1])!=digest):
            raise MonitorError('observation_head_mismatch')
        return started,sequence,digest

    def _verify_chain(self,db):
        started,head_seq,head_digest=self._head(db);previous='0'*64;count=0
        for sequence,payload,prior,digest in db.execute('SELECT seq,payload,previous_digest,digest FROM observations ORDER BY seq'):
            count+=1
            if (count>100000 or sequence!=count or len(payload)>16384 or prior!=previous
                    or _hash(prior+'\0'+payload)!=digest): raise MonitorError('observation_integrity_failed')
            previous=digest
        if count!=head_seq or previous!=head_digest: raise MonitorError('observation_integrity_failed')
        return started,head_seq,head_digest

    def append(self,report):
        try:
            if self.path.is_symlink(): raise MonitorError('observation_link_refused')
            try:
                fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600);os.close(fd)
                created=True
            except FileExistsError: created=False
            with closing(sqlite3.connect(self.path,timeout=10,isolation_level=None)) as db:
                db.execute('PRAGMA synchronous=FULL');db.execute('BEGIN IMMEDIATE')
                if created:
                    db.execute('CREATE TABLE metadata(version INTEGER,started_at REAL,head_seq INTEGER,head_digest TEXT)')
                    db.execute('INSERT INTO metadata VALUES(1,?,0,?)',(report['generated_at'],'0'*64))
                    db.execute('CREATE TABLE observations(seq INTEGER PRIMARY KEY,payload TEXT,previous_digest TEXT,digest TEXT)')
                # Stream the committed prefix before healthy publication;
                # checking only its tail misses older corruption.
                started,sequence,previous=self._verify_chain(db)
                if sequence>=100000: raise MonitorError('observation_retention_budget')
                if sequence:
                    last=decode_json(db.execute('SELECT payload FROM observations WHERE seq=?',(sequence,)).fetchone()[0])
                    if report['generated_at']<last['generated_at']: raise MonitorError('observation_clock_regression')
                value={**report,'observing_started_at':started}
                payload=_json(value)
                if len(payload)>16384: raise MonitorError('observation_size_budget')
                digest=_hash(previous+'\0'+payload)
                db.execute('INSERT INTO observations VALUES(?,?,?,?)',(sequence+1,payload,previous,digest))
                db.execute('UPDATE metadata SET head_seq=?,head_digest=?',(sequence+1,digest));db.commit()
            if created and os.name!='nt':
                fd=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY)
                try: os.fsync(fd)
                finally: os.close(fd)
            return {**value,'audit_sha256':digest}
        except Exception: raise MonitorError('observation_archive_failed') from None

    def verify(self):
        try:
            if self.path.is_symlink() or not self.path.is_file(): raise MonitorError('observation_missing')
            with closing(sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True)) as db:
                _,count,_=self._verify_chain(db)
            return dict(records=count,integrity_verified=True)
        except Exception: raise MonitorError('observation_verify_failed') from None


def _base(now):
    return dict(schema=1,mode='paper',currency='USD',capital_usd=10000,phase='observing',execution_enabled=False,
        generated_at=now,account_received_at=None,observing_started_at=None,paper_started_at=None,
        status='account_unavailable',accounting_ok=False,equity_usd=None,cash_usd=None,profit_usd=None,
        return_pct=None,positions_count=None,open_orders_count=None,audit_sha256=None,
        details_ok=False,holdings=[],recent_trades=[],holdings_limited=False,history_window_limited=False,
        scan=dict(status='unavailable',as_of=None,pool_size=None,checked_count=None,candidate_count=None),
        model_status='pending_secure_rotation')


def _scan(path,now):
    if path is None: return _base(now)['scan']
    try:
        from .audit import ScanArchive,replay
        archive=ScanArchive(path)
        with closing(sqlite3.connect(archive.path.as_uri()+'?mode=ro',uri=True,timeout=10)) as db:
            row=db.execute('SELECT run_id FROM scans ORDER BY rowid DESC LIMIT 1').fetchone()
        if row is None: raise ValueError('no scan')
        record=archive.load(row[0]);replay(record)
        result=record['result'];at=utc_time(result['as_of']).timestamp();selection=record['inputs']['selection']
        if not 0<=now-at<=180 or result['status'] not in ('ok','market_closed'): raise ValueError('scan age')
        return dict(status=result['status'],as_of=at,pool_size=selection['total'],
                    checked_count=selection['count'] if result['status']=='ok' else 0,candidate_count=len(result['candidates']))
    except Exception: return _base(now)['scan']


def _publish(path,value):
    path=Path(path)
    if path.is_symlink(): raise MonitorError('public_projection_link_refused')
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o640)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as handle:
            handle.write(_json(value));handle.flush()
            if os.name!='nt': os.fchmod(handle.fileno(),0o644)
            os.fsync(handle.fileno())
        os.replace(temporary,path)
        if os.name!='nt':
            fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
    finally:
        if temporary.exists(): temporary.unlink()


def observe_once(*,reader,archive,publish,scan_archive,clock=None):
    clock=clock or (lambda:datetime.now(timezone.utc))
    report=_base(utc_time(clock()).timestamp())
    public_fields=set(report)
    report.update(code_sha256=_source_hash(),account_evidence=None)
    try:
        account=reader.read();now=utc_time(clock()).timestamp()
        report['account_evidence']={name:account[name] for name in ('positions_count','orders_count','received_at','blocked')}
        report['account_evidence'].update(cash_usd=str(account['cash_usd']),equity_usd=str(account['equity_usd']))
        fresh=0<=now-account['received_at']<=60
        if fresh:
            report.update(details_ok=True,**{name:account[name] for name in ('holdings','recent_trades','holdings_limited','history_window_limited')})
        report.update(generated_at=now,account_received_at=account['received_at'],
                      positions_count=account['positions_count'],open_orders_count=account['orders_count'])
        # No fill book is silently invented. Until the execution/reconciliation
        # stage exists, only the verified empty 10,000 USD baseline is reconciled.
        valid=(not account['blocked'] and account['cash_usd']==account['equity_usd']==10000
               and account['positions_count']==account['orders_count']==0 and fresh
               and not account['recent_trades'] and not account['history_window_limited'])
        report.update(accounting_ok=valid,status='observing' if valid else 'account_unreconciled')
        if valid: report.update(equity_usd=10000,cash_usd=10000,profit_usd=0,return_pct=0)
    except Exception: pass
    report['scan']=_scan(scan_archive,report['generated_at'])
    try: report=archive.append(report)
    except Exception:
        report=_base(utc_time(clock()).timestamp());report['status']='audit_unavailable'
    report={name:report[name] for name in public_fields}
    _publish(publish,report)
    return report


def main():
    parser=argparse.ArgumentParser(description='Paper account GET-only observation; no orders')
    parser.add_argument('--archive',required=True);parser.add_argument('--scan-archive',required=True);parser.add_argument('--publish',required=True)
    args=parser.parse_args()
    try:
        result=observe_once(reader=AccountReader.from_environment(),archive=ObservationArchive(args.archive),
                            publish=args.publish,scan_archive=args.scan_archive)
        print(_json(dict(status=result['status'],phase='observing',orders_submitted=0)),flush=True)
        return 0 if result['accounting_ok'] else 1
    except Exception:
        print(_json(dict(status='observation_unavailable',orders_submitted=0)),flush=True)
        return 1


if __name__=='__main__': sys.exit(main())
