"""Durable paper intents, reservations and cumulative order observations.

No broker/network calls. A coordinator may POST only the one ticket returned by
prepare_submission; every unknown outcome requires broker lookup/reconciliation,
never a blind retry. Evidence hashes detect corruption, not privileged rewrites.
"""
from contextlib import contextmanager,closing
from dataclasses import fields,replace
from datetime import date,datetime
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid

from .alpaca import decode_json
from .data import utc_time,DataQualityError
from .risk import (OrderIntent,Portfolio,Position,PendingOrder,StopReservation,RiskState,evaluate,
                   RiskDecision,Quote,RiskLimits,_identity,_money,_price)

ACTIVE=('reserved','submission_unknown','open','partially_filled')
TERMINAL=('filled','canceled','rejected','expired','rejected_by_risk')
BROKER_STATES=('open','partially_filled','filled','canceled','rejected','expired')


class LedgerError(RuntimeError): pass


def _json(value):
    def encode(v):
        if type(v) is Decimal: return format(v,'f')
        if isinstance(v,(date,datetime)): return v.isoformat()
        if type(v) is frozenset: return sorted(v)
        raise TypeError('unsupported private evidence field')
    result=json.dumps(value,default=encode,sort_keys=True,separators=(',',':'),allow_nan=False)
    if len(result.encode())>2*1024*1024: raise LedgerError('ledger_record_too_large')
    return result


def _hash(raw): return hashlib.sha256(raw.encode()).hexdigest()
def _normalize(value): return decode_json(_json(value))
def _base(value,base): return {f.name:getattr(value,f.name) for f in fields(base)}
def _ts(value): return utc_time(value).isoformat(timespec='microseconds')


def _code_hash():
    digest=hashlib.sha256()
    for name in ('risk.py','orders.py','native_orders.py','native_broker.py','native_recovery.py','protection.py','data.py',
                 'native_cleanup.py','risk_view.py','risk_baselines.py','native_reconciler.py',
                 'cleanup_execution.py','decision_journal.py','controls.py','native_close.py','cleanup_recovery.py','repair_orders.py','repair_decisions.py','repair_execution.py','compensation.py','native_admission.py','native_execution.py'):
        digest.update(name.encode()+b'\0'+Path(__file__).with_name(name).read_bytes()+b'\0')
    return digest.hexdigest()


def recheck_ticket(ticket,*,now):
    """Pure final clock check of the exact already-audited effective portfolio.

    It runs AFTER the durable claim, with no disk/network operation between
    this calculation and the broker request. Other admissions see the claim
    as unknown and cannot treat its cash or shares as available.
    """
    from .data import Calendar,Session
    context=ticket['context']
    p=context['portfolio'];q=context['quote'];s=context['state'];c=context['calendar']
    portfolio=Portfolio(**{k:p[k] for k in ('as_of','cash_usd','equity_usd','day_start_equity_usd','high_water_equity_usd','reconciled')},
        positions=tuple(Position(**value) for value in p['positions']),
        pending_orders=tuple(PendingOrder(**value) for value in p['pending_orders']),
        stop_reservations=tuple(StopReservation(**value) for value in p['stop_reservations']),
        valuation_quotes=tuple(Quote(**{k:value[k] for k in ('symbol','bid','ask','as_of','received_at')})
                               for value in p['valuation_quotes']))
    quote=Quote(**{k:q[k] for k in ('symbol','bid','ask','as_of','received_at')})
    state=RiskState(s['pause_entries'],s['halt_orders'],frozenset(s['seen_intent_ids']))
    calendar=Calendar([Session(date.fromisoformat(value['day']),value['opens'],value['closes']) for value in c['sessions']],
                      start=c['start'],end=c['end'])
    return evaluate(OrderIntent(**ticket['intent']),portfolio=portfolio,quote=quote,state=state,
                    limits=RiskLimits(**context['limits']),calendar=calendar,now=now)


class OrderLedger:
    def __init__(self,path): self.path=Path(path).absolute()

    def initialize(self):
        """Explicit first creation only. Existing databases are never migrated."""
        try:
            if self.path.is_symlink(): raise LedgerError('ledger_link_refused')
            if self.path.exists():
                with self._db(False): pass
                return
            self.path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
            os.close(fd)
            with closing(sqlite3.connect(self.path)) as db:
                db.executescript('''
                    PRAGMA journal_mode=DELETE;
                    PRAGMA synchronous=FULL;
                    BEGIN IMMEDIATE;
                    CREATE TABLE metadata(version INTEGER NOT NULL);
                    INSERT INTO metadata VALUES(1);
                    CREATE TABLE orders(intent_id TEXT PRIMARY KEY,client_order_id TEXT UNIQUE NOT NULL,
                        state TEXT NOT NULL,updated_at TEXT NOT NULL,payload TEXT NOT NULL,digest TEXT NOT NULL);
                    CREATE INDEX active_orders ON orders(state);
                    CREATE TABLE events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,intent_id TEXT NOT NULL,
                        payload TEXT NOT NULL,previous_digest TEXT NOT NULL,digest TEXT NOT NULL);
                    CREATE INDEX order_events ON events(intent_id,sequence);
                    COMMIT;
                ''')
            if os.name!='nt':
                fd=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY)
                try: os.fsync(fd)
                finally: os.close(fd)
        except LedgerError: raise
        except Exception: raise LedgerError('ledger_initialization_failed') from None

    @contextmanager
    def _db(self,write):
        db=None
        try:
            if self.path.is_symlink() or not self.path.is_file(): raise LedgerError('ledger_missing_or_link_refused')
            db=sqlite3.connect(self.path.as_uri()+('?mode=rw' if write else '?mode=ro'),uri=True,timeout=10,isolation_level=None)
            db.row_factory=sqlite3.Row
            db.execute('PRAGMA synchronous=FULL')
            db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            if [r[0] for r in db.execute('SELECT version FROM metadata')]!=[1]: raise LedgerError('unsupported_ledger_version')
            db.execute('SELECT intent_id,client_order_id,state,updated_at,payload,digest FROM orders LIMIT 0')
            db.execute('SELECT sequence,intent_id,payload,previous_digest,digest FROM events LIMIT 0')
            yield db
            db.commit()
        except LedgerError:
            if db is not None: db.rollback()
            raise
        except Exception:
            if db is not None: db.rollback()
            raise LedgerError('ledger_operation_failed') from None
        finally:
            if db is not None: db.close()

    def _load(self,db,identity,missing=False):
        row=db.execute('SELECT * FROM orders WHERE intent_id=?',(identity,)).fetchone()
        if row is None:
            if missing: return None
            raise LedgerError('intent_not_found')
        if _hash(row['payload'])!=row['digest']: raise LedgerError('order_integrity_mismatch')
        record=decode_json(row['payload'])
        if (record['schema']!=1 or record['intent_id']!=identity or record['state']!=row['state']
                or record['updated_at']!=row['updated_at'] or record['client_order_id']!=row['client_order_id']
                or record['client_order_id']!='pb3-'+identity[:40]):
            raise LedgerError('order_identity_mismatch')
        event=db.execute('SELECT payload,previous_digest,digest FROM events WHERE intent_id=? ORDER BY sequence DESC LIMIT 1',(identity,)).fetchone()
        if event is None or _hash(event['previous_digest']+'\0'+event['payload'])!=event['digest'] or decode_json(event['payload'])['record']!=record:
            raise LedgerError('order_audit_link_mismatch')
        return record

    def load(self,identity):
        _identity(identity)
        with self._db(False) as db: return self._load(db,identity)

    def _persist(self,db,record,kind,at):
        record=_normalize(record)
        payload=_json(record)
        db.execute('INSERT INTO orders VALUES(?,?,?,?,?,?) ON CONFLICT(intent_id) DO UPDATE SET state=excluded.state,updated_at=excluded.updated_at,payload=excluded.payload,digest=excluded.digest',
                   (record['intent_id'],record['client_order_id'],record['state'],record['updated_at'],payload,_hash(payload)))
        last=db.execute('SELECT digest FROM events ORDER BY sequence DESC LIMIT 1').fetchone()
        previous=last[0] if last else '0'*64
        event=_json(dict(kind=kind,at=_ts(at),record=record))
        db.execute('INSERT INTO events(intent_id,payload,previous_digest,digest) VALUES(?,?,?,?)',
                   (record['intent_id'],event,previous,_hash(previous+'\0'+event)))
        return record

    def _risk(self,db,intent,*,portfolio,quote,state,limits,calendar,now,exclude=None,_native_cleanup=None,_native_entry=None):
        if type(portfolio) is not Portfolio or type(state) is not RiskState: raise DataQualityError('risk contracts required')
        if _native_cleanup is not None:
            from .native_cleanup import _RISK_TOKEN
            if _native_cleanup is not _RISK_TOKEN or intent.side!='sell' or intent.stop_price_usd is not None:
                raise DataQualityError('native cleanup scope required')
        if _native_entry is not None:
            from .native_admission import _RISK_TOKEN
            if _native_entry is not _RISK_TOKEN or intent.side!='buy' or intent.stop_price_usd is None or _native_cleanup is not None:
                raise DataQualityError('native entry scope required')
        rows=db.execute('SELECT intent_id FROM orders WHERE state IN (?,?,?,?) LIMIT 1001',ACTIVE).fetchall()
        if len(rows)>1000: raise LedgerError('active_order_budget_exceeded')
        local={row[0]:self._load(db,row[0]) for row in rows if row[0]!=exclude}
        # Exclude only our own local reservation during rechecking. A broker
        # order with that identity is evidence to reconcile, never to resubmit.
        broker={p.intent_id:p for p in portfolio.pending_orders}
        combined=dict(broker);findings=[]
        # The legacy portfolio contract has no native stop reservation. Until
        # a group-aware coordinator supplies that contract, never regard those
        # shares as free for a second sell or silently ignore a missing child.
        for row in db.execute('SELECT intent_id FROM orders WHERE intent_id!=? LIMIT 10001',(exclude or '',)):
            if self._load(db,row[0]).get('native_stop') is not None and _native_cleanup is None and _native_entry is None:
                findings.append('native_group_coordinator_required')
                break
        for identity in broker:
            if identity not in local: findings.append('unrecognized_broker_order')
        for identity,record in local.items():
            original=OrderIntent(**record['intent'])
            remaining=original.quantity-record['filled_quantity']
            pending=PendingOrder(identity,original.symbol,original.side,remaining,original.limit_price_usd)
            supplied=broker.get(identity)
            if supplied is not None and supplied!=pending: findings.append('broker_order_mismatch')
            if record['state'] in ('open','partially_filled') and supplied is None: findings.append('missing_broker_order')
            if record['state']=='submission_unknown': findings.append('unconfirmed_submission')
            combined[identity]=pending
        # A provider snapshot predating a locally confirmed fill cannot be used
        # as fresh purchasing power, even if it is under the nominal age limit.
        for row in db.execute('SELECT intent_id FROM orders WHERE updated_at>? AND state IN (?,?,?,?)',
                              (_ts(portfolio.as_of),'filled','partially_filled','canceled','expired')):
            if self._load(db,row[0])['filled_quantity']>0: findings.append('account_before_confirmed_fill')
        effective=replace(portfolio,pending_orders=tuple(combined[key] for key in sorted(combined)),
                          reconciled=portfolio.reconciled and not findings)
        effective_state=replace(state,seen_intent_ids=state.seen_intent_ids-{exclude}) if exclude else state
        decision=evaluate(intent,portfolio=effective,quote=quote,state=effective_state,limits=limits,calendar=calendar,now=now)
        from .risk import Quote,RiskLimits
        from .data import Session
        portfolio_record=_base(effective,Portfolio)
        portfolio_record['positions']=[_base(p,Position) for p in effective.positions]
        portfolio_record['pending_orders']=[_base(p,PendingOrder) for p in effective.pending_orders]
        portfolio_record['stop_reservations']=[_base(p,StopReservation) for p in effective.stop_reservations]
        portfolio_record['valuation_quotes']=[_base(q,Quote) for q in effective.valuation_quotes]
        context=dict(portfolio=portfolio_record,quote=_base(quote,Quote),state=_base(effective_state,RiskState),
                     limits=_base(limits,RiskLimits),observed_at=utc_time(now),
                     broker_pending_orders=[_base(p,PendingOrder) for p in portfolio.pending_orders],
                     reconciliation_findings=sorted(set(findings)),
                     calendar=dict(start=calendar.start,end=calendar.end,sessions=[_base(s,Session) for s in calendar.sessions.values()]))
        return decision,context

    def admit(self,intent,**risk_inputs):
        if '_native_cleanup' in risk_inputs: raise LedgerError('dedicated_native_cleanup_admission_required')
        if '_native_entry' in risk_inputs: raise LedgerError('dedicated_native_entry_admission_required')
        if type(intent) is not OrderIntent: raise LedgerError('exact_order_intent_required')
        identity=intent.intent_id
        with self._db(True) as db:
            existing=self._load(db,identity,missing=True)
            serialized=_normalize(_base(intent,OrderIntent))
            if existing is not None:
                if existing['intent']!=serialized: raise LedgerError('idempotency_payload_conflict')
                return existing
            decision,context=self._risk(db,intent,**risk_inputs)
            at=_ts(risk_inputs['now'])
            record=dict(schema=1,intent_id=identity,client_order_id='pb3-'+identity[:40],intent=serialized,
                        state='reserved' if decision.allowed else 'rejected_by_risk',
                        risk=_base(decision,type(decision)),filled_quantity=0,filled_notional_usd='0',
                        broker_order_id=None,broker_created_at=None,broker_updated_at=None,submission_claimed_at=None,cancel_requested_at=None,
                        created_at=at,updated_at=at,
                        context=context,code_sha256=_code_hash())
            return self._persist(db,record,'admission',risk_inputs['now'])

    def prepare_submission(self,identity,*,gateway='simple_limit',**risk_inputs):
        if '_native_cleanup' in risk_inputs: raise LedgerError('dedicated_native_cleanup_claim_required')
        if '_native_entry' in risk_inputs: raise LedgerError('dedicated_native_entry_claim_required')
        _identity(identity)
        with self._db(True) as db:
            record=self._load(db,identity)
            if record.get('native_cleanup') is not None: raise LedgerError('native_cleanup_gateway_required')
            if record.get('native_entry') is not None: raise LedgerError('native_entry_gateway_required')
            if record['state']!='reserved': return None
            expected='native_oto_stop' if record.get('native_stop') is not None else 'simple_limit'
            if gateway!=expected: raise LedgerError('submission_gateway_mismatch')
            now=utc_time(risk_inputs['now'])
            intent=OrderIntent(**record['intent'])
            portfolio=risk_inputs['portfolio']
            if type(portfolio) is not Portfolio: raise DataQualityError('risk contracts required')
            already_at_broker=any(p.intent_id==identity for p in portfolio.pending_orders)
            if now>=intent.expires_at and not already_at_broker:
                record.update(state='expired',updated_at=_ts(now))
                self._persist(db,record,'expired_before_submission',now)
                return None
            decision,context=self._risk(db,intent,exclude=identity,**risk_inputs)
            if already_at_broker:
                context['reconciliation_findings']=sorted(set(context['reconciliation_findings'])|{'broker_order_already_exists'})
                record.update(state='submission_unknown',risk=_base(decision,type(decision)),context=context,updated_at=_ts(now))
                self._persist(db,record,'broker_order_requires_reconciliation',now)
                return None
            record.update(state='submission_unknown' if decision.allowed else 'rejected_by_risk',
                          risk=_base(decision,type(decision)),context=context,updated_at=_ts(now),
                          submission_claimed_at=_ts(now) if decision.allowed else None)
            result=self._persist(db,record,'submission_claim' if decision.allowed else 'submission_veto',now)
            return result if decision.allowed else None

    def prepare_cancellation(self,identity,*,at):
        """Claim a cancellation once; its outcome never frees a reservation."""
        _identity(identity);at=utc_time(at)
        with self._db(True) as db:
            record=self._load(db,identity)
            if record.get('native_stop') is not None: raise LedgerError('native_group_cancellation_gateway_required')
            if record.get('native_cleanup') is not None: raise LedgerError('native_cleanup_cancellation_gateway_required')
            if (record['state'] not in ACTIVE or record['broker_order_id'] is None
                    or record.get('cancel_requested_at') is not None): return None
            if at<utc_time(record['updated_at']): raise LedgerError('cancellation_clock_regression')
            record.update(cancel_requested_at=_ts(at),updated_at=_ts(at))
            return self._persist(db,record,'cancel_claim',at)

    def reject_unsubmitted(self,ticket,decision,*,at,observed_at=None):
        """Only the claimant may record a pre-network veto of its unchanged ticket."""
        if type(decision) is not RiskDecision or decision.allowed: raise LedgerError('explicit_veto_required')
        with self._db(True) as db:
            record=self._load(db,ticket['intent_id'])
            if (record!=ticket or record['state']!='submission_unknown' or record['broker_order_id'] is not None
                    or utc_time(at)<utc_time(record['updated_at'])): raise LedgerError('submission_ticket_changed')
            observed=utc_time(observed_at) if observed_at is not None else utc_time(at)
            if observed>utc_time(at) or (observed<utc_time(at) and 'submission_clock_regression' not in decision.violations):
                raise LedgerError('invalid_veto_observation_time')
            record.update(state='rejected_by_risk',risk=_base(decision,RiskDecision),updated_at=_ts(at))
            record['context']['observed_at']=_ts(observed)
            record['context']['veto_recorded_at']=_ts(at)
            return self._persist(db,record,'pre_network_veto',at)

    def observe(self,identity,*,status,filled_quantity,average_fill_price_usd,broker_order_id,broker_created_at,broker_event_at,received_at):
        _identity(identity)
        with self._db(True) as db:
            return self._observe(db,identity,status=status,filled_quantity=filled_quantity,
                average_fill_price_usd=average_fill_price_usd,broker_order_id=broker_order_id,
                broker_created_at=broker_created_at,broker_event_at=broker_event_at,received_at=received_at)

    def _observe(self,db,identity,*,status,filled_quantity,average_fill_price_usd,broker_order_id,broker_created_at,broker_event_at,received_at,_allow_cleanup_terminal_update=False):
        record=self._load(db,identity);intent=OrderIntent(**record['intent'])
        if record.get('native_stop') is not None: raise LedgerError('native_group_observation_required')
        created,event_at,received=utc_time(broker_created_at),utc_time(broker_event_at),utc_time(received_at)
        if (status not in BROKER_STATES or type(filled_quantity) is not int or not 0<=filled_quantity<=intent.quantity
                or event_at>received or received<utc_time(record['updated_at'])
                or not utc_time(record.get('submission_claimed_at') or record['created_at'])<=created<=event_at
                or (record.get('broker_created_at') is not None and utc_time(record['broker_created_at'])!=created)):
            raise LedgerError('invalid_broker_observation')
        if not isinstance(broker_order_id,str) or str(uuid.UUID(broker_order_id))!=broker_order_id.lower():
            raise LedgerError('invalid_broker_order_identity')
        if record['broker_order_id'] not in (None,broker_order_id): raise LedgerError('broker_order_identity_conflict')
        from .native_orders import refuse_reused_identity
        refuse_reused_identity(self,db,identity,[broker_order_id])
        average=_price(average_fill_price_usd) if filled_quantity else Decimal(0)
        if not filled_quantity and average_fill_price_usd is not None and _money(average_fill_price_usd)!=0:
            raise LedgerError('price_without_fills')
        notional=average*filled_quantity
        if record['broker_updated_at'] and event_at<utc_time(record['broker_updated_at']):
            if filled_quantity<=record['filled_quantity']: return record
            raise LedgerError('old_observation_conflicts_with_fills')
        if filled_quantity<record['filled_quantity']: raise LedgerError('fill_counter_regression')
        if filled_quantity and ((intent.side=='buy' and average>intent.limit_price_usd) or (intent.side=='sell' and average<intent.limit_price_usd)):
            raise LedgerError('fill_violates_limit_price')
        if (status=='open' and filled_quantity!=0 or status=='partially_filled' and not 0<filled_quantity<intent.quantity
                or status=='filled' and filled_quantity!=intent.quantity or status=='rejected' and filled_quantity!=0):
            raise LedgerError('broker_status_fill_mismatch')
        identical=(status==record['state'] and filled_quantity==record['filled_quantity']
                   and notional==Decimal(record['filled_notional_usd']) and broker_order_id==record['broker_order_id'])
        if identical:
            if record.get('broker_created_at') is None or event_at>utc_time(record['broker_updated_at']):
                record.update(broker_created_at=_ts(created),broker_updated_at=_ts(event_at),updated_at=_ts(received))
                return self._persist(db,record,'broker_observation_watermark',received)
            return record
        terminal_update=(_allow_cleanup_terminal_update is True and record.get('native_cleanup') is not None
            and record['state'] in ('canceled','expired') and status==record['state']
            and filled_quantity>record['filled_quantity'])
        if record['state'] in TERMINAL and not terminal_update or record['state']=='reserved': raise LedgerError('invalid_order_transition')
        if record['state']=='partially_filled' and status=='open': raise LedgerError('invalid_order_transition')
        old_notional=Decimal(record['filled_notional_usd'])
        if (filled_quantity==record['filled_quantity'] and notional!=old_notional
                or filled_quantity>record['filled_quantity'] and notional<=old_notional):
            raise LedgerError('invalid_cumulative_fill_value')
        delta_quantity=filled_quantity-record['filled_quantity']
        delta_notional=notional-old_notional
        if delta_quantity and ((intent.side=='buy' and delta_notional>intent.limit_price_usd*delta_quantity)
                               or (intent.side=='sell' and delta_notional<intent.limit_price_usd*delta_quantity)):
            raise LedgerError('incremental_fill_violates_limit_price')
        record.update(state=status,filled_quantity=filled_quantity,filled_notional_usd=str(notional),
                      broker_order_id=broker_order_id,broker_created_at=_ts(created),broker_updated_at=_ts(event_at),updated_at=_ts(received))
        return self._persist(db,record,'broker_terminal_fill_observation' if terminal_update else 'broker_observation',received)

    def attach_native_stop(self,identity,*,plan,at):
        from .native_orders import attach
        return attach(self,identity,plan=plan,at=at)

    def prepare_native_submission(self,identity,**risk_inputs):
        return self.prepare_submission(identity,gateway='native_oto_stop',**risk_inputs)

    def observe_native_group(self,identity,*,row,received_at):
        from .native_orders import observe
        return observe(self,identity,row=row,received_at=received_at)

    def prepare_stop_cancellation(self,identity,*,at):
        from .native_orders import prepare_stop_cancellation
        return prepare_stop_cancellation(self,identity,at=at)

    def prepare_native_entry_cancellation(self,identity,**inputs):
        from .native_orders import prepare_entry_cancellation
        return prepare_entry_cancellation(self,identity,**inputs)

    def finish_native_entry_cancellation(self,ticket,**inputs):
        from .native_orders import finish_entry_cancellation
        return finish_entry_cancellation(self,ticket,**inputs)

    def active(self):
        with self._db(False) as db:
            rows=db.execute('SELECT intent_id FROM orders WHERE state IN (?,?,?,?) LIMIT 1001',ACTIVE).fetchall()
            if len(rows)>1000: raise LedgerError('active_order_budget_exceeded')
            return [self._load(db,row[0]) for row in rows]

    def _verify(self,db):
        previous='0'*64;latest={};count=0
        for row in db.execute('SELECT intent_id,payload,previous_digest,digest FROM events ORDER BY sequence'):
            count+=1
            if count>100000 or len(row['payload'].encode())>2*1024*1024: raise LedgerError('ledger_verification_budget')
            if row['previous_digest']!=previous or _hash(previous+'\0'+row['payload'])!=row['digest']:
                raise LedgerError('event_chain_integrity_mismatch')
            event=decode_json(row['payload'])
            if event['record']['intent_id']!=row['intent_id']: raise LedgerError('event_identity_mismatch')
            latest[row['intent_id']]=event['record'];previous=row['digest']
        identities=[row[0] for row in db.execute('SELECT intent_id FROM orders LIMIT 10001')]
        if len(identities)>10000: raise LedgerError('order_history_budget')
        if set(identities)!=set(latest): raise LedgerError('order_event_set_mismatch')
        for identity in identities:
            if self._load(db,identity)!=latest[identity]: raise LedgerError('order_event_state_mismatch')
        return dict(orders=len(identities),events=count,integrity_verified=True)

    def verify(self):
        with self._db(False) as db: return self._verify(db)

    def all_orders(self):
        """One verified read transaction, including terminal orders needed for fills."""
        with self._db(False) as db:
            self._verify(db)
            return [self._load(db,row[0]) for row in db.execute('SELECT intent_id FROM orders ORDER BY intent_id')]

    def snapshot_prefix(self):
        """Atomically bind records to their immutable verified event prefix."""
        with self._db(False) as db:
            self._verify(db)
            head=db.execute('SELECT sequence,digest FROM events ORDER BY sequence DESC LIMIT 1').fetchone()
            prefix=dict(sequence=head[0] if head else 0,digest=head[1] if head else '0'*64)
            records=[self._load(db,r[0]) for r in db.execute('SELECT intent_id FROM orders ORDER BY intent_id')]
            return prefix,records

    def records_at_prefix(self,prefix):
        if (type(prefix) is not dict or set(prefix)!={'sequence','digest'} or type(prefix['sequence']) is not int
                or not 0<=prefix['sequence']<=100000):raise LedgerError('invalid_ledger_prefix')
        _identity(prefix['digest'])
        with self._db(False) as db:
            self._verify(db);latest={};sequence=0;digest='0'*64
            for row in db.execute('SELECT sequence,payload,digest FROM events WHERE sequence<=? ORDER BY sequence',
                                  (prefix['sequence'],)):
                record=decode_json(row['payload'])['record'];latest[record['intent_id']]=record
                sequence,digest=row['sequence'],row['digest']
            if sequence!=prefix['sequence'] or digest!=prefix['digest']:raise LedgerError('ledger_prefix_mismatch')
            return [latest[k] for k in sorted(latest)]
