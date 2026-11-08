"""GET-only group/account reconciliation; readiness is not order permission.

The legacy reconciler/risk path stays separate and conservative. This module
joins group evidence, flat open-order visibility, activities, available shares
and a replayable account book. It cannot submit, cancel or replace an order.
"""
from datetime import datetime,timezone
from decimal import Decimal
import hashlib
from pathlib import Path
import time

from .accountbook import AccountBook
from .accounting import AccountSnapshot
from .broker import PaperBroker
from .data import Calendar,utc_time
from .monitor import ObservationArchive
from .native_broker import NativePaperReader
from .native_orders import TERMINAL,accounting_evidence
from .orders import OrderLedger,ACTIVE
from .protection import GroupLeg,OTOGroup,NativeStopPlan,assess_protection
from .reconciler import ReconciliationCycle,_digest,_matches
from .runner import _exclusive


def _version():
    digest=hashlib.sha256()
    for name in ('native_reconciler.py','native_broker.py','native_orders.py','protection.py',
                 'reconciler.py','accountbook.py','accounting.py','orders.py','broker.py','risk.py','data.py','native_cleanup.py','native_close.py','cleanup_recovery.py','repair_orders.py','repair_decisions.py','repair_execution.py','compensation.py','native_admission.py','native_execution.py'):
        digest.update(name.encode()+b'\0'+Path(__file__).with_name(name).read_bytes()+b'\0')
    return digest.hexdigest()


def _settled(native):
    parent,stop=native['parent'],native['stop']
    if parent is None or stop is None or not native['child_visible']: return False
    return ((parent['status']=='filled' and stop['status']=='filled'
             and parent['filled_at'] is not None and stop['filled_at'] is not None
             and parent['filled_quantity']==stop['filled_quantity']==native['plan']['quantity'])
            or (parent['status'] in ('canceled','expired','rejected') and stop['status'] in ('canceled','expired','rejected')
                and parent['filled_quantity']==stop['filled_quantity']==0))


def _group(record):
    native=record['native_stop']
    def leg(value):
        row=dict(value)
        for key in ('created_at','updated_at','filled_at'):
            if row[key] is not None: row[key]=utc_time(row[key])
        if row['average_fill_price_usd'] is not None: row['average_fill_price_usd']=Decimal(row['average_fill_price_usd'])
        return GroupLeg(**row)
    return OTOGroup(NativeStopPlan(**native['plan']),leg(native['parent']),
        leg(native['stop']) if native['stop'] is not None and native['child_visible'] else None,
        utc_time(native['observed_at']))


def _expected(records):
    result={}
    def add(key,value):
        if key in result: raise ValueError('duplicate owned broker identity')
        result[key]=value
    for record in records:
        if record['broker_order_id'] is None: continue
        native=record.get('native_stop');intent=record['intent']
        if native is None:
            add(record['broker_order_id'],dict(symbol=intent['symbol'],side=intent['side'],quantity=intent['quantity'],
                filled_quantity=record['filled_quantity'],notional=Decimal(record['filled_notional_usd']),
                kind='limit',limit=Decimal(intent['limit_price_usd']),stop=None,tif='day',status=record.get('cleanup_broker_status',record['state']),
                created=utc_time(record['broker_created_at']),updated=utc_time(record['broker_updated_at']),
                client=record['client_order_id'],classes=('simple',''),visible=True))
            continue
        for role in ('parent','stop'):
            order=native[role]
            if order is None: continue
            add(order['broker_order_id'],dict(symbol=intent['symbol'],side='buy' if role=='parent' else 'sell',
                quantity=order['quantity'],filled_quantity=order['filled_quantity'],
                notional=Decimal(order['average_fill_price_usd'] or '0')*order['filled_quantity'],
                kind='limit' if role=='parent' else 'stop',
                limit=Decimal(intent['limit_price_usd']) if role=='parent' else None,
                stop=Decimal(native['plan']['stop_price_usd']) if role=='stop' else None,
                tif=native['plan']['time_in_force'],status=order['status'],created=utc_time(order['created_at']),
                updated=utc_time(order['updated_at']),client=record['client_order_id'] if role=='parent' else None,
                classes=('oto',) if role=='parent' else ('oto','simple',''),
                visible=role=='parent' or native['child_visible']))
        from .repair_orders import repair_rows
        for row in repair_rows(record):
            order=row['order']
            if order is None:continue
            add(order['broker_order_id'],dict(symbol=intent['symbol'],side='sell',quantity=row['quantity'],
                filled_quantity=order['filled_quantity'],notional=Decimal(order['average_fill_price_usd'] or '0')*order['filled_quantity'],
                kind='stop',limit=None,stop=Decimal(row['stop_price_usd']),tif='gtc',status=order['status'],
                created=utc_time(order['created_at']),updated=utc_time(order['updated_at']),client=row['client_order_id'],
                classes=('simple',''),visible=True))
    return result


def _status(value):
    return 'open' if value in ('new','accepted','pending_new','accepted_for_bidding') else value


def _assess_records(records,calendar,at):
    from .native_cleanup import project_groups
    from .repair_orders import repair_assessment
    groups=project_groups(records)
    states=[];findings=set()
    for record in records:
        if record.get('native_cleanup') is not None and record.get('cleanup_lookup_state')=='unknown':
            findings.add('cleanup_lookup_unconfirmed')
        native=record.get('native_stop')
        if native is None or native['parent'] is None or _settled(native) or groups[record['intent_id']].settled: continue
        group=_group(record)
        if not 0<=(at-group.received_at).total_seconds()<=30:
            findings.add('native_protection_stale')
            states.append(dict(symbol=record['intent']['symbol'],status='evidence_stale',
                remaining_quantity=None,required_actions=['pause_entries','reconcile_group']))
            continue
        state=repair_assessment(record,groups[record['intent_id']],calendar=calendar,now=at) or assess_protection(group,calendar=calendar,now=at)
        close=native.get('close')
        if (native['stop_cancel_requested_at'] is not None and (close is None or close['outcome']!='rejected')
                and native['stop'] is not None and native['stop']['status'] not in TERMINAL):
            findings.add('native_protection_close_in_progress')
        states.append(dict(symbol=record['intent']['symbol'],status=state.status,
            remaining_quantity=groups[record['intent_id']].remaining_quantity,required_actions=list(state.required_actions)))
        if state.pause_entries: findings.add('native_protection_'+state.status)
    return states,findings


def _ref_matches(order,expected):
    return (order.symbol==expected['symbol'] and order.side==expected['side']
        and order.quantity==expected['quantity'] and order.filled_quantity==expected['filled_quantity']
        and abs((order.average_fill_price_usd or Decimal(0))*order.filled_quantity-expected['notional'])
            <=Decimal('.00000001')*max(1,order.filled_quantity)
        and order.order_type==expected['kind'] and order.limit_price_usd==expected['limit']
        and order.stop_price_usd==expected['stop'] and order.time_in_force==expected['tif']
        and order.order_class in expected['classes']
        and _status(order.status)==_status(expected['status'])
        and order.created_at==expected['created'] and order.updated_at==expected['updated']
        and (expected['client'] is None or order.client_order_id==expected['client']) and expected['visible'])


class NativePaperReconciler:
    def __init__(self,*,broker,ledger,book,journal,calendar,clock=None,monotonic=None):
        for value,exact in ((broker,PaperBroker),(ledger,OrderLedger),(book,AccountBook),(calendar,Calendar)):
            if type(value) is not exact: raise ValueError('exact native reconciliation dependencies required')
        self.broker,self.ledger,self.book,self.calendar=broker,ledger,book,calendar
        self.journal=ObservationArchive(journal)
        self._clock=clock or (lambda:datetime.now(timezone.utc))
        self._monotonic=monotonic or time.monotonic

    def refresh(self):
        now=utc_time(self._clock());deadline=self._monotonic()+25
        def check():
            if self._monotonic()>=deadline: raise ValueError('native reconciliation deadline')
        try:
            with _exclusive(self.journal.path):
                if self.journal.path.exists(): self.journal.verify()
                records=self.ledger.all_orders();anchor=self.book.read_anchor();findings=set();states=[]
                receipts=[now,*(utc_time(r['updated_at']) for r in records)]
                from .native_cleanup import project_groups
                groups=project_groups(records);active=[]
                for record in records:
                    native=record.get('native_stop')
                    if native is None:
                        if record['state'] in ACTIVE and record['state']!='reserved': active.append(record)
                    elif (record['submission_claimed_at'] is not None and not _settled(native)
                          and not groups[record['intent_id']].settled):
                        if native['parent'] is not None or record['state']=='submission_unknown': active.append(record)
                if len(active)>20: raise ValueError('active family lookup budget')
                reader=NativePaperReader(self.broker,monotonic=self._monotonic)
                check();window=self.broker.activities(after=anchor.as_of,until=now)
                receipts.extend((window.received_at,utc_time(self._clock())))
                # A canceled/expired cleanup can receive delayed fill evidence.
                # Requery only terminal discrepancies, retaining immutable fills
                # and avoiding an unbounded lookup of all historical orders.
                activity_quantities={}
                for activity in window.activities:
                    if activity.kind=='FILL':
                        activity_quantities[activity.broker_order_id]=activity_quantities.get(activity.broker_order_id,0)+activity.quantity
                active_ids={r['intent_id'] for r in active}
                for record in records:
                    if (record.get('native_cleanup') is not None and record['broker_order_id'] is not None
                            and record['intent_id'] not in active_ids
                            and activity_quantities.get(record['broker_order_id'],0)!=record['filled_quantity']):
                        active.append(record);active_ids.add(record['intent_id'])
                from .repair_orders import repair_rows,filled_quantity as repair_fills,CLOSED as REPAIR_CLOSED,RepairPaperReader
                repair_targets=[]
                for record in records:
                    for repair in repair_rows(record):
                        if repair['submission_claimed_at'] is None or repair['state']=='aborted_unsent':continue
                        order=repair['order']
                        if (repair['state'] not in REPAIR_CLOSED or repair['lookup_state']=='unknown'
                                or order is not None and activity_quantities.get(order['broker_order_id'],0)!=repair_fills(repair)):
                            repair_targets.append((record['intent_id'],repair['identity']))
                if len(active)+len(repair_targets)>20:raise ValueError('owned order lookup budget')
                check();visible=reader.open_orders()
                receipts.extend((utc_time(self._clock()),*(o.received_at for o in visible)))
                for record in active:
                    check()
                    if record.get('native_stop') is not None:
                        recovered=reader.recover(self.ledger,record['intent_id'])
                        if recovered is None: findings.add('unconfirmed_submission' if record['broker_order_id'] is None else 'missing_known_order')
                        else: receipts.append(utc_time(recovered['native_stop']['observed_at']))
                    elif record.get('native_cleanup') is not None:
                        from .cleanup_recovery import CleanupPaperReader
                        recovered=CleanupPaperReader(self.broker).recover(self.ledger,record['intent_id'])
                        if recovered is None:findings.add('unconfirmed_submission' if record['broker_order_id'] is None else 'missing_known_order')
                        else:receipts.append(utc_time(recovered['cleanup_observed_at']))
                    else:
                        order=self.broker.lookup(record['client_order_id'])
                        if order is None: findings.add('unconfirmed_submission' if record['state']=='submission_unknown' else 'missing_known_order')
                        elif not _matches(record,order): findings.add('broker_order_mismatch')
                        else:
                            receipts.append(order.received_at)
                            self.ledger.observe(record['intent_id'],status=order.status,filled_quantity=order.filled_quantity,
                                average_fill_price_usd=order.average_fill_price_usd,broker_order_id=order.broker_order_id,
                                broker_created_at=order.created_at,broker_event_at=order.updated_at,received_at=order.received_at)
                    receipts.append(utc_time(self._clock()))
                repair_reader=RepairPaperReader(self.broker)
                for family_id,repair_id in repair_targets:
                    check();recovered=repair_reader.recover(self.ledger,family_id,repair_id)
                    if recovered is None:findings.add('repair_lookup_unconfirmed')
                    receipts.append(utc_time(self._clock()))
                records=self.ledger.all_orders();expected=_expected(records)
                for order in visible:
                    known=expected.get(order.broker_order_id)
                    if known is None: findings.add('unrecognized_broker_order')
                    elif known['status'] in TERMINAL: findings.add('terminal_order_reported_open')
                    elif not _ref_matches(order,known): findings.add('non_atomic_order_snapshot')
                visible_ids={o.broker_order_id for o in visible}
                for identity,order in expected.items():
                    if order['status'] not in TERMINAL and not (order['kind']=='stop' and order['status']=='held'):
                        if identity not in visible_ids: findings.add('missing_known_order')
                at=utc_time(self._clock())
                for record in records:
                    if record['state']=='submission_unknown': findings.add('unconfirmed_submission')
                states,protection_findings=_assess_records(records,self.calendar,at)
                findings.update(protection_findings)
                check();available=self.broker.positions();receipts.append(utc_time(self._clock()))
                check();marks=self.broker.account_marks();receipts.append(utc_time(self._clock()))
                check();account=self.broker.account();at=utc_time(self._clock());check()
                if {p.symbol:p.quantity for p in available}!={p.symbol:p.quantity for p in marks}: findings.add('position_snapshot_mismatch')
                reserved={}
                for order in visible:
                    if order.side=='sell' and order.status not in TERMINAL and order.status!='held':
                        reserved[order.symbol]=reserved.get(order.symbol,0)+order.quantity-order.filled_quantity
                held={p.symbol:p.quantity for p in available}
                if any(quantity>held.get(symbol,0) for symbol,quantity in reserved.items()): findings.add('inconsistent_sell_reservations')
                for position in available:
                    if position.available_quantity!=position.quantity-reserved.get(position.symbol,0): findings.add('available_share_mismatch')
                snapshot=AccountSnapshot(account.received_at,account.cash_usd,account.equity_usd,account.blocked,marks,account.account_sha256)
                receipt=self.book.refresh(window=window,orders=accounting_evidence(records),account=snapshot,now=at)
                findings.update(receipt.result.violations);digest=_digest(records)
                if _digest(self.ledger.all_orders())!=digest: findings.add('order_snapshot_changed')
                floor=max(*receipts,at,account.received_at,receipt.as_of,
                    *(p.received_at for p in marks),*(p.received_at for p in available))
                at=utc_time(self._clock())
                if at<floor or not 0<=(at-window.until).total_seconds()<=30: findings.add('stale_reconciliation')
                states,protection_findings=_assess_records(records,self.calendar,at)
                findings.update(protection_findings)
                floor=max(floor,at);good=receipt.result.reconciled and not findings
                report=self.journal.append(dict(kind='native_reconciliation',generated_at=at.timestamp(),code_sha256=_version(),
                    status='reconciled' if good else 'vetoed',ready_for_risk=good,findings=sorted(findings),
                    orders_sha256=digest,orders_count=len(records),positions_count=len(marks),native_group_states=states,
                    book_sequence=receipt.sequence,book_sha256=receipt.digest,activity_until=window.until.timestamp()))
                at=utc_time(self._clock())
                if at<floor or not 0<=(at-window.until).total_seconds()<=30: findings.add('stale_reconciliation');good=False
                states,protection_findings=_assess_records(records,self.calendar,at)
                findings.update(protection_findings);good=good and not findings
                return ReconciliationCycle(at,'reconciled' if good else 'vetoed',good,tuple(sorted(findings)),receipt,
                    digest,report['audit_sha256'],len(records),len(marks),available,tuple(states))
        except Exception:
            return ReconciliationCycle(utc_time(self._clock()),'unavailable',False,('reconciliation_unavailable',),None,None,None,None,None)
