"""Audited1us broker timestamp representation adapter, outside pinned core.

No order authority. Non-GET requests pass through without adapter disk I/O.
Only already-owned native parent/stop identities can be normalized. Raw and
canonical times are committed separately before input reaches the old core.
"""
from copy import deepcopy
from contextlib import closing
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit

from .broker import BrokerError, HttpReply, PAPER
from .data import utc_time
from .monitor import ObservationArchive
from .native_broker import _reference
from .native_reconciler import _expected
from .orders import OrderLedger, _hash, _json
from .runner import _exclusive

POLICY='bounded_1us_no_anchor_walk_v1'
TIMES=('created_at','updated_at','filled_at')


def _source():
    paths=sorted(Path(__file__).parent.glob('*.py'))
    if not 51<=len(paths)<=64 or any(p.is_symlink() or p.stat().st_size>524288 for p in paths):
        raise BrokerError('broker_time_source_unavailable')
    return _hash(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},sort_keys=True))


def _near(a,b):
    return a is not None and b is not None and abs(utc_time(a)-utc_time(b))<=timedelta(microseconds=1)


def _advance(history,observation):
    identity=observation['order_sha256'];event=utc_time(observation['raw_updated_at'])
    signature=observation['semantic_sha256'];prior=history.get(identity)
    if prior is not None:
        if event<=prior['at'] and signature!=prior['signature']:
            raise BrokerError('broker_raw_event_reused_or_reversed')
        if event<=prior['at']:return
    history[identity]=dict(at=event,signature=signature)


class CanonicalBrokerTimes:
    def __init__(self,*,ledger,journal,transport,clock=None):
        if type(ledger) is not OrderLedger or not callable(transport):
            raise BrokerError('broker_time_dependencies_required')
        self.ledger=ledger;self.journal=Path(journal).absolute();self.transport=transport
        self.clock=clock or (lambda:datetime.now(timezone.utc));self.version=_source()
        if self.journal.parent!=self.ledger.path.parent or self.journal==self.ledger.path:
            raise BrokerError('broker_time_private_state_required')

    def __call__(self,method,url,headers,body):
        # Keep the final order check adjacent to the underlying transport:
        # no source, ledger, clock, journal or file access on this branch.
        if method!='GET':
            return self.transport(method,url,headers,body)
        parts=urlsplit(url)
        if parts.scheme!='https' or parts.netloc!=urlsplit(PAPER).netloc or not parts.path.startswith('/v2/orders'):
            return self.transport(method,url,headers,body)
        reply=self.transport(method,url,headers,body)
        if type(reply) is not HttpReply or reply.status!=200:return reply
        try:
            received=utc_time(self.clock())
            with _exclusive(self.journal):
                return self._canonical_reply(reply,received)
        except Exception:
            raise BrokerError('broker_time_canonicalization_unavailable') from None

    def _history(self):
        history={}
        if self.journal.exists():
            if (self.journal.is_symlink() or not self.journal.is_file()
                    or self.journal.stat().st_size>268435456):raise BrokerError('broker_time_audit_unavailable')
            with closing(sqlite3.connect(self.journal.as_uri()+'?mode=ro',uri=True,isolation_level=None)) as db:
                db.execute('BEGIN');ObservationArchive(self.journal)._verify_chain(db)
                for payload, in db.execute('SELECT payload FROM observations ORDER BY seq'):
                    report=json.loads(payload)
                    if report['kind']!='broker_time_canonicalization' or report['policy']!=POLICY:
                        raise BrokerError('broker_time_audit_policy_mismatch')
                    for observation in report['observations']:_advance(history,observation)
                db.rollback()
        return history

    def _canonical_reply(self,reply,received):
            history=self._history()
            records=self.ledger.all_orders();expected=_expected(records);known={}
            for record in records:
                native=record.get('native_stop')
                if native is None:continue
                for role in ('parent','stop'):
                    leg=native[role]
                    if leg is not None:
                        if leg['broker_order_id'] in known:raise ValueError('duplicate owned identity')
                        known[leg['broker_order_id']]=leg
            payload=deepcopy(reply.payload);changes=[];observations=[]
            def visit(row,depth=0):
                if type(row) is not dict or depth>1:raise ValueError('order response shape')
                identity=row.get('id');old=known.get(identity);contract=expected.get(identity)
                if old is not None and contract is not None:
                    ref=_reference({**row,'legs':None},received)
                    static=(ref.symbol==contract['symbol'] and ref.side==contract['side']
                        and ref.quantity==contract['quantity'] and ref.order_type==contract['kind']
                        and ref.limit_price_usd==contract['limit'] and ref.stop_price_usd==contract['stop']
                        and ref.time_in_force==contract['tif'] and ref.order_class in contract['classes']
                        and (contract['client'] is None or ref.client_order_id==contract['client']))
                    if static:
                        identity_sha=_hash(identity)
                        signature=_hash(_json(dict(status=ref.status,quantity=ref.quantity,
                            filled_quantity=ref.filled_quantity,average_fill_price_usd=ref.average_fill_price_usd,
                            group_id=row.get('group_id'))))
                        observation=dict(order_sha256=identity_sha,raw_updated_at=ref.updated_at.isoformat(),semantic_sha256=signature)
                        # Durable raw-event high water is separate from canonical
                        # time. A changed event cannot reuse a previously seen
                        # provider timestamp, including after a process restart.
                        _advance(history,observation);observations.append(observation)
                        raw={k:row.get(k) for k in TIMES}
                        for field in ('created_at','filled_at'):
                            if _near(old[field],row.get(field)):row[field]=old[field]
                        same_event=(ref.status==old['status'] and ref.filled_quantity==old['filled_quantity']
                            and ref.average_fill_price_usd==(Decimal(old['average_fill_price_usd']) if old['average_fill_price_usd'] is not None else None)
                            and row.get('group_id')==old['group_id'] and row.get('filled_at')==old['filled_at'])
                        if same_event and _near(old['updated_at'],row['updated_at']):row['updated_at']=old['updated_at']
                        canonical={k:row.get(k) for k in TIMES}
                        if raw!=canonical:
                            changes.append(dict(order_sha256=_hash(identity),raw=raw,canonical=canonical))
                legs=row.get('legs')
                if legs is not None:
                    if type(legs) is not list or len(legs)>1:raise ValueError('single native child required')
                    for leg in legs:visit(leg,depth+1)
            if type(payload) is list:
                if len(payload)>=500:raise ValueError('complete order bound')
                for row in payload:visit(row)
            else:visit(payload)
            for start in range(0,len(observations),8):
                batch=observations[start:start+8];keys={o['order_sha256'] for o in batch}
                report=dict(kind='broker_time_canonicalization',policy=POLICY,generated_at=received.timestamp(),
                    code_sha256=self.version,raw_response_sha256=_hash(_json(reply.payload)),
                    canonical_response_sha256=_hash(_json(payload)),observations=batch,
                    changes=[c for c in changes if c['order_sha256'] in keys])
                ObservationArchive(self.journal).append(report)
            return HttpReply(reply.status,payload)
