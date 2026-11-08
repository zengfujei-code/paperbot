"""Bounded close/cleanup/repair advancement under the shared execution lock.

No CLI/service, entry signal or strategy acceptance. Each call advances one owned
close; uncertain mutations only recover by GET. The production coordinator and
its outer process deadline remain activation gates.
"""
from dataclasses import dataclass
from datetime import timedelta

from .broker import PaperBroker
from .controls import ControlJournal
from .data import Calendar,utc_time
from .decision_journal import DecisionJournal,capture_cleanup_decision
from .native_close import NativeCloseGateway,_target
from .native_cleanup import (project_groups,validate_accounting,prepare_cleanup_risk_view,
    admit_cleanup,retire_cleanup_reservation)
from .native_reconciler import NativePaperReconciler
from .orders import OrderLedger,_hash,_json,_ts
from .repair_decisions import RepairDecisionJournal
from .repair_execution import RepairPaperGateway
from .repair_orders import (RepairPaperReader,repair_rows,reserve_repair,retire_repair,
    CLOSED as REPAIR_CLOSED)
from .repair_protection import prepare_repair
from .risk import RiskState,RiskLimits,OrderIntent,_identity
from .runner import _exclusive


class CompensationError(RuntimeError):pass


@dataclass(frozen=True)
class CompensationResult:
    status:str
    remaining_quantity:int|None
    required_actions:tuple

    def public_summary(self):
        return dict(mode='paper',phase='close_compensation',status=self.status,remaining_quantity=self.remaining_quantity,
            required_actions=list(self.required_actions),execution_enabled=False,ready_for_execution=False)


class CloseCompensator:
    def __init__(self,*,broker,ledger,reconciler,controls,decisions,repair_decisions,policy,limits,calendar):
        from .execution_policy import PolicySnapshot
        for value,exact in ((broker,PaperBroker),(ledger,OrderLedger),(reconciler,NativePaperReconciler),
                (controls,ControlJournal),(decisions,DecisionJournal),(repair_decisions,RepairDecisionJournal),
                (policy,PolicySnapshot),(limits,RiskLimits),(calendar,Calendar)):
            if type(value) is not exact:raise CompensationError('exact_compensation_dependencies_required')
        if reconciler.broker is not broker or reconciler.ledger is not ledger:
            raise CompensationError('compensation_scope_mismatch')
        self.broker=broker;self.ledger=ledger;self.reconciler=reconciler;self.book=reconciler.book
        self.controls=controls;self.decisions=decisions;self.repair_decisions=repair_decisions
        self.policy=policy;self.limits=limits;self.calendar=calendar
        self.lock_path=ledger.path.with_name('execution-global')

    def _now(self):return utc_time(self.broker._clock())

    def _load_cleanup(self,identity,family):
        with self.ledger._db(False) as db:
            self.ledger._verify(db);record=self.ledger._load(db,identity,missing=True)
        if record is not None and (record.get('native_cleanup') is None
                or record['native_cleanup']['family_intent_id']!=family):raise ValueError('cleanup owner mismatch')
        return record

    def _mark(self,family,repair_id,close_id,status,remaining,*,complete=False,book_digest=None):
        now=self._now()
        with self.ledger._db(True) as db:
            self.ledger._verify(db);record=self.ledger._load(db,family);slot,_=_target(record,repair_id)
            if slot['close'] is None or slot['close']['identity']!=close_id:raise ValueError('compensation close changed')
            slot['close']['recovery']=dict(schema=1,status=status,remaining_quantity=remaining,complete=complete,
                checked_at=_ts(now),book_digest=book_digest,clock_regressed=now<utc_time(record['updated_at']))
            record['updated_at']=_ts(max(now,utc_time(record['updated_at'])))
            self.ledger._persist(db,record,'close_compensation_observation',utc_time(record['updated_at']))
        actions=() if complete else ('pause_entries','reconcile_group','reconcile_account')
        return CompensationResult(status,remaining,actions)

    def _pause(self,*,halt=False):
        current=self.controls.read(now=self._now())
        if not current.pause_entries or halt and not current.halt_orders:
            current=self.controls.update(expected_revision=current.revision,pause_entries=True,
                halt_orders=current.halt_orders or halt,reason='fault_review',now=self._now())
        return current

    def _cycle(self):
        cycle=self.reconciler.refresh();records=self.ledger.all_orders()
        validate_accounting(self.book,records,cycle,self.calendar,self._now())
        return cycle,records,project_groups(records)

    def _quotes(self,quotes):return quotes() if callable(quotes) else quotes

    def _view(self,family,cycle,quotes,control):
        return prepare_cleanup_risk_view(book=self.book,ledger=self.ledger,baselines=self._baselines,
            cycle=cycle,quotes=self._quotes(quotes),family_id=family,
            operator_state=RiskState(pause_entries=control.pause_entries,halt_orders=control.halt_orders),
            calendar=self.calendar,now=self._now())

    def advance(self,family_id,*,baselines,target_repair_id=None,quotes=None):
        from .risk_baselines import RiskBaselineStore
        if type(baselines) is not RiskBaselineStore:raise CompensationError('exact_compensation_baseline_required')
        _identity(family_id)
        try:
            with _exclusive(self.lock_path):
                self._baselines=baselines
                return self._advance(family_id,target_repair_id,quotes)
        except Exception:raise CompensationError('close_compensation_unavailable') from None

    def _advance(self,family,repair_id,quotes):
        from .cleanup_recovery import CleanupRecovery
        from .cleanup_execution import CleanupPaperGateway
        record=self.ledger.load(family);slot,_=_target(record,repair_id);close=slot.get('close')
        if close is None:raise ValueError('existing close claim required')
        close_id=close['identity'];_identity(close_id)
        if close.get('recovery',{}).get('complete'):
            return CompensationResult('previously_resolved',None,('observe_account',))
        cleanup_id=_hash('paperbot-close-cleanup-v1\0'+close_id)
        cleanup=self._load_cleanup(cleanup_id,family)
        result=NativeCloseGateway(self.broker).recover(self.ledger,family,target_repair_id=repair_id)
        control=self.controls.read(now=self._now())
        overdue=self._now()>=utc_time(close['deadline_at'])
        if overdue:control=self._pause()
        if cleanup is None and not result.target_terminal and result.status not in ('group_closed','close_rejected_protected','close_not_sent'):
            if overdue:self._pause()
            return self._mark(family,repair_id,close_id,'escalated_unknown_stop' if overdue else 'awaiting_stop_terminal',None)
        if cleanup is not None and cleanup['submission_claimed_at'] is not None and cleanup['state'] in (
                'open','partially_filled','submission_unknown','filled','canceled','expired','rejected'):
            recovery=CleanupRecovery(self.broker)
            working=cleanup['state'] in ('open','partially_filled','submission_unknown')
            if working and (control.halt_orders or overdue or self._now()>=utc_time(cleanup['intent']['expires_at'])):
                recovery.cancel(self.ledger,cleanup_id,controls=self.controls,control_snapshot=control)
            else:recovery.recover(self.ledger,cleanup_id)
            cleanup=self._load_cleanup(cleanup_id,family)
            if cleanup['state'] in ('open','partially_filled','submission_unknown') or cleanup.get('cleanup_lookup_state')!='observed':
                if overdue:self._pause()
                return self._mark(family,repair_id,close_id,'cleanup_unknown_or_working',None)
        elif cleanup is not None and cleanup['state']=='reserved':
            # Restart with a known-unsent reservation gets an explicit abort;
            # an old decision is never upgraded using new source observations.
            retire_cleanup_reservation(self.ledger,cleanup_id,expected_sha256=_hash(_json(cleanup)),
                reason='pre_dispatch_abort',now=self._now())
            cleanup=self._load_cleanup(cleanup_id,family)
        try:cycle,records,groups=self._cycle()
        except Exception:
            if overdue:self._pause()
            return self._mark(family,repair_id,close_id,'accounting_unconfirmed',None)
        group=groups[family];book_digest=cycle.book_receipt.digest
        if group.remaining_quantity==0:
            return self._mark(family,repair_id,close_id,'closed',0,complete=True,book_digest=book_digest)
        if result.status in ('close_rejected_protected','close_not_sent') and cycle.ready_for_risk:
            self._pause()
            status='close_failed_protected' if result.status=='close_rejected_protected' else 'close_not_sent_protected'
            return self._mark(family,repair_id,close_id,status,group.remaining_quantity,
                complete=True,book_digest=book_digest)
        owned=next(r for r in records if r['intent_id']==family)
        descendants=[r for r in repair_rows(owned) if r['identity']!=repair_id and utc_time(r['allocated_at'])>=utc_time(close['claimed_at'])
            and r['state'] not in ('retired','aborted_unsent')]
        if descendants:
            for row in descendants:
                if row['submission_claimed_at'] is not None:RepairPaperReader(self.broker).recover(self.ledger,family,row['identity'])
                elif row['state']=='reserved':retire_repair(self.ledger,family,row['identity'],
                    expected_sha256=_hash(_json(self.ledger.all_orders())),reason='pre_dispatch_abort',now=self._now())
            try:cycle,records,groups=self._cycle()
            except Exception:return self._mark(family,repair_id,close_id,'repair_unconfirmed',None)
            group=groups[family]
            if cycle.ready_for_risk:
                return self._mark(family,repair_id,close_id,'restored',group.remaining_quantity,
                    complete=True,book_digest=cycle.book_receipt.digest)
            if group.reserved_quantity:
                if overdue:self._pause()
                return self._mark(family,repair_id,close_id,'repair_unconfirmed',None)
        control=self.controls.read(now=self._now())
        if control.halt_orders:
            return self._mark(family,repair_id,close_id,'unprotected_global_halt',group.remaining_quantity)
        expires=utc_time(close['decision_expires_at']) if close.get('decision_expires_at') is not None else self._now()
        if cleanup is None and not overdue and self._now()<expires and group.available_quantity==group.remaining_quantity:
            try:view=self._view(family,cycle,quotes,control)
            except Exception:
                self._pause()
                return self._mark(family,repair_id,close_id,'cleanup_inputs_unavailable',None)
            at=self._now()
            intent=OrderIntent(cleanup_id,group.symbol,'sell',group.remaining_quantity,view.risk.quote.bid,None,
                min(expires,at+timedelta(seconds=20)))
            frame=capture_cleanup_decision(ledger=self.ledger,prepared=view,intent=intent,policy=self.policy,
                controls=control,limits=self.limits,calendar=self.calendar,now=at)
            receipt=self.decisions.append(frame,ledger=self.ledger)
            admit_cleanup(self.ledger,intent,prepared=view,limits=self.limits,calendar=self.calendar,now=self._now(),
                decision=receipt,journal=self.decisions)
            cycle,_,_=self._cycle();fresh=self._view(family,cycle,quotes,control)
            try:
                CleanupPaperGateway(self.broker).submit(self.ledger,cleanup_id,prepared=fresh,limits=self.limits,
                    calendar=self.calendar,controls=self.controls,control_snapshot=control,journal=self.decisions,
                    book=self.book,baselines=self._baselines)
            except Exception:
                pending=self._load_cleanup(cleanup_id,family)
                status=('cleanup_submission_unknown' if pending['submission_claimed_at'] is not None
                    and pending['state']=='submission_unknown' else 'cleanup_dispatch_unconfirmed')
                return self._mark(family,repair_id,close_id,status,None)
            return self._mark(family,repair_id,close_id,'cleanup_dispatched',group.remaining_quantity)
        if group.available_quantity!=group.remaining_quantity:
            return self._mark(family,repair_id,close_id,'shares_not_available',None)
        if sum(utc_time(r['allocated_at'])>=utc_time(close['claimed_at']) for r in repair_rows(owned))>=3:
            self._pause();return self._mark(family,repair_id,close_id,'repair_attempt_budget',group.remaining_quantity)
        # Explicitly refresh retained terminal repair evidence for a new handoff.
        for row in repair_rows(owned):
            if row['submission_claimed_at'] is not None and row['state'] in REPAIR_CLOSED:
                RepairPaperReader(self.broker).recover(self.ledger,family,row['identity'])
        cycle,_,_=self._cycle()
        try:
            plan=prepare_repair(book=self.book,ledger=self.ledger,baselines=self._baselines,cycle=cycle,
                quotes=self._quotes(quotes),family_id=family,operator_state=RiskState(pause_entries=True),
                calendar=self.calendar,limits=self.limits,now=self._now())
        except Exception:
            self._pause()
            return self._mark(family,repair_id,close_id,'repair_inputs_unavailable',None)
        decision=self.repair_decisions.record(plan,ledger=self.ledger,book=self.book,baselines=self._baselines,
            controls=control,calendar=self.calendar,limits=self.limits)
        reservation=reserve_repair(ledger=self.ledger,plan=plan,book=self.book,baselines=self._baselines,
            calendar=self.calendar,limits=self.limits,now=self._now(),decision=decision,journal=self.repair_decisions)
        RepairPaperGateway(self.broker).submit(self.ledger,family,reservation.repair_id,journal=self.repair_decisions,
            book=self.book,baselines=self._baselines,controls=self.controls,control_snapshot=control,
            calendar=self.calendar,limits=self.limits)
        return self._mark(family,repair_id,close_id,'repair_dispatched',group.remaining_quantity)
