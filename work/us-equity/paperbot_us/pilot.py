"""Explicit experimental paper pilot; never a claim of strategy acceptance.

The administrator pins this release and a 30-day paper-only authorization. The
coordinator retains independent accounting, risk, controls and original decision
replay. It never uses a model's confidence as a probability or bypasses a veto.
"""
from datetime import datetime,timedelta,timezone
from decimal import Decimal,ROUND_DOWN
import hashlib
from pathlib import Path
from zoneinfo import ZoneInfo

from .accountbook import AccountBook
from .broker import PaperBroker
from .controls import ControlJournal
from .data import utc_time
from .decision_journal import (DecisionJournal,capture_decision,capture_close_decision,strategy_digest)
from .execution_policy import PolicySnapshot
from .monitor import _publish
from .native_admission import NativeEntrySources,admit_entry
from .native_cleanup import project_groups
from .native_close import prepare_close,NativeCloseGateway
from .native_execution import NativePaperGateway
from .native_recovery import NativeEntryRecovery
from .native_risk_view import prepare_native_risk_view
from .orders import OrderLedger,_hash,_json,_base,_ts
from .replay_bridge import regular_bytes,strict_json
from .risk import Quote,RiskLimits,RiskState,OrderIntent,_identity
from .runner import _exclusive


class PilotError(RuntimeError):pass


class PilotAuthorization:
    def __init__(self,path,expected_sha256):
        self.path=Path(path).absolute();_identity(expected_sha256);self.digest=expected_sha256

    def verify(self,*,policy,limits,now,allow_expired=False):
        try:
            raw=regular_bytes(self.path,8192);value=strict_json(raw);now=utc_time(now)
            if hashlib.sha256(raw).hexdigest()!=self.digest:raise ValueError('authorization changed')
            if (set(value)!={'schema','kind','mode','capital_usd','approved_at','expires_at',
                    'strategy_sha256','risk_limits_sha256','formal_strategy_accepted','live_authorized'}
                    or type(value['schema']) is not int or value['schema']!=1
                    or value['kind']!='experimental_paper_pilot' or value['mode']!='paper'
                    or type(value['capital_usd']) is not int or value['capital_usd']!=10000
                    or value['formal_strategy_accepted'] is not False or value['live_authorized'] is not False
                    or type(policy) is not PolicySnapshot or type(limits) is not RiskLimits
                    or not policy.policy.execution_enabled or policy.policy.acceptance_sha256!=self.digest
                    or value['strategy_sha256']!=policy.policy.strategy_sha256
                    or value['risk_limits_sha256']!=policy.policy.risk_limits_sha256
                    or value['risk_limits_sha256']!=_hash(_json(_base(limits,RiskLimits)))):
                raise ValueError('paper pilot contract')
            start,end=utc_time(value['approved_at']),utc_time(value['expires_at'])
            if not start<=now or not timedelta(0)<end-start<=timedelta(days=30) or not allow_expired and now>=end:
                raise ValueError('paper pilot window')
            return value
        except Exception:raise PilotError('paper_pilot_authorization_unavailable') from None


def repair_status(result):
    if result.status=='submitted' and result.request_sent:return 'protection_repair_dispatched'
    if result.status=='not_sent' and not result.request_sent:return 'protection_repair_not_sent'
    return 'protection_repair_unknown'


def retire_unsent_entry(ledger,identity,*,now):
    """Retain history and release only an expired, never-dispatched entry."""
    now=utc_time(now)
    with ledger._db(True) as db:
        ledger._verify(db);row=ledger._load(db,identity)
        if (row.get('native_entry') is None or row['state']!='reserved'
                or row['submission_claimed_at'] is not None or row['broker_order_id'] is not None
                or row['filled_quantity']!=0 or now<utc_time(row['intent']['expires_at'])):return None
        if now<utc_time(row['updated_at']):raise PilotError('pilot_clock_regression')
        row['risk'].update(allowed=False,violations=list(dict.fromkeys(
            (*row['risk']['violations'],'native_entry_expired_unsubmitted'))))
        row.update(state='expired',updated_at=_ts(now))
        return ledger._persist(db,row,'native_entry_retired',now)


class PilotCoordinator:
    def __init__(self,*,broker,ledger,book,baselines,controls,decisions,repair_decisions,
                 reconciler,policy,authorization,continuity,sources,calendar,limits,snapshots,clock=None):
        for value,exact in ((broker,PaperBroker),(ledger,OrderLedger),(book,AccountBook),
                (controls,ControlJournal),(decisions,DecisionJournal),(authorization,PilotAuthorization)):
            if type(value) is not exact:raise PilotError('exact_pilot_dependencies_required')
        self.broker,self.ledger,self.book,self.baselines=broker,ledger,book,baselines
        self.controls,self.decisions,self.repair_decisions=controls,decisions,repair_decisions
        self.reconciler,self.policy,self.authorization=reconciler,policy,authorization
        self.continuity,self.sources,self.calendar,self.limits=continuity,sources,calendar,limits
        self.snapshots=snapshots;self.clock=clock or (lambda:datetime.now(timezone.utc))
        self.lock_path=ledger.path.with_name('execution-global')
        self.result=None

    def now(self):return utc_time(self.clock())

    def _report(self,status,*,symbol=None,action='wait',reasons=(),cycle=None):
        now=self.now();report=dict(schema=1,mode='paper',phase='experimental_pilot',capital_usd=10000,
            generated_at=now.timestamp(),status=status,action=action,symbol=symbol,reasons=list(reasons),
            execution_enabled=True,formal_strategy_accepted=False,live_authorized=False,
            accounting_ok=False,cash_usd=None,equity_usd=None,profit_usd=None,return_pct=None,
            positions_count=None,open_orders_count=None,holdings=[],recent_trades=[],
            entries_paused=None,orders_halted=None,started_at=None)
        try:
            c=self.controls.read(now=now);report.update(entries_paused=c.pause_entries,orders_halted=c.halt_orders)
            approval=self.authorization.verify(policy=self.policy,limits=self.limits,now=now,allow_expired=True)
            report['started_at']=utc_time(approval['approved_at']).timestamp()
            report['execution_enabled']=now<utc_time(approval['expires_at']) and not c.halt_orders and status not in (
                'continuity_unavailable','authorization_unavailable','cycle_unavailable','strategy_changed')
            if cycle is not None:
                book=self.book.snapshot();a=book.account;r=book.receipt.result
                if (book.receipt==cycle.book_receipt and r.reconciled
                        and cycle.orders_sha256==_hash(_json(self.ledger.all_orders()))
                        and 0<=(now-book.receipt.as_of).total_seconds()<=60):
                    from .native_reconciler import _expected
                    costs={p.symbol:p.average_cost_usd for p in r.inventory}
                    orders=_expected(self.ledger.all_orders())
                    open_count=sum(o['status'] not in ('filled','canceled','expired','rejected','held') for o in orders.values())
                    report.update(accounting_ok=True,cash_usd=float(a.cash_usd),equity_usd=float(a.equity_usd),
                        profit_usd=float(r.profit_usd),return_pct=float(r.profit_usd)/100,
                        positions_count=len(a.positions),open_orders_count=open_count,
                        holdings=[dict(symbol=p.symbol,quantity=p.quantity,average_cost_usd=float(costs[p.symbol]),
                            reference_price_usd=float(p.reference_price_usd),unrealized_usd=float(
                            p.market_value_usd-costs[p.symbol]*p.quantity)) for p in a.positions][:20])
                    trades=[]
                    for o in orders.values():
                        if o['filled_quantity']:
                            trades.append(dict(symbol=o['symbol'],side=o['side'],quantity=o['filled_quantity'],
                                average_price_usd=float(o['notional']/o['filled_quantity']),reported_at=o['updated'].timestamp(),
                                status='filled' if o['filled_quantity']==o['quantity'] else 'partially_filled',
                                time_kind='updated_at'))
                    report['recent_trades']=sorted(trades,key=lambda t:t['reported_at'],reverse=True)[:20]
        except Exception:report['execution_enabled']=False
        self.result=report;return report

    def _quotes(self,symbols):
        raw=self.snapshots(sorted(set(symbols))) if symbols else {}
        if set(raw)!=set(symbols):raise PilotError('required_quotes_unavailable')
        return raw,tuple(Quote(s,str(v.bid),str(v.ask),v.quote_at,v.received_at) for s,v in sorted(raw.items()))

    def _view(self,cycle,quotes,symbol,control):
        return prepare_native_risk_view(book=self.book,ledger=self.ledger,baselines=self.baselines,
            cycle=cycle,quotes=quotes,symbol=symbol,operator_state=RiskState(
                pause_entries=control.pause_entries,halt_orders=control.halt_orders),calendar=self.calendar,now=self.now())

    def _sources(self,view,cycle,control):
        return NativeEntrySources(view,cycle,self.book,self.baselines,self.decisions,self.sources,
            control,self.limits,self.calendar)

    def _pending_close(self,rows):
        from .repair_orders import repair_rows
        for row in rows:
            if row.get('native_stop') is None:continue
            for slot,repair in [(row['native_stop'],None)]+[(r,r['identity']) for r in repair_rows(row)]:
                close=slot.get('close')
                if close and not close.get('recovery',{}).get('complete'):
                    return row['intent_id'],repair
        return None

    def tick(self):
        # No implicit state initialization/reset is permitted in a running tick.
        try:
            with _exclusive(self.lock_path):
                try:self.continuity()
                except Exception:return self._report('continuity_unavailable')
                try:self.authorization.verify(policy=self.policy,limits=self.limits,now=self.now(),allow_expired=True)
                except Exception:return self._report('authorization_unavailable')
                pending=self._pending_close(self.ledger.all_orders())
                if pending is None:return self._tick_locked()
            # CloseCompensator owns the same global lock; never nest it.
            from .compensation import CloseCompensator
            helper=CloseCompensator(broker=self.broker,ledger=self.ledger,reconciler=self.reconciler,
                controls=self.controls,decisions=self.decisions,repair_decisions=self.repair_decisions,
                policy=self.policy,limits=self.limits,calendar=self.calendar)
            def quotes():
                symbols={p.symbol for p in self.book.snapshot().account.positions}
                return self._quotes(symbols)[1]
            result=helper.advance(pending[0],target_repair_id=pending[1],baselines=self.baselines,quotes=quotes)
            return self._report('close_recovery',reasons=(result.status,))
        except Exception:return self._report('cycle_unavailable')

    def _tick_locked(self):
        control=self.controls.read(now=self.now())
        # GET recovery always precedes any new admission. Expiry never resolves
        # a claimed submission. Existing broker stops survive pause/halt.
        from .repair_orders import RepairPaperReader,repair_rows,retire_repair
        from .native_reconciler import _settled
        initial_groups=project_groups(self.ledger.all_orders())
        for row in self.ledger.all_orders():
            if row.get('native_entry') is not None and row['state']=='reserved':
                retire_unsent_entry(self.ledger,row['intent_id'],now=self.now())
            native=row.get('native_stop')
            if (native is not None and row['submission_claimed_at'] is not None
                    and not _settled(native) and not initial_groups[row['intent_id']].settled):
                NativeEntryRecovery(self.broker).cancel_entry(self.ledger,row['intent_id'],
                    controls=self.controls,control_snapshot=control,calendar=self.calendar)
                for repair in repair_rows(self.ledger.load(row['intent_id'])):
                    if repair['state']=='reserved' and self.now()>=utc_time(repair['expires_at']):
                        retire_repair(self.ledger,row['intent_id'],repair['identity'],
                            expected_sha256=_hash(_json(self.ledger.all_orders())),reason='expired',now=self.now())
                    if repair['submission_claimed_at'] is not None:
                        RepairPaperReader(self.broker).recover(self.ledger,row['intent_id'],repair['identity'])
        cycle=self.reconciler.refresh()
        if not cycle.ready_for_risk:
            return self._repair_or_wait(cycle,control)
        self.baselines.observe(book=self.book,cycle=cycle,calendar=self.calendar,now=self.now())
        if self.calendar.at(self.now()) is None:return self._report('market_closed',cycle=cycle)
        if control.halt_orders:return self._report('orders_halted',cycle=cycle)
        closing=self._account_risk_close(cycle,control)
        if closing is not None:return closing
        try:self.authorization.verify(policy=self.policy,limits=self.limits,now=self.now())
        except Exception:return self._report('trial_complete',cycle=cycle)
        try:source=self.sources.bridge.latest_review(self.sources.model_archive,source_archive=self.sources.quant_archive)
        except Exception:return self._report('waiting_for_model',cycle=cycle)
        q=source.read()['quant']['record'];symbol=q['symbol'];action=source.read()['model']['record']['receipt']['action']
        symbols={p.symbol for p in self.book.snapshot().account.positions}|{symbol}
        symbols|={r['intent']['symbol'] for r in self.ledger.all_orders() if r['state'] in ('reserved','open','partially_filled','submission_unknown')}
        try:
            raw,quotes=self._quotes(symbols);view=self._view(cycle,quotes,symbol,control)
        except Exception:return self._report('inputs_unavailable',symbol=symbol,cycle=cycle)
        if strategy_digest(source)!=self.policy.policy.strategy_sha256:
            return self._report('strategy_changed',symbol=symbol,cycle=cycle)
        groups=project_groups(self.ledger.all_orders())
        owned=[g for g in groups.values() if g.symbol==symbol and g.remaining_quantity>0 and g.parent_terminal]
        # Quantitative exits and account loss exits use the owned stop-cancel /
        # fresh terminal / cleanup protocol, never a competing sell reservation.
        if owned and action=='sell' and q['exit_candidate']:
            plan=prepare_close(ledger=self.ledger,prepared=view,family_id=owned[0].family_id,
                calendar=self.calendar,now=self.now(),reason='quantitative')
            frame=capture_close_decision(ledger=self.ledger,prepared=plan,source=source,
                policy=self.policy,controls=control,limits=self.limits,calendar=self.calendar,now=self.now())
            decision=self.decisions.append(frame,sources=self.sources,ledger=self.ledger)
            if decision.read()['outcome']['action']=='sell':
                NativeCloseGateway(self.broker).request(self.ledger,plan,decision=decision,journal=self.decisions,
                    controls=self.controls,control_snapshot=control,limits=self.limits,calendar=self.calendar,
                    book=self.book,baselines=self.baselines,sources=self.sources)
                return self._report('close_requested',symbol=symbol,action='sell',cycle=None)
        if action!='buy' or not q['entry_candidate']:
            return self._report('observing_signal',symbol=symbol,action='hold' if owned else 'wait',cycle=cycle)
        now=self.now();day=now.astimezone(ZoneInfo('America/New_York')).date()
        entries=[r for r in self.ledger.all_orders() if r.get('native_entry') is not None
            and utc_time(r['created_at']).astimezone(ZoneInfo('America/New_York')).date()==day]
        if len(entries)>=self.policy.policy.max_daily_entries:
            return self._report('daily_entry_limit',symbol=symbol,cycle=cycle)
        identity=_hash(_json(['paper-pilot-entry-v1',symbol,q['bar_ends_at'],strategy_digest(source)]))
        if any(r['intent_id']==identity for r in self.ledger.all_orders()):
            return self._report('signal_already_processed',symbol=symbol,cycle=cycle)
        ask=Decimal(str(raw[symbol].ask));tick=Decimal('.01') if ask>=1 else Decimal('.0001')
        # $750 target within $1,000 hard per-symbol/order cap; whole shares,
        # 2% stop plus modeled round-trip cost still checked independently.
        price=ask.quantize(tick,rounding=ROUND_DOWN)
        stop=(price*Decimal('.98')).quantize(tick,rounding=ROUND_DOWN)
        qty=int(Decimal('750')/price)
        if qty<1 or stop<=0:return self._report('whole_share_budget',symbol=symbol,cycle=cycle)
        intent=OrderIntent(identity,symbol,'buy',qty,price,stop,now+timedelta(seconds=20))
        frame=capture_decision(source=source,prepared=view,intent=intent,policy=self.policy,controls=control,
            limits=self.limits,calendar=self.calendar,now=now,requested_action='buy',authorization_sha256=self.authorization.digest)
        decision=self.decisions.append(frame,sources=self.sources,ledger=self.ledger)
        outcome=decision.read()['outcome']
        if outcome['action']!='buy':return self._report('risk_veto',symbol=symbol,reasons=outcome['vetoes'],cycle=cycle)
        admit_entry(self.ledger,intent,decision=decision,sources=self._sources(view,cycle,control),now=self.now(),time_in_force='gtc')
        # Admission changes reservations: reread actual account/ledger/quotes
        # before the independent one-shot sender claims and rechecks again.
        cycle=self.reconciler.refresh();raw,quotes=self._quotes(symbols);view=self._view(cycle,quotes,symbol,control)
        row=NativePaperGateway(self.broker).submit(self.ledger,identity,controls=self.controls,
            control_snapshot=control,market_price_usd=str(raw[symbol].price),market_at=raw[symbol].trade_at,
            entry_sources=self._sources(view,cycle,control))
        self.decisions.observe_order(decision.sequence,ledger=self.ledger)
        return self._report('entry_submitted' if row is not None else 'entry_not_sent',symbol=symbol,action='buy' if row else 'wait')

    def _account_risk_close(self,cycle,control):
        baseline=self.baselines.replay();equity=self.book.snapshot().account.equity_usd
        if (baseline.day_start_equity_usd-equity<self.limits.max_daily_loss_usd
                and baseline.high_water_equity_usd-equity<self.limits.max_drawdown_usd):return None
        if not control.pause_entries:
            control=self.controls.update(expected_revision=control.revision,pause_entries=True,
                halt_orders=False,reason='fault_review',now=self.now())
        groups=project_groups(self.ledger.all_orders())
        targets=[g for g in groups.values() if g.parent_terminal and g.remaining_quantity>0]
        if not targets:return self._report('account_loss_limit',cycle=cycle)
        symbols={p.symbol for p in self.book.snapshot().account.positions};_,quotes=self._quotes(symbols)
        target=sorted(targets,key=lambda g:g.symbol)[0]
        view=self._view(cycle,quotes,target.symbol,control)
        plan=prepare_close(ledger=self.ledger,prepared=view,family_id=target.family_id,
            calendar=self.calendar,now=self.now(),reason='account_risk')
        frame=capture_close_decision(ledger=self.ledger,prepared=plan,source=None,policy=self.policy,
            controls=control,limits=self.limits,calendar=self.calendar,now=self.now())
        decision=self.decisions.append(frame,ledger=self.ledger)
        if decision.read()['outcome']['action']!='sell':
            return self._report('risk_veto',symbol=target.symbol,reasons=decision.read()['outcome']['vetoes'],cycle=cycle)
        NativeCloseGateway(self.broker).request(self.ledger,plan,decision=decision,journal=self.decisions,
            controls=self.controls,control_snapshot=control,limits=self.limits,calendar=self.calendar,
            book=self.book,baselines=self.baselines)
        return self._report('close_requested',symbol=target.symbol,action='sell')

    def _repair_or_wait(self,cycle,control):
        # Standalone replacement protection only after actual terminal evidence
        # and complete accounting. The dedicated builder and sender own checks.
        from .repair_protection import prepare_repair
        from .repair_orders import reserve_repair
        from .repair_execution import RepairPaperGateway
        groups=project_groups(self.ledger.all_orders())
        targets=[g for g in groups.values() if g.parent_terminal and g.child_terminal
            and g.remaining_quantity>0 and g.available_quantity==g.remaining_quantity]
        if not targets or control.halt_orders or self.calendar.at(self.now()) is None:
            return self._report('reconciliation_pending',reasons=cycle.findings,cycle=cycle)
        if not control.pause_entries:
            control=self.controls.update(expected_revision=control.revision,pause_entries=True,
                halt_orders=False,reason='fault_review',now=self.now())
        self.baselines.observe_accounting(book=self.book,ledger=self.ledger,cycle=cycle,calendar=self.calendar,now=self.now())
        symbols={p.symbol for p in self.book.snapshot().account.positions};_,quotes=self._quotes(symbols)
        family=targets[0].family_id
        plan=prepare_repair(book=self.book,ledger=self.ledger,baselines=self.baselines,cycle=cycle,
            quotes=quotes,family_id=family,operator_state=RiskState(pause_entries=True),
            calendar=self.calendar,limits=self.limits,now=self.now())
        decision=self.repair_decisions.record(plan,ledger=self.ledger,book=self.book,baselines=self.baselines,
            controls=control,calendar=self.calendar,limits=self.limits)
        reservation=reserve_repair(ledger=self.ledger,plan=plan,book=self.book,baselines=self.baselines,
            calendar=self.calendar,limits=self.limits,now=self.now(),decision=decision,journal=self.repair_decisions)
        result=RepairPaperGateway(self.broker).submit(self.ledger,family,reservation.repair_id,journal=self.repair_decisions,
            book=self.book,baselines=self.baselines,controls=self.controls,control_snapshot=control,
            calendar=self.calendar,limits=self.limits)
        return self._report(repair_status(result),symbol=targets[0].symbol)
