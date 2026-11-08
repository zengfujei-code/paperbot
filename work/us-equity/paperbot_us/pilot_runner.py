"""Deployment entrypoint for the authorized experimental paper pilot.

Initialize is a separate, explicit, first-use operation. Normal runs never
initialize, migrate or reset existing state. The supervisor bounds whole worker
process groups and schedules a near-close accounting observation.
"""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from zoneinfo import ZoneInfo

from .accountbook import AccountBook
from .alpaca import AlpacaDataClient
from .book_continuity import carry_forward_empty_book
from .book_replay import BookReplay
from .broker import PaperBroker
from .broker_time_transport import CanonicalBrokerTimes
from .controls import ControlJournal
from .decision_journal import DecisionJournal,ReplaySources
from .execution_gates import ExecutionStateGate
from .execution_policy import PolicyFile
from .monitor import _publish
from .native_reconciler import NativePaperReconciler
from .orders import OrderLedger,_json
from .pilot import PilotAuthorization,PilotCoordinator
from .repair_decisions import RepairDecisionJournal
from .replay_bridge import PinnedRelease,ReplayBridge,regular_bytes,strict_json
from .risk import RiskLimits
from .risk_baselines import RiskBaselineStore


def configuration(path,digest):
    raw=regular_bytes(Path(path).absolute(),32768)
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('deployment configuration changed')
    value=strict_json(raw)
    if value['schema']!=1 or value['mode']!='paper' or value['capital_usd']!=10000:raise ValueError('paper deployment required')
    return value


def build(value):
    state=Path(value['state']);now=datetime.now(timezone.utc)
    data=AlpacaDataClient.from_environment();day=now.astimezone(ZoneInfo('America/New_York')).date()
    calendar=data.calendar((day-timedelta(days=14)).isoformat(),(day+timedelta(days=7)).isoformat())
    def release(kind):
        v=value[kind];raw=regular_bytes(Path(v['manifest']),32768)
        if hashlib.sha256(raw).hexdigest()!=v['manifest_sha256']:raise ValueError('original manifest changed')
        manifest=strict_json(raw)
        return PinnedRelease(v['source'],files=manifest['files'],package_sha256=manifest['package_sha256'])
    old=Path(value['original_book_state'])
    replay=BookReplay(release('original_book'),book=old/'book.sqlite',baseline=old/'baseline.sqlite',ledger=old/'orders.sqlite')
    sources=ReplaySources(ReplayBridge(release('original_model')),value['quant_archive'],value['model_archive'])
    ledger=OrderLedger(state/'orders.sqlite');book=AccountBook(state/'book.sqlite')
    controls=ControlJournal(state/'controls.sqlite');baselines=RiskBaselineStore(state/'baselines.sqlite')
    decisions=DecisionJournal(state/'decisions.sqlite');repairs=RepairDecisionJournal(state/'repair-decisions.sqlite',clock=lambda:datetime.now(timezone.utc))
    broker=PaperBroker.from_environment()
    broker._transport=CanonicalBrokerTimes(ledger=ledger,journal=state/'broker-times.sqlite',
        transport=broker._transport,clock=broker._clock)
    reconciler=NativePaperReconciler(broker=broker,ledger=ledger,book=book,journal=state/'reconciliation.sqlite',calendar=calendar)
    policy=PolicyFile(value['policy'],value['policy_sha256']).read()
    approval=PilotAuthorization(value['authorization'],value['authorization_sha256'])
    return locals()


def initialize(value):
    d=build(value);state=d['state']
    if any(state.iterdir()):raise ValueError('new exclusive state required')
    d['ledger'].initialize()
    proof=carry_forward_empty_book(replay=d['replay'],broker=d['broker'],ledger=d['ledger'],
        book=d['book'],journal=state/'continuity.sqlite')
    cycle=d['reconciler'].refresh()
    if not cycle.ready_for_risk:raise ValueError('initial account reconciliation failed')
    d['baselines'].initialize(book=d['book'],cycle=cycle,now=datetime.now(timezone.utc))
    d['decisions'].initialize()
    d['controls'].initialize(now=datetime.now(timezone.utc))
    d['controls'].update(expected_revision=1,pause_entries=False,halt_orders=False,
        reason='operator_resume',now=datetime.now(timezone.utc))
    registry=dict(proof_sha256=proof.proof_sha256,completion_sha256=proof.audit_sha256,
        baseline_seed_sha256=d['baselines'].replay().audit_sha256)
    _publish(state/'registry.json',registry)
    # Private account trace and original evidence stay in the service directory.
    os.chmod(state/'registry.json',0o600)
    return dict(status='pilot_initialized',mode='paper',capital_usd=10000,orders_submitted=0,
        anchor_preserved=True,formal_strategy_accepted=False,
        registry_sha256=hashlib.sha256((state/'registry.json').read_bytes()).hexdigest())


def once(value):
    d=build(value);state=d['state'];raw=regular_bytes(state/'registry.json',8192)
    if hashlib.sha256(raw).hexdigest()!=value['registry_sha256']:raise ValueError('state origin changed')
    registry=strict_json(raw)
    gate=ExecutionStateGate(replay=d['replay'],continuity_journal=state/'continuity.sqlite',
        expected_proof_sha256=registry['proof_sha256'],expected_completion_sha256=registry['completion_sha256'],
        baseline_seed_sha256=registry['baseline_seed_sha256'],paths=dict(book=d['book'].path,
            ledger=d['ledger'].path,baselines=d['baselines'].path,decisions=d['decisions'].path))
    def continuity():return gate.verify(book=d['book'],ledger=d['ledger'],baselines=d['baselines'],
        decisions=d['decisions'],now=datetime.now(timezone.utc))
    coordinator=PilotCoordinator(**{k:d[k] for k in ('broker','ledger','book','baselines','controls',
        'decisions','reconciler','policy','calendar','sources')},repair_decisions=d['repairs'],
        authorization=d['approval'],continuity=continuity,limits=RiskLimits(),snapshots=d['data'].snapshots)
    result=coordinator.tick();_publish(value['publish'],result)
    session=d['calendar'].at(datetime.now(timezone.utc))
    # The pristine daily baseline is retained; held-account days require an
    # actual observation within 30s of close, not an invented closing price.
    delay=180 if session else 900
    if session:
        remaining=(session.closes-datetime.now(timezone.utc)).total_seconds()
        if 15<remaining<195:delay=max(5,remaining-15)
        elif 0<remaining<=15:delay=15
    if result['status'] in ('entry_submitted','close_requested','close_recovery','protection_repair_dispatched'):
        delay=5
    return dict(status=result['status'],mode='paper',next_check_seconds=delay)


def supervise(args,value):
    while True:
        started=time.monotonic()
        process=subprocess.Popen([sys.executable,'-B','-m','paperbot_us.pilot_runner','--config',args.config,
            '--config-sha256',args.config_sha256,'--once'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        status='worker_unavailable';delay=60
        try:
            output,errors=process.communicate(timeout=110)
            if process.returncode==0 and not errors and len(output)<8192:
                r=strict_json(output);delay=r['next_check_seconds'];status=r['status']
                if type(delay) not in (int,float) or not 5<=delay<=900:raise ValueError('invalid schedule')
        except Exception:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.communicate(timeout=5)
        # No raw error, provider output, credentials or account identifiers enter
        # systemd logs/public projections. A deadline never clears a claim.
        print(_json(dict(mode='paper',status=status)),flush=True)
        deadline=time.monotonic()+delay
        while True:
            _publish(value['heartbeat'],dict(mode='paper',generated_at=time.time(),last_cycle_status=status,
                cycle_elapsed_seconds=round(time.monotonic()-started,2)))
            remaining=deadline-time.monotonic()
            if remaining<=0:break
            time.sleep(min(30,remaining))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--config-sha256',required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--initialize',action='store_true');group.add_argument('--once',action='store_true')
    group.add_argument('--supervise',action='store_true');args=parser.parse_args()
    try:
        value=configuration(args.config,args.config_sha256)
        if args.supervise:supervise(args,value)
        else:print(_json(initialize(value) if args.initialize else once(value)),flush=True)
        return 0
    except Exception:
        print(_json(dict(mode='paper',status='pilot_operation_unavailable')),flush=True);return 1


if __name__=='__main__':sys.exit(main())
