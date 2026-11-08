"""Pinned paper-only acceptance evidence. No generation, approval or broker I/O.

An administrator must pin the manifest after independent evidence review. This
reader checks its content and reports; it cannot establish that an untrusted
report is true or that historical performance will recur. Production must keep
the source evidence and its independent reproducible validation available.
"""
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
import hashlib,os,stat
import sqlite3
from contextlib import closing
from pathlib import Path

from .alpaca import decode_json
from .data import utc_time
from .execution_policy import PolicySnapshot
from .orders import _hash,_json,_base
from .risk import RiskLimits,_identity,_symbol


class GateError(RuntimeError):pass


def _read(path,expected):
    _identity(expected);path=Path(path).absolute()
    if path.is_symlink() or path.resolve()!=path:raise ValueError('canonical gate path required')
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
    with os.fdopen(fd,'rb') as f:
        s=os.fstat(f.fileno())
        if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_size>65536:raise ValueError('bounded gate file required')
        raw=f.read(65537)
    if len(raw)>65536 or hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('gate content changed')
    value=decode_json(raw)
    if type(value) is not dict:raise ValueError('gate object required')
    return value


def _exact(value,keys):
    if type(value) is not dict or set(value)!=set(keys.split()):raise ValueError('gate schema mismatch')


def _integer(value,minimum=0,maximum=10000000):
    if type(value) is not int or not minimum<=value<=maximum:raise ValueError('gate integer invalid')
    return value


def _number(value):
    if type(value) not in (int,float):raise ValueError('gate number required')
    result=Decimal(str(value))
    if not result.is_finite() or abs(result)>10000000:raise ValueError('bounded gate number required')
    return result


_ZERO={
    'security':('secret_findings','unauthed_write_paths'),
    'data':('invalid_timestamps','future_rows','unresolved_actions','missing_calendar_rows','unlicensed_rows','survivorship_unresolved'),
    'quant':('failures',),'backtest':('leaked_rows','unpriced_costs','omitted_fill_constraints'),
    'decision':('untraced_decisions','unsafe_fallbacks'),
    'risk':('failed_limits','duplicate_mutations','kill_switch_failures'),
    'paper_lifecycle':('unexplained_cash_cents','unexplained_shares','live_requests','unresolved_orders'),
    'reproducibility':('failures','missing_inputs')}
_POSITIVE={'data':('observations',),'quant':('cases',),'decision':('cases',),'risk':('cases',),
    'paper_lifecycle':('completed_order_cycles',),'reproducibility':('cases',)}
_TRUE={'quant':('units_reviewed',),'backtest':('calibrated_costs',),'paper_lifecycle':('real_provider',)}
_GATES=frozenset((*_ZERO,'evaluation'))


@dataclass(frozen=True)
class AcceptanceReceipt:
    digest:str
    strategy_sha256:str
    dataset_sha256:str
    symbols:tuple
    expires_at:object

    def public_summary(self):
        return dict(mode='paper',acceptance_evidence_verified=True,live_authorized=False,
            symbol_count=len(self.symbols),expires_at=self.expires_at.timestamp())


class PaperAcceptanceFile:
    def __init__(self,path,expected_sha256):
        _identity(expected_sha256);self.path=Path(path).absolute();self.expected_sha256=expected_sha256

    def verify(self,*,policy,limits,strategy_sha256,now):
        try:
            if type(policy) is not PolicySnapshot or type(limits) is not RiskLimits:raise ValueError('exact gate dependencies required')
            now=utc_time(now);_identity(strategy_sha256);v=_read(self.path,self.expected_sha256)
            _exact(v,'schema kind mode capital_usd approved_at expires_at strategy_sha256 dataset_sha256 risk_limits_sha256 symbols criteria evidence')
            if (type(v['schema']) is not int or v['schema']!=1 or v['kind']!='paper_strategy_acceptance'
                    or v['mode']!='paper' or type(v['capital_usd']) is not int or v['capital_usd']!=10000):
                raise ValueError('paper-only acceptance required')
            _identity(v['dataset_sha256'])
            if (policy.policy.acceptance_sha256!=self.expected_sha256 or policy.policy.strategy_sha256!=strategy_sha256
                    or v['strategy_sha256']!=strategy_sha256 or v['risk_limits_sha256']!=policy.policy.risk_limits_sha256
                    or v['risk_limits_sha256']!=_hash(_json(_base(limits,RiskLimits)))):
                raise ValueError('acceptance policy/source mismatch')
            approved=utc_time(v['approved_at']);expires=utc_time(v['expires_at'])
            if not approved<=now<expires<=approved+timedelta(days=30):raise ValueError('acceptance expired or future')
            symbols=v['symbols']
            if type(symbols) is not list or not 1<=len(symbols)<=2000 or symbols!=sorted(set(symbols)):
                raise ValueError('bounded accepted universe required')
            for symbol in symbols:_symbol(symbol)
            _exact(v['evidence'],' '.join(_GATES));reports={}
            for name in sorted(_GATES):
                ref=v['evidence'][name];_exact(ref,'path sha256')
                if ref['path']!=name+'.json':raise ValueError('fixed sibling evidence path required')
                report=_read(self.path.with_name(ref['path']),ref['sha256'])
                _exact(report,'schema kind mode strategy_sha256 dataset_sha256 metrics')
                if (type(report['schema']) is not int or report['schema']!=1 or report['kind']!=name or report['mode']!='paper'
                        or report['strategy_sha256']!=strategy_sha256 or report['dataset_sha256']!=v['dataset_sha256']):
                    raise ValueError('acceptance evidence identity mismatch')
                metrics=report['metrics'];reports[name]=metrics
                if name=='evaluation':continue
                keys=(*_ZERO[name],*_POSITIVE.get(name,()),*_TRUE.get(name,()))
                if name=='paper_lifecycle':keys=(*keys,'provider_mode')
                _exact(metrics,' '.join(keys))
                if any(_integer(metrics[k])!=0 for k in _ZERO[name]):raise ValueError('acceptance safety finding')
                for key in _POSITIVE.get(name,()):_integer(metrics[key],1)
                if any(metrics[k] is not True for k in _TRUE.get(name,())):raise ValueError('acceptance evidence incomplete')
                if name=='paper_lifecycle' and metrics['provider_mode']!='paper':raise ValueError('actual paper provider required')
            self._evaluation(v['criteria'],reports['evaluation'],approved,limits)
            return AcceptanceReceipt(self.expected_sha256,strategy_sha256,v['dataset_sha256'],tuple(symbols),expires)
        except Exception:raise GateError('paper_acceptance_unavailable_or_not_passed') from None

    @staticmethod
    def _evaluation(criteria,metrics,approved,limits):
        _exact(criteria,'registered_at min_test_closed_trades min_walk_forward_folds min_net_return_fraction max_drawdown_fraction')
        _exact(metrics,'train_end validation_start validation_end test_start test_end test_closed_trades walk_forward_folds '
            'net_return_fraction maximum_drawdown_fraction daily_volatility_fraction turnover_fraction total_cost_usd multiple_testing_corrected')
        registered=utc_time(criteria['registered_at']);end=utc_time(metrics['test_end'])
        if not (registered<utc_time(metrics['train_end'])<utc_time(metrics['validation_start'])
                <=utc_time(metrics['validation_end'])<utc_time(metrics['test_start'])<end<=approved):
            raise ValueError('predeclared ordered independent research required')
        trades=_integer(criteria['min_test_closed_trades'],1);folds=_integer(criteria['min_walk_forward_folds'],2)
        min_return=_number(criteria['min_net_return_fraction']);max_dd=_number(criteria['max_drawdown_fraction'])
        if not -1<=min_return<=1 or not 0<max_dd<=limits.max_drawdown_usd/limits.capital_usd:
            raise ValueError('invalid predeclared performance criteria')
        if (_integer(metrics['test_closed_trades'])<trades or _integer(metrics['walk_forward_folds'])<folds
                or _number(metrics['net_return_fraction'])<min_return
                or not 0<=_number(metrics['maximum_drawdown_fraction'])<=max_dd
                or metrics['multiple_testing_corrected'] is not True):raise ValueError('paper performance criteria not met')
        for k in ('daily_volatility_fraction','turnover_fraction','total_cost_usd'):
            if _number(metrics[k])<0:raise ValueError('invalid evaluation metric')


@dataclass(frozen=True)
class StateGateReceipt:
    proof_sha256:str
    completion_sha256:str
    baseline_seed_sha256:str


class ExecutionStateGate:
    """Read-only replay of original continuity and the pinned current stores.

    This never creates/migrates a store, changes a source reference or resets
    capital/high-water state. A source upgrade requires a separately verified
    continuity procedure, not pointing the worker at a fresh empty database.
    """
    def __init__(self,*,replay,continuity_journal,expected_proof_sha256,
            expected_completion_sha256,baseline_seed_sha256,paths):
        from .book_replay import BookReplay
        try:
            if type(replay) is not BookReplay:raise ValueError('exact original book replay required')
            _exact(paths,'book ledger baselines decisions')
            for h in (expected_proof_sha256,expected_completion_sha256,baseline_seed_sha256):_identity(h)
            self.replay=replay;self.journal=Path(continuity_journal).absolute()
            self.proof_sha256=expected_proof_sha256;self.completion_sha256=expected_completion_sha256
            self.baseline_seed_sha256=baseline_seed_sha256
            self.paths={k:Path(v).absolute() for k,v in paths.items()}
            all_paths=(*self.paths.values(),*replay.paths,self.journal)
            if len(set(all_paths))!=len(all_paths):raise ValueError('separate pinned state required')
        except Exception:raise GateError('invalid_execution_state_registry') from None

    def verify(self,*,book,ledger,baselines,decisions,now):
        from .accountbook import AccountBook
        from .orders import OrderLedger
        from .risk_baselines import RiskBaselineStore,_digest
        from .decision_journal import DecisionJournal
        from .book_continuity import replay_continuity
        try:
            for key,obj,exact in (('book',book,AccountBook),('ledger',ledger,OrderLedger),
                    ('baselines',baselines,RiskBaselineStore),('decisions',decisions,DecisionJournal)):
                if type(obj) is not exact or obj.path!=self.paths[key]:raise ValueError('registered state path changed')
                if obj.path.resolve()!=obj.path or obj.path.is_symlink() or obj.path.stat().st_nlink!=1:
                    raise ValueError('canonical single-link state required')
            proof=replay_continuity(replay=self.replay,book=book,journal=self.journal)
            if (proof.proof_sha256!=self.proof_sha256 or proof.audit_sha256!=self.completion_sha256
                    or proof.as_of>utc_time(now)):raise ValueError('original continuity identity changed')
            state,_,_=baselines._load();receipt=baselines.replay()
            with closing(sqlite3.connect(baselines.path.as_uri()+'?mode=ro',uri=True)) as db:
                row=db.execute('SELECT payload,digest FROM observations ORDER BY seq LIMIT 1').fetchone()
            if (row is None or row[1]!=self.baseline_seed_sha256 or decode_json(row[0])['kind']!='risk_initialize'
                    or state['anchor_sha256']!=_digest(proof.anchor)
                    or state['account_sha256']!=proof.anchor.account_sha256
                    or receipt.high_water_equity_usd<10000 or receipt.book_sequence<proof.book_receipt.sequence):
                raise ValueError('baseline origin or history changed')
            snapshot=book.snapshot_by_digest(receipt.book_digest)
            if snapshot.anchor!=proof.anchor:raise ValueError('current baseline account changed')
            ledger.all_orders();decisions.verify()
            return StateGateReceipt(proof.proof_sha256,proof.audit_sha256,self.baseline_seed_sha256)
        except Exception:raise GateError('execution_state_continuity_unavailable') from None
