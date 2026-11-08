"""Bounded train-only selection and a durable precommit before holdout use.

The local registry detects repeated use while preserved. It is not protection
against an owner deleting history or rewriting content and hashes. No strategy
is accepted automatically from this small fixed-symbol reference study.
"""
from dataclasses import fields
import hashlib
import json
import os
import sys
from pathlib import Path

from .alpaca import decode_json,ProviderError
from .backtest import backtest,ReferencePolicy,CostModel,policy_record,research_code_hash,BacktestError
from .data import DataQualityError
from .research import ResearchDataset,SessionSplit,time_split
from .runner import _exclusive,RunnerError


class StudyError(RuntimeError):
    pass


def _json(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def _hash(value): return hashlib.sha256(_json(value).encode()).hexdigest()


def _write(path,value):
    payload=_json(value).encode()
    if len(payload)>64*1024*1024: raise StudyError('study_artifact_too_large')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
    with os.fdopen(fd,'wb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if os.name!='nt':
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)


def _read(path):
    if path.is_symlink(): raise StudyError('study_artifact_link_refused')
    with path.open('rb') as handle: raw=handle.read(64*1024*1024+1)
    if len(raw)>64*1024*1024: raise StudyError('study_artifact_too_large')
    value=decode_json(raw)
    if set(value)!={'content','sha256'} or _hash(value['content'])!=value['sha256']:
        raise StudyError('study_artifact_integrity_mismatch')
    return value['content']


def _save(path,value): _write(path,dict(content=value,sha256=_hash(value)))


def run_study(dataset,*,split,policies,costs,directory):
    try:
        if type(dataset) is not ResearchDataset or type(split) is not SessionSplit or type(costs) is not CostModel:
            raise StudyError('explicit_study_contract_required')
        if not isinstance(policies,(tuple,list)) or not 1<=len(policies)<=16 or any(type(p) is not ReferencePolicy for p in policies):
            raise StudyError('bounded_parameter_search_required')
        if len(dataset.bars)*len(policies)>100000: raise StudyError('study_compute_budget_exceeded')
        days=tuple(s.day for s in dataset.sessions)
        expected=time_split(days,train_sessions=len(split.train),validation_sessions=len(split.validation),
                            test_sessions=len(split.test),embargo_sessions=len(split.embargo)//2)
        if split!=expected or len(split.embargo)<2:
            raise StudyError('invalid_or_overlapping_session_split_or_missing_embargo')
        candidates={_hash(policy_record(policy)):policy for policy in policies}
        if len(candidates)!=len(policies): raise StudyError('duplicate_policy_candidate')
        configuration=dict(data_sha256=dataset.manifest()['data_sha256'],
            data_identity=dict(symbol=dataset.bars[0].symbol,source='alpaca',feed='iex',adjustment='raw'),
            policies={key:policy_record(value) for key,value in sorted(candidates.items())},
            costs={f.name:getattr(costs,f.name) for f in fields(CostModel)},python_version=sys.version.split()[0],
            split={name:[day.isoformat() for day in getattr(split,name)] for name in ('train','validation','test','embargo')},
            code_sha256=hashlib.sha256((research_code_hash()+Path(__file__).read_text()+Path(__file__).with_name('runner.py').read_text()).encode()).hexdigest())
        study_id=_hash(configuration)
        # Receipt timestamps and revised prices cannot turn the same symbol /
        # dates into a newly unseen holdout in this preserved registry.
        holdout=_hash(dict(symbol=dataset.bars[0].symbol,source='alpaca',feed='iex',adjustment='raw',
                           test_days=configuration['split']['test']))
        directory=Path(directory).absolute()
        if directory.is_symlink(): raise StudyError('study_directory_link_refused')
        directory.mkdir(mode=0o700,exist_ok=True)
        locked=directory/(holdout+'.locked.json')
        report_path=directory/(study_id+'.report.json')
        # Serialize the complete registry, not merely an exact set of test days.
        # A changed split must not reuse previously exposed dates as untouched.
        with _exclusive(directory/'registry'):
            current_used=set(configuration['split']['train']+configuration['split']['validation']+configuration['split']['test'])
            current_holdout=set(configuration['split']['validation']+configuration['split']['test'])
            for prior_path in sorted(directory.glob('*.locked.json')):
                prior=_read(prior_path)
                if prior['study_id']==study_id: continue
                prior_config=prior['configuration']
                if prior_config['data_identity']!=dict(symbol=dataset.bars[0].symbol,source='alpaca',feed='iex',adjustment='raw'):
                    continue
                prior_split=prior_config['split']
                prior_used=set(prior_split['train']+prior_split['validation']+prior_split['test'])
                prior_holdout=set(prior_split['validation']+prior_split['test'])
                if current_holdout&prior_used or current_used&prior_holdout:
                    raise StudyError('overlapping_previously_exposed_research_days')
            if locked.exists():
                commitment=_read(locked)
                if commitment['study_id']!=study_id or commitment['configuration']!=configuration:
                    raise StudyError('holdout_already_exposed_to_other_study')
                if report_path.exists():
                    report=_read(report_path)
                    if report['study_id']!=study_id: raise StudyError('study_report_identity_mismatch')
                    return report
            else:
                train=dataset.subset(split.train)
                results={key:backtest(train,policy=policy,costs=costs) for key,policy in sorted(candidates.items())}
                # Only train results select the rule. Tie breaks are explicit
                # and stable; no validation/test information enters this rank.
                winner=sorted(results,key=lambda key:(-results[key]['net_return_fraction'],
                              results[key]['maximum_drawdown_fraction'],key))[0]
                commitment=dict(study_id=study_id,configuration=configuration,selected_policy_id=winner,
                                training_results=results,stage='locked_before_validation_and_test')
                _save(locked,commitment)
            winner=commitment['selected_policy_id']
            if winner not in candidates: raise StudyError('invalid_selected_policy')
            validation=backtest(dataset.subset(split.validation),policy=candidates[winner],costs=costs)
            test=backtest(dataset.subset(split.test),policy=candidates[winner],costs=costs)
            report=dict(study_id=study_id,configuration=configuration,selected_policy_id=winner,
                selection_locked_before_holdout=True,train=commitment['training_results'][winner],
                training_results=commitment['training_results'],validation=validation,test=test,
                strategy_accepted=False,acceptance_status='engineering_reference_only',
                limitations=['fixed_symbol_not_historical_market_universe','vendor_history_is_not_original_receipt_evidence',
                             'uncalibrated_costs_and_bar_fill_assumptions','insufficient_independent_oos_evidence',
                             'corporate_actions_and_market_benchmark_gates_incomplete'])
            _save(report_path,report)
            return report
    except StudyError:
        raise
    except (DataQualityError,BacktestError,ProviderError,RunnerError,OSError,ValueError,TypeError,KeyError):
        raise StudyError('study_failed_no_strategy_acceptance') from None
