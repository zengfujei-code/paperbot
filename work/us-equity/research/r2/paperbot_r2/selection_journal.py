"""One source-bound validation choice, with no strategy/order authority.

Scalar projections verify deterministic rules only. Current admission always
reloads the original physical validation book and all upstream source proofs.
"""
from contextlib import contextmanager
from dataclasses import asdict,dataclass
from datetime import datetime
import math
from pathlib import Path

from paperbot_us.runner import _exclusive
from . import canonical_history,common_registration as registered,common_scores
from . import evaluation_book as books,test_protocol as protocol,training_codec as codec
from . import validation_issue_journal as issuer,validation_scores as scores,validation_settlements as settlements
from .contracts import aware,digest,is_sha
from .model_training import FAMILIES
from .training_data import SYMBOLS


class SelectionError(ValueError):pass


def _need(value):
    if not value:raise SelectionError('original_validation_selection_unavailable')


def _projection_valid(value):
    _need(type(value) is dict and set(value)=={'report_sha256','report_as_of','report_source_sha256',
        'training_fits','coverage','macro_brier','engineering_gate_passed','selected_family','reason'})
    _need(is_sha(value['report_sha256']) and aware(value['report_as_of'])>protocol.SELECTION_AFTER
          and value['report_source_sha256']==common_scores.source_sha256())
    fits=value['training_fits']
    _need(type(fits) is list and all(type(r) is dict and set(r)=={'symbol','family','status'} for r in fits)
          and tuple((r['symbol'],r['family']) for r in fits)==registered.PAIRS
          and all(r['status'] in ('succeeded','failed') for r in fits))
    coverage=value['coverage'];expected=tuple((day,symbol) for day in registered.VALIDATION_DAYS for symbol in SYMBOLS)
    _need(type(coverage) is list and all(type(r) is dict and set(r)=={'day','symbol','planned','common'} for r in coverage)
          and tuple((r['day'],r['symbol']) for r in coverage)==expected)
    for row in coverage:
        _need(type(row['planned']) is int and row['planned']==(4 if row['day']=='2026-11-27' else 16)
              and type(row['common']) is int and 0<=row['common']<=row['planned'])
    passed=all(r['common']*5>=r['planned']*4 for r in coverage)
    macro=value['macro_brier']
    _need(type(macro) is dict and set(macro)==set(FAMILIES)
          and all(v is None or (type(v) is float and math.isfinite(v) and 0<=v<=1.) for v in macro.values())
          and type(value['engineering_gate_passed']) is bool and value['engineering_gate_passed']==passed)
    if passed:
        _need(all(v is not None for v in macro.values()) and all(r['status']=='succeeded' for r in fits))
        selected=min(FAMILIES,key=lambda f:macro[f])
    else:selected=None
    _need(value['selected_family']==selected and value['reason']==('minimum_macro_brier' if passed else 'engineering_gate_failed'))


def project_report(report):
    """Pure rule projection, never source-backed admission on its own."""
    try:
        _need(type(report) is dict and report['schema']==1 and report['phase']=='validation'
              and report['math_only'] is report['complete'] is True and report['trial_units']==15
              and report['selection_admission'] is report['test_admission'] is report['orders'] is False)
        value=dict(report_sha256=digest(report),report_as_of=report['as_of'],report_source_sha256=report['source_sha256'],
            training_fits=[dict(r) for r in report['training_fits']],
            coverage=[dict(day=g['day'],symbol=g['symbol'],planned=g['counts']['planned'],common=g['counts']['common'])
                      for g in report['groups']],macro_brier={f:report['macro'][f]['brier'] for f in FAMILIES},
            engineering_gate_passed=report['engineering_gate_passed'],selected_family=report['proposed_family'],
            reason='minimum_macro_brier' if report['engineering_gate_passed'] else 'engineering_gate_failed')
        _projection_valid(value)
        _need(all(g['coverage_passed']==(g['counts']['common']*5>=g['counts']['planned']*4) for g in report['groups']))
        return value
    except Exception:raise SelectionError('frozen_validation_projection_invalid') from None


@contextmanager
def validation_view(score_book,*,timezone_file,guard):
    """Protect original score and all upstream sources over the entire read."""
    guard();_need(type(score_book) is scores.ValidationScoreJournal)
    source=protocol.source_sha256()  # An unfinished protocol never opens source SQL.
    with issuer._verified_epoch((score_book.path,),guard) as own_stable:
        state=score_book.snapshot(clock=guard);_need(state['initialized'])
        bound=state['binding'];settled=settlements.ValidationSettlementJournal(bound['settlement_book']['path'])
        with scores._score_view(settled,timezone_file=timezone_file,guard=guard) as view:
            score_book._check_binding(bound,view)
            _need(view['policy'].prospective_protocol_sha256==source)
            def stable():
                own_stable();view['stable']();_need(protocol.source_sha256()==source)
            stable()
            yield dict(score_book=score_book,score_state=state,source=view,policy=view['policy'],stable=stable)
            stable();guard()


def _frozen(view,*,timezone_file,guard):
    book=view['score_book'];state=view['score_state']
    receipt=book.load_frozen(timezone_file=timezone_file,clock=guard)
    _need(type(receipt) is scores.FrozenValidationReceipt and receipt.head_seq==state['head_seq']
          and receipt.head_digest==state['head_digest'] and receipt.book_path==str(book.path)
          and receipt.report_json==codec.encode(state['domain']['report'])
          and receipt.policy_json==codec.encode(asdict(view['policy'])))
    view['stable']();return receipt,project_report(codec._raw(receipt.report_json))


def _source_book(value):
    _need(type(value) is dict and set(value)=={'path','device','inode','head_seq','head_digest'}
          and type(value['path']) is str and Path(value['path']).is_absolute()
          and type(value['device']) is type(value['inode']) is int
          and value['head_seq']==4 and is_sha(value['head_digest']))


@dataclass(frozen=True)
class TestAdmission:
    registration_identity:str
    selected_family:str
    validation_report_sha256:str
    policy_json:str
    selection_book_path:str
    selection_device:int
    selection_inode:int
    selection_head_seq:int
    selection_head_digest:str
    score_book_json:str
    decided_at:datetime
    commit_observed_at:datetime
    available_at:datetime
    source_sha256:str
    test_only:bool
    promotion:bool
    orders:bool
    seal:str

    def __post_init__(self):
        policy=settlements.decode_policy(self.policy_json);_source_book(codec._raw(self.score_book_json))
        _need(self.selected_family in FAMILIES and all(is_sha(v) for v in
              (self.registration_identity,self.validation_report_sha256,self.selection_head_digest,self.source_sha256,self.seal))
              and self.selection_head_seq==4 and Path(self.selection_book_path).is_absolute()
              and type(self.selection_device) is type(self.selection_inode) is int
              and protocol.SELECTION_AFTER<self.decided_at<self.commit_observed_at<protocol.TEST_OPEN
              and self.available_at>=self.commit_observed_at and self.test_only is True
              and self.promotion is self.orders is False
              and policy.prospective_protocol_sha256==self.source_sha256==protocol.source_sha256()
              and self.seal==digest({k:v for k,v in asdict(self).items() if k!='seal'}))


class SelectionJournal(books.EvaluationBook):
    def source_sha256(self):return protocol.source_sha256()
    def domain_initial(self):
        return dict(decision=None,score_book=None,score_commit_observed_at=None,score_available_at=None,
                    decided_at=None,decision_digest=None,confirmed_at=None,confirmation_digest=None)

    def domain_apply(self,domain,kind,body,at,sha,binding):
        _need(protocol.SELECTION_AFTER<at<protocol.TEST_OPEN)
        if kind=='decided':
            _need(domain['decision'] is None and set(body)=={'decision','score_book','score_commit_observed_at','score_available_at'})
            _projection_valid(body['decision']);_source_book(body['score_book'])
            observed,available=aware(body['score_commit_observed_at']),aware(body['score_available_at'])
            _need(protocol.SELECTION_AFTER<aware(body['decision']['report_as_of'])<observed<=available<=at)
            domain.update(decision=body['decision'],score_book=body['score_book'],score_commit_observed_at=observed,
                          score_available_at=available,decided_at=at,decision_digest=sha)
        elif kind=='confirmed':
            _need(domain['decision'] is not None and domain['confirmed_at'] is None and domain['decided_at']<at
                  and body==dict(decision_digest=domain['decision_digest']))
            domain.update(confirmed_at=at,confirmation_digest=sha)
        else:_need(False)

    def initialize(self,validation_score_journal,*,timezone_file,clock):
        try:
            guard=settlements.ValidationClock(timezone_file,clock,initial=True);guard()
            with validation_view(validation_score_journal,timezone_file=timezone_file,guard=guard) as view:
                identity=view['source']['issue']['registration'].identity
                _need(self.path==Path(view['source']['issue']['binding']['canonical']['path']).parent/('r2-selection-'+identity+'.sqlite'))
                binding=dict(score_book=scores._descriptor(validation_score_journal,view['score_state']),
                    score_binding=view['score_state']['binding'],policy=asdict(view['policy']),registration_identity=identity,
                    timezone_file=str(Path(timezone_file).absolute()),protocol_sha256=protocol.source_sha256())
                return self.create(binding,clock=guard,check=view['stable'])
        except Exception:raise SelectionError('selection_initialization_refused') from None

    def _check_binding(self,binding,view):
        _need(binding['protocol_sha256']==protocol.source_sha256() and binding['policy']==asdict(view['policy'])
              and binding['score_binding']==view['score_state']['binding']
              and binding['registration_identity']==view['source']['issue']['registration'].identity)
        scores._check_descriptor(binding['score_book'],view['score_book'],view['score_state'])

    def _check_decision(self,domain,view,receipt,decision):
        _need(domain['decision']==decision and domain['score_book']==scores._descriptor(view['score_book'],view['score_state'])
              and domain['score_commit_observed_at']==receipt.commit_observed_at
              and domain['score_available_at']>=receipt.commit_observed_at)

    def select_once(self,*,timezone_file,clock):
        try:
            guard=protocol.SelectionClock(timezone_file,clock);guard()  # Before SQL/locks.
            with _exclusive(self.path):
                state=self.snapshot(clock=guard);_need(state['initialized']);binding=state['binding']
                score_book=scores.ValidationScoreJournal(binding['score_book']['path'])
                with validation_view(score_book,timezone_file=timezone_file,guard=guard) as view:
                    self._check_binding(binding,view)
                    receipt,decision=_frozen(view,timezone_file=timezone_file,guard=guard)
                    domain=state['domain']
                    if domain['decision'] is None:
                        body=dict(decision=decision,score_book=scores._descriptor(score_book,view['score_state']),
                                  score_commit_observed_at=receipt.commit_observed_at,score_available_at=receipt.available_at)
                        state=self.append(state,'decided',body,clock=guard,check=view['stable']);domain=state['domain']
                    self._check_decision(domain,view,receipt,decision)
                    if domain['confirmed_at'] is None:
                        state=self.append(state,'confirmed',dict(decision_digest=domain['decision_digest']),clock=guard,check=view['stable'])
                    view['stable']();return state
        except Exception:raise SelectionError('selection_decision_refused') from None

    def load_current(self,*,timezone_file,clock):
        try:
            base=canonical_history.Clock(timezone_file,clock)
            def guard():
                at=base();_need(at>protocol.SELECTION_AFTER);return at
            guard()  # Before SQL or source reads.
            with issuer._verified_epoch((self.path,),guard) as own_stable:
                state=self.snapshot(clock=guard);_need(state['initialized']);binding=state['binding'];domain=state['domain']
                _need(domain['confirmed_at'] is not None and domain['decision']['selected_family'] in FAMILIES)
                score_book=scores.ValidationScoreJournal(binding['score_book']['path'])
                with validation_view(score_book,timezone_file=timezone_file,guard=guard) as view:
                    self._check_binding(binding,view)
                    receipt,decision=_frozen(view,timezone_file=timezone_file,guard=guard)
                    self._check_decision(domain,view,receipt,decision);view['stable']();own_stable()
                    identity=self._identity();available=guard()
                    fields=dict(registration_identity=binding['registration_identity'],selected_family=decision['selected_family'],
                        validation_report_sha256=decision['report_sha256'],policy_json=codec.encode(binding['policy']),
                        selection_book_path=str(self.path),selection_device=identity[0],selection_inode=identity[1],
                        selection_head_seq=state['head_seq'],selection_head_digest=state['head_digest'],
                        score_book_json=codec.encode(domain['score_book']),decided_at=domain['decided_at'],
                        commit_observed_at=domain['confirmed_at'],available_at=available,source_sha256=protocol.source_sha256(),
                        test_only=True,promotion=False,orders=False)
                    result=TestAdmission(**fields,seal=digest(fields));view['stable']();own_stable()
                own_stable()
            return result
        except Exception:raise SelectionError('current_test_admission_unavailable') from None
