"""First test publications from original current selection and causal inputs."""
from contextlib import contextmanager
from dataclasses import asdict,dataclass
from datetime import datetime,timedelta
from pathlib import Path

from paperbot_us.runner import _exclusive
from . import evaluation_book as books,forecast_inputs as inputs,candidate_forecasts as maths
from . import selection_journal as selection,test_protocol as protocol,training_codec as codec
from . import validation_scores as scores,validation_settlements as settlements,validation_issue_journal as issuer
from .contracts import aware,digest,is_sha
from .model_training import FAMILIES


class TestIssueError(ValueError):pass


def _need(value):
    if not value:raise TestIssueError('prospective_test_publication_unavailable')


def _key(key):
    try:return protocol._test_key(key)
    except Exception:raise TestIssueError('registered_test_key_required') from None


def _reference(raw):
    _need(type(raw) is dict and set(raw)=={'selected_family','selection_head_digest','selection_commit_observed_at','validation_report_sha256'}
          and raw['selected_family'] in FAMILIES and is_sha(raw['selection_head_digest']) and is_sha(raw['validation_report_sha256']))
    at=aware(raw['selection_commit_observed_at']);_need(protocol.SELECTION_AFTER<at<protocol.TEST_OPEN)
    return dict(raw,selection_commit_observed_at=at)


def admission_reference(admission):
    _need(type(admission) is selection.TestAdmission);admission.__post_init__()
    return _reference(dict(selected_family=admission.selected_family,selection_head_digest=admission.selection_head_digest,
        selection_commit_observed_at=admission.commit_observed_at,validation_report_sha256=admission.validation_report_sha256))


@contextmanager
def selection_view(book,*,timezone_file,guard,require_admission=True):
    guard();_need(type(book) is selection.SelectionJournal)
    source=protocol.source_sha256()
    with issuer._verified_epoch((book.path,),guard) as own_stable:
        state=book.snapshot(clock=guard);_need(state['initialized']);binding=state['binding']
        score_book=scores.ValidationScoreJournal(binding['score_book']['path'])
        with selection.validation_view(score_book,timezone_file=timezone_file,guard=guard) as validation:
            book._check_binding(binding,validation)
            admission=book.load_current(timezone_file=timezone_file,clock=guard) if require_admission else None
            if admission is not None:
                _need(admission.selection_head_seq==state['head_seq'] and admission.selection_head_digest==state['head_digest']
                      and admission.selection_book_path==str(book.path))
            def stable():
                own_stable();validation['stable']();_need(protocol.source_sha256()==source)
            stable()
            yield dict(selection_book=book,selection_state=state,validation=validation,
                       issue=validation['source']['issue'],admission=admission,stable=stable)
            stable();guard()


@contextmanager
def issue_view(book,*,timezone_file,guard,require_admission=True):
    at=guard()
    _need(type(require_admission) is bool and (require_admission or at<settlements.registered.FIRST_OPEN))
    _need(type(book) is TestIssueJournal)
    with issuer._verified_epoch((book.path,),guard) as own_stable:
        state=book.snapshot(clock=guard);_need(state['initialized']);binding=state['binding']
        source=selection.SelectionJournal(binding['selection_book']['path'])
        with selection_view(source,timezone_file=timezone_file,guard=guard,require_admission=require_admission) as view:
            book._check_binding(binding,view)
            def stable():own_stable();view['stable']()
            stable()
            yield dict(book=book,state=state,binding=binding,selection=view,issue=view['issue'],stable=stable)
            stable();guard()


@dataclass(frozen=True)
class TestIssuedReceipt:
    issue_input:inputs.IssueInput
    panel:maths.CandidatePanel
    admission_json:str
    issued_at:datetime
    commit_observed_at:datetime
    available_at:datetime
    book_path:str
    book_device:int
    book_inode:int
    head_seq:int
    head_digest:str
    confirmation_digest:str
    source_sha256:str
    publication_only:bool
    promotion:bool
    orders:bool
    seal:str

    def __post_init__(self):
        _need(type(self.issue_input) is inputs.IssueInput and type(self.panel) is maths.CandidatePanel)
        self.issue_input.__post_init__();self.panel.__post_init__()
        _reference(codec._raw(self.admission_json));event=_key(self.issue_input.symbol+'|'+self.issue_input.bundle.vector.event_at.isoformat())
        _need(self.panel.input_sha256==self.issue_input.seal and self.panel.fits_sha256==self.issue_input.fits_sha256
              and self.panel.symbol==self.issue_input.symbol and self.panel.target_start==self.issue_input.bundle.target_start
              and self.panel.target_end==self.issue_input.bundle.target_end
              and self.available_at>=self.commit_observed_at>self.issued_at>=self.panel.completed_at
              and self.commit_observed_at<=event+timedelta(seconds=120) and self.commit_observed_at<self.panel.target_start
              and Path(self.book_path).is_absolute() and type(self.book_device) is type(self.book_inode) is int
              and type(self.head_seq) is int and self.head_seq>=6 and is_sha(self.head_digest) and is_sha(self.confirmation_digest)
              and self.source_sha256==protocol.source_sha256() and self.publication_only is True
              and self.promotion is self.orders is False
              and self.seal==digest({k:v for k,v in asdict(self).items() if k!='seal'}))


class TestIssueJournal(books.EvaluationBook):
    def source_sha256(self):return protocol.source_sha256()
    def domain_initial(self):return dict(frames={})

    def domain_apply(self,domain,kind,body,at,sha,binding):
        _need(type(body.get('key')) is str);key=body['key'];event=_key(key);frame=domain['frames'].get(key)
        _need(event<=at<=event+timedelta(seconds=120) and at<event+timedelta(seconds=900))
        if kind=='reserved':
            _need(frame is None and set(body)=={'key','input','candidates','admission'})
            captured=inputs.decode_input(body['input']);bound=binding['validation_issue_binding'];reference=_reference(body['admission'])
            _need(key==captured.symbol+'|'+captured.bundle.vector.event_at.isoformat()
                  and captured.registration.identity==bound['identity'] and captured.registration.plan_json==bound['plan_json']
                  and captured.fits_sha256==bound['training']['fits_sha256'] and captured.available_at<=at
                  and body['candidates']==list(FAMILIES)
                  and key in {s.key for s in protocol.planned_slots(captured.registration.plan())})
            _need(all(old['admission']==reference for old in domain['frames'].values()))
            domain['frames'][key]=dict(phase='reserved',input=captured,panel=None,admission=reference,
                reserved_at=at,reservation_digest=sha,result_digest=None,issued_at=None,issue_digest=None,
                commit_observed_at=None,confirmation_digest=None,consumed_slots=5)
        elif kind=='computed':
            _need(frame is not None and frame['phase']=='reserved' and set(body)=={'key','panel','reservation_digest'}
                  and body['reservation_digest']==frame['reservation_digest'])
            panel=maths.decode_panel(body['panel']);captured=frame['input']
            _need(panel.input_sha256==captured.seal and panel.fits_sha256==captured.fits_sha256
                  and panel.feature_sha256==captured.bundle.seal and panel.symbol==captured.symbol
                  and panel.target_start==captured.bundle.target_start and panel.target_end==captured.bundle.target_end
                  and frame['reserved_at']<=panel.started_at<=panel.completed_at<=at)
            frame.update(phase='computed',panel=panel,result_digest=sha)
        elif kind=='issued':
            _need(frame is not None and frame['phase']=='computed' and set(body)=={'key','result_digest'}
                  and body['result_digest']==frame['result_digest'] and at>=frame['panel'].completed_at)
            frame.update(phase='issued',issued_at=at,issue_digest=sha)
        elif kind=='confirmed':
            _need(frame is not None and frame['phase']=='issued' and set(body)=={'key','issue_digest'}
                  and body['issue_digest']==frame['issue_digest'] and at>frame['issued_at'])
            frame.update(phase='confirmed',commit_observed_at=at,confirmation_digest=sha)
        elif kind=='abandoned':
            _need(frame is not None and frame['phase']=='reserved' and set(body)=={'key','reservation_digest'}
                  and body['reservation_digest']==frame['reservation_digest'])
            frame.update(phase='abandoned')
        else:_need(False)
        _need(len(domain['frames'])<=240)

    def initialize(self,selection_journal,*,timezone_file,clock):
        try:
            guard=settlements.ValidationClock(timezone_file,clock,initial=True);guard()
            with selection_view(selection_journal,timezone_file=timezone_file,guard=guard,require_admission=False) as view:
                iv=view['issue'];identity=iv['registration'].identity
                _need(self.path==Path(iv['binding']['canonical']['path']).parent/('r2-test-issues-'+identity+'.sqlite'))
                binding=dict(selection_book=scores._descriptor(selection_journal,view['selection_state']),
                    selection_binding=view['selection_state']['binding'],validation_issue_binding=iv['binding'],
                    policy=asdict(view['validation']['policy']),timezone_file=str(Path(timezone_file).absolute()))
                return self.create(binding,clock=guard,check=view['stable'])
        except Exception:raise TestIssueError('test_issue_initialization_refused') from None

    def _check_binding(self,binding,view):
        _need(binding['selection_binding']==view['selection_state']['binding']
              and binding['validation_issue_binding']==view['issue']['binding']
              and binding['policy']==asdict(view['validation']['policy']))
        scores._check_descriptor(binding['selection_book'],view['selection_book'],view['selection_state'])

    def _check_frame(self,frame,view,guard,*,exact=False):
        _need(frame['admission']==admission_reference(view['admission']))
        iv=view['issue']
        issuer._sources(iv['binding'],iv['pairs'],iv['training'],guard,issue_input=frame['input'],exact=exact)
        view['stable']()

    def issue_once(self,key,*,selection_journal,timezone_file,clock):
        try:
            guard=protocol.TestIssueClock(key,timezone_file,clock);guard()  # Before any lock/SQL.
            with _exclusive(self.path):
                state=self.snapshot(clock=guard);_need(state['initialized']);binding=state['binding']
                with selection_view(selection_journal,timezone_file=timezone_file,guard=guard) as view:
                    self._check_binding(binding,view);frame=state['domain']['frames'].get(key)
                    if frame is not None:
                        self._check_frame(frame,view,guard)
                        if frame['phase'] in ('confirmed','abandoned'):return state
                        if frame['phase']=='reserved':
                            return self.append(state,'abandoned',dict(key=key,reservation_digest=frame['reservation_digest']),
                                               clock=guard,check=view['stable'])
                    iv=view['issue']
                    if frame is None:
                        captured=inputs.capture_issue_input(iv['training'],iv['registration'],iv['pairs'],key,
                                                           timezone_file=timezone_file,clock=guard)
                        body=dict(key=key,input=inputs.encode_input(captured),candidates=list(FAMILIES),admission=admission_reference(view['admission']))
                        state=self.append(state,'reserved',body,clock=guard,check=view['stable']);frame=state['domain']['frames'][key]
                        self._check_frame(frame,view,guard,exact=True)
                        panel=maths.compute_candidates(iv['training'],captured,clock=guard)
                        state=self.append(state,'computed',dict(key=key,panel=codec.encode(panel),reservation_digest=frame['reservation_digest']),
                                          clock=guard,check=lambda:self._check_frame(frame,view,guard,exact=True))
                        frame=state['domain']['frames'][key]
                    def check():self._check_frame(frame,view,guard)
                    if frame['phase']=='computed':
                        state=self.append(state,'issued',dict(key=key,result_digest=frame['result_digest']),clock=guard,check=check)
                        frame=state['domain']['frames'][key]
                    _need(frame['phase']=='issued')
                    return self.append(state,'confirmed',dict(key=key,issue_digest=frame['issue_digest']),clock=guard,check=check)
        except Exception:raise TestIssueError('test_publication_refused') from None

    def load_issued(self,key,*,timezone_file,clock):
        try:
            event=_key(key);base=issuer.canonical.Clock(timezone_file,clock)
            def guard():
                at=base();_need(at>=event);return at
            guard()  # Future publications are refused before any source SQL.
            with issue_view(self,timezone_file=timezone_file,guard=guard) as view:
                frame=view['state']['domain']['frames'].get(key)
                _need(frame is not None and frame['phase']=='confirmed')
                self._check_frame(frame,view['selection'],guard);identity=self._identity()
                fields=dict(issue_input=frame['input'],panel=frame['panel'],admission_json=codec.encode(frame['admission']),
                    issued_at=frame['issued_at'],commit_observed_at=frame['commit_observed_at'],available_at=guard(),book_path=str(self.path),
                    book_device=identity[0],book_inode=identity[1],head_seq=view['state']['head_seq'],head_digest=view['state']['head_digest'],
                    confirmation_digest=frame['confirmation_digest'],source_sha256=protocol.source_sha256(),publication_only=True,promotion=False,orders=False)
                receipt=TestIssuedReceipt(**fields,seal=digest(fields));view['stable']()
            return receipt
        except Exception:raise TestIssueError('current_test_publication_unavailable') from None
