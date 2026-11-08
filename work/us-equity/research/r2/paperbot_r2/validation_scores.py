"""Runtime code/policy commitment for the fixed validation evaluator.

Score/report admission is implemented independently of order authority.
"""
from contextlib import closing,contextmanager
from dataclasses import asdict,dataclass
from datetime import datetime,timedelta
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from paperbot_us.runner import _exclusive
from . import common_scores,validation_issue_journal,common_outcome,evaluation_book as books
from . import validation_settlements as settlements,model_training,training_codec as codec
from .contracts import aware,digest,is_sha


def source_sha256():
    names=('common_scores.py','evaluation_book.py','validation_settlements.py','validation_scores.py')
    paths=tuple(Path(__file__).with_name(name) for name in names)
    if not all(p.is_file() and not p.is_symlink() and p.stat().st_size<=524288 for p in paths):
        raise common_scores.CommonScoreError('evaluation_code_commitment_unavailable')
    return digest(dict(files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        issuer=validation_issue_journal.source_sha256(),outcome=common_outcome._source()))


def make_policy(prospective_protocol_sha256):
    return common_scores.ScorePolicy.build(source_sha256(),prospective_protocol_sha256)


FINAL_DEADLINE=aware('2026-12-04T21:02:00+00:00')


class ValidationScoreError(ValueError):pass


def _need(value):
    if not value:raise ValidationScoreError('frozen_validation_score_unavailable')


class FreezeClock(settlements.ValidationClock):
    def __call__(self):
        at=super().__call__();_need(at>FINAL_DEADLINE);return at


def _descriptor(book,state):
    identity=book._identity()
    return dict(path=str(book.path),device=identity[0],inode=identity[1],head_seq=state['head_seq'],head_digest=state['head_digest'])


def _check_descriptor(descriptor,book,state):
    identity=book._identity()
    _need(descriptor['path']==str(book.path) and (descriptor['device'],descriptor['inode'])==identity[:2]
          and descriptor['head_seq']<=state['head_seq'])
    with closing(sqlite3.connect(book.path.as_uri()+'?mode=ro',uri=True)) as db:
        validation_issue_journal._prefix(db,'events',descriptor['head_seq'],descriptor['head_digest'])


@contextmanager
def _score_view(settlement_book,*,timezone_file,guard):
    guard();_need(type(settlement_book) is settlements.ValidationSettlementJournal)
    with validation_issue_journal._verified_epoch((settlement_book.path,),guard) as settlement_stable:
        state=settlement_book.snapshot(clock=guard);_need(state['initialized'])
        binding=state['binding'];policy=settlements.decode_policy(binding['policy'])
        issues=validation_issue_journal.ValidationIssueJournal(binding['issue_book']['path'])
        with settlements.issue_view(issues,policy,timezone_file=timezone_file,guard=guard) as issue_view:
            _need(issue_view['binding']==binding['issue_binding'])
            settlements.check_issue_descriptor(binding['issue_book'],issue_view)
            def stable():settlement_stable();issue_view['stable']()
            stable()
            yield dict(settlements=settlement_book,settlement_state=state,issue=issue_view,policy=policy,stable=stable)
            stable();guard()


def _source_prefix(proof,old,pair):
    issuer=validation_issue_journal
    _need(issuer._same_source(proof,old))
    for path,table,seq,sha,current in (
        (pair.raw_book.path,'events',old.raw_head_seq,old.raw_head_digest,proof.raw_head_seq),
        (pair.feature_journal.path,'observations',old.feature_head_seq,old.feature_head_digest,proof.feature_head_seq)):
        _need(current>=seq)
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:issuer._prefix(db,table,seq,sha)


def _current_quarantine(issue_view):
    plan=issue_view['registration'].plan();slots=issue_view['slots'];keys={s.key for s in slots}
    statuses={};quarantined=set()
    for session in plan['calendar']['sessions'][:8]:
        day=session['day'];index=next(i for i,p in enumerate(issue_view['proofs']) if p.start_day<=day<=p.end_day)
        pair=issue_view['pairs'][index]
        with closing(sqlite3.connect(pair.raw_book.path.as_uri()+'?mode=ro',uri=True)) as db:
            values=db.execute("SELECT json_extract(payload,'$.body.session') FROM events "
                "WHERE json_extract(payload,'$.kind')='calendar' AND json_extract(payload,'$.body.day')=?",(day,)).fetchall()
        if not values:status='missing'
        elif any(value[0] is None or json.loads(value[0])!=session for value in values):status='quarantined'
        else:status='verified'
        statuses[day]=status
        if status=='quarantined':quarantined.update(s.key for s in slots if s.day==day)
    for pair in issue_view['pairs']:
        with closing(sqlite3.connect(pair.feature_journal.path.as_uri()+'?mode=ro',uri=True)) as db:
            rows=db.execute("SELECT json_extract(payload,'$.body.key') FROM observations WHERE json_extract(payload,'$.kind')='quarantined'").fetchall()
            quarantined.update(key for (key,) in rows if key in keys)
    issue_view['stable']();return statuses,quarantined


def _check_frame_sources(key,frame,issue_view):
    captured=frame['input'];trained=issue_view['training'];registration=issue_view['registration']
    _need(captured.registration.identity==registration.identity and captured.registration.plan_json==registration.plan_json
          and captured.training_snapshot_sha256==trained.training.seal and captured.fits_sha256==trained.fits.seal)
    for proof,old,pair in zip(issue_view['proofs'],captured.segments,issue_view['pairs']):_source_prefix(proof,old,pair)
    with closing(sqlite3.connect(issue_view['registry'].path.as_uri()+'?mode=ro',uri=True)) as db:
        validation_issue_journal._prefix(db,'canonical_v2_events',captured.registration.canonical_head_seq,captured.registration.canonical_head_digest)
    _need(key==captured.symbol+'|'+captured.bundle.vector.event_at.isoformat())


def _rows(frame,trained):
    panel=frame['panel']
    if panel is None:return ()
    found=[]
    for row in panel.rows:
        attempt=next(a for a in trained.fits.attempts if a.symbol==panel.symbol and a.family==row.family)
        _need(row.model_sha256==attempt.seal)
        if row.status!='computed':found.append(common_scores.ScoreRow(row.family,row.failure_code,None,None,None));continue
        payload=model_training.bound_model(attempt,symbol=panel.symbol,snapshot_sha256=trained.training.seal,as_of=panel.started_at)
        mean=None
        if row.family=='ewma':mean=0.
        elif row.family=='garch':mean=float(payload.model.mean)
        elif row.family=='hmm':mean=float(math.fsum(w*m for w,m in zip(row.target_weights,payload.model.parameters.means)))
        found.append(common_scores.ScoreRow(row.family,'computed',row.probability,mean,row.target_variance))
    return tuple(found)


def _verify_settlement(key,entry,frame,issue_view):
    _need(frame['phase']=='confirmed');witness=entry['witness'];captured=frame['input']
    issues=issue_view['issue_journal'];identity=validation_issue_journal.inputs._identity(issues.path)
    _need(witness.key==key and witness.symbol==captured.symbol and witness.input_sha256==captured.seal
          and witness.panel_sha256==frame['panel'].seal and witness.frame_confirmation_digest==frame['confirmation_digest']
          and witness.issued_at==frame['issued_at'] and witness.issue_commit_observed_at==frame['commit_observed_at']
          and witness.issue_book_path==str(issues.path) and (witness.issue_book_device,witness.issue_book_inode)==identity[:2]
          and witness.issue_book_head_seq<=issue_view['state']['head_seq'])
    with closing(sqlite3.connect(issues.path.as_uri()+'?mode=ro',uri=True)) as db:
        validation_issue_journal._prefix(db,'events',witness.issue_book_head_seq,witness.issue_book_head_digest)
    result=entry['outcome']
    if result is not None:
        _need(result.feature_key==key and result.symbol==captured.symbol and result.feature_bundle_sha256==captured.bundle.seal
              and result.example.features==captured.bundle.vector and result.target_start==frame['panel'].target_start
              and result.target_end==frame['panel'].target_end and result.observed_at>=witness.available_at)
        day=captured.bundle.session_day;index=next(i for i,p in enumerate(issue_view['proofs']) if p.start_day<=day<=p.end_day)
        pair=issue_view['pairs'][index];proof=issue_view['proofs'][index]
        _need((result.raw_plan_sha256,result.raw_source_sha256)==(proof.raw_plan_sha256,proof.raw_source_sha256)
              and result.raw_head_seq<=proof.raw_head_seq and result.feature_head_seq<=proof.feature_head_seq)
        for path,table,seq,sha in ((pair.raw_book.path,'events',result.raw_head_seq,result.raw_head_digest),
                                  (pair.feature_journal.path,'observations',result.feature_head_seq,result.feature_head_digest)):
            with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:validation_issue_journal._prefix(db,table,seq,sha)


def _build_frames(view,*,as_of):
    iv=view['issue'];original=iv['state']['frames'];settled=view['settlement_state']['domain']['frames']
    expected={s.key:s for s in iv['slots']};_need(set(original)<=set(expected) and set(settled)<=set(original))
    statuses,quarantined=_current_quarantine(iv);frames=[]
    for key,frame in original.items():
        _check_frame_sources(key,frame,iv);slot=expected[key];issued=frame['phase']=='confirmed'
        entry=settled.get(key)
        if entry is not None:_verify_settlement(key,entry,frame,iv)
        confirmed=entry is not None and entry['status']=='confirmed'
        outcome=entry['outcome'] if confirmed else None
        complete=outcome is not None
        missing=bool(entry['missing']) if entry is not None else False
        late=bool(entry['effective_late']) if confirmed else bool(entry is not None and as_of>slot.target_end+timedelta(seconds=120))
        frames.append(common_scores.ScoreFrame(key,slot.symbol,slot.event_at,slot.target_start,slot.target_end,
            _rows(frame,iv['training']),issued,complete,outcome.label if complete else None,
            outcome.return_fraction if complete else None,missing,late,key in quarantined))
    # Unissued calendar quarantines are visible too; they remain unissued.
    for key in sorted(quarantined-set(original)):
        slot=expected[key];frames.append(common_scores.ScoreFrame(key,slot.symbol,slot.event_at,slot.target_start,slot.target_end,quarantined=True))
    view['stable']();return tuple(frames),statuses,quarantined


def _basis(view,calendar_status,quarantined):
    iv=view['issue'];trained=iv['training']
    return dict(settlement_book=_descriptor(view['settlements'],view['settlement_state']),
        issue_book=settlements._issue_descriptor(iv['issue_journal'],iv['state']),source_proofs=[asdict(p) for p in iv['proofs']],
        quarantined_keys=sorted(quarantined),calendar_status=calendar_status,registration_identity=iv['registration'].identity,
        policy_seal=view['policy'].seal,training_head_digest=trained.head_digest,
        training_fits_sha256=trained.fits.seal,training_snapshot_sha256=trained.training.seal,
        training_outcome=iv['training_outcome_proof'])


def verify_basis(basis,view):
    _need(type(basis) is dict and set(basis)=={'settlement_book','issue_book','source_proofs','quarantined_keys',
        'calendar_status','registration_identity','policy_seal','training_head_digest','training_fits_sha256','training_snapshot_sha256','training_outcome'})
    iv=view['issue'];trained=iv['training']
    _check_descriptor(basis['settlement_book'],view['settlements'],view['settlement_state'])
    settlements.check_issue_descriptor(basis['issue_book'],iv)
    _need(basis['registration_identity']==iv['registration'].identity and basis['policy_seal']==view['policy'].seal
          and basis['training_head_digest']==trained.head_digest and basis['training_fits_sha256']==trained.fits.seal
          and basis['training_snapshot_sha256']==trained.training.seal and len(basis['source_proofs'])==2)
    original=basis['training_outcome'];current=iv['training_outcome_proof']
    _need(all(original.get(k)==current.get(k) for k in ('kind','path','device','inode','plan_sha256','source_sha256',
        'consumed_head_seq','consumed_head_digest')) and original['head_seq']<=current['head_seq'])
    with closing(sqlite3.connect(Path(current['path']).as_uri()+'?mode=ro',uri=True)) as db:
        validation_issue_journal._prefix(db,'observations',original['head_seq'],original['head_digest'])
    for proof,raw,pair in zip(iv['proofs'],basis['source_proofs'],iv['pairs']):
        _source_prefix(proof,validation_issue_journal._segment(raw),pair)
    _,quarantined=_current_quarantine(iv)
    _need(sorted(quarantined)==basis['quarantined_keys'])
    view['stable']()


@dataclass(frozen=True)
class FrozenValidationReceipt:
    registration_identity:str
    policy_json:str
    report_json:str
    basis_json:str
    book_path:str
    book_device:int
    book_inode:int
    head_seq:int
    head_digest:str
    evaluated_at:datetime
    commit_observed_at:datetime
    available_at:datetime
    requires_selection_admission:bool
    test_admission:bool
    orders:bool
    source_sha256:str
    seal:str

    def __post_init__(self):
        policy=settlements.decode_policy(self.policy_json);report=json.loads(self.report_json)
        _need(is_sha(self.registration_identity) and is_sha(self.head_digest) and self.head_seq==4
            and self.requires_selection_admission is True and self.test_admission is False and self.orders is False
            and self.available_at>=self.commit_observed_at>self.evaluated_at>FINAL_DEADLINE
            and self.commit_observed_at<settlements.TEST_OPEN and report['math_only'] is True
            and report['selection_admission'] is report['test_admission'] is report['orders'] is False
            and policy.evaluation_source_sha256==source_sha256() and self.source_sha256==source_sha256()
            and self.seal==digest({k:v for k,v in asdict(self).items() if k!='seal'}))


class ValidationScoreJournal(books.EvaluationBook):
    def source_sha256(self):return source_sha256()
    def domain_initial(self):
        return dict(report=None,basis=None,evaluated_at=None,evaluated_digest=None,confirmed_at=None,confirmation_digest=None)

    def domain_apply(self,domain,kind,body,at,sha,binding):
        if kind=='evaluated':
            _need(domain['report'] is None and set(body)=={'report','basis'} and FINAL_DEADLINE<at<settlements.TEST_OPEN)
            report=body['report'];basis=body['basis']
            _need(type(report) is dict and report['schema']==1 and report['phase']=='validation' and report['math_only'] is True
                and report['selection_admission'] is report['test_admission'] is report['orders'] is False
                and report['trial_units']==15 and report['source_sha256']==common_scores.source_sha256()
                and FINAL_DEADLINE<aware(report['as_of'])<=at
                and basis['policy_seal']==binding['policy']['seal'] and basis['registration_identity']==binding['registration_identity'])
            domain.update(report=report,basis=basis,evaluated_at=at,evaluated_digest=sha)
        elif kind=='confirmed':
            _need(domain['report'] is not None and domain['confirmed_at'] is None
                and body==dict(evaluated_digest=domain['evaluated_digest'])
                and domain['evaluated_at']<at<settlements.TEST_OPEN)
            domain.update(confirmed_at=at,confirmation_digest=sha)
        else:_need(False)

    def initialize(self,settlement_journal,*,timezone_file,clock):
        try:
            guard=settlements.ValidationClock(timezone_file,clock,initial=True);guard()
            with _score_view(settlement_journal,timezone_file=timezone_file,guard=guard) as view:
                iv=view['issue'];identity=iv['registration'].identity
                _need(self.path==Path(iv['binding']['canonical']['path']).parent/('r2-validation-scores-'+identity+'.sqlite'))
                binding=dict(settlement_book=_descriptor(settlement_journal,view['settlement_state']),policy=asdict(view['policy']),
                    registration_identity=identity,timezone_file=str(Path(timezone_file).absolute()))
                return self.create(binding,clock=guard,check=view['stable'])
        except Exception:raise ValidationScoreError('validation_score_initialization_failed') from None

    def _check_binding(self,binding,view):
        _need(binding['registration_identity']==view['issue']['registration'].identity and binding['policy']==asdict(view['policy']))
        _check_descriptor(binding['settlement_book'],view['settlements'],view['settlement_state'])

    def freeze_once(self,*,timezone_file,clock):
        try:
            guard=FreezeClock(timezone_file,clock);guard()  # Before SQL or any label access.
            with _exclusive(self.path):
                state=self.snapshot(clock=guard);_need(state['initialized']);domain=state['domain']
                if domain['confirmed_at'] is not None:return state
                binding=state['binding'];settled=settlements.ValidationSettlementJournal(binding['settlement_book']['path'])
                with _score_view(settled,timezone_file=timezone_file,guard=guard) as view:
                    self._check_binding(binding,view)
                    if domain['report'] is None:
                        as_of=guard();frames,statuses,quarantined=_build_frames(view,as_of=as_of)
                        fits=tuple((a.symbol,a.family,a.status) for a in view['issue']['training'].fits.attempts)
                        report=common_scores.evaluate_frames(view['issue']['registration'].plan(),frames,as_of=as_of,fit_statuses=fits)
                        basis=_basis(view,statuses,quarantined);view['stable']()
                        state=self.append(state,'evaluated',dict(report=report,basis=basis),clock=guard,check=view['stable'])
                        domain=state['domain']
                    verify_basis(domain['basis'],view)
                    state=self.append(state,'confirmed',dict(evaluated_digest=domain['evaluated_digest']),clock=guard,check=view['stable'])
                    return state
        except Exception:raise ValidationScoreError('validation_score_freeze_refused') from None

    def load_frozen(self,*,timezone_file,clock):
        try:
            guard=settlements.ValidationClock(timezone_file,clock,allow_after_test=True);started=guard()
            with validation_issue_journal._verified_epoch((self.path,),guard) as own_stable:
                state=self.snapshot(clock=guard);_need(state['initialized']);domain=state['domain']
                _need(domain['confirmed_at'] is not None)
                binding=state['binding'];settled=settlements.ValidationSettlementJournal(binding['settlement_book']['path'])
                with _score_view(settled,timezone_file=timezone_file,guard=guard) as view:
                    self._check_binding(binding,view);verify_basis(domain['basis'],view);own_stable()
                    available=guard();_need(available>=started>=domain['confirmed_at'])
                own_stable()
            identity=self._identity()
            fields=dict(registration_identity=binding['registration_identity'],policy_json=codec.encode(binding['policy']),
                report_json=codec.encode(domain['report']),basis_json=codec.encode(domain['basis']),
                book_path=str(self.path),book_device=identity[0],book_inode=identity[1],head_seq=state['head_seq'],head_digest=state['head_digest'],
                evaluated_at=domain['evaluated_at'],commit_observed_at=domain['confirmed_at'],available_at=available,
                requires_selection_admission=True,test_admission=False,orders=False,source_sha256=source_sha256())
            return FrozenValidationReceipt(**fields,seal=digest(fields))
        except Exception:raise ValidationScoreError('frozen_validation_report_unavailable') from None
