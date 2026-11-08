"""One immutable, source-bound final test comparison; never order authority."""
from contextlib import closing
from dataclasses import asdict,dataclass
from datetime import datetime,timedelta
import json
from pathlib import Path
import sqlite3

from paperbot_us.runner import _exclusive
from . import test_protocol as protocol,test_issue_journal as issuer,test_settlements as settlements
from . import validation_scores as previous,validation_settlements as validation,common_scores as maths
from . import evaluation_book as books,training_codec as codec
from .contracts import aware,digest,is_sha
from .model_training import FAMILIES
from .training_data import SYMBOLS


class TestScoreError(ValueError):pass


def _need(value):
    if not value:raise TestScoreError('frozen_test_comparison_unavailable')


def validate_report(report,selected_family):
    """Validate the declared projection, without granting source authority."""
    try:
        _need(type(report) is dict and report['schema']==1 and report['phase']=='test'
              and report['math_only'] is report['complete'] is True
              and report['selection_admission'] is report['test_admission'] is report['promotion'] is report['orders'] is False
              and report['trial_units']==15 and report['test_sessions']==5
              and selected_family in FAMILIES and report['selected_family']==selected_family and report['proposed_family'] is None
              and report['evidence_class']=='exploratory_risk_forecast_only'
              and report['source_sha256']==protocol.math_source_sha256()
              and aware(report['as_of'])>protocol.TEST_FINAL_DEADLINE)
        fits=report['training_fits'];groups=report['groups']
        _need(type(fits) is list and all(type(r) is dict and set(r)=={'symbol','family','status'} for r in fits)
              and tuple((r['symbol'],r['family']) for r in fits)==protocol.registered.PAIRS
              and all(r['status'] in ('succeeded','failed') for r in fits)
              and type(groups) is list and tuple((g['day'],g['symbol']) for g in groups)==
                  tuple((day,symbol) for day in protocol.registered.TEST_DAYS for symbol in SYMBOLS))
        for group in groups:
            counts=group['counts']
            _need(type(counts['planned']) is int and counts['planned']==16
                  and all(type(v) is int and 0<=v<=16 for v in counts.values())
                  and counts['due']+counts['not_yet_due']==16
                  and counts['issued']+counts['unissued']==16
                  and counts['common']<=counts['settled']<=counts['issued']
                  and type(group['coverage_passed']) is bool
                  and group['coverage_passed']==(counts['common']*5>=64)
                  and set(group['families'])==set(FAMILIES)
                  and all(group['families'][f]['count']==counts['common'] for f in FAMILIES))
        _need(type(report['engineering_gate_passed']) is bool
              and report['engineering_gate_passed']==all(g['coverage_passed'] for g in groups)
              and set(report['macro'])==set(FAMILIES) and set(report['symbols'])==set(SYMBOLS))
        codec.encode(report)  # Strict finite JSON; infinity only via declared tag.
    except Exception:raise TestScoreError('test_report_projection_invalid') from None


def _current_quarantine(view):
    original=view['issue'];slots=protocol.planned_slots(original['registration'].plan());keys={s.key for s in slots}
    statuses={};quarantined=set()
    for session in original['registration'].plan()['calendar']['sessions'][8:]:
        day=session['day'];index=next(i for i,p in enumerate(original['proofs']) if p.start_day<=day<=p.end_day)
        pair=original['pairs'][index]
        with closing(sqlite3.connect(pair.raw_book.path.as_uri()+'?mode=ro',uri=True)) as db:
            values=db.execute("SELECT json_extract(payload,'$.body.session') FROM events "
                "WHERE json_extract(payload,'$.kind')='calendar' AND json_extract(payload,'$.body.day')=?",(day,)).fetchall()
        if not values:status='missing'
        elif any(v[0] is None or json.loads(v[0])!=session for v in values):status='quarantined'
        else:status='verified'
        statuses[day]=status
        if status=='quarantined':quarantined.update(s.key for s in slots if s.day==day)
    for pair in original['pairs']:
        with closing(sqlite3.connect(pair.feature_journal.path.as_uri()+'?mode=ro',uri=True)) as db:
            rows=db.execute("SELECT json_extract(payload,'$.body.key') FROM observations "
                "WHERE json_extract(payload,'$.kind')='quarantined'").fetchall()
        quarantined.update(key for (key,) in rows if key in keys)
    view['stable']();return statuses,quarantined


def _build_frames(view,*,as_of):
    tv=view['test_issue'];original=tv['state']['domain']['frames'];settled=view['settlement_state']['domain']['frames']
    slots=protocol.planned_slots(view['issue']['registration'].plan());expected={s.key:s for s in slots}
    _need(set(original)<=set(expected) and set(settled)<=set(original))
    reference=issuer.admission_reference(view['selection']['admission'])
    statuses,quarantined=_current_quarantine(view);frames=[]
    for key,frame in original.items():
        _need(frame['admission']==reference)
        previous._check_frame_sources(key,frame,view['issue']);slot=expected[key]
        entry=settled.get(key)
        if entry is not None:settlements.verify_record(key,entry,tv)
        confirmed=entry is not None and entry['status']=='confirmed'
        outcome=entry['outcome'] if confirmed else None
        missing=bool(entry['missing']) if entry is not None else False
        late=bool(entry['effective_late']) if confirmed else bool(entry is not None and as_of>slot.target_end+timedelta(seconds=120))
        frames.append(maths.ScoreFrame(key,slot.symbol,slot.event_at,slot.target_start,slot.target_end,
            previous._rows(frame,view['issue']['training']),frame['phase']=='confirmed',outcome is not None,
            outcome.label if outcome is not None else None,outcome.return_fraction if outcome is not None else None,
            missing,late,key in quarantined))
    for key in sorted(quarantined-set(original)):
        slot=expected[key]
        frames.append(maths.ScoreFrame(key,slot.symbol,slot.event_at,slot.target_start,slot.target_end,quarantined=True))
    view['stable']();return tuple(frames),statuses,quarantined


def _basis(view,statuses,quarantined):
    tv=view['test_issue'];original=dict(view['issue'],issue_journal=tv['book'],state=tv['state'])
    adapted=dict(view,issue=original)
    return dict(source_basis=previous._basis(adapted,statuses,quarantined),
        selection_book=previous._descriptor(view['selection']['selection_book'],view['selection']['selection_state']),
        selection_reference=issuer.admission_reference(view['selection']['admission']),protocol_sha256=protocol.source_sha256())


def verify_basis(basis,view):
    """Current physical/prefix/quarantine checks without new numerical prices."""
    _need(type(basis) is dict and set(basis)=={'source_basis','selection_book','selection_reference','protocol_sha256'}
          and basis['protocol_sha256']==protocol.source_sha256()
          and issuer._reference(basis['selection_reference'])==issuer.admission_reference(view['selection']['admission']))
    previous._check_descriptor(basis['selection_book'],view['selection']['selection_book'],view['selection']['selection_state'])
    source=basis['source_basis'];iv=view['issue'];tv=view['test_issue'];trained=iv['training']
    _need(type(source) is dict and set(source)=={'settlement_book','issue_book','source_proofs','quarantined_keys',
        'calendar_status','registration_identity','policy_seal','training_head_digest','training_fits_sha256',
        'training_snapshot_sha256','training_outcome'})
    previous._check_descriptor(source['settlement_book'],view['settlements'],view['settlement_state'])
    previous._check_descriptor(source['issue_book'],tv['book'],tv['state'])
    _need(source['registration_identity']==iv['registration'].identity and source['policy_seal']==view['policy'].seal
          and source['training_head_digest']==trained.head_digest and source['training_fits_sha256']==trained.fits.seal
          and source['training_snapshot_sha256']==trained.training.seal and len(source['source_proofs'])==2)
    old=source['training_outcome'];current=iv['training_outcome_proof']
    _need(all(old.get(k)==current.get(k) for k in ('kind','path','device','inode','plan_sha256','source_sha256',
        'consumed_head_seq','consumed_head_digest')) and old['head_seq']<=current['head_seq'])
    with closing(sqlite3.connect(Path(current['path']).as_uri()+'?mode=ro',uri=True)) as db:
        issuer.issuer._prefix(db,'observations',old['head_seq'],old['head_digest'])
    for proof,raw,pair in zip(iv['proofs'],source['source_proofs'],iv['pairs']):
        previous._source_prefix(proof,issuer.issuer._segment(raw),pair)
    _,quarantined=_current_quarantine(view)
    _need(sorted(quarantined)==source['quarantined_keys'])
    view['stable']()


@dataclass(frozen=True)
class FrozenTestReceipt:
    registration_identity:str
    selected_family:str
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
    test_only:bool
    promotion:bool
    orders:bool
    source_sha256:str
    seal:str

    def __post_init__(self):
        for name in ('evaluated_at','commit_observed_at','available_at'):
            object.__setattr__(self,name,aware(getattr(self,name)))
        policy=validation.decode_policy(self.policy_json);report=codec._raw(self.report_json)
        validate_report(report,self.selected_family)
        _need(is_sha(self.registration_identity) and is_sha(self.head_digest) and type(self.head_seq) is int and self.head_seq==4
              and Path(self.book_path).is_absolute() and type(self.book_device) is type(self.book_inode) is int
              and self.available_at>=self.commit_observed_at>self.evaluated_at>=aware(report['as_of'])>protocol.TEST_FINAL_DEADLINE
              and self.test_only is True and self.promotion is self.orders is False
              and policy.evaluation_source_sha256==previous.source_sha256()
              and policy.prospective_protocol_sha256==self.source_sha256==protocol.source_sha256()
              and self.seal==digest({k:v for k,v in asdict(self).items() if k!='seal'}))


class TestScoreJournal(books.EvaluationBook):
    def source_sha256(self):return protocol.source_sha256()
    def domain_initial(self):
        return dict(report=None,basis=None,evaluated_at=None,evaluated_digest=None,confirmed_at=None,confirmation_digest=None)

    def domain_apply(self,domain,kind,body,at,sha,binding):
        _need(at>protocol.TEST_FINAL_DEADLINE)
        if kind=='evaluated':
            _need(domain['report'] is None and set(body)=={'report','basis'})
            report=body['report'];basis=body['basis'];reference=issuer._reference(basis['selection_reference'])
            validate_report(report,reference['selected_family'])
            _need(aware(report['as_of'])<=at and basis['source_basis']['policy_seal']==binding['policy']['seal']
                  and basis['source_basis']['registration_identity']==binding['registration_identity']
                  and basis['protocol_sha256']==binding['policy']['prospective_protocol_sha256']==protocol.source_sha256())
            domain.update(report=report,basis=basis,evaluated_at=at,evaluated_digest=sha)
        elif kind=='confirmed':
            _need(domain['report'] is not None and domain['confirmed_at'] is None
                  and body==dict(evaluated_digest=domain['evaluated_digest']) and at>domain['evaluated_at'])
            domain.update(confirmed_at=at,confirmation_digest=sha)
        else:_need(False)

    def initialize(self,test_settlement_journal,*,timezone_file,clock):
        try:
            guard=validation.ValidationClock(timezone_file,clock,initial=True);guard()
            with settlements.settlement_view(test_settlement_journal,timezone_file=timezone_file,guard=guard,require_admission=False) as view:
                original=view['issue'];identity=original['registration'].identity
                _need(self.path==Path(original['binding']['canonical']['path']).parent/('r2-test-scores-'+identity+'.sqlite'))
                binding=dict(settlement_book=previous._descriptor(test_settlement_journal,view['settlement_state']),
                    policy=asdict(view['policy']),registration_identity=identity,timezone_file=str(Path(timezone_file).absolute()))
                return self.create(binding,clock=guard,check=view['stable'])
        except Exception:raise TestScoreError('test_score_initialization_refused') from None

    def _check_binding(self,binding,view):
        _need(binding['registration_identity']==view['issue']['registration'].identity and binding['policy']==asdict(view['policy']))
        previous._check_descriptor(binding['settlement_book'],view['settlements'],view['settlement_state'])

    def freeze_once(self,*,timezone_file,clock):
        try:
            guard=protocol.TestFreezeClock(timezone_file,clock);guard()  # Before SQL/locks.
            with _exclusive(self.path):
                state=self.snapshot(clock=guard);_need(state['initialized']);domain=state['domain'];binding=state['binding']
                source=settlements.TestSettlementJournal(binding['settlement_book']['path'])
                with settlements.settlement_view(source,timezone_file=timezone_file,guard=guard) as view:
                    self._check_binding(binding,view)
                    if domain['report'] is None:
                        as_of=guard();frames,statuses,quarantined=_build_frames(view,as_of=as_of)
                        fits=tuple((a.symbol,a.family,a.status) for a in view['issue']['training'].fits.attempts)
                        report=protocol.evaluate_test_frames(view['issue']['registration'].plan(),frames,as_of=as_of,
                            fit_statuses=fits,selected_family=view['selection']['admission'].selected_family)
                        basis=_basis(view,statuses,quarantined);verify_basis(basis,view)
                        state=self.append(state,'evaluated',dict(report=report,basis=basis),clock=guard,check=view['stable'])
                        domain=state['domain']
                    verify_basis(domain['basis'],view)
                    if domain['confirmed_at'] is not None:return state
                    return self.append(state,'confirmed',dict(evaluated_digest=domain['evaluated_digest']),
                        clock=guard,check=lambda:verify_basis(domain['basis'],view))
        except Exception:raise TestScoreError('test_comparison_freeze_refused') from None

    def load_frozen(self,*,timezone_file,clock):
        try:
            guard=protocol.TestFreezeClock(timezone_file,clock);started=guard()  # Before SQL.
            with issuer.issuer._verified_epoch((self.path,),guard) as own_stable:
                state=self.snapshot(clock=guard);_need(state['initialized']);domain=state['domain'];binding=state['binding']
                _need(domain['confirmed_at'] is not None)
                source=settlements.TestSettlementJournal(binding['settlement_book']['path'])
                with settlements.settlement_view(source,timezone_file=timezone_file,guard=guard) as view:
                    self._check_binding(binding,view);verify_basis(domain['basis'],view);own_stable()
                    selected=view['selection']['admission'].selected_family;validate_report(domain['report'],selected)
                    available=guard();_need(available>=started>=domain['confirmed_at']);identity=self._identity()
                    fields=dict(registration_identity=binding['registration_identity'],selected_family=selected,
                        policy_json=codec.encode(binding['policy']),report_json=codec.encode(domain['report']),basis_json=codec.encode(domain['basis']),
                        book_path=str(self.path),book_device=identity[0],book_inode=identity[1],head_seq=state['head_seq'],head_digest=state['head_digest'],
                        evaluated_at=domain['evaluated_at'],commit_observed_at=domain['confirmed_at'],available_at=available,
                        test_only=True,promotion=False,orders=False,source_sha256=protocol.source_sha256())
                    receipt=FrozenTestReceipt(**fields,seal=digest(fields));view['stable']();own_stable()
                own_stable();guard()
            return receipt
        except Exception:raise TestScoreError('current_test_report_unavailable') from None
