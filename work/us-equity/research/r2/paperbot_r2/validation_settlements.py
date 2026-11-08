"""First durable validation outcomes against confirmed prospective issues.

Stored results are immutable historical evidence, not current score or phase
permission. A separate report loader rechecks all current lineage/quarantine.
"""
from contextlib import closing,contextmanager
from dataclasses import asdict,dataclass
from datetime import datetime,timedelta,timezone
import hashlib
from pathlib import Path
import sqlite3

from paperbot_us.runner import _exclusive
from . import common_scores as scores,evaluation_book as books,validation_issue_journal as issuer
from . import common_registration as registered,common_outcome,common_outcome_journal as outcomes
from . import canonical_history as canonical,training_journal,training_codec as codec
from .contracts import aware,digest,is_sha

TEST_OPEN=aware('2026-12-07T14:30:00+00:00')


class SettlementError(ValueError):pass


def _need(value):
    if not value:raise SettlementError('validation_settlement_unavailable')


def source_sha256():
    return digest(dict(protocol=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        storage=books.source_sha256(),scores=scores.source_sha256(),issuer=issuer.source_sha256(),outcome=outcomes._source()))


class ValidationClock:
    def __init__(self,timezone_file,clock,*,key=None,initial=False,allow_after_test=False):
        self.base=canonical.Clock(timezone_file,clock);self.key=key;self.initial=initial;self.allow_after_test=allow_after_test
        self.event_at=self.target_end=None
        if key is not None:
            symbol,raw=key.split('|',1);self.event_at=aware(raw)
            _need(symbol in scores.SYMBOLS and key==symbol+'|'+self.event_at.isoformat()
                  and self.event_at.astimezone(self.base.zone).date().isoformat() in registered.VALIDATION_DAYS)
            self.target_end=self.event_at+timedelta(seconds=1800)

    def __call__(self):
        at=self.base()
        _need((self.allow_after_test or at<TEST_OPEN) and (not self.initial or at<registered.FIRST_OPEN)
              and (self.target_end is None or at>=self.target_end))
        return at


def decode_policy(raw):return scores.ScorePolicy(**codec._raw(raw))


def source_pairs(binding):
    return tuple(issuer.inputs.SourcePair(issuer.inputs.receipt_book.ReceiptBook(p['raw_path']),
        issuer.inputs.feature_pipeline.FeatureJournal(p['feature_path'])) for p in binding['segments'])


def _verify_training_outcomes(bound,trained,guard):
    """Authenticate original label provenance without decoding later labels."""
    path=Path(bound.path);identity=issuer.inputs._identity(path)
    _need(path.is_absolute() and identity[:2]==(bound.device,bound.inode) and identity[2]<=268435456
          and bound.source_sha256==outcomes._source())
    now=guard()
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,isolation_level=None)) as db:
        db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
        rows=db.execute('SELECT version,started_at,head_seq,head_digest FROM metadata').fetchall()
        _need(len(rows)==1 and rows[0][0]==1 and type(rows[0][2]) is int and bound.head_seq<=rows[0][2]<=20001)
        latest=db.execute("SELECT MAX(json_extract(payload,'$.generated_at')) FROM observations").fetchone()[0]
        _need(type(latest) in (float,int) and latest<=now.timestamp())
        forbidden=db.execute("SELECT COUNT(*) FROM observations WHERE "
            "substr(json_extract(payload,'$.body.outcome.example.features.event_at'),1,10) BETWEEN '2026-11-10' AND '2026-11-23' "
            "OR substr(json_extract(payload,'$.body.outcome.target_start'),1,10) BETWEEN '2026-11-10' AND '2026-11-23'").fetchone()[0]
        _need(forbidden==0)
        created=issuer.inputs.alpaca.decode_json(db.execute('SELECT payload FROM observations WHERE seq=1').fetchone()[0])
        _need(created['kind']=='created' and created['source_sha256']==bound.source_sha256
              and created['generated_at']==created['observing_started_at']==rows[0][1]
              and digest(created['body']['binding'])==bound.plan_sha256)
        binding=created['body']['binding']
        for kind in ('raw','feature'):
            original=next(b for b in trained.training.bindings if b.kind==kind)
            _need((binding[kind+'_path'],binding[kind+'_device'],binding[kind+'_inode'],binding[kind+'_plan_sha256'],binding[kind+'_source_sha256'])==
                  (original.path,original.device,original.inode,original.plan_sha256,original.source_sha256))
        _,count,head=outcomes.ObservationArchive(path)._verify_chain(db)
        _need((count,head)==(rows[0][2],rows[0][3]))
        issuer._prefix(db,'observations',bound.head_seq,bound.head_digest)
        db.rollback()
    guard()
    return dict(kind='outcome',path=str(path),device=identity[0],inode=identity[1],
        plan_sha256=bound.plan_sha256,source_sha256=bound.source_sha256,head_seq=count,head_digest=head,
        consumed_head_seq=bound.head_seq,consumed_head_digest=bound.head_digest,
        last_at=datetime.fromtimestamp(latest,timezone.utc))


@contextmanager
def issue_view(issue_journal,policy,*,timezone_file,guard):
    """One entire-read epoch, with metadata-first vetoes and no price decoding."""
    guard();_need(type(issue_journal) is issuer.ValidationIssueJournal and type(policy) is scores.ScorePolicy)
    policy.__post_init__()
    from . import validation_scores
    _need(policy.evaluation_source_sha256==validation_scores.source_sha256())
    with issuer._verified_epoch((issue_journal.path,),guard) as issue_stable:
        issue_journal._before_read(guard());state=issue_journal.snapshot();binding=state['binding']
        _need(guard()>=state['last_at'] and binding['validation_only'] is True)
        pairs=source_pairs(binding)
        registry=canonical.CanonicalRegistry(binding['canonical']['path'])
        training=training_journal.TrainingJournal(binding['training']['path'])
        paths=tuple(p for pair in pairs for p in (pair.raw_book.path,pair.feature_journal.path))+(registry.path,training.path)
        with issuer._verified_epoch(paths,guard) as source_stable:
            receipt,trained=issuer._current(binding,registry,training,timezone_file=timezone_file,guard=guard)
            issue_stable();source_stable()
            original_outcomes=tuple(b for b in trained.training.bindings if b.kind=='outcome')
            _need(len(original_outcomes)==1)
            # Discovery reads the already watched, fully verified training book.
            # No original-outcome payload is inspected before this extra watch.
            with issuer._verified_epoch((Path(original_outcomes[0].path),),guard) as outcome_stable:
                def stable():issue_stable();source_stable();outcome_stable()
                _need(receipt.plan()['configuration_sha256']==policy.seal)
                slots=scores.planned_slots(receipt.plan())
                outcome_proof=_verify_training_outcomes(original_outcomes[0],trained,guard)
                proofs=issuer._sources(binding,pairs,trained,guard);stable()
                yield dict(issue_journal=issue_journal,state=state,binding=binding,pairs=pairs,registry=registry,training_book=training,
                    registration=receipt,training=trained,training_outcome_proof=outcome_proof,proofs=proofs,slots=slots,stable=stable)
                stable();guard()


def _issue_descriptor(issue_journal,state):
    identity=issuer.inputs._identity(issue_journal.path)
    return dict(path=str(issue_journal.path),device=identity[0],inode=identity[1],head_seq=state['head_seq'],head_digest=state['head_digest'])


def check_issue_descriptor(descriptor,view):
    issue_journal=view['issue_journal'];identity=issuer.inputs._identity(issue_journal.path)
    _need(descriptor['path']==str(issue_journal.path) and (descriptor['device'],descriptor['inode'])==identity[:2]
          and descriptor['head_seq']<=view['state']['head_seq'])
    with closing(sqlite3.connect(issue_journal.path.as_uri()+'?mode=ro',uri=True)) as db:
        issuer._prefix(db,'events',descriptor['head_seq'],descriptor['head_digest'])
    view['stable']()


@dataclass(frozen=True)
class ForecastWitness:
    key:str
    symbol:str
    input_sha256:str
    panel_sha256:str
    issue_book_path:str
    issue_book_device:int
    issue_book_inode:int
    issue_book_head_seq:int
    issue_book_head_digest:str
    frame_confirmation_digest:str
    issued_at:datetime
    issue_commit_observed_at:datetime
    available_at:datetime
    event_at:datetime
    target_start:datetime
    target_end:datetime
    seal:str

    def __post_init__(self):
        for name in ('issued_at','issue_commit_observed_at','available_at','event_at','target_start','target_end'):
            object.__setattr__(self,name,aware(getattr(self,name)))
        _need(self.symbol in scores.SYMBOLS and self.key==self.symbol+'|'+self.event_at.isoformat()
            and self.target_start==self.event_at+timedelta(seconds=900) and self.target_end==self.target_start+timedelta(seconds=900)
            and self.event_at<=self.issued_at<self.issue_commit_observed_at<=self.event_at+timedelta(seconds=120)
            and self.issue_commit_observed_at<self.target_start and self.available_at>=self.issue_commit_observed_at
            and Path(self.issue_book_path).is_absolute() and type(self.issue_book_head_seq) is int and self.issue_book_head_seq>=5
            and all(type(v) is int and v>=0 for v in (self.issue_book_device,self.issue_book_inode))
            and all(is_sha(v) for v in (self.input_sha256,self.panel_sha256,self.issue_book_head_digest,self.frame_confirmation_digest,self.seal))
            and self.seal==digest({k:v for k,v in asdict(self).items() if k!='seal'}))


def witness_from_issued(key,receipt):
    _need(type(receipt) is issuer.IssuedReceipt);receipt.__post_init__()
    identity=issuer.inputs._identity(Path(receipt.book_path))
    fields=dict(key=key,symbol=receipt.issue_input.symbol,input_sha256=receipt.issue_input.seal,panel_sha256=receipt.panel.seal,
        issue_book_path=receipt.book_path,issue_book_device=identity[0],issue_book_inode=identity[1],
        issue_book_head_seq=receipt.head_seq,issue_book_head_digest=receipt.head_digest,
        frame_confirmation_digest=receipt.confirmation_digest,issued_at=receipt.issued_at,
        issue_commit_observed_at=receipt.commit_observed_at,available_at=receipt.available_at,
        event_at=receipt.issue_input.bundle.vector.event_at,target_start=receipt.panel.target_start,target_end=receipt.panel.target_end)
    return ForecastWitness(**fields,seal=digest(fields))


def decode_witness(value):return ForecastWitness(**codec._raw(value))


def _first_pair_present(pair,witness):
    with closing(sqlite3.connect(pair.raw_book.path.as_uri()+'?mode=ro',uri=True)) as db:
        for start in (witness.event_at,witness.target_start):
            found=db.execute("SELECT COUNT(*) FROM events WHERE json_extract(payload,'$.kind')='receipt' "
                "AND json_extract(payload,'$.body.bar.symbol')=? AND json_extract(payload,'$.body.bar.starts')=?",
                (witness.symbol,start.isoformat())).fetchone()[0]
            _need(found in (0,1))
            if not found:return False
    return True


class ValidationSettlementJournal(books.EvaluationBook):
    def source_sha256(self):return source_sha256()
    def domain_initial(self):return dict(frames={})

    def domain_apply(self,domain,kind,body,at,sha,binding):
        frames=domain['frames']
        if kind=='observed':
            _need(set(body)=={'key','witness','outcome','missing'} and body['key'] not in frames)
            witness=decode_witness(body['witness']);key=body['key']
            _need(key==witness.key and at>=witness.available_at and type(body['missing']) is bool)
            slots={s.key:s for s in scores.planned_slots(issuer.inputs.alpaca.decode_json(binding['issue_binding']['plan_json']))}
            _need(key in slots and witness.target_end==slots[key].target_end and at>=witness.target_end)
            if body['outcome'] is None:
                _need(body['missing'] and at>witness.target_end+timedelta(seconds=120));result=None
            else:
                result=outcomes._outcome(body['outcome'])
                _need(result.feature_key==key and result.symbol==witness.symbol and result.target_start==witness.target_start
                      and result.target_end==witness.target_end and result.observed_at<=at
                      and result.example.features.event_at==witness.event_at
                      and body['missing']==result.missing)
            frames[key]=dict(status='observed',witness=witness,outcome=result,missing=body['missing'],observed_at=at,
                observed_digest=sha,commit_observed_at=None,confirmation_digest=None,effective_late=None,current_scoring_permission=False)
        elif kind=='confirmed':
            _need(set(body)=={'key','observed_digest'} and body['key'] in frames)
            entry=frames[body['key']]
            _need(entry['status']=='observed' and body['observed_digest']==entry['observed_digest'] and at>entry['observed_at'])
            late=entry['outcome'] is None or entry['outcome'].late or at>entry['witness'].target_end+timedelta(seconds=120)
            entry.update(status='confirmed',commit_observed_at=at,confirmation_digest=sha,effective_late=bool(late))
        else:_need(False)
        _need(len(frames)<=348)

    def initialize(self,issue_journal,policy,*,timezone_file,clock):
        try:
            guard=ValidationClock(timezone_file,clock,initial=True);guard()
            with issue_view(issue_journal,policy,timezone_file=timezone_file,guard=guard) as view:
                expected=Path(view['binding']['canonical']['path']).parent/('r2-validation-settlements-'+view['registration'].identity+'.sqlite')
                _need(self.path==expected)
                binding=dict(issue_book=_issue_descriptor(issue_journal,view['state']),issue_binding=view['binding'],
                    policy=asdict(policy),timezone_file=str(Path(timezone_file).absolute()))
                return self.create(binding,clock=guard,check=view['stable'])
        except Exception:raise SettlementError('validation_settlement_initialization_failed') from None

    def settle_once(self,key,*,timezone_file,clock):
        try:
            guard=ValidationClock(timezone_file,clock,key=key);guard()  # Before any SQL.
            with _exclusive(self.path):
                state=self.snapshot(clock=guard);_need(state['initialized'])
                entry=state['domain']['frames'].get(key)
                if entry is not None and entry['status']=='confirmed':return entry
                binding=state['binding'];policy=decode_policy(binding['policy'])
                issue_journal=issuer.ValidationIssueJournal(binding['issue_book']['path'])
                with issue_view(issue_journal,policy,timezone_file=timezone_file,guard=guard) as view:
                    _need(view['binding']==binding['issue_binding'])
                    check_issue_descriptor(binding['issue_book'],view)
                    if entry is None:
                        receipt=issue_journal.load_issued(key,clock=guard)
                        witness=witness_from_issued(key,receipt)
                        day=witness.event_at.astimezone(guard.base.zone).date().isoformat()
                        index=next(i for i,p in enumerate(view['proofs']) if p.start_day<=day<=p.end_day)
                        pair=view['pairs'][index]
                        present=_first_pair_present(pair,witness);view['stable']()
                        if not present and guard()<=witness.target_end+timedelta(seconds=120):
                            return dict(status='waiting',key=key,available_at=guard().isoformat(),current_scoring_permission=False)
                        result=common_outcome.observe_common_outcome(pair.feature_journal,pair.raw_book,key,clock=guard,
                            read_guard=guard) if present else None
                        view['stable']()
                        body=dict(key=key,witness=asdict(witness),outcome=asdict(result) if result is not None else None,
                            missing=result.missing if result is not None else True)
                        state=self.append(state,'observed',body,clock=guard,check=view['stable'])
                        entry=state['domain']['frames'][key]
                    _need(entry['status']=='observed')
                    # Recovered results are never recalculated from later market values.
                    receipt=issue_journal.load_issued(key,clock=guard)
                    _need(receipt.issue_input.seal==entry['witness'].input_sha256 and receipt.panel.seal==entry['witness'].panel_sha256)
                    state=self.append(state,'confirmed',dict(key=key,observed_digest=entry['observed_digest']),
                        clock=guard,check=view['stable'])
                    return state['domain']['frames'][key]
        except Exception:raise SettlementError('validation_settlement_refused') from None
