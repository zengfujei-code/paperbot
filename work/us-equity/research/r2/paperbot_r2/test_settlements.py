"""First durable test outcomes, requiring current original test publications.

Historical results never grant current scoring or order permission. Recovery
keeps the first stored label and acknowledges its COMMIT without new prices.
"""
from contextlib import contextmanager
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

from paperbot_us.runner import _exclusive
from . import evaluation_book as books,test_protocol as protocol,test_issue_journal as issuer
from . import validation_settlements as previous,validation_scores as scores
from . import common_outcome,common_outcome_journal as outcomes,training_codec as codec
from .contracts import digest


class TestSettlementError(ValueError):pass


def _need(value):
    if not value:raise TestSettlementError('test_settlement_unavailable')


def witness_from_issued(key,receipt):
    """Physical identities were captured inside the current issue-book epoch."""
    _need(type(receipt) is issuer.TestIssuedReceipt);receipt.__post_init__()
    event=protocol._test_key(key)
    _need(key==receipt.issue_input.symbol+'|'+receipt.issue_input.bundle.vector.event_at.isoformat())
    fields=dict(key=key,symbol=receipt.issue_input.symbol,input_sha256=receipt.issue_input.seal,
        panel_sha256=receipt.panel.seal,issue_book_path=receipt.book_path,
        issue_book_device=receipt.book_device,issue_book_inode=receipt.book_inode,
        issue_book_head_seq=receipt.head_seq,issue_book_head_digest=receipt.head_digest,
        frame_confirmation_digest=receipt.confirmation_digest,issued_at=receipt.issued_at,
        issue_commit_observed_at=receipt.commit_observed_at,available_at=receipt.available_at,
        event_at=event,target_start=receipt.panel.target_start,target_end=receipt.panel.target_end)
    return previous.ForecastWitness(**fields,seal=digest(fields))


@contextmanager
def settlement_view(book,*,timezone_file,guard,require_admission=True):
    """Hold this book and every original upstream source throughout permission."""
    at=guard()
    _need(type(require_admission) is bool and (require_admission or at<previous.registered.FIRST_OPEN))
    _need(type(book) is TestSettlementJournal)
    with issuer.issuer._verified_epoch((book.path,),guard) as own_stable:
        state=book.snapshot(clock=guard);_need(state['initialized']);binding=state['binding']
        source=issuer.TestIssueJournal(binding['issue_book']['path'])
        with issuer.issue_view(source,timezone_file=timezone_file,guard=guard,require_admission=require_admission) as view:
            book._check_binding(binding,view)
            def stable():own_stable();view['stable']()
            stable()
            yield dict(settlements=book,settlement_state=state,test_issue=view,issue=view['issue'],
                       selection=view['selection'],policy=previous.decode_policy(binding['policy']),stable=stable)
            stable();guard()


def verify_record(key,entry,view):
    """Reuse source/prefix verification, never the validation-only phase gate."""
    frame=view['state']['domain']['frames'].get(key)
    _need(frame is not None and frame['phase']=='confirmed')
    # The generic verifier checks the exact input, first panel, confirmation,
    # physical issue book, original raw/feature ancestors and feature bundle.
    # Its phase-neutral checks receive the actual TEST book and watched state.
    original=dict(view['issue'],issue_journal=view['book'],state=view['state'])
    scores._verify_settlement(key,entry,frame,original)
    view['stable']()


class TestSettlementJournal(books.EvaluationBook):
    def source_sha256(self):return protocol.source_sha256()
    def domain_initial(self):return dict(frames={})

    def domain_apply(self,domain,kind,body,at,sha,binding):
        frames=domain['frames'];key=body.get('key');protocol._test_key(key)
        if kind=='observed':
            _need(set(body)=={'key','witness','outcome','missing'} and key not in frames)
            witness=previous.decode_witness(body['witness'])
            _need(key==witness.key and at>=witness.available_at and type(body['missing']) is bool)
            bound=binding['issue_binding']['validation_issue_binding']
            slots={s.key:s for s in protocol.planned_slots(issuer.inputs.alpaca.decode_json(bound['plan_json']))}
            _need(key in slots and witness.target_end==slots[key].target_end and at>=witness.target_end)
            if body['outcome'] is None:
                _need(body['missing'] and at>witness.target_end+timedelta(seconds=120));result=None
            else:
                result=outcomes._outcome(body['outcome'])
                _need(result.feature_key==key and result.symbol==witness.symbol
                      and result.target_start==witness.target_start and result.target_end==witness.target_end
                      and witness.available_at<=result.observed_at<=at
                      and result.example.features.event_at==witness.event_at and body['missing']==result.missing)
            frames[key]=dict(status='observed',witness=witness,outcome=result,missing=body['missing'],observed_at=at,
                observed_digest=sha,commit_observed_at=None,confirmation_digest=None,effective_late=None,
                current_scoring_permission=False)
        elif kind=='confirmed':
            _need(set(body)=={'key','observed_digest'} and key in frames)
            entry=frames[key]
            _need(entry['status']=='observed' and body['observed_digest']==entry['observed_digest'] and at>entry['observed_at'])
            late=entry['outcome'] is None or entry['outcome'].late or at>entry['witness'].target_end+timedelta(seconds=120)
            entry.update(status='confirmed',commit_observed_at=at,confirmation_digest=sha,effective_late=bool(late))
        else:_need(False)
        _need(len(frames)<=240)

    def initialize(self,test_issue_journal,*,timezone_file,clock):
        try:
            guard=previous.ValidationClock(timezone_file,clock,initial=True);guard()
            with issuer.issue_view(test_issue_journal,timezone_file=timezone_file,guard=guard,require_admission=False) as view:
                original=view['issue'];identity=original['registration'].identity
                expected=Path(original['binding']['canonical']['path']).parent/('r2-test-settlements-'+identity+'.sqlite')
                _need(self.path==expected)
                binding=dict(issue_book=scores._descriptor(test_issue_journal,view['state']),issue_binding=view['binding'],
                    policy=view['binding']['policy'],timezone_file=str(Path(timezone_file).absolute()))
                return self.create(binding,clock=guard,check=view['stable'])
        except Exception:raise TestSettlementError('test_settlement_initialization_refused') from None

    def _check_binding(self,binding,view):
        _need(binding['issue_binding']==view['binding'] and binding['policy']==view['binding']['policy'])
        scores._check_descriptor(binding['issue_book'],view['book'],view['state'])

    def settle_once(self,key,*,timezone_file,clock):
        try:
            guard=protocol.TestOutcomeClock(key,timezone_file,clock);guard()  # Before any SQL or lock.
            with _exclusive(self.path):
                state=self.snapshot(clock=guard);_need(state['initialized']);binding=state['binding']
                entry=state['domain']['frames'].get(key)
                source=issuer.TestIssueJournal(binding['issue_book']['path'])
                with issuer.issue_view(source,timezone_file=timezone_file,guard=guard) as view:
                    self._check_binding(binding,view)
                    receipt=source.load_issued(key,timezone_file=timezone_file,clock=guard)
                    if entry is None:
                        witness=witness_from_issued(key,receipt)
                        day=witness.event_at.astimezone(guard.base.zone).date().isoformat();original=view['issue']
                        index=next(i for i,p in enumerate(original['proofs']) if p.start_day<=day<=p.end_day)
                        pair=original['pairs'][index]
                        present=previous._first_pair_present(pair,witness);view['stable']()
                        if not present and guard()<=witness.target_end+timedelta(seconds=120):
                            return dict(status='waiting',key=key,available_at=guard().isoformat(),current_scoring_permission=False)
                        result=common_outcome.observe_common_outcome(pair.feature_journal,pair.raw_book,key,
                            clock=guard,read_guard=guard) if present else None
                        if result is not None:
                            _need(result.feature_bundle_sha256==receipt.issue_input.bundle.seal
                                  and result.example.features==receipt.issue_input.bundle.vector)
                        view['stable']()
                        body=dict(key=key,witness=asdict(witness),outcome=asdict(result) if result is not None else None,
                            missing=result.missing if result is not None else True)
                        state=self.append(state,'observed',body,clock=guard,check=view['stable'])
                        entry=state['domain']['frames'][key]
                    verify_record(key,entry,view)
                    # Even historical confirmed returns require CURRENT original
                    # selection and source integrity; no new label is observed.
                    if entry['status']=='confirmed':return entry
                    _need(entry['status']=='observed')
                    state=self.append(state,'confirmed',dict(key=key,observed_digest=entry['observed_digest']),
                        clock=guard,check=lambda:verify_record(key,entry,view))
                    return state['domain']['frames'][key]
        except Exception:raise TestSettlementError('test_settlement_refused') from None
