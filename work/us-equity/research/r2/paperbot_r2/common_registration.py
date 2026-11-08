"""Fixed common-comparison reservations inheriting canonical history and trials.

Reservation evidence is not phase admission, a forecast, a score or order
authority. Training history includes interrupted/unconfirmed consumed batches.
"""
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path

from paperbot_us import data
from . import canonical_history as canonical, model_training as models, training_journal as training_book
from .training_data import TRAINING_DAYS, SYMBOLS, TIMEZONE_SHA256
from .contracts import aware, digest, is_sha

VALIDATION_DAYS = ('2026-11-24','2026-11-25','2026-11-27','2026-11-30',
                   '2026-12-01','2026-12-02','2026-12-03','2026-12-04')
TEST_DAYS = ('2026-12-07','2026-12-08','2026-12-09','2026-12-10','2026-12-11')
PAIRS = tuple((s, f) for s in SYMBOLS for f in models.FAMILIES)
PROPOSAL = digest(dict(training=TRAINING_DAYS,validation=VALIDATION_DAYS,test=TEST_DAYS,
                       pairs=PAIRS,event='abs(target_first_close/gap_first_close-1)>=.005',gap=900,horizon=900))
FIRST_OPEN = aware('2026-11-24T14:30:00+00:00')
TRAINING_CUTOFF = aware('2026-11-10T04:55:00+00:00')


class CommonRegistrationError(ValueError):
    pass


def _need(value):
    if not value:
        raise CommonRegistrationError('common_registration_unavailable')


def _source():
    return digest(dict(protocol=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       training=training_book.source_sha256()))


def _key(record):
    return digest(record['book'])


def _project(journal, clock):
    _need(type(journal) is training_book.TrainingJournal)
    state = journal.snapshot(); stamp = aware(clock())
    _need(state['reserved_trial_units'] == 15 and state['training'] is not None
          and stamp >= state['last_at'] and state['last_at'] < TRAINING_CUTOFF)
    identity = journal._identity()
    if state['phase'] == 'acknowledged':
        receipt = journal.load_acknowledged(clock=clock)
        _need(receipt.head_digest == state['head_digest'])
    values = []
    symbols = {s.symbol:s for s in state['training'].symbols}
    for i, (symbol, family) in enumerate(PAIRS):
        source = symbols[symbol]
        rows = source.examples if family in ('prevalence','gaussian_nb') else source.returns
        attempt = state['fits'].attempts[i] if state['fits'] is not None else None
        values.append(dict(symbol=symbol,family=family,fit_status=attempt.status if attempt else 'unknown',
            failure_code=attempt.failure_code if attempt else 'interrupted_or_uncommitted_fit',
            model_sha256=digest(asdict(attempt.payload)) if attempt and attempt.payload is not None else None,
            attempt_sha256=attempt.seal if attempt else None,input_count=len(rows),
            input_rows_sha256=digest([asdict(r) for r in rows])))
    _need(journal._identity() == identity)
    return dict(book=dict(path=str(journal.path),device=identity[0],inode=identity[1],
                          reservation_digest=state['reserved_digest']),
        journal_head_seq=state['head_seq'],journal_head_digest=state['head_digest'],
        journal_source_sha256=training_book.source_sha256(),phase=state['phase'],
        proposal_id=PROPOSAL,training_snapshot_sha256=state['training'].seal,
        fits_sha256=state['fits'].seal if state['fits'] is not None else None,
        train_days=list(TRAINING_DAYS),trial_units=15,candidates=values,
        last_at=state['last_at'].isoformat(),
        ack_commit_observed_at=state['commit_observed_at'].isoformat() if state['commit_observed_at'] else None)


def _validate_training(record):
    _need(type(record) is dict and set(record) == {'book','journal_head_seq','journal_head_digest',
        'journal_source_sha256','phase','proposal_id','training_snapshot_sha256','fits_sha256',
        'train_days','trial_units','candidates','last_at','ack_commit_observed_at'})
    book = record['book']
    _need(type(book) is dict and set(book)=={'path','device','inode','reservation_digest'}
          and Path(book['path']).is_absolute() and is_sha(book['reservation_digest'])
          and all(type(book[k]) is int and book[k]>=0 for k in ('device','inode'))
          and record['proposal_id']==PROPOSAL and record['train_days']==list(TRAINING_DAYS)
          and record['trial_units']==15 and type(record['journal_head_seq']) is int
          and 2<=record['journal_head_seq']<=5 and is_sha(record['journal_head_digest'])
          and record['journal_source_sha256']==training_book.source_sha256()
          and is_sha(record['training_snapshot_sha256'])
          and (record['fits_sha256'] is None or is_sha(record['fits_sha256']))
          and record['phase'] in ('reserved','abandoned','result_pending_ack','ack_pending_confirmation','acknowledged')
          and aware(record['last_at'])<TRAINING_CUTOFF)
    _need(type(record['candidates']) is list and len(record['candidates'])==15)
    for row, pair in zip(record['candidates'],PAIRS):
        _need(type(row) is dict and set(row)=={'symbol','family','fit_status','failure_code','model_sha256',
            'attempt_sha256','input_count','input_rows_sha256'} and (row['symbol'],row['family'])==pair
            and row['fit_status'] in ('unknown','succeeded','failed') and is_sha(row['input_rows_sha256'])
            and type(row['input_count']) is int and 0<=row['input_count']<=2000
            and all(row[k] is None or is_sha(row[k]) for k in ('model_sha256','attempt_sha256'))
            and row['failure_code'] in models.SAFE_FAILURES | {None,'fixed_fit_failed','interrupted_or_uncommitted_fit'})
    if record['phase']=='acknowledged':
        _need(record['journal_head_seq']==5 and record['fits_sha256'] is not None
              and record['ack_commit_observed_at'] is not None
              and aware(record['ack_commit_observed_at'])<TRAINING_CUTOFF)


def _original(record, clock):
    _validate_training(record)
    _need(_project(training_book.TrainingJournal(record['book']['path']),clock)==record)


def _heldout(plan):
    return set(plan['split']['validation']+plan['split']['test'])


def invalidate_exposure(state, domain, days):
    for item in state['reservations'].values():
        if item['plan']['domain']==domain and _heldout(item['plan']) & set(days):
            item['invalidated']=True


def _calendar(value):
    _need(type(value) is data.Calendar)
    sessions = []
    for day in VALIDATION_DAYS+TEST_DAYS:
        session = value.sessions.get(datetime.fromisoformat(day).date())
        _need(session is not None)
        opens = day+'T14:30:00+00:00'
        closes = day+('T18:00:00+00:00' if day=='2026-11-27' else 'T21:00:00+00:00')
        _need(session.opens==aware(opens) and session.closes==aware(closes))
        sessions.append(dict(day=day,opens=opens,closes=closes))
    return sessions


def build_common_plan(training_import, calendar, *, calendar_received_at, issue_source_sha256, configuration_sha256):
    try:
        _validate_training(training_import)
        _need(is_sha(issue_source_sha256) and is_sha(configuration_sha256))
        sessions = _calendar(calendar)
        return dict(schema=2,proposal_id=PROPOSAL,domain='us_equity',family='comparison',
            objective='brier_macro_min',selection_stage='validation',trial_budget=15,
            training_import_id=_key(training_import),data_sha256=training_import['training_snapshot_sha256'],
            code_sha256=issue_source_sha256,configuration_sha256=configuration_sha256,
            candidate_ids=[digest(dict(training=_key(training_import),symbol=s,family=f,
                                      configuration_sha256=configuration_sha256)) for s,f in PAIRS],
            pairs=[list(p) for p in PAIRS],symbols=list(SYMBOLS),
            split=dict(train=list(TRAINING_DAYS),validation=list(VALIDATION_DAYS),test=list(TEST_DAYS),context=[]),
            event=dict(formula='abs(target_first_close/gap_first_close-1)>=threshold',threshold_fraction=.005,
                       gap_seconds=900,horizon_seconds=900,feed='iex',adjustment='raw',currency='USD',volume_unit='shares'),
            calendar=dict(sessions=sessions,sha256=digest(sessions),
                          received_at=aware(calendar_received_at).isoformat(),timezone_sha256=TIMEZONE_SHA256))
    except Exception:
        raise CommonRegistrationError('common_plan_invalid') from None


def _validate_plan(state, plan):
    record = state['training_imports'].get(plan['training_import_id']); _need(record is not None)
    value = plan['calendar']; _need(type(value) is dict and set(value)=={'sessions','sha256','received_at','timezone_sha256'})
    calendar = data.Calendar([data.Session(datetime.fromisoformat(s['day']).date(),s['opens'],s['closes'])
                              for s in value['sessions']],start=VALIDATION_DAYS[0],end=TEST_DAYS[-1])
    expected = build_common_plan(record,calendar,calendar_received_at=value['received_at'],
        issue_source_sha256=plan['code_sha256'],configuration_sha256=plan['configuration_sha256'])
    _need(plan==expected and record['phase']=='acknowledged'
          and len([v for v in state['training_imports'].values() if v['proposal_id']==PROPOSAL])==1)
    return record


def _all_used(state, domain):
    days = set(canonical._used(state['legacy_state'])[domain])
    for value in state['new_exposures'].values():
        if value['domain']==domain:days.update(value['days'])
    if domain=='us_equity':
        for value in state['training_imports'].values():days.update(value['train_days'])
    for item in state['reservations'].values():
        if item['plan']['domain']==domain:
            for values in item['plan']['split'].values():days.update(values)
    return days


def apply_event(state, kind, body, at, sha):
    if kind=='training_imported':
        _need(set(body)=={'record'}); record=body['record']; _validate_training(record)
        _need(aware(record['last_at'])<=at)
        key=_key(record)
        if key in state['training_imports']:
            _need(state['training_imports'][key]==record); return False
        _need(len(state['training_imports'])<128)
        state['training_imports'][key]=record
        if len([v for v in state['training_imports'].values() if v['proposal_id']==PROPOSAL])>1:
            for item in state['reservations'].values():
                if item['plan']['proposal_id']==PROPOSAL:item['invalidated']=True
        return True
    if kind=='common_registered':
        _need(set(body)=={'plan'}); plan=body['plan']; identity=digest(plan)
        _validate_plan(state,plan)
        existing=state['reservations'].get(identity)
        if existing is not None:
            _need(existing['plan']==plan and not existing['invalidated']); return False
        _need(at<FIRST_OPEN and not (_all_used(state,plan['domain']) & _heldout(plan)))
        received=aware(plan['calendar']['received_at'])
        _need(received<=at and (at-received).total_seconds()<=86400)
        state['reservations'][identity]=dict(plan=plan,registered_at=at,registration_digest=sha,
            confirmed_at=None,confirmation_digest=None,invalidated=False)
        return True
    if kind=='common_confirmed':
        _need(set(body)=={'identity','registration_digest'})
        item=state['reservations'].get(body['identity'])
        _need(item is not None and not item['invalidated'] and body['registration_digest']==item['registration_digest'])
        if item['confirmed_at'] is not None:return False
        _need(item['registered_at']<at<FIRST_OPEN)
        item.update(confirmed_at=at,confirmation_digest=sha)
        return True
    _need(False)


def import_training(registry, journal, *, timezone_file, clock):
    try:
        guard=canonical.Clock(timezone_file,clock); guard()
        record=_project(journal,guard)
        registry.append('training_imported',dict(record=record),timezone_file=timezone_file,clock=guard)
        return _key(record)
    except Exception:
        raise CommonRegistrationError('training_history_import_failed') from None


def import_exposure(registry, *, domain, days, source_sha256, timezone_file, clock):
    try:
        return registry.append('exposure',dict(domain=domain,days=days,source_sha256=source_sha256),
                               timezone_file=timezone_file,clock=clock)
    except Exception:
        raise CommonRegistrationError('common_exposure_import_failed') from None


def reserve_common(registry, plan, *, timezone_file, clock):
    try:
        guard=canonical.Clock(timezone_file,clock); guard()
        state=registry.snapshot(timezone_file=timezone_file,clock=guard)
        record=_validate_plan(state,plan); _original(record,guard)
        identity=digest(plan)
        registry.append('common_registered',dict(plan=plan),timezone_file=timezone_file,clock=guard)
        state=registry.snapshot(timezone_file=timezone_file,clock=guard)
        item=state['reservations'][identity]
        if item['confirmed_at'] is None:
            registry.append('common_confirmed',dict(identity=identity,registration_digest=item['registration_digest']),
                            timezone_file=timezone_file,clock=guard)
        return identity
    except Exception:
        raise CommonRegistrationError('common_reservation_refused') from None


@dataclass(frozen=True)
class RegistrationReceipt:
    identity: str
    plan_json: str
    canonical_path: str
    canonical_head_seq: int
    canonical_head_digest: str
    confirmed_at: datetime
    loaded_at: datetime
    requires_phase_admission: bool
    source_sha256: str
    seal: str

    def __post_init__(self):
        _need(digest(json.loads(self.plan_json))==self.identity and is_sha(self.canonical_head_digest)
              and self.requires_phase_admission is True and self.confirmed_at<FIRST_OPEN
              and self.loaded_at>=self.confirmed_at and self.source_sha256==canonical.source_sha256()
              and self.seal==digest({k:v for k,v in asdict(self).items() if k!='seal'}))

    def plan(self):
        return json.loads(self.plan_json)


def load_registration(registry, identity, *, timezone_file, clock):
    try:
        guard=canonical.Clock(timezone_file,clock); started=guard()
        state=registry.snapshot(timezone_file=timezone_file,clock=guard)
        item=state['reservations'].get(identity)
        _need(item is not None and item['confirmed_at'] is not None and not item['invalidated'])
        record=_validate_plan(state,item['plan']); _original(record,guard)
        final=registry.snapshot(timezone_file=timezone_file,clock=guard)
        _need((final['head_seq'],final['head_digest'])==(state['head_seq'],state['head_digest']))
        loaded=guard(); _need(loaded>=started and started>=item['confirmed_at'])
        fields=dict(identity=identity,plan_json=codec_plan(item['plan']),canonical_path=str(registry.path),
            canonical_head_seq=state['head_seq'],canonical_head_digest=state['head_digest'],
            confirmed_at=item['confirmed_at'],loaded_at=loaded,requires_phase_admission=True,
            source_sha256=canonical.source_sha256())
        return RegistrationReceipt(**fields,seal=digest(fields))
    except Exception:
        raise CommonRegistrationError('common_registration_read_failed') from None


def codec_plan(plan):
    return json.dumps(plan,sort_keys=True,separators=(',',':'),allow_nan=False)
