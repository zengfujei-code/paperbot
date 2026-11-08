"""Fixed prospective test clocks and scalar diagnostics, never phase authority.

The complete source commitment remains unavailable while any declared protocol
module is missing. Arithmetic reports have a separate math fingerprint and
cannot substitute for original-book admission or an observed publication.
"""
from datetime import timedelta
import hashlib
from pathlib import Path

from . import canonical_history,common_scores as scores,common_registration as registered
from . import validation_scores
from .contracts import aware,digest
from .model_training import FAMILIES
from .training_data import SYMBOLS

PROTOCOL_FILES=('test_protocol.py','selection_journal.py','test_issue_journal.py','test_settlements.py','test_scores.py','replay_cache.py')
SELECTION_AFTER=aware('2026-12-04T21:02:00+00:00')
TEST_OPEN=aware('2026-12-07T14:30:00+00:00')
TEST_FINAL_DEADLINE=aware('2026-12-11T21:02:00+00:00')


class TestProtocolError(ValueError):pass


def _need(value):
    if not value:raise TestProtocolError('prospective_test_protocol_unavailable')


def _file_identity(path):
    _need(path.is_file() and not path.is_symlink() and not path.parent.is_symlink())
    stat=path.stat();_need(0<stat.st_size<=524288)
    return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns


def _file_hash(path):
    before=_file_identity(path)
    result=hashlib.sha256(path.read_bytes()).hexdigest()
    _need(_file_identity(path)==before)
    return result


def math_source_sha256():
    return digest(dict(test_math=_file_hash(Path(__file__)),scalar_losses=scores.source_sha256()))


def source_sha256():
    """Commit every real protocol module without recursive function calls."""
    try:
        root=Path(__file__).parent
        paths=tuple(root/name for name in PROTOCOL_FILES)
        before=tuple(_file_identity(p) for p in paths)
        dependency=validation_scores.source_sha256()
        files={p.name:_file_hash(p) for p in paths}
        _need(validation_scores.source_sha256()==dependency and tuple(_file_identity(p) for p in paths)==before)
        return digest(dict(files=files,validation_evaluator=dependency))
    except Exception:raise TestProtocolError('complete_test_protocol_commitment_unavailable') from None


def planned_slots(plan):
    try:
        scores.planned_slots(plan)  # Authenticate all declared calendar/split/event rules.
        slots=[]
        for session in plan['calendar']['sessions'][8:]:
            opens,closes=aware(session['opens']),aware(session['closes'])
            for symbol in SYMBOLS:
                for completed in range(9,int((closes-opens).total_seconds()/900)-1):
                    at=opens+timedelta(seconds=completed*900)
                    slots.append(scores.PlannedSlot(symbol+'|'+at.isoformat(),symbol,session['day'],at,
                                                   at+timedelta(seconds=900),at+timedelta(seconds=1800)))
        _need(len(slots)==240 and all(sum(s.symbol==symbol for s in slots)==80 for symbol in SYMBOLS))
        return tuple(slots)
    except Exception:raise TestProtocolError('registered_test_grid_invalid') from None


class _Clock:
    def __init__(self,timezone_file,clock):
        try:self.base=canonical_history.Clock(timezone_file,clock)
        except Exception:raise TestProtocolError('test_clock_unavailable') from None

    def _now(self):
        try:return self.base()
        except Exception:raise TestProtocolError('test_clock_unavailable') from None


class SelectionClock(_Clock):
    def __call__(self):
        at=self._now();_need(SELECTION_AFTER<at<TEST_OPEN);return at


def _test_key(key):
    try:
        _need(type(key) is str);symbol,raw=key.split('|',1);event=aware(raw)
        day=event.date().isoformat();opens=aware(day+'T14:30:00+00:00')
        seconds=(event-opens).total_seconds()
        _need(symbol in SYMBOLS and key==symbol+'|'+event.isoformat() and day in registered.TEST_DAYS
              and 9*900<=seconds<=24*900 and seconds%900==0)
        return event
    except Exception:raise TestProtocolError('declared_test_key_required') from None


class TestIssueClock(_Clock):
    def __init__(self,key,timezone_file,clock):
        self.event_at=_test_key(key);super().__init__(timezone_file,clock)
    def __call__(self):
        at=self._now();_need(self.event_at<=at<=self.event_at+timedelta(seconds=120)
                            and at<self.event_at+timedelta(seconds=900));return at


class TestOutcomeClock(_Clock):
    def __init__(self,key,timezone_file,clock):
        self.event_at=_test_key(key);self.target_end=self.event_at+timedelta(seconds=1800)
        super().__init__(timezone_file,clock)
    def __call__(self):
        at=self._now();_need(at>=self.target_end);return at


class TestFreezeClock(_Clock):
    def __call__(self):
        at=self._now();_need(at>TEST_FINAL_DEADLINE);return at


def evaluate_test_frames(plan,frames,*,as_of,fit_statuses,selected_family):
    """Describe the fixed selected family and all-five common comparison only."""
    try:
        slots=planned_slots(plan);as_of=aware(as_of)
        _need(type(selected_family) is str and selected_family in FAMILIES
              and type(frames) is tuple and len(frames)<=len(slots) and type(fit_statuses) is tuple
              and tuple((s,f) for s,f,_ in fit_statuses)==registered.PAIRS
              and all(status in ('succeeded','failed') for _,_,status in fit_statuses))
        training={(s,f):status for s,f,status in fit_statuses}
        expected={s.key:s for s in slots};by_key={}
        for frame in frames:
            _need(type(frame) is scores.ScoreFrame and frame.key in expected and frame.key not in by_key)
            slot=expected[frame.key]
            _need((frame.symbol,frame.event_at,frame.target_start,frame.target_end)==
                  (slot.symbol,slot.event_at,slot.target_start,slot.target_end)
                  and (not frame.issued or frame.event_at<=as_of)
                  and (not frame.settled or frame.target_end<=as_of)
                  and (not frame.missing or frame.settled or as_of>frame.target_end+timedelta(seconds=120)))
            for row in frame.rows:
                _need((row.status=='training_fit_failed')==(training[(frame.symbol,row.family)]=='failed'))
            by_key[frame.key]=frame
        groups=[dict(day=day,symbol=symbol,**scores._group(tuple(s for s in slots if s.symbol==symbol and s.day==day),by_key,as_of))
                for day in registered.TEST_DAYS for symbol in SYMBOLS]
        symbols={symbol:scores._group(tuple(s for s in slots if s.symbol==symbol),by_key,as_of) for symbol in SYMBOLS}
        macro={}
        for family in FAMILIES:
            values=[symbols[s]['families'][family] for s in SYMBOLS]
            macro[family]={name:scores._mean([v[name] for v in values])
                          if all(v['count'] and v[name] is not None for v in values) else None for name in scores.METRICS}
        complete=as_of>TEST_FINAL_DEADLINE
        report=dict(schema=1,phase='test',as_of=as_of.isoformat(),math_only=True,
            selection_admission=False,test_admission=False,promotion=False,orders=False,
            selected_family=selected_family,proposed_family=None,trial_units=15,test_sessions=5,
            evidence_class='exploratory_risk_forecast_only',
            training_fits=[dict(symbol=s,family=f,status=status) for s,f,status in fit_statuses],
            complete=complete,engineering_gate_passed=complete and all(g['coverage_passed'] for g in groups),
            groups=groups,symbols=symbols,macro=macro,
            units=dict(return_unit='fraction',variance_unit='fraction_squared',variance_proxy_mse='fraction_fourth_power',
                       qlike='log_fraction_squared_plus_squared_error_over_predicted_variance',annualized=False),
            source_sha256=math_source_sha256())
        return scores._loss_json(report)
    except Exception:raise TestProtocolError('test_comparison_evaluation_failed') from None
