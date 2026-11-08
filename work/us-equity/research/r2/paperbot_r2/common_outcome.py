"""Read-only close-to-close outcome from committed R2 first receipts.

This module derives a *matured label*, never a prospective forecast. Its
post-read timestamp is the current observation time, not a backdated proof that
the result was available to an earlier trading decision.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import hashlib
import math

from paperbot_us import data
from . import feature_pipeline, receipt_book
from .classifier import ClassifierExample
from .contracts import aware, digest, is_sha
from .source_views import VerifiedFeatureView


class CommonOutcomeError(ValueError):
    pass


def _need(condition, code='common_outcome_unavailable'):
    if not condition:
        raise CommonOutcomeError(code)


def _source():
    paths = (Path(__file__), Path(feature_pipeline.__file__), Path(receipt_book.__file__),
             Path(data.__file__), Path(__file__).with_name('source_views.py'))
    _need(all(p.is_file() and not p.is_symlink() and p.stat().st_size <= 524288 for p in paths))
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def _seal(value):
    return digest({k: v for k, v in asdict(value).items() if k != 'seal'})


@dataclass(frozen=True)
class CommonOutcome:
    feature_key: str
    symbol: str
    feature_bundle_sha256: str
    feature_head_seq: int
    feature_head_digest: str
    raw_plan_sha256: str
    raw_source_sha256: str
    raw_head_seq: int
    raw_head_digest: str
    gap_bar_sha256: str
    target_bar_sha256: str
    gap_recorded_at: datetime
    target_recorded_at: datetime
    observed_at: datetime
    target_start: datetime
    target_end: datetime
    return_fraction: float
    label: int
    missing: bool
    late: bool
    quarantined: bool
    example: ClassifierExample
    source_sha256: str
    seal: str
    formula_version: str = 'first-receipt-close-to-close-risk-event-v1'

    def __post_init__(self):
        for name in ('gap_recorded_at', 'target_recorded_at', 'observed_at', 'target_start', 'target_end'):
            object.__setattr__(self, name, aware(getattr(self, name)))
        _need(type(self.feature_key) is str and type(self.symbol) is str
              and self.feature_key.startswith(self.symbol + '|'))
        _need(all(is_sha(v) for v in (self.feature_bundle_sha256, self.feature_head_digest,
              self.raw_plan_sha256, self.raw_source_sha256, self.raw_head_digest,
              self.gap_bar_sha256, self.target_bar_sha256, self.source_sha256, self.seal)))
        _need(type(self.feature_head_seq) is int and self.feature_head_seq >= 1
              and type(self.raw_head_seq) is int and self.raw_head_seq >= 1)
        _need(self.target_start < self.target_end <= self.observed_at
              and self.gap_recorded_at <= self.observed_at
              and self.target_recorded_at <= self.observed_at)
        _need(type(self.return_fraction) is float and math.isfinite(self.return_fraction)
              and -1. < self.return_fraction <= 100.)
        _need(type(self.label) is int and self.label in (0, 1)
              and all(type(v) is bool for v in (self.missing, self.late, self.quarantined))
              and self.quarantined is False)
        _need(type(self.example) is ClassifierExample and self.example.label == self.label
              and self.example.target_start == self.target_start
              and self.example.target_end == self.target_end
              and self.example.outcome_available_at == self.observed_at)
        _need(self.formula_version == 'first-receipt-close-to-close-risk-event-v1'
              and self.source_sha256 == _source() and self.seal == _seal(self))


def observe_common_outcome(feature_journal, raw_book, key, *, clock,
                           threshold_fraction=.005, read_guard=None):
    """Observe a matured outcome now; never infer an earlier issue/availability.

    Both journals are read-only. A caller must persist the first returned
    observation before treating it as a prospective evaluation record.
    """
    try:
        _need(type(feature_journal) in (feature_pipeline.FeatureJournal, VerifiedFeatureView)
              and type(raw_book) is receipt_book.ReceiptBook
              and type(key) is str and callable(clock)
              and type(threshold_fraction) is float and threshold_fraction == .005)
        if read_guard is not None:
            read_guard()
        state = feature_journal.snapshot(deep=True)
        entry = state['entries'].get(key)
        _need(entry is not None and entry['phase'] == 'active' and not entry['quarantined'])
        bundle = feature_pipeline._bundle(entry['accepted_bundle'])
        _need(bundle.availability_policy == 'archived_receipts'
              and key == feature_pipeline._entry_key(bundle.price_identity[0], bundle.vector.event_at))
        if read_guard is not None:
            read_guard()
        with raw_book._transaction() as (db, raw, meta):
            plan = state['plan']
            _need(digest(asdict(raw['plan'])) == plan.raw_plan_sha256
                  and receipt_book._source() == plan.raw_source_sha256
                  and raw['plan'].timezone_sha256 == plan.timezone_sha256
                  and bundle.price_identity[0] in raw['plan'].symbols)
            context = entry['raw_context']
            _need(type(context['sequence']) is int and context['sequence'] <= raw['head_seq'])
            ancestor = db.execute('SELECT digest FROM events WHERE seq=?',
                                  (context['sequence'],)).fetchone()
            _need(ancestor is not None and ancestor[0] == context['digest'])
            proof = raw['calendars'].get(bundle.session_day)
            _need(proof is not None and not proof['changed']
                  and proof['session'] == entry['session'])
            for original in entry['bars']:
                slot = raw['slots'].get(receipt_book._key(original['symbol'], aware(original['starts'])))
                _need(slot is not None and slot['first'] == original)
            gap_slot = raw['slots'].get(receipt_book._key(bundle.price_identity[0], bundle.vector.event_at))
            target_slot = raw['slots'].get(receipt_book._key(bundle.price_identity[0], bundle.target_start))
            _need(gap_slot is not None and target_slot is not None
                  and gap_slot['first'] is not None and target_slot['first'] is not None)
            gap_raw = gap_slot['first']
            target_raw = target_slot['first']
            gap_recorded = aware(gap_slot['first_recorded_at'])
            target_recorded = aware(target_slot['first_recorded_at'])
            missing = bool(gap_slot['missed'] or target_slot['missed'])
            late = bool(not gap_slot['on_time'] or not target_slot['on_time'])
            raw_head_seq, raw_head_digest = raw['head_seq'], raw['head_digest']
            raw_last_at = aware(meta['last_at'])

        # Recheck the feature journal after the raw read so a concurrent
        # quarantine or phase change cannot be silently ignored.
        if read_guard is not None:
            read_guard()
        final_state = feature_journal.snapshot(deep=True)
        _need(final_state['head_digest'] == state['head_digest'])
        observed = aware(clock())
        _need(observed >= raw_last_at and observed >= aware(final_state['last_at'])
              and observed >= gap_recorded and observed >= target_recorded)
        gap = data.Bar(**gap_raw)
        target = data.Bar(**target_raw)
        session = receipt_book._session(dict(entry['session']))
        _need(session is not None and gap.symbol == target.symbol == bundle.price_identity[0]
              and gap.starts == bundle.vector.event_at and gap.ends == bundle.target_start
              and target.starts == bundle.target_start and target.ends == bundle.target_end
              and gap_slot['day'] == target_slot['day'] == bundle.session_day)
        gap.validate(as_of=observed, session=session)
        target.validate(as_of=observed, session=session)
        _need(gap.volume > 0 and target.volume > 0
              and all((b.source, b.feed, b.adjustment, b.interval_seconds, b.currency, b.volume_unit)
                      == bundle.price_identity[1:] for b in (gap, target)))
        late = late or (observed - target.ends).total_seconds() > 120
        prior, current, cutoff = (Fraction(str(v)) for v in
                                  (gap.close, target.close, threshold_fraction))
        _need(prior > 0 and current > 0)
        label = int(abs(current - prior) >= cutoff * prior)
        value = float((current - prior) / prior)
        source = _source()
        gap_sha, target_sha = digest(gap_raw), digest(target_raw)
        schema = digest(dict(formula='abs(target_first_close/gap_first_close-1)>=threshold',
                             threshold_fraction=threshold_fraction,
                             source_sha256=source, price_identity=bundle.price_identity,
                             horizon_seconds=900, gap_seconds=900))
        outcome_id = digest(dict(feature_bundle_sha256=bundle.seal, gap_bar_sha256=gap_sha,
                                 target_bar_sha256=target_sha, schema=schema))
        example = ClassifierExample(bundle.vector, schema, bundle.target_start,
                                    bundle.target_end, observed, label, outcome_id)
        fields = dict(feature_key=key, symbol=gap.symbol,
                      feature_bundle_sha256=bundle.seal,
                      feature_head_seq=state['head_seq'], feature_head_digest=state['head_digest'],
                      raw_plan_sha256=plan.raw_plan_sha256,
                      raw_source_sha256=plan.raw_source_sha256,
                      raw_head_seq=raw_head_seq, raw_head_digest=raw_head_digest,
                      gap_bar_sha256=gap_sha, target_bar_sha256=target_sha,
                      gap_recorded_at=gap_recorded, target_recorded_at=target_recorded,
                      observed_at=observed, target_start=bundle.target_start,
                      target_end=bundle.target_end, return_fraction=value,
                      label=label, missing=missing, late=late, quarantined=False,
                      example=example, source_sha256=source,
                      formula_version='first-receipt-close-to-close-risk-event-v1')
        return CommonOutcome(**fields, seal=digest({
            k: asdict(v) if k == 'example' else v for k, v in fields.items()}))
    except CommonOutcomeError:
        raise
    except Exception:
        raise CommonOutcomeError('common_outcome_unavailable') from None
