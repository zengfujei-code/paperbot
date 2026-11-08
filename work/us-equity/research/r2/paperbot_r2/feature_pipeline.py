"""Committed raw receipts to causal features and matured labels; no network I/O.

Activation acknowledges an already committed readiness record. Its timestamp
does not purport to measure its own commit. Consumers must observe their own
read/issue time before using a feature. This journal makes no forecasts.
"""
import argparse
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import zoneinfo

from paperbot_us import data
from paperbot_us.monitor import ObservationArchive
from paperbot_us.runner import _exclusive
from . import features, receipt_book
from .classifier import FeatureVector
from .contracts import aware, digest, is_sha, json_value, number


class PipelineError(RuntimeError):
    pass


def _need(ok, code='invalid_pipeline_evidence'):
    if not ok:
        raise PipelineError(code)


def _source():
    roots = {'paperbot_us': Path(data.__file__).parent,
             'paperbot_r2': Path(__file__).parent}
    names = {
        'paperbot_us': ('__init__', 'data', 'alpaca', 'regime_baselines',
                       'monitor', 'risk', 'runner', 'audit', 'scanner', 'quant'),
        'paperbot_r2': ('__init__', 'contracts', 'classifier', 'features',
                       'receipt_book', 'receipt_runner', 'feature_pipeline')}
    values = {}
    for package, modules in names.items():
        for module in modules:
            path = roots[package] / (module + '.py')
            _need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 524288)
            values[package + '/' + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest(values)


@dataclass(frozen=True)
class PipelinePlan:
    symbols: tuple
    start_day: str
    end_day: str
    raw_plan_sha256: str
    raw_source_sha256: str
    timezone_sha256: str
    lookback_returns: int = 8
    threshold_fraction: float = .005
    paused_days: tuple = ()
    max_records: int = 20000

    def __post_init__(self):
        receipt_book.ReceiptPlan(self.symbols, self.start_day, self.end_day, self.timezone_sha256)
        _need(is_sha(self.raw_plan_sha256) and is_sha(self.raw_source_sha256))
        _need(type(self.lookback_returns) is int and 2 <= self.lookback_returns <= 20)
        _need(number(self.threshold_fraction, minimum=0., maximum=.2) > 0)
        _need(type(self.max_records) is int and 2 <= self.max_records <= 20000)
        _need(type(self.paused_days) is tuple and tuple(sorted(set(self.paused_days))) == self.paused_days)
        for day in self.paused_days:
            _need(date.fromisoformat(day).isoformat() == day and self.start_day <= day <= self.end_day)


def _plan(raw):
    values = dict(raw)
    for key in ('symbols', 'paused_days'):
        values[key] = tuple(values[key])
    return PipelinePlan(**values)


def _bundle(raw):
    values = dict(raw)
    vector = dict(values['vector'])
    vector['feature_names'] = tuple(vector['feature_names'])
    vector['values'] = tuple(vector['values'])
    values['vector'] = FeatureVector(**vector)
    for key in ('price_identity', 'bar_sha256'):
        values[key] = tuple(values[key])
    return features.FeatureBundle(**values)


def _calendar(raw):
    return data.Calendar([receipt_book._session(raw)], start=raw['day'], end=raw['day'])


def _built(entry, plan, at):
    return features.build_features([data.Bar(**b) for b in entry['bars']],
        calendar=_calendar(entry['session']), issued_at=at,
        lookback_returns=plan.lookback_returns, availability_policy='archived_receipts',
        record_sha256=digest(entry['raw_context']))


def _context(state, context, at):
    _need(set(context) == {'sequence', 'digest', 'read_at', 'last_at', 'calendar_day', 'calendar'})
    _need(type(context['sequence']) is int and context['sequence'] >= 1 and is_sha(context['digest']))
    _need(aware(context['last_at']) <= aware(context['read_at']) <= at)
    _need(date.fromisoformat(context['calendar_day']).isoformat() == context['calendar_day'])
    if context['calendar'] is not None:
        proof = context['calendar']
        _need(set(proof) == {'session', 'received_at', 'changed'} and type(proof['changed']) is bool
              and aware(proof['received_at']) <= aware(context['read_at']))
        session = receipt_book._session(proof['session'])
        _need(session is None or session.day.isoformat() == context['calendar_day'])
    old = state['raw_context']
    if old is not None:
        _need(context['sequence'] >= old['sequence'] and aware(context['last_at']) >= aware(old['last_at']))
        _need(context['sequence'] != old['sequence'] or context['digest'] == old['digest'])
        _need(aware(context['read_at']) >= aware(old['read_at']))


def _entry_key(symbol, end):
    return symbol + '|' + aware(end).isoformat()


def _apply(state, event, *, deep=False):
    """Fast typed replay; new records additionally recompute their mathematics."""
    _need(set(event) == {'kind', 'body', 'at', 'generated_at', 'source_sha256', 'observing_started_at'})
    at = aware(event['at'])
    _need(at.timestamp() == event['generated_at'] and event['source_sha256'] == state['source_sha256'])
    _need(state['last_at'] is None or at >= aware(state['last_at']))
    kind, body = event['kind'], event['body']
    _need(type(body) is dict)
    if kind == 'created':
        _need(state['plan'] is None and set(body) == {'plan'})
        state['plan'] = _plan(body['plan'])
    else:
        plan = state['plan']
        _need(plan is not None)
        if kind == 'raw_anchor':
            _context(state, body, at)
            state['raw_context'] = body
        elif kind == 'feature_unavailable':
            _need(set(body) == {'key', 'symbol', 'event_end', 'reason', 'raw_context'})
            end = aware(body['event_end'])
            _need(body['key'] == _entry_key(body['symbol'], end) and body['symbol'] in plan.symbols
                  and body['key'] not in state['failed_windows'] and end <= at
                  and body['reason'] in ('missing_input', 'invalid_input', 'stale_input'))
            _context(state, body['raw_context'], at)
            state['failed_windows'][body['key']] = dict(body, observed_at=at.isoformat())
        elif kind == 'prepared':
            _need(set(body) == {'key', 'bars', 'session', 'raw_context', 'prepared_bundle'})
            key = body['key']; bundle = body['prepared_bundle']; vector = bundle['vector']
            _need(key not in state['entries'] and vector['available_at'] == at.isoformat())
            _need(key == _entry_key(bundle['price_identity'][0], vector['event_at']))
            _need(bundle['price_identity'][0] in plan.symbols and plan.start_day <= bundle['session_day'] <= plan.end_day
                  and bundle['session_day'] not in plan.paused_days)
            _context(state, body['raw_context'], at)
            _need(bundle['seal'] == digest({k: v for k, v in bundle.items() if k != 'seal'}))
            if deep:
                _need(json_value(asdict(_built(body, plan, at))) == bundle)
            state['entries'][key] = dict(body, phase='prepared', prepared_at=at.isoformat(),
                ready_at=None, ready_digest=None, accepted_bundle=None, missing=False,
                label_prepared=None, label_example=None, label_late=False, quarantined=False)
        else:
            key = body['key']; entry = state['entries'][key]
            bundle = entry['prepared_bundle']
            if kind == 'quarantined':
                _need(set(body) == {'key', 'raw_context', 'calendar'} and not entry['quarantined'])
                _context(state, body['raw_context'], at)
                proof = body['calendar']
                _need(proof is None or proof['changed'] or proof['session'] != entry['session'])
                entry.update(quarantined=True, quarantined_at=at.isoformat(), quarantine_evidence=body)
            elif kind == 'ready':
                _need(set(body) == {'key', 'prepared_sha256'} and entry['phase'] == 'prepared')
                _need(body['prepared_sha256'] == digest(entry['prepared_bundle']))
                _need(0 <= (at - aware(bundle['vector']['event_at'])).total_seconds() <= 120
                      and at < aware(bundle['target_start']))
                entry.update(phase='ready', ready_at=at.isoformat(), ready_digest=digest(event))
            elif kind == 'active':
                _need(set(body) == {'key', 'ready_sha256', 'accepted_bundle'} and entry['phase'] == 'ready')
                _need(body['ready_sha256'] == entry['ready_digest'])
                accepted = body['accepted_bundle']
                _need(accepted['vector']['available_at'] == at.isoformat()
                      and accepted['seal'] == digest({k: v for k, v in accepted.items() if k != 'seal'})
                      and accepted['bar_sha256'] == bundle['bar_sha256']
                      and accepted['vector']['values'] == bundle['vector']['values']
                      and accepted['target_start'] == bundle['target_start']
                      and accepted['target_end'] == bundle['target_end']
                      and 0 <= (at - aware(bundle['vector']['event_at'])).total_seconds() <= 120
                      and at < aware(bundle['target_start']))
                if deep:
                    _need(json_value(asdict(_built(entry, plan, at))) == accepted)
                entry.update(phase='active', accepted_bundle=accepted)
            elif kind == 'aborted':
                _need(set(body) == {'key', 'reason'} and entry['phase'] in ('prepared', 'ready'))
                _need(body['reason'] in ('readiness_expired', 'calendar_quarantined'))
                entry.update(phase='aborted', aborted_at=at.isoformat(), reason=body['reason'])
            elif kind == 'missing':
                _need(set(body) == {'key'} and entry['phase'] == 'active' and not entry['missing']
                      and entry['label_example'] is None and at > aware(bundle['target_end']) + timedelta(seconds=120))
                entry.update(missing=True, missing_at=at.isoformat())
            elif kind == 'label_prepared':
                _need(set(body) == {'key', 'bar', 'raw_context', 'candidate'} and entry['phase'] == 'active'
                      and entry['label_prepared'] is None and entry['label_example'] is None)
                _context(state, body['raw_context'], at)
                _need(body['candidate']['features'] == entry['accepted_bundle']['vector']
                      and body['candidate']['outcome_available_at'] == at.isoformat())
                if deep:
                    candidate = features.label_example(_bundle(entry['accepted_bundle']), data.Bar(**body['bar']),
                        record_sha256=digest(body['raw_context']), threshold_fraction=plan.threshold_fraction,
                        observed_at=at, availability_policy='archived_receipts')
                    _need(json_value(asdict(candidate)) == body['candidate'])
                entry['label_prepared'] = body
            elif kind == 'labeled':
                _need(set(body) == {'key', 'example', 'prepared_sha256'} and entry['label_prepared'] is not None
                      and entry['label_example'] is None and body['prepared_sha256'] == digest(entry['label_prepared']))
                example = body['example']; prepared = entry['label_prepared']
                _need(example['features'] == entry['accepted_bundle']['vector']
                      and example['label'] == prepared['candidate']['label']
                      and example['outcome_available_at'] == at.isoformat())
                late = at > aware(bundle['target_end']) + timedelta(seconds=120)
                _need(not late or entry['missing'])
                if deep:
                    expected = features.label_example(_bundle(entry['accepted_bundle']), data.Bar(**prepared['bar']),
                        record_sha256=digest(prepared['raw_context']), threshold_fraction=plan.threshold_fraction,
                        observed_at=at, availability_policy='archived_receipts')
                    _need(json_value(asdict(expected)) == example)
                entry.update(label_example=example, label_late=late)
            else:
                raise PipelineError('unknown_pipeline_event')
    state['last_at'] = at.isoformat()


class FeatureJournal:
    def __init__(self, path):
        self.path = Path(path).absolute()

    def _path(self):
        _need(self.path.is_file() and not self.path.is_symlink() and not self.path.parent.is_symlink()
              and self.path.stat().st_size <= 268435456, 'pipeline_missing_or_oversized')

    def initialize(self, plan, *, now):
        try:
            _need(type(plan) is PipelinePlan and self.path.parent.is_dir() and not self.path.parent.is_symlink())
            with _exclusive(self.path):
                _need(not self.path.exists() and not self.path.is_symlink())
                at = aware(now)
                ObservationArchive(self.path).append(dict(kind='created', body=json_value(dict(plan=asdict(plan))),
                    at=at.isoformat(), generated_at=at.timestamp(), source_sha256=_source()))
        except Exception:
            raise PipelineError('pipeline_initialization_failed') from None

    def snapshot(self, *, deep=False):
        try:
            self._path()
            state = dict(plan=None, entries={}, failed_windows={}, raw_context=None, head_seq=0,
                         last_at=None, source_sha256=_source())
            with closing(sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True, isolation_level=None)) as db:
                db.execute('BEGIN')
                _, head, sha = ObservationArchive(self.path)._verify_chain(db)
                for seq, payload in db.execute('SELECT seq,payload FROM observations ORDER BY seq'):
                    _apply(state, json.loads(payload), deep=deep)
                    _need(seq <= state['plan'].max_records)
                _need(head >= 1)
                state.update(head_seq=head, head_digest=sha)
                db.rollback()
            return state
        except Exception:
            raise PipelineError('pipeline_integrity_or_storage_failed') from None

    def _append(self, kind, body, at):
        # Runtime calls hold the exclusive lock for the entire derive cycle.
        state = self.snapshot(); at = aware(at)
        _need(state['head_seq'] < state['plan'].max_records, 'pipeline_record_budget_exhausted')
        event = dict(kind=kind, body=json_value(body), at=at.isoformat(), generated_at=at.timestamp(),
                     source_sha256=state['source_sha256'], observing_started_at=0)
        _apply(state, event, deep=True)
        event.pop('observing_started_at')
        return ObservationArchive(self.path).append(event)


def _guard(plan, at):
    day = at.astimezone(zoneinfo.ZoneInfo('America/New_York')).date().isoformat()
    if day in plan.paused_days:
        return 'paused_reserved_dates'
    if day < plan.start_day:
        return 'awaiting_start'
    if day > plan.end_day:
        return 'collection_closed'
    return None


def _projection(state, status, at):
    entries = list(state['entries'].values()) if state else []
    return dict(schema=1, component='r2_feature_pipeline', research_only=True,
        models_fitted=0, predictions_issued=0, orders_submitted=0, execution_enabled=False,
        generated_at=at.isoformat(), status=status, head_seq=state['head_seq'] if state else None,
        active_features=sum(e['phase'] == 'active' and not e['quarantined'] for e in entries) if state else None,
        quarantined_features=sum(e['quarantined'] for e in entries) if state else None,
        failed_windows=len(state['failed_windows']) if state else None,
        aborted_features=sum(e['phase'] == 'aborted' for e in entries) if state else None,
        pending_targets=sum(e['phase'] == 'active' and not e['quarantined'] and e['label_example'] is None for e in entries) if state else None,
        labels_observed=sum(e['label_example'] is not None for e in entries) if state else None,
        usable_labels=sum(e['label_example'] is not None and not e['quarantined'] for e in entries) if state else None,
        late_labels=sum(e['label_late'] for e in entries) if state else None,
        missing_targets=sum(e['missing'] for e in entries) if state else None,
        training_or_context_only=True, full_market_coverage=False)


def _derive(journal, raw_book, clock):
    state = journal.snapshot(); plan = state['plan']; at = aware(clock())
    _need(at >= aware(state['last_at']), 'clock_regression')
    guard = _guard(plan, at)
    if guard:
        return _projection(state, guard, at)
    day = at.astimezone(zoneinfo.ZoneInfo('America/New_York')).date().isoformat()
    old = state['raw_context']
    # Read all regular-session cycles and the raw collector's one-hour recovery
    # window. Outside it, check every30 minutes so durable metadata acknowledgments
    # remain inside the frozen20000-event budget. No unconsumed raw clock is claimed.
    if old and old['calendar_day'] == day and old['calendar'] is not None:
        known = receipt_book._session(old['calendar']['session'])
        outside = known is None or at < known.opens or at > known.closes + timedelta(seconds=3600)
        if outside and (at - aware(old['read_at'])).total_seconds() < 1800:
            return _projection(state, 'market_closed', at)
    try:
        with raw_book._transaction() as (db, raw, meta):
            _need(digest(asdict(raw['plan'])) == plan.raw_plan_sha256
                  and receipt_book._source() == plan.raw_source_sha256
                  and raw['plan'].timezone_sha256 == plan.timezone_sha256
                  and set(plan.symbols).issubset(raw['plan'].symbols))
            old = state['raw_context']
            if old:
                _need(raw['head_seq'] >= old['sequence'] and aware(meta['last_at']) >= aware(old['last_at']))
                row = db.execute('SELECT digest FROM events WHERE seq=?', (old['sequence'],)).fetchone()
                _need(row is not None and row[0] == old['digest'])
        # This clock is deliberately after the read transaction has closed.
        at = aware(clock())
        _need(at >= aware(meta['last_at']) and at >= aware(state['last_at']))
    except Exception:
        return _projection(state, 'raw_evidence_unavailable', aware(clock()))
    guard = _guard(plan, at)
    if guard:
        return _projection(state, guard, at)
    day = at.astimezone(zoneinfo.ZoneInfo('America/New_York')).date().isoformat()
    context = dict(sequence=raw['head_seq'], digest=raw['head_digest'], read_at=at.isoformat(), last_at=meta['last_at'],
                   calendar_day=day, calendar=raw['calendars'].get(day))
    if (old is None or context['sequence'] != old['sequence'] or context['last_at'] != old['last_at']
            or context['calendar_day'] != old['calendar_day']
            or (at - aware(old['read_at'])).total_seconds() >= 1800):
        journal._append('raw_anchor', context, at)

    def now():
        current = aware(clock())
        _need(current >= at and not _guard(plan, current), 'clock_or_date_boundary')
        return current

    day = at.astimezone(zoneinfo.ZoneInfo('America/New_York')).date().isoformat()
    proof = raw['calendars'].get(day)
    session = None if not proof or proof['changed'] else receipt_book._session(proof['session'])
    # Only the latest expected closed interval may become a live feature.
    if session and session.opens <= at < session.closes:
        count = int((at - session.opens).total_seconds() // 900)
        end = session.opens + timedelta(seconds=count * 900)
        if count >= plan.lookback_returns + 1 and end + timedelta(seconds=1800) <= session.closes:
            for symbol in plan.symbols:
                key = _entry_key(symbol, end)
                if key in state['entries']:
                    continue
                bars = []
                for i in range(count - plan.lookback_returns - 1, count):
                    slot = raw['slots'].get(receipt_book._key(symbol, session.opens + timedelta(seconds=i * 900)))
                    if not slot or slot['first'] is None:
                        break
                    bars.append(slot['first'])
                entry = dict(key=key, bars=bars, session=proof['session'], raw_context=context)
                reason = None
                if (at - end).total_seconds() > 120:
                    reason = 'stale_input'
                elif len(bars) != plan.lookback_returns + 1:
                    reason = 'missing_input'
                try:
                    bundle = None if reason else _built(entry, plan, at)
                except Exception:
                    reason = 'invalid_input'
                if reason:
                    if key not in state['failed_windows']:
                        journal._append('feature_unavailable', dict(key=key, symbol=symbol, event_end=end.isoformat(),
                            reason=reason, raw_context=context), at)
                    continue
                journal._append('prepared', dict(entry, prepared_bundle=asdict(bundle)), at)

    state = journal.snapshot()
    for key, entry in state['entries'].items():
        bundle = entry['prepared_bundle']
        proof = raw['calendars'].get(bundle['session_day'])
        valid_calendar = proof is not None and not proof['changed'] and proof['session'] == entry['session']
        if not valid_calendar and not entry['quarantined']:
            journal._append('quarantined', dict(key=key, raw_context=context, calendar=proof), now())
            entry['quarantined'] = True
        if entry['quarantined']:
            if entry['phase'] in ('prepared', 'ready'):
                journal._append('aborted', dict(key=key, reason='calendar_quarantined'), now())
            continue
        if entry['phase'] in ('prepared', 'ready'):
            current = now()
            if not valid_calendar or current > aware(bundle['vector']['event_at']) + timedelta(seconds=120) or current >= aware(bundle['target_start']):
                journal._append('aborted', dict(key=key, reason='readiness_expired' if valid_calendar else 'calendar_quarantined'), current)
                continue
            if entry['phase'] == 'prepared':
                journal._append('ready', dict(key=key, prepared_sha256=digest(bundle)), current)
                entry = journal.snapshot()['entries'][key]
            # A fresh clock after the ready COMMIT, including on restart.
            current = now()
            if current > aware(bundle['vector']['event_at']) + timedelta(seconds=120) or current >= aware(bundle['target_start']):
                journal._append('aborted', dict(key=key, reason='readiness_expired'), current)
                continue
            accepted = _built(entry, plan, current)
            journal._append('active', dict(key=key, ready_sha256=entry['ready_digest'], accepted_bundle=asdict(accepted)), current)
            entry = journal.snapshot()['entries'][key]
        if entry['phase'] != 'active' or entry['label_example'] is not None:
            continue
        current = now(); deadline = aware(bundle['target_end']) + timedelta(seconds=120)
        if current > deadline and not entry['missing']:
            journal._append('missing', dict(key=key), current)
            entry['missing'] = True
        if not valid_calendar:
            continue
        if entry['label_prepared'] is None:
            slot = raw['slots'].get(receipt_book._key(bundle['price_identity'][0], bundle['target_start']))
            if not slot or slot['first'] is None:
                continue
            try:
                candidate = features.label_example(_bundle(entry['accepted_bundle']), data.Bar(**slot['first']),
                    record_sha256=digest(context), threshold_fraction=plan.threshold_fraction,
                    observed_at=current, availability_policy='archived_receipts')
            except Exception:
                continue
            prepared = dict(key=key, bar=slot['first'], raw_context=context, candidate=asdict(candidate))
            journal._append('label_prepared', prepared, current)
            entry['label_prepared'] = json_value(prepared)
        current = now()  # candidate COMMIT acknowledged before label availability
        if current > deadline and not entry['missing']:
            journal._append('missing', dict(key=key), current)
            entry['missing'] = True
        prepared = entry['label_prepared']
        example = features.label_example(_bundle(entry['accepted_bundle']), data.Bar(**prepared['bar']),
            record_sha256=digest(prepared['raw_context']), threshold_fraction=plan.threshold_fraction,
            observed_at=current, availability_policy='archived_receipts')
        journal._append('labeled', dict(key=key, example=asdict(example), prepared_sha256=digest(prepared)), current)
    status = 'observing' if session and session.opens <= at < session.closes else 'market_closed'
    return _projection(journal.snapshot(), status, now())


def derive_once(journal, *, raw_book, clock=None):
    clock = clock or (lambda: datetime.now(timezone.utc))
    try:
        # Reject secret variable NAMES; never read or project their values.
        _need(not any(k.upper().startswith(('ALPACA_', 'APCA_', 'DEEPSEEK_', 'OPENAI_', 'AWS_', 'ALIYUN_'))
                      or k.upper().endswith(('_TOKEN', '_SECRET', '_PASSWORD', '_API_KEY', '_ACCESS_KEY')) for k in os.environ),
              'credential_environment_refused')
        journal._path()
        with _exclusive(journal.path):
            return _derive(journal, raw_book, clock)
    except Exception:
        return _projection(None, 'pipeline_unavailable', datetime.now(timezone.utc))


def main():
    parser = argparse.ArgumentParser(description='Local-only causal feature and target journal')
    for name in ('plan', 'plan-sha256', 'book', 'raw-book', 'publish', 'timezone-root'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--initialize-only', action='store_true')
    args = parser.parse_args()
    result = _projection(None, 'pipeline_unavailable', datetime.now(timezone.utc))
    try:
        path = Path(args.plan)
        _need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 16384)
        content = path.read_bytes()
        _need(hashlib.sha256(content).hexdigest() == args.plan_sha256)
        plan = _plan(json.loads(content)); root = Path(args.timezone_root); tz = root / 'America/New_York'
        _need(not root.is_symlink() and tz.is_file() and not tz.is_symlink()
              and hashlib.sha256(tz.read_bytes()).hexdigest() == plan.timezone_sha256)
        zoneinfo.reset_tzpath([str(root)]); zoneinfo.ZoneInfo.clear_cache()
        journal = FeatureJournal(args.book)
        if args.initialize_only:
            journal.initialize(plan, now=datetime.now(timezone.utc))
            result = _projection(journal.snapshot(), 'initialized', datetime.now(timezone.utc))
        else:
            _need(journal.snapshot()['plan'] == plan)
            result = derive_once(journal, raw_book=receipt_book.ReceiptBook(args.raw_book))
        from .receipt_runner import _publish
        try:
            _publish(args.publish, result)
        except Exception:
            result = _projection(None, 'publication_unavailable', datetime.now(timezone.utc))
    except Exception:
        pass
    print(json.dumps(result, sort_keys=True), flush=True)
    return int(result['status'] in ('pipeline_unavailable', 'raw_evidence_unavailable', 'publication_unavailable'))


if __name__ == '__main__':
    raise SystemExit(main())
