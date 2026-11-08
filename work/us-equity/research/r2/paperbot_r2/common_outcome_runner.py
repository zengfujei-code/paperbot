"""Bounded local first-outcome worker; no credentials or network operations."""

import argparse
from dataclasses import asdict
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import zoneinfo

from paperbot_us.runner import _exclusive
from . import common_outcome_journal, feature_pipeline, receipt_book
from .contracts import aware, digest, is_sha
from .receipt_runner import _publish
from .source_views import VerifiedFeatureView


def _need(value):
    if not value:
        raise ValueError('outcome_worker_unavailable')


def _credentials_present():
    return any(k.upper().startswith(('ALPACA_', 'APCA_', 'DEEPSEEK_', 'OPENAI_', 'AWS_', 'ALIYUN_'))
               or k.upper().endswith(('_TOKEN', '_SECRET', '_PASSWORD', '_API_KEY', '_ACCESS_KEY'))
               for k in os.environ)


def _guard(at):
    day = at.astimezone(zoneinfo.ZoneInfo('America/New_York')).date()
    return 'reserved_dates_paused' if date(2026, 11, 10) <= day <= date(2026, 11, 23) else None


def _projection(state, features, status, at, *, attempted=0, recorded=0, unavailable=0):
    rows = None if state is None else state['entries']
    entries = None if features is None else features['entries']
    acknowledged = None if rows is None else {k: r for k, r in rows.items() if r['status'] == 'acknowledged'}
    active = None if entries is None else {k: e for k, e in entries.items() if e['phase'] == 'active'}
    return dict(schema=1, component='r2_common_outcomes', research_only=True,
        execution_enabled=False, orders_submitted=0, predictions_issued=0,
        score_eligible_count=0, generated_at=at.isoformat(), status=status,
        feature_count=None if entries is None else len(entries),
        acknowledged_count=None if acknowledged is None else len(acknowledged),
        pending_ack_count=None if rows is None else sum(r['status'] == 'pending_ack' for r in rows.values()),
        late_count=None if acknowledged is None else sum(r['effective_late'] for r in acknowledged.values()),
        missing_count=None if acknowledged is None else sum(r['outcome']['missing'] for r in acknowledged.values()),
        quarantined_count=None if entries is None else sum(e['quarantined'] for e in entries.values()),
        quarantined_ack_count=None if acknowledged is None or entries is None else
            sum(k not in entries or entries[k]['quarantined'] for k in acknowledged),
        unobserved_matured_count=None if active is None or acknowledged is None else
            sum(not e['quarantined'] and aware(e['prepared_bundle']['target_end']) <= at
                and k not in acknowledged for k, e in active.items()),
        pending_target_count=None if active is None else
            sum(not e['quarantined'] and aware(e['prepared_bundle']['target_end']) > at for e in active.values()),
        failed_windows_count=None if features is None else len(features['failed_windows']),
        cycle_attempted=attempted, cycle_recorded=recorded, cycle_unavailable=unavailable,
        head_seq=None if state is None else state['head_seq'])


def collect_once(journal, *, feature_journal, raw_book, clock=None, max_per_cycle=3):
    clock = clock or (lambda: datetime.now(timezone.utc))
    view = None
    try:
        at = aware(clock())
        if _credentials_present():
            return _projection(None, None, 'credential_environment_refused', at)
        paused = _guard(at)
        if paused:
            return _projection(None, None, paused, at)
        _need(type(journal) is common_outcome_journal.OutcomeJournal
              and type(feature_journal) is feature_pipeline.FeatureJournal
              and type(raw_book) is receipt_book.ReceiptBook
              and type(max_per_cycle) is int and 1 <= max_per_cycle <= 3)
        # Different from the journal's append lock: serialize the whole worker
        # while record_first owns its own commit/recovery boundary.
        with _exclusive(journal.path.with_name(journal.path.name + '.worker')):
            state = journal.snapshot()
            _need(at >= state['last_at'])
            day = at.astimezone(zoneinfo.ZoneInfo('America/New_York')).date().isoformat()
            binding = state['binding']
            if not binding['start_day'] <= day <= binding['end_day']:
                return _projection(state, None,
                    'awaiting_start' if day < binding['start_day'] else 'collection_closed', at)
            _need(str(feature_journal.path) == binding['feature_path'])
            stat = feature_journal.path.stat()
            _need(stat.st_dev == binding['feature_device'] and stat.st_ino == binding['feature_inode'])
            _need(not _guard(aware(clock())))
            view = VerifiedFeatureView(feature_journal,
                read_guard=lambda: _need(not _guard(aware(clock()))))
            feature_journal = view
            features = feature_journal.snapshot(deep=True)
            _need(digest(asdict(features['plan'])) == binding['feature_plan_sha256']
                  and features['source_sha256'] == binding['feature_source_sha256'])
            after_read = aware(clock())
            _need(after_read >= at and after_read >= aware(features['last_at']) and not _guard(after_read))
            candidates = []
            for key, entry in features['entries'].items():
                existing = state['entries'].get(key)
                end = aware(entry['prepared_bundle']['target_end'])
                if (entry['phase'] != 'active' or entry['quarantined'] or end > after_read
                        or (existing is not None and existing['status'] == 'acknowledged')):
                    continue
                fresh = (after_read - end).total_seconds() <= 120
                rank = 0 if fresh else 1 if existing is not None else 2
                candidates.append((rank, end, key))
            candidates.sort()
            attempted = recorded = unavailable = 0
            ready = []
            if candidates:
                _need(not _guard(aware(clock())))
                common_outcome_journal._binding(feature_journal, raw_book, binding, clock=clock)
                common_outcome_journal._verify_consumed(feature_journal, raw_book, state['entries'], clock=clock)
                _need(not _guard(aware(clock())))
                raw = raw_book.snapshot()
                for candidate in candidates:
                    key = candidate[2]
                    bundle = features['entries'][key]['accepted_bundle']
                    starts = (aware(bundle['vector']['event_at']), aware(bundle['target_start']))
                    slots = [raw['slots'].get(receipt_book._key(bundle['price_identity'][0], start))
                             for start in starts]
                    if any(s is None or s['first'] is None or s['first']['volume'] <= 0 for s in slots):
                        unavailable += 1
                    else:
                        ready.append(candidate)
            for _, _, key in ready[:max_per_cycle]:
                current = aware(clock())
                _need(current >= after_read and not _guard(current))
                attempted += 1
                try:
                    journal.record_first(feature_journal, raw_book, key, clock=clock)
                    recorded += 1
                except Exception:
                    unavailable += 1
            final_at = aware(clock())
            _need(final_at >= after_read and not _guard(final_at))
            final = journal.snapshot()
            _need(not _guard(aware(clock())))
            final_features = feature_journal.snapshot(deep=True)
            published_at = aware(clock())
            _need(published_at >= final_at and published_at >= final['last_at']
                  and published_at >= aware(final_features['last_at']) and not _guard(published_at))
            status = 'observing' if recorded else 'awaiting_receipts' if unavailable else 'idle'
            if not final_features['entries']:
                status = 'awaiting_features'
            result = _projection(final, final_features, status, published_at,
                                 attempted=attempted, recorded=recorded, unavailable=unavailable)
            if status == 'idle' and result['pending_target_count']:
                result['status'] = 'awaiting_targets'
            return result
    except Exception:
        return _projection(None, None, 'outcome_worker_unavailable', datetime.now(timezone.utc))
    finally:
        if view is not None:
            view.close()


def main():
    parser = argparse.ArgumentParser(description='Local-only common risk-event outcome recorder')
    for name in ('outcome-book', 'feature-book', 'raw-book', 'publish',
                 'timezone-root', 'timezone-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--initialize-only', action='store_true')
    args = parser.parse_args()
    result = _projection(None, None, 'outcome_worker_unavailable', datetime.now(timezone.utc))
    try:
        _need(not _credentials_present() and is_sha(args.timezone_sha256))
        root = Path(args.timezone_root)
        path = root / 'America/New_York'
        _need(not root.is_symlink() and path.is_file() and not path.is_symlink()
              and path.stat().st_size <= 65536
              and hashlib.sha256(path.read_bytes()).hexdigest() == args.timezone_sha256)
        zoneinfo.reset_tzpath([str(root)])
        zoneinfo.ZoneInfo.clear_cache()
        now = datetime.now(timezone.utc)
        if _guard(now):
            result = _projection(None, None, 'reserved_dates_paused', now)
        else:
            journal = common_outcome_journal.OutcomeJournal(args.outcome_book)
            features = feature_pipeline.FeatureJournal(args.feature_book)
            raw = receipt_book.ReceiptBook(args.raw_book)
            if args.initialize_only:
                journal.initialize(features, raw, clock=lambda: datetime.now(timezone.utc))
                state = journal.snapshot()
                fstate = features.snapshot(deep=True)
                result = _projection(state, fstate, 'initialized', datetime.now(timezone.utc))
            else:
                result = collect_once(journal, feature_journal=features, raw_book=raw)
        try:
            _publish(args.publish, result)
        except Exception:
            result = _projection(None, None, 'publication_unavailable', datetime.now(timezone.utc))
    except Exception:
        pass
    print(json.dumps(result, sort_keys=True), flush=True)
    return int(result['status'] in ('outcome_worker_unavailable', 'publication_unavailable',
                                   'credential_environment_refused'))


if __name__ == '__main__':
    raise SystemExit(main())
