"""Synthetic common comparison registration; no live transfer/model/market calls."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import zoneinfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_training_data import Fixture, Tick, ROOT, TZ
from test_canonical_history import legacy_fixture, HANDOFF
from paperbot_us.data import Calendar, Session
from paperbot_r2 import canonical_history as canonical, common_registration as common
from paperbot_r2.training_journal import TrainingJournal


class CommonRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_tz = zoneinfo.TZPATH
        zoneinfo.reset_tzpath([str(ROOT / 'outputs/zoneinfo')]); zoneinfo.ZoneInfo.clear_cache()
        cls.addClassCleanup(zoneinfo.reset_tzpath, cls.old_tz)
        cls.addClassCleanup(zoneinfo.ZoneInfo.clear_cache)
        cls.temp = tempfile.TemporaryDirectory(); cls.addClassCleanup(cls.temp.cleanup)
        with patch.dict(os.environ, {}, clear=True):
            cls.f = Fixture(Path(cls.temp.name)); cls.f.finish()
            cls.books = []
            for n in range(2):
                book = TrainingJournal(Path(cls.temp.name) / ('training-' + str(n) + '.sqlite'))
                clock = Tick(cls.f.at + timedelta(seconds=n + 1))
                book.initialize(timezone_file=TZ, clock=clock)
                book.run_once(cls.f.raw, cls.f.features, cls.f.outcomes, timezone_file=TZ, clock=clock)
                cls.books.append(book)
            cls.interrupted = TrainingJournal(Path(cls.temp.name) / 'interrupted.sqlite')
            clock = Tick(cls.f.at + timedelta(seconds=4))
            cls.interrupted.initialize(timezone_file=TZ, clock=clock)
            from paperbot_r2 import model_training
            with patch.object(model_training, 'fit_training_snapshot', side_effect=RuntimeError('synthetic interruption')):
                try:
                    cls.interrupted.run_once(cls.f.raw, cls.f.features, cls.f.outcomes, timezone_file=TZ, clock=clock)
                except ValueError:
                    pass

    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True); env.start(); self.addCleanup(env.stop)
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.old, _ = legacy_fixture(self.path / 'shared.sqlite')
        self.store = canonical.CanonicalRegistry(self.old.path); self.clock = Tick(HANDOFF)
        anchor = canonical.inspect_legacy(self.old, timezone_file=TZ, clock=self.clock)
        self.store.continue_legacy(expected_history=anchor, timezone_file=TZ, clock=self.clock)
        days = common.VALIDATION_DAYS + common.TEST_DAYS
        self.calendar = Calendar([Session.from_alpaca(dict(date=d,open='09:30',close='13:00' if d=='2026-11-27' else '16:00'))
                                  for d in days], start=days[0], end=days[-1])

    def state(self):
        return self.store.snapshot(timezone_file=TZ, clock=self.clock)

    def import_book(self, book=None):
        return common.import_training(self.store, book or self.books[0], timezone_file=TZ, clock=self.clock)

    def plan(self, key=None, calendar=None):
        key = key or self.import_book()
        record = self.state()['training_imports'][key]
        return common.build_common_plan(record, calendar or self.calendar, calendar_received_at=HANDOFF,
                                        issue_source_sha256='e' * 64, configuration_sha256='f' * 64)

    def reserve(self, plan=None, clock=None):
        return common.reserve_common(self.store, plan or self.plan(), timezone_file=TZ, clock=clock or self.clock)

    def test_import_preserves_fifteen_fit_cells_and_is_idempotent(self):
        key = self.import_book(); state = self.state()
        record = state['training_imports'][key]
        self.assertEqual(record['trial_units'], 15)
        self.assertEqual(len(record['candidates']), 15)
        self.assertTrue(any(r['fit_status']=='failed' for r in record['candidates']))
        self.assertEqual(record['phase'], 'acknowledged')
        self.assertEqual(self.import_book(), key)
        self.assertEqual(self.state()['head_digest'], state['head_digest'])

    def test_interrupted_batch_is_counted_and_cannot_supply_models(self):
        key = self.import_book(self.interrupted)
        record = self.state()['training_imports'][key]
        self.assertEqual(record['trial_units'], 15)
        self.assertEqual(record['phase'], 'reserved')
        self.assertTrue(all(r['fit_status']=='unknown' for r in record['candidates']))
        with self.assertRaises(common.CommonRegistrationError):
            self.reserve(self.plan(key))

    def test_confirmed_registration_retains_history_and_actual_read_availability(self):
        plan = self.plan(); identity = self.reserve(plan)
        record = self.state()['reservations'][identity]
        self.assertGreater(record['confirmed_at'], record['registered_at'])
        receipt = common.load_registration(self.store, identity, timezone_file=TZ, clock=self.clock)
        self.assertEqual(receipt.plan(), plan)
        self.assertGreaterEqual(receipt.loaded_at, receipt.confirmed_at)
        self.assertTrue(receipt.requires_phase_admission)
        self.assertEqual(self.reserve(plan), identity)
        self.assertEqual(len(self.state()['legacy_state']['experiments']), 1)

    def test_second_distinct_training_batch_is_not_a_budget_reset(self):
        self.import_book(); second = self.import_book(self.books[1])
        self.assertEqual(sum(r['trial_units'] for r in self.state()['training_imports'].values()), 30)
        with self.assertRaises(common.CommonRegistrationError):
            self.reserve(self.plan(second))

    def test_late_additional_training_invalidates_existing_reservation(self):
        identity = self.reserve(); self.import_book(self.books[1])
        self.assertTrue(self.state()['reservations'][identity]['invalidated'])
        with self.assertRaises(common.CommonRegistrationError):
            common.load_registration(self.store, identity, timezone_file=TZ, clock=self.clock)

    def test_heldout_exposure_blocks_or_invalidates_regardless_family(self):
        common.import_exposure(self.store, domain='us_equity',days=['2026-12-07'],source_sha256='b'*64,
                               timezone_file=TZ,clock=self.clock)
        with self.assertRaises(common.CommonRegistrationError):
            self.reserve()

    def test_late_exposure_invalidates_and_synthetic_domain_is_separate(self):
        identity = self.reserve()
        common.import_exposure(self.store,domain='synthetic',days=['2026-11-24'],source_sha256='c'*64,
                               timezone_file=TZ,clock=self.clock)
        common.load_registration(self.store,identity,timezone_file=TZ,clock=self.clock)
        common.import_exposure(self.store,domain='us_equity',days=['2026-11-24'],source_sha256='d'*64,
                               timezone_file=TZ,clock=self.clock)
        with self.assertRaises(common.CommonRegistrationError):
            common.load_registration(self.store,identity,timezone_file=TZ,clock=self.clock)

    def test_fixed_calendar_and_objective_are_not_silently_changed(self):
        plan = self.plan()
        self.assertEqual(next(s for s in plan['calendar']['sessions'] if s['day']=='2026-11-27')['closes'],
                         '2026-11-27T18:00:00+00:00')
        altered = dict(plan, objective='net_return_max')
        with self.assertRaises(common.CommonRegistrationError):
            self.reserve(altered)
        sessions = list(self.calendar.sessions.values())
        sessions = [Session(s.day,s.opens,s.closes+timedelta(hours=3)) if s.day.isoformat()=='2026-11-27' else s for s in sessions]
        calendar = Calendar(sessions,start=common.VALIDATION_DAYS[0],end=common.TEST_DAYS[-1])
        with self.assertRaises(common.CommonRegistrationError):
            self.plan(calendar=calendar)

    def test_commit_crossing_first_open_never_produces_confirmed_reservation(self):
        plan = self.plan(); clock = self.clock; target = self.store.path
        first_open = datetime(2026,11,24,14,30,tzinfo=timezone.utc)
        class DelayedCommit(sqlite3.Connection):
            def commit(db):
                file = Path(db.execute('PRAGMA database_list').fetchone()[2])
                kind = None
                if file==target:
                    row=db.execute('SELECT payload FROM canonical_v2_events ORDER BY seq DESC LIMIT 1').fetchone()
                    kind=json.loads(row[0])['kind'] if row else None
                super().commit()
                if kind=='common_registered':clock.value=first_open+timedelta(seconds=1)
        original=sqlite3.connect
        with patch('sqlite3.connect',lambda *a,**kw:original(*a,**kw,factory=DelayedCommit)):
            with self.assertRaises(common.CommonRegistrationError):
                self.reserve(plan)
        record=next(iter(self.state()['reservations'].values()))
        self.assertIsNone(record['confirmed_at'])

    def test_exposure_arriving_during_training_source_recheck_invalidates_load(self):
        identity=self.reserve(); original=common._original
        def change(record,clock):
            original(record,clock)
            common.import_exposure(self.store,domain='us_equity',days=['2026-11-24'],source_sha256='9'*64,
                                   timezone_file=TZ,clock=self.clock)
        with patch.object(common,'_original',change):
            with self.assertRaises(common.CommonRegistrationError):
                common.load_registration(self.store,identity,timezone_file=TZ,clock=self.clock)

    def test_confirmation_failure_recovers_original_reservation_without_new_trial(self):
        plan=self.plan(); append=self.store.append
        def fail(kind,body,**kwargs):
            if kind=='common_confirmed':raise OSError('synthetic confirmation interruption')
            return append(kind,body,**kwargs)
        with patch.object(self.store,'append',fail):
            with self.assertRaises(common.CommonRegistrationError):self.reserve(plan)
        state=self.state(); identity=next(iter(state['reservations']))
        first=state['reservations'][identity]
        with self.assertRaises(common.CommonRegistrationError):
            common.load_registration(self.store,identity,timezone_file=TZ,clock=self.clock)
        self.assertEqual(self.reserve(plan),identity)
        after=self.state()['reservations'][identity]
        self.assertEqual(first['registration_digest'],after['registration_digest'])
        self.assertEqual(first['registered_at'],after['registered_at'])
        self.assertEqual(len(self.state()['training_imports']),1)

    def test_copied_training_book_is_not_a_fresh_original_batch(self):
        copied=self.path/'training-copy.sqlite';shutil.copyfile(self.books[0].path,copied)
        with self.assertRaises(common.CommonRegistrationError):
            self.import_book(TrainingJournal(copied))
        self.assertEqual(self.state()['training_imports'],{})

    def test_backwards_registration_clock_cannot_reserve(self):
        plan=self.plan();values=iter((HANDOFF+timedelta(seconds=2),HANDOFF+timedelta(seconds=1)))
        with self.assertRaises(common.CommonRegistrationError):
            self.reserve(plan,clock=lambda:next(values))
        self.assertEqual(self.state()['reservations'],{})


if __name__=='__main__':
    unittest.main()
