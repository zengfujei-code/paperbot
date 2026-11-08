"""Original temporary books with a REAL full-protocol policy before reservation.

These synthetic fixtures are not market evidence. In particular, the default
small training set fails NB; a failed validation gate must never admit test.
"""
from contextlib import ExitStack,closing,nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zoneinfo

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[3]),str(Path(__file__).resolve().parents[1])]
import issue_fixture
from test_training_data import ROOT,TZ,Tick,Fixture,DAYS,SYMBOLS
from test_canonical_history import HANDOFF
from paperbot_r2 import common_registration as common,validation_issue_journal as validation_issuer
from paperbot_r2 import validation_settlements,validation_scores,selection_journal
from paperbot_r2 import test_protocol,test_issue_journal,test_settlements,test_scores,common_outcome
from paperbot_r2 import receipt_book,feature_pipeline,common_outcome_journal
from paperbot_r2 import replay_cache,forecast_inputs,candidate_forecasts
from paperbot_r2.contracts import aware,digest
from paperbot_us.data import Calendar
from dataclasses import asdict


class PhaseWorld:
    def __init__(self,directory,*,training_fixture=None,market_class=issue_fixture.IssueWorld,use_replay_cache=False):
        self.directory=Path(directory);self.policy=validation_scores.make_policy(test_protocol.source_sha256())
        self.runtime=replay_cache.ReplayRuntime() if use_replay_cache else None
        original=common.build_common_plan
        def configured(*args,**kwargs):
            kwargs['configuration_sha256']=self.policy.configuration_sha256
            return original(*args,**kwargs)
        training_context=patch.object(issue_fixture,'Fixture',training_fixture) if training_fixture is not None else nullcontext()
        with training_context,patch.object(common,'build_common_plan',side_effect=configured):
            self.w=market_class(self.directory,issue_source_sha256=validation_issuer.source_sha256())
        identity=self.w.identity
        self.issues=validation_issuer.ValidationIssueJournal(self.directory/('r2-validation-'+identity+'.sqlite'))
        self.validation_settlements=validation_settlements.ValidationSettlementJournal(self.directory/('r2-validation-settlements-'+identity+'.sqlite'))
        self.validation_scores=validation_scores.ValidationScoreJournal(self.directory/('r2-validation-scores-'+identity+'.sqlite'))
        self.selection=selection_journal.SelectionJournal(self.directory/('r2-selection-'+identity+'.sqlite'))
        self.test_issues=test_issue_journal.TestIssueJournal(self.directory/('r2-test-issues-'+identity+'.sqlite'))
        self.test_settlements=test_settlements.TestSettlementJournal(self.directory/('r2-test-settlements-'+identity+'.sqlite'))
        self.test_scores=test_scores.TestScoreJournal(self.directory/('r2-test-scores-'+identity+'.sqlite'))
        self.clock=Tick(HANDOFF+timedelta(minutes=1))
        self.call(self.issues.initialize,self.w.registry,identity,self.w.training,self.w.source_pairs(),timezone_file=TZ,clock=self.clock)
        self.call(self.validation_settlements.initialize,self.issues,self.policy,timezone_file=TZ,clock=self.clock)
        self.call(self.validation_scores.initialize,self.validation_settlements,timezone_file=TZ,clock=self.clock)
        self.call(self.selection.initialize,self.validation_scores,timezone_file=TZ,clock=self.clock)
        self.call(self.test_issues.initialize,self.selection,timezone_file=TZ,clock=self.clock)
        self.call(self.test_settlements.initialize,self.test_issues,timezone_file=TZ,clock=self.clock)
        self.call(self.test_scores.initialize,self.test_settlements,timezone_file=TZ,clock=self.clock)

    def call(self,method,*args,clock,**kwargs):
        """Select the real operation's preflight, without mocking admission."""
        if self.runtime is None:return method(*args,clock=clock,**kwargs)
        owner=type(method.__self__);name=method.__name__;tz=kwargs.get('timezone_file',TZ)
        if name=='initialize':guard=validation_settlements.ValidationClock(tz,clock,initial=True)
        elif owner is validation_issuer.ValidationIssueJournal and name=='issue_once':
            guard=validation_issuer.IssueClock(args[0],tz,clock)
        elif owner is validation_settlements.ValidationSettlementJournal and name=='settle_once':
            guard=validation_settlements.ValidationClock(tz,clock,key=args[0])
        elif owner is validation_scores.ValidationScoreJournal and name=='freeze_once':
            guard=validation_scores.FreezeClock(tz,clock)
        elif owner is validation_scores.ValidationScoreJournal and name=='load_frozen':
            guard=validation_settlements.ValidationClock(tz,clock,allow_after_test=True)
        elif owner is selection_journal.SelectionJournal and name=='select_once':
            guard=test_protocol.SelectionClock(tz,clock)
        elif owner is test_issue_journal.TestIssueJournal and name=='issue_once':
            guard=test_protocol.TestIssueClock(args[0],tz,clock)
        elif owner is test_settlements.TestSettlementJournal and name=='settle_once':
            guard=test_protocol.TestOutcomeClock(args[0],tz,clock)
        elif owner is test_scores.TestScoreJournal and name in ('freeze_once','load_frozen'):
            guard=test_protocol.TestFreezeClock(tz,clock)
        elif ((owner is selection_journal.SelectionJournal and name=='load_current') or
              (owner in (validation_issuer.ValidationIssueJournal,test_issue_journal.TestIssueJournal) and name=='load_issued')):
            base=validation_issuer.canonical.Clock(tz,clock)
            if owner is selection_journal.SelectionJournal:boundary=test_protocol.SELECTION_AFTER;strict=True
            else:
                boundary=test_protocol._test_key(args[0]) if owner is test_issue_journal.TestIssueJournal else aware(args[0].split('|',1)[1])
                strict=False
            def guard():
                at=base()
                if not (at>boundary if strict else at>=boundary):raise ValueError('original phase preflight refused')
                return at
        else:raise ValueError('unsupported original phase operation')
        return self.runtime.call(guard,method,*args,**kwargs)


class OriginalPhaseInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_tz=zoneinfo.TZPATH;zoneinfo.reset_tzpath([str(ROOT/'outputs/zoneinfo')]);zoneinfo.ZoneInfo.clear_cache()
        cls.addClassCleanup(zoneinfo.reset_tzpath,cls.previous_tz);cls.addClassCleanup(zoneinfo.ZoneInfo.clear_cache)
        cls.stack=ExitStack();cls.addClassCleanup(cls.stack.close)
        cls.stack.enter_context(patch.dict(os.environ,{},clear=True))
        cls.stack.enter_context(patch.object(socket,'create_connection',side_effect=AssertionError('no network')))
        cls.stack.enter_context(patch.object(socket,'socket',side_effect=AssertionError('no network')))
        cls.temp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        cls.c=PhaseWorld(Path(cls.temp.name))

    def test_all_original_headers_bind_actual_complete_protocol_before_validation(self):
        c=self.c
        self.assertEqual(c.w.registration.plan()['configuration_sha256'],c.policy.configuration_sha256)
        self.assertEqual(c.policy.prospective_protocol_sha256,test_protocol.source_sha256())
        for book in (c.validation_settlements,c.validation_scores,c.selection,c.test_issues,c.test_settlements,c.test_scores):
            state=book.snapshot();self.assertTrue(state['initialized']);self.assertEqual(state['head_seq'],2)
            self.assertLess(state['initialized_at'],common.FIRST_OPEN)
        self.assertEqual(c.w.training.snapshot()['reserved_trial_units'],15)

    def test_failed_validation_is_confirmed_once_but_cannot_open_test_prices(self):
        c=self.c;c.clock=Tick(aware('2026-12-04T21:03:00+00:00'))
        c.validation_scores.freeze_once(timezone_file=TZ,clock=c.clock)
        state=c.selection.select_once(timezone_file=TZ,clock=c.clock)
        self.assertIsNone(state['domain']['decision']['selected_family']);self.assertEqual(state['head_seq'],4)
        self.assertEqual(state['domain']['decision']['reason'],'engineering_gate_failed')
        with self.assertRaises(selection_journal.SelectionError):c.selection.load_current(timezone_file=TZ,clock=c.clock)
        with patch.object(common_outcome,'observe_common_outcome',side_effect=AssertionError('no admission, no outcomes')) as observe:
            with self.assertRaises(test_issue_journal.TestIssueError):
                c.test_issues.issue_once('AAPL|2026-12-07T16:45:00+00:00',selection_journal=c.selection,
                    timezone_file=TZ,clock=Tick(aware('2026-12-07T16:45:01+00:00')))
            with self.assertRaises(test_settlements.TestSettlementError):
                c.test_settlements.settle_once('AAPL|2026-12-07T16:45:00+00:00',
                    timezone_file=TZ,clock=Tick(aware('2026-12-07T17:15:30+00:00')))
            observe.assert_not_called()
        self.assertEqual(c.test_issues.snapshot()['head_seq'],2);self.assertEqual(c.test_settlements.snapshot()['head_seq'],2)
        self.assertEqual(c.selection.select_once(timezone_file=TZ,clock=c.clock)['head_seq'],4)

    def test_early_test_receipt_read_does_not_open_sql(self):
        c=self.c;before_test=Tick(HANDOFF+timedelta(minutes=2))
        with patch('sqlite3.connect',side_effect=AssertionError('future test receipt must precede SQL')) as connect:
            with self.assertRaises(test_issue_journal.TestIssueError):
                c.test_issues.load_issued('AAPL|2026-12-07T16:45:00+00:00',timezone_file=TZ,clock=before_test)
            connect.assert_not_called()

    def test_no_copied_current_history_or_after_open_header_creation(self):
        c=self.c
        for book in (c.selection,c.test_issues,c.test_settlements,c.test_scores):
            copy=Path(self.temp.name)/('copy-'+book.path.name);shutil.copy2(book.path,copy)
            with self.assertRaises(Exception):type(book)(copy).snapshot()
        with patch('sqlite3.connect',side_effect=AssertionError('phase before original SQL')) as connect:
            with self.assertRaises(test_scores.TestScoreError):
                c.test_scores.initialize(c.test_settlements,timezone_file=TZ,clock=lambda:common.FIRST_OPEN)
            connect.assert_not_called()

    def test_original_outcome_wal_only_commit_during_current_selection_is_rejected(self):
        c=self.c;path=c.w.f.outcomes.path;original=selection_journal._frozen;changed=[]
        with closing(sqlite3.connect(path,isolation_level=None)) as keep:
            self.assertEqual(keep.execute('PRAGMA journal_mode=WAL').fetchone()[0].lower(),'wal')
            def concurrent(*args,**kwargs):
                value=original(*args,**kwargs);stat=path.stat();before=(stat.st_size,stat.st_mtime_ns)
                keep.execute('BEGIN IMMEDIATE');keep.execute('UPDATE metadata SET started_at=started_at+1')
                keep.execute('UPDATE metadata SET started_at=started_at-1');keep.commit()
                stat=path.stat();changed.append(before==(stat.st_size,stat.st_mtime_ns))
                c.w.f.outcomes.snapshot()  # Same values and valid chain, new data_version.
                return value
            with patch.object(selection_journal,'_frozen',side_effect=concurrent):
                with self.assertRaises(selection_journal.SelectionError):c.selection.select_once(timezone_file=TZ,clock=c.clock)
            self.assertEqual(changed,[True])
        self.assertEqual(c.selection.snapshot()['head_seq'],4)

    def test_original_outcome_path_replacement_after_report_read_is_rejected(self):
        c=self.c;path=c.w.f.outcomes.path;copy=path.with_name('replacement-source.sqlite')
        saved=path.with_name('held-original-source.sqlite');shutil.copy2(path,copy)
        frozen=selection_journal._frozen;identity=validation_issuer.inputs._identity;mode=[]
        def replaced(*args,**kwargs):
            value=frozen(*args,**kwargs)
            try:
                os.replace(path,saved);os.replace(copy,path);mode.append('native')
            except PermissionError:mode.append('simulated-sharing-restriction')
            return value
        def changed(current_path):
            value=identity(current_path)
            if mode==['simulated-sharing-restriction'] and current_path==path:return value[0],value[1]+1,*value[2:]
            return value
        try:
            with (patch.object(selection_journal,'_frozen',side_effect=replaced),
                  patch.object(validation_issuer.inputs,'_identity',side_effect=changed)):
                with self.assertRaises(selection_journal.SelectionError):c.selection.select_once(timezone_file=TZ,clock=c.clock)
            self.assertEqual(len(mode),1)
            if os.name!='nt':self.assertEqual(mode,['native'])
        finally:
            if saved.exists():os.replace(saved,path)  # Restore the original temporary inode for the next test.
        self.assertEqual(c.selection.snapshot()['head_seq'],4)

    def test_z_original_training_outcome_corruption_vetoes_current_selection(self):
        c=self.c;path=c.w.f.outcomes.path
        with closing(sqlite3.connect(path,isolation_level=None)) as db:
            db.execute('UPDATE observations SET payload=payload||? WHERE seq=(SELECT MAX(seq) FROM observations)',(' ',))
        with patch.object(common_outcome,'observe_common_outcome',side_effect=AssertionError('no new labels')) as observe:
            with self.assertRaises(selection_journal.SelectionError):c.selection.select_once(timezone_file=TZ,clock=c.clock)
            observe.assert_not_called()
        self.assertEqual(c.selection.snapshot()['head_seq'],4)


class CachedOriginalPhaseTests(unittest.TestCase):
    """Real original-book authority with cached mathematical decoding only."""
    def test_cached_original_lifecycle_keeps_current_source_and_phase_vetoes(self):
        import inspect
        self.assertIn('use_replay_cache',inspect.signature(PhaseWorld).parameters)
        self.assertIn('replay_cache.py',test_protocol.PROTOCOL_FILES)
        previous_tz=zoneinfo.TZPATH;zoneinfo.reset_tzpath([str(ROOT/'outputs/zoneinfo')]);zoneinfo.ZoneInfo.clear_cache()
        self.addCleanup(zoneinfo.reset_tzpath,previous_tz);self.addCleanup(zoneinfo.ZoneInfo.clear_cache)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        originals=(forecast_inputs.decode_input,candidate_forecasts.decode_panel)
        with (patch.dict(os.environ,{},clear=True),
              patch.object(socket,'socket',side_effect=AssertionError('no network')),
              patch.object(socket,'create_connection',side_effect=AssertionError('no network'))):
            c=PhaseWorld(Path(temp.name),use_replay_cache=True)
            self.assertEqual(c.policy.prospective_protocol_sha256,test_protocol.source_sha256())
            self.assertEqual(c.runtime.stats()['runtime_code_sha256'],test_protocol.source_sha256())
            for book in (c.selection,c.test_issues,c.test_settlements,c.test_scores):
                self.assertEqual(book.snapshot()['head_seq'],2)
            key=c.w.receive_until('2026-11-24',9);c.clock=Tick(c.w.at)
            c.call(c.issues.issue_once,key,registry=c.w.registry,training_journal=c.w.training,
                   pairs=c.w.source_pairs(),timezone_file=TZ,clock=c.clock)
            first=c.call(c.issues.load_issued,key,clock=c.clock)
            second=c.call(c.issues.load_issued,key,clock=c.clock)
            self.assertEqual(first.panel.seal,second.panel.seal)
            self.assertEqual(first.issue_input.seal,second.issue_input.seal)
            self.assertEqual(first.commit_observed_at,second.commit_observed_at)
            self.assertGreaterEqual(second.available_at,first.available_at)
            c.w.receive_until('2026-11-24',11);c.clock=Tick(c.w.at)
            settled=c.call(c.validation_settlements.settle_once,key,timezone_file=TZ,clock=c.clock)
            self.assertEqual(settled['status'],'confirmed');self.assertFalse(settled['effective_late'])
            c.clock=Tick(aware('2026-12-04T21:03:00+00:00'))
            frozen=c.call(c.validation_scores.freeze_once,timezone_file=TZ,clock=c.clock)
            self.assertFalse(frozen['domain']['report']['engineering_gate_passed'])
            decision=c.call(c.selection.select_once,timezone_file=TZ,clock=c.clock)
            self.assertIsNone(decision['domain']['decision']['selected_family'])
            self.assertEqual(decision['head_seq'],4)
            self.assertGreater(c.runtime.stats()['hits'],0)
            for kind in ('training_snapshot','training_fits'):
                self.assertGreater(c.runtime.stats()['kind_hits'][kind],0)
            self.assertLessEqual(c.runtime.stats()['charged_bytes'],replay_cache.MAX_BYTES)
            self.assertEqual((forecast_inputs.decode_input,candidate_forecasts.decode_panel),originals)
            test_key='AAPL|2026-12-07T16:45:00+00:00'
            with patch('sqlite3.connect',side_effect=AssertionError('future phase before SQL')) as sql:
                with self.assertRaises(ValueError):
                    c.call(c.test_issues.load_issued,test_key,timezone_file=TZ,clock=c.clock)
                sql.assert_not_called()
            with self.assertRaises(test_issue_journal.TestIssueError):
                c.call(c.test_issues.issue_once,test_key,selection_journal=c.selection,timezone_file=TZ,
                       clock=Tick(aware('2026-12-07T16:45:01+00:00')))
            self.assertEqual(c.test_issues.snapshot()['head_seq'],2)
            # Repopulate only mathematical records, then corrupt a real watched
            # original. The next current selection must refuse and discard it.
            c.runtime.memo.decode('panel',candidate_forecasts.codec.encode(first.panel))
            self.assertGreater(c.runtime.stats()['entries'],0)
            with closing(sqlite3.connect(c.w.f.outcomes.path,isolation_level=None)) as db:
                db.execute('UPDATE observations SET payload=payload||? WHERE seq=(SELECT MAX(seq) FROM observations)',(' ',))
            with self.assertRaises(selection_journal.SelectionError):
                c.call(c.selection.select_once,timezone_file=TZ,clock=c.clock)
            self.assertEqual(c.runtime.stats()['entries'],0)
            self.assertEqual((forecast_inputs.decode_input,candidate_forecasts.decode_panel),originals)
            self.assertEqual(c.selection.snapshot()['head_seq'],4)


class ForwardAllSymbolsTraining(Fixture):
    """192 genuine on-time synthetic labels:64for EACH declared symbol."""
    def __init__(self,directory):
        self.directory=directory;self.sessions=tuple(issue_fixture.IssueWorld.session(day) for day in DAYS)
        self.before=self.sessions[0].opens-timedelta(minutes=30)
        raw_plan=receipt_book.ReceiptPlan(SYMBOLS,DAYS[0],'2026-11-25',validation_issuer.inputs.TIMEZONE_SHA256)
        self.raw=receipt_book.ReceiptBook(directory/'raw.sqlite');self.raw.initialize(raw_plan,now=self.before)
        calendar=Calendar(list(self.sessions),start=DAYS[0],end='2026-11-25')
        for day in DAYS:self.raw.record_calendar(calendar,day=day,received_at=self.before)
        paused=tuple((aware('2026-11-10T00:00:00+00:00')+timedelta(days=i)).date().isoformat() for i in range(14))
        plan=feature_pipeline.PipelinePlan(SYMBOLS,DAYS[0],'2026-11-25',digest(asdict(raw_plan)),
            receipt_book._source(),raw_plan.timezone_sha256,paused_days=paused)
        self.features=feature_pipeline.FeatureJournal(directory/'features.sqlite');self.features.initialize(plan,now=self.before)
        self.outcomes=common_outcome_journal.OutcomeJournal(directory/'common.sqlite')
        self.outcomes.initialize(self.features,self.raw,clock=lambda:self.before);self.at=self.before
        for session in self.sessions[:4]:
            for offset in range(26):
                bars=[self.bar(session,offset,symbol) for symbol in SYMBOLS]
                self.raw.observe(bars,now=bars[0].received_at+timedelta(seconds=1))
                self.at=bars[0].received_at+timedelta(seconds=2)
                feature_pipeline.derive_once(self.features,raw_book=self.raw,clock=Tick(self.at))
                clock=Tick(self.at+timedelta(seconds=1))
                if offset>=10:
                    for symbol in SYMBOLS:
                        key=feature_pipeline._entry_key(symbol,session.opens+timedelta(seconds=900*(offset-1)))
                        self.outcomes.record_first(self.features,self.raw,key,clock=clock)
                self.at=clock.value+timedelta(seconds=1)
                if offset%4==3 or offset==25:
                    print('synthetic training progress: day='+session.day.isoformat()+' completed_bars='+str(offset+1)+'/26',flush=True)
            print('synthetic training session completed: '+session.day.isoformat(),flush=True)
        self.keys=tuple(self.features.snapshot()['entries'])


class ForwardPhaseMarket(issue_fixture.IssueWorld):
    """Same stationary synthetic bar law in training/validation/test, no tuning."""
    def receive_until(self,day,count,*,omit=(),derive=True):
        session=self.session(day);raw,features=self.pair(day)
        if day not in raw.snapshot()['calendars']:
            raw.record_calendar(Calendar([session],start=day,end=day),day=day,received_at=session.opens-timedelta(minutes=30))
        seen=self.fed.setdefault(day,set())
        for offset in range(count):
            if offset in seen:continue
            seen.add(offset)
            if offset in omit:continue
            bars=[Fixture.bar(self,session,offset,symbol) for symbol in SYMBOLS]
            raw.observe(bars,now=bars[0].received_at+timedelta(seconds=1));self.at=bars[0].received_at+timedelta(seconds=2)
        if derive:feature_pipeline.derive_once(features,raw_book=raw,clock=Tick(self.at))
        self.at+=timedelta(seconds=1)
        return feature_pipeline._entry_key('AAPL',session.opens+timedelta(seconds=900*count))


class OriginalPositivePhaseTests(unittest.TestCase):
    """Full original-source comparison, without mocked admission or fits."""
    def test_full_validation_single_selection_and_prospective_test_report(self):
        previous_tz=zoneinfo.TZPATH;zoneinfo.reset_tzpath([str(ROOT/'outputs/zoneinfo')]);zoneinfo.ZoneInfo.clear_cache()
        self.addCleanup(zoneinfo.reset_tzpath,previous_tz);self.addCleanup(zoneinfo.ZoneInfo.clear_cache)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        with (patch.dict(os.environ,{},clear=True),
              patch.object(socket,'create_connection',side_effect=AssertionError('no network')),
              patch.object(socket,'socket',side_effect=AssertionError('no network'))):
            c=PhaseWorld(Path(temp.name),training_fixture=ForwardAllSymbolsTraining,market_class=ForwardPhaseMarket,
                         use_replay_cache=True)
            trained=c.w.training_receipt
            self.assertEqual(len(trained.fits.attempts),15)
            self.assertTrue(all(a.status=='succeeded' for a in trained.fits.attempts))
            for source in trained.training.symbols:
                self.assertEqual(len(source.examples),64)
            print('synthetic all15fixed fits and all original headers verified',flush=True)
            for days,test_phase in ((common.VALIDATION_DAYS,False),(common.TEST_DAYS,True)):
                for day in days:
                    session=c.w.session(day);bar_count=int((session.closes-session.opens).total_seconds()/900)
                    for completed in range(1,bar_count+1):
                        batch_started=time.monotonic()
                        c.w.receive_until(day,completed);c.clock=Tick(c.w.at)
                        if completed>=11:
                            target_event=session.opens+timedelta(seconds=900*(completed-2))
                            for symbol in SYMBOLS:
                                key=feature_pipeline._entry_key(symbol,target_event)
                                book=c.test_settlements if test_phase else c.validation_settlements
                                result=c.call(book.settle_once,key,timezone_file=TZ,clock=c.clock)
                                self.assertEqual(result['status'],'confirmed');self.assertFalse(result['effective_late'])
                        if 9<=completed<=bar_count-2:
                            event=session.opens+timedelta(seconds=900*completed)
                            for symbol in SYMBOLS:
                                key=feature_pipeline._entry_key(symbol,event)
                                if test_phase:
                                    c.call(c.test_issues.issue_once,key,selection_journal=c.selection,timezone_file=TZ,clock=c.clock)
                                else:
                                    c.call(c.issues.issue_once,key,registry=c.w.registry,training_journal=c.w.training,
                                        pairs=c.w.source_pairs(),timezone_file=TZ,clock=c.clock)
                        if completed>=9:
                            print('synthetic '+('test' if test_phase else 'validation')+' progress: day='+day+
                                ' completed_bars='+str(completed)+'/'+str(bar_count)+
                                ' batch_seconds='+format(time.monotonic()-batch_started,'.3f')+
                                ' cache_hits='+str(c.runtime.stats()['hits'])+
                                ' cache_evictions='+str(c.runtime.stats()['evictions'])+
                                ' cache_charged_bytes='+str(c.runtime.stats()['charged_bytes']),flush=True)
                    print('synthetic '+('test' if test_phase else 'validation')+' session completed: '+day,flush=True)
                if not test_phase:
                    c.clock=Tick(aware('2026-12-04T21:03:00+00:00'))
                    state=c.call(c.validation_scores.freeze_once,timezone_file=TZ,clock=c.clock);report=state['domain']['report']
                    self.assertTrue(report['engineering_gate_passed'])
                    self.assertEqual(sum(g['counts']['planned'] for g in report['groups']),348)
                    self.assertEqual(sum(g['counts']['common'] for g in report['groups']),348)
                    c.call(c.selection.select_once,timezone_file=TZ,clock=c.clock)
                    admission=c.call(c.selection.load_current,timezone_file=TZ,clock=c.clock)
                    self.assertEqual(admission.selected_family,report['proposed_family']);self.assertFalse(admission.orders)
                    # Genuine admitted sources, not a scalar/admission mock:
                    # even a value-preserving WAL commit must end this epoch.
                    original_frozen=selection_journal._frozen;changes=[]
                    with closing(sqlite3.connect(c.w.f.outcomes.path,isolation_level=None)) as writer:
                        self.assertEqual(writer.execute('PRAGMA journal_mode=WAL').fetchone()[0].lower(),'wal')
                        def race(*args,**kwargs):
                            value=original_frozen(*args,**kwargs);before=c.w.f.outcomes.path.stat()
                            writer.execute('BEGIN IMMEDIATE');writer.execute('UPDATE metadata SET started_at=started_at+1')
                            writer.execute('UPDATE metadata SET started_at=started_at-1');writer.commit()
                            after=c.w.f.outcomes.path.stat()
                            changes.append((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns))
                            return value
                        with patch.object(selection_journal,'_frozen',side_effect=race):
                            with self.assertRaises(selection_journal.SelectionError):
                                c.call(c.selection.load_current,timezone_file=TZ,clock=c.clock)
                    self.assertEqual(changes,[True]);self.assertEqual(c.runtime.stats()['entries'],0)
                    recovered=c.call(c.selection.load_current,timezone_file=TZ,clock=c.clock)
                    self.assertEqual(recovered.selected_family,admission.selected_family)
                    self.assertEqual(c.selection.snapshot()['head_seq'],4)
                    with patch('sqlite3.connect',side_effect=AssertionError('admitted future test still precedes SQL')) as sql:
                        with self.assertRaises(ValueError):
                            c.call(c.test_issues.load_issued,'AAPL|2026-12-07T16:45:00+00:00',timezone_file=TZ,clock=c.clock)
                        with self.assertRaises(ValueError):
                            c.call(c.test_scores.freeze_once,timezone_file=TZ,clock=c.clock)
                        sql.assert_not_called()
                    print('synthetic validation348common and original single selection verified',flush=True)
            c.clock=Tick(aware('2026-12-11T21:03:00+00:00'))
            state=c.call(c.test_scores.freeze_once,timezone_file=TZ,clock=c.clock);report=state['domain']['report']
            self.assertEqual(report['selected_family'],admission.selected_family);self.assertIsNone(report['proposed_family'])
            self.assertTrue(report['engineering_gate_passed'])
            self.assertEqual(sum(g['counts']['common'] for g in report['groups']),240)
            with (patch.object(common_outcome,'observe_common_outcome',side_effect=AssertionError('no new test prices')),
                  patch.object(test_protocol,'evaluate_test_frames',side_effect=AssertionError('no second test comparison')),
                  patch.object(receipt_book.ReceiptBook,'_transaction',side_effect=AssertionError('opaque source checks only'))):
                first=c.call(c.test_scores.load_frozen,timezone_file=TZ,clock=c.clock)
                # Independent original decoder replay: no wrapper/runtime is
                # active, and no numerical source/refit/rescore is permitted.
                self.assertIsNone(replay_cache._ACTIVE_SCOPE)
                original=c.test_scores.load_frozen(timezone_file=TZ,clock=c.clock)
                self.assertEqual(first.report_json,original.report_json)
                c.clock=Tick(c.clock.value+timedelta(days=1))
                later=c.call(c.test_scores.load_frozen,timezone_file=TZ,clock=c.clock)
            self.assertEqual(first.report_json,later.report_json);self.assertFalse(first.promotion);self.assertFalse(first.orders)
            self.assertEqual(c.w.training.snapshot()['reserved_trial_units'],15)
            self.assertEqual(c.runtime.stats()['runtime_code_sha256'],c.policy.prospective_protocol_sha256)
            self.assertGreater(c.runtime.stats()['hits'],0)
            self.assertLessEqual(c.runtime.stats()['charged_bytes'],replay_cache.MAX_BYTES)
            print('synthetic replay cache summary: '+json.dumps(c.runtime.stats(),sort_keys=True),flush=True)
            copied=c.test_scores.path.with_name('copied-final-test-report.sqlite');shutil.copy2(c.test_scores.path,copied)
            with self.assertRaises(test_scores.TestScoreError):
                c.call(test_scores.TestScoreJournal(copied).load_frozen,timezone_file=TZ,clock=c.clock)
            with patch('sqlite3.connect',side_effect=AssertionError('test outcomes cannot reopen selection')) as sql:
                with self.assertRaises(ValueError):c.call(c.selection.select_once,timezone_file=TZ,clock=c.clock)
                sql.assert_not_called()
            with closing(sqlite3.connect(c.w.f.outcomes.path,isolation_level=None)) as db:
                db.execute('UPDATE observations SET payload=payload||? WHERE seq=(SELECT MAX(seq) FROM observations)',(' ',))
            with self.assertRaises(test_scores.TestScoreError):
                c.call(c.test_scores.load_frozen,timezone_file=TZ,clock=c.clock)
            self.assertEqual(c.runtime.stats()['entries'],0);self.assertIsNone(replay_cache._ACTIVE_SCOPE)
            print('synthetic test240common and immutable final report verified; no market evidence',flush=True)


if __name__=='__main__':unittest.main()
