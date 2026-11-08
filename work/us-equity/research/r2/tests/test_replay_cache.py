"""Pure immutable replay memo only; no current source/phase authority."""
from dataclasses import asdict
from datetime import timedelta
import json
import hashlib
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import zoneinfo

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[3]),str(Path(__file__).resolve().parents[1])]
from issue_fixture import IssueWorld,Tick,TZ
from test_training_data import ROOT
from paperbot_r2 import forecast_inputs as inputs,candidate_forecasts as panels,training_codec
from paperbot_r2.contracts import aware
try:
    from paperbot_r2 import replay_cache
except ImportError:
    replay_cache=None


class ActualFileGuard:
    """Private real-file epoch for scope primitives, NEVER phase admission."""
    def __init__(self,path):
        self.path=path;self.expected=hashlib.sha256(path.read_bytes()).hexdigest();self.calls=0;self.failed=False
    def check(self):
        self.calls+=1
        if self.failed or hashlib.sha256(self.path.read_bytes()).hexdigest()!=self.expected:
            self.failed=True;raise replay_cache.ReplayCacheError('private_code_epoch_changed')


class ImmutableReplayMemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if replay_cache is None:return
        old=zoneinfo.TZPATH;zoneinfo.reset_tzpath([str(ROOT/'outputs/zoneinfo')]);zoneinfo.ZoneInfo.clear_cache()
        cls.addClassCleanup(zoneinfo.reset_tzpath,old);cls.addClassCleanup(zoneinfo.ZoneInfo.clear_cache)
        cls.temp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        with patch.dict(os.environ,{},clear=True),patch.object(socket,'socket',side_effect=AssertionError('no network')):
            w=IssueWorld(Path(cls.temp.name));key=w.receive_until('2026-12-07',9)
            captured=inputs.capture_issue_input(w.training_receipt,w.registration,w.source_pairs(),key,timezone_file=TZ,clock=Tick(w.at))
            panel=panels.compute_candidates(w.training_receipt,captured,clock=Tick(w.at+timedelta(seconds=1)))
        cls.input_json=inputs.encode_input(captured);cls.panel_json=training_codec.encode(panel)
        cls.input_seal=captured.seal;cls.panel_seal=panel.seal
    def setUp(self):
        self.assertIsNotNone(replay_cache,'immutable replay memo not implemented')
        self.memo=replay_cache._DecodeMemo()

    def test_repeated_exact_inputs_and_panels_keep_original_seals(self):
        for _ in range(3):
            self.assertEqual(self.memo.decode('input',self.input_json).seal,self.input_seal)
            self.assertEqual(self.memo.decode('panel',self.panel_json).seal,self.panel_seal)
        stats=self.memo.stats();self.assertEqual(stats['hits'],4);self.assertEqual(stats['misses'],2)
        self.assertEqual(stats['entries'],2);self.assertFalse(stats['phase_authority'])
        self.assertLessEqual(stats['charged_bytes'],stats['byte_budget'])

    def test_returned_objects_cannot_poison_the_private_cached_value(self):
        first=self.memo.decode('input',self.input_json)
        object.__setattr__(first,'symbol','SPY')
        second=self.memo.decode('input',self.input_json)
        self.assertIsNot(first,second);self.assertEqual(second.symbol,'AAPL');second.__post_init__()
        object.__setattr__(second.bundle.vector,'values',tuple(0. for _ in second.bundle.vector.values))
        third=self.memo.decode('input',self.input_json)
        self.assertEqual(third.seal,self.input_seal);third.__post_init__()
        self.assertNotEqual(third.bundle.vector.values,second.bundle.vector.values)

    def test_changed_payload_is_validated_instead_of_reusing_an_old_record(self):
        self.memo.decode('input',self.input_json)
        bad=json.loads(self.input_json);bad['source_sha256']='0'*64
        with self.assertRaises(ValueError):self.memo.decode('input',json.dumps(bad))
        self.assertEqual(self.memo.decode('input',self.input_json).seal,self.input_seal)
        self.assertEqual(self.memo.stats()['entries'],1)

    def test_nonstring_objects_are_not_retained_and_mutation_is_revalidated(self):
        value=json.loads(self.input_json)
        self.assertEqual(self.memo.decode('input',value).seal,self.input_seal)
        self.assertEqual(self.memo.stats()['entries'],0)
        value['symbol']='SPY'
        with self.assertRaises(ValueError):self.memo.decode('input',value)
        self.assertEqual(self.memo.stats()['entries'],0)

    def test_unknown_kinds_and_invalid_budgets_are_refused(self):
        with self.assertRaises(ValueError):self.memo.decode('admission',self.input_json)
        for options in (dict(max_bytes=0),dict(max_items=0),dict(max_bytes=True),dict(max_items=True),
                        dict(max_bytes=33554433),dict(max_items=2049)):
            with self.assertRaises(ValueError):replay_cache._DecodeMemo(**options)

    def test_oversized_values_decode_normally_without_retention(self):
        memo=replay_cache._DecodeMemo(max_bytes=1)
        for _ in range(2):self.assertEqual(memo.decode('input',self.input_json).seal,self.input_seal)
        self.assertEqual(memo.stats()['entries'],0);self.assertEqual(memo.stats()['hits'],0)

    def test_lru_eviction_and_clear_change_only_decoding_cost(self):
        memo=replay_cache._DecodeMemo(max_items=1)
        a=memo.decode('input',self.input_json);memo.decode('panel',self.panel_json);b=memo.decode('input',self.input_json)
        self.assertEqual(a.seal,b.seal);self.assertEqual(memo.stats()['misses'],3)
        self.assertEqual(memo.stats()['entries'],1)
        memo.clear();self.assertEqual(memo.stats()['entries'],0);self.assertEqual(memo.stats()['charged_bytes'],0)
        self.assertEqual(memo.decode('panel',self.panel_json).seal,self.panel_seal)

    def test_codec_scope_can_delegate_without_recursive_lookup(self):
        with (patch.object(inputs,'decode_input',side_effect=lambda value:self.memo.decode('input',value)),
              patch.object(panels,'decode_panel',side_effect=lambda value:self.memo.decode('panel',value))):
            self.assertEqual(inputs.decode_input(self.input_json).seal,self.input_seal)
            self.assertEqual(panels.decode_panel(self.panel_json).seal,self.panel_seal)

    def scope_parts(self,clock=None):
        self.assertTrue(hasattr(replay_cache,'_CodecScope'),'protected codec scope not implemented')
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        path=Path(temp.name)/'actual-code.py';path.write_bytes(b'ALPHA')
        guard=ActualFileGuard(path);clock=clock or Tick(aware('2026-12-07T16:45:01+00:00'))
        return path,guard,clock

    def test_scope_reuses_only_data_and_restores_original_functions(self):
        _,guard,clock=self.scope_parts();original_input=inputs.decode_input;original_panel=panels.decode_panel
        with replay_cache._CodecScope(self.memo,guard,clock) as scope:
            self.assertIsNot(inputs.decode_input,original_input)
            for _ in range(2):self.assertEqual(inputs.decode_input(self.input_json).seal,self.input_seal)
            self.assertEqual(panels.decode_panel(self.panel_json).seal,self.panel_seal)
            scope.clock()
        self.assertIs(inputs.decode_input,original_input);self.assertIs(panels.decode_panel,original_panel)
        self.assertGreaterEqual(self.memo.stats()['hits'],1);self.assertGreaterEqual(guard.calls,3)

    def test_scope_exception_clears_data_and_restores_original_functions(self):
        _,guard,clock=self.scope_parts();original=inputs.decode_input
        with self.assertRaisesRegex(RuntimeError,'private body failure'):
            with replay_cache._CodecScope(self.memo,guard,clock):
                inputs.decode_input(self.input_json);raise RuntimeError('private body failure')
        self.assertIs(inputs.decode_input,original);self.assertEqual(self.memo.stats()['entries'],0)

    def test_restored_mtime_code_edit_is_refused_at_next_clock(self):
        path,guard,clock=self.scope_parts();original=inputs.decode_input
        with self.assertRaises(replay_cache.ReplayCacheError):
            with replay_cache._CodecScope(self.memo,guard,clock) as scope:
                inputs.decode_input(self.input_json);before=path.stat();path.write_bytes(b'BRAVO')
                os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns));scope.clock()
        self.assertIs(inputs.decode_input,original);self.assertEqual(self.memo.stats()['entries'],0)

    def test_final_source_check_refuses_change_without_another_clock(self):
        path,guard,clock=self.scope_parts();original=panels.decode_panel
        with self.assertRaises(replay_cache.ReplayCacheError):
            with replay_cache._CodecScope(self.memo,guard,clock):
                panels.decode_panel(self.panel_json);path.write_bytes(b'changed')
        self.assertIs(panels.decode_panel,original);self.assertEqual(self.memo.stats()['entries'],0)

    def test_nested_scope_refusal_does_not_destroy_the_outer_scope(self):
        _,guard,clock=self.scope_parts();original=inputs.decode_input
        with replay_cache._CodecScope(self.memo,guard,clock) as outer:
            installed=inputs.decode_input
            with self.assertRaises(replay_cache.ReplayCacheError):
                with replay_cache._CodecScope(self.memo,guard,clock):self.fail('nested owner admitted')
            self.assertIs(inputs.decode_input,installed);outer.clock()
            self.assertEqual(inputs.decode_input(self.input_json).seal,self.input_seal)
        self.assertIs(inputs.decode_input,original)

    def test_foreign_thread_cannot_read_cached_objects(self):
        _,guard,clock=self.scope_parts();errors=[];original=inputs.decode_input
        with self.assertRaises(replay_cache.ReplayCacheError):
            with replay_cache._CodecScope(self.memo,guard,clock) as scope:
                inputs.decode_input(self.input_json)
                def foreign():
                    try:inputs.decode_input(self.input_json)
                    except Exception as error:errors.append(type(error))
                thread=threading.Thread(target=foreign);thread.start();thread.join()
                self.assertEqual(errors,[replay_cache.ReplayCacheError]);scope.clock()
        self.assertIs(inputs.decode_input,original);self.assertEqual(self.memo.stats()['entries'],0)

    def test_replaced_codec_is_refused_and_known_original_is_restored(self):
        _,guard,clock=self.scope_parts();original=inputs.decode_input
        with self.assertRaises(replay_cache.ReplayCacheError):
            with replay_cache._CodecScope(self.memo,guard,clock) as scope:
                inputs.decode_input=lambda value:None;scope.clock()
        self.assertIs(inputs.decode_input,original)

    def test_clock_returns_time_after_fresh_code_validation(self):
        _,guard,clock=self.scope_parts();real_check=guard.check
        def advances_time():real_check();clock.value+=timedelta(seconds=1)
        guard.check=advances_time
        with replay_cache._CodecScope(self.memo,guard,clock) as scope:
            before=clock.value;observed=scope.clock()
            self.assertGreaterEqual(observed,before+timedelta(seconds=1))

    def test_clock_crossing_deadline_during_code_validation_is_refused(self):
        _,guard,_=self.scope_parts();now=aware('2026-12-07T16:45:00+00:00');deadline=now+timedelta(seconds=120)
        value=[now]
        def phase_clock():
            if value[0]>deadline:raise ValueError('private phase deadline')
            return value[0]
        with self.assertRaises(ValueError):
            with replay_cache._CodecScope(self.memo,guard,phase_clock) as scope:
                actual=guard.check
                def crosses():actual();value[0]=deadline+timedelta(seconds=1)
                guard.check=crosses;scope.clock()

    def test_code_change_after_private_commit_cannot_be_backdated(self):
        from contextlib import closing
        import sqlite3
        path,guard,clock=self.scope_parts();database=path.with_name('private-commit.sqlite')
        with closing(sqlite3.connect(database,isolation_level=None)) as db:
            db.execute('CREATE TABLE observation(value INTEGER)')
            with self.assertRaises(replay_cache.ReplayCacheError):
                with replay_cache._CodecScope(self.memo,guard,clock) as scope:
                    scope.clock();db.execute('INSERT INTO observation VALUES(1)');db.commit()
                    path.write_bytes(b'BRAVO');scope.clock()
            self.assertEqual(db.execute('SELECT COUNT(*) FROM observation').fetchone()[0],1)
        self.assertEqual(self.memo.stats()['entries'],0)

    def test_runtime_future_preflight_precedes_source_and_sql(self):
        self.assertTrue(hasattr(replay_cache,'ReplayRuntime'),'bound runtime not implemented')
        from paperbot_r2.test_issue_journal import TestIssueJournal
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);book=TestIssueJournal(root/'unused.sqlite');runtime=replay_cache.ReplayRuntime()
            def refused():raise ValueError('private phase refusal')
            with (patch('sqlite3.connect',side_effect=AssertionError('phase before SQL')) as sql,
                  patch.object(Path,'read_bytes',side_effect=AssertionError('phase before code')) as reads):
                with self.assertRaisesRegex(ValueError,'private phase refusal'):
                    runtime.call(refused,book.load_issued,'AAPL|2026-12-07T16:45:00+00:00',timezone_file=TZ)
                sql.assert_not_called();reads.assert_not_called()
            self.assertEqual(list(root.iterdir()),[])

    def test_runtime_refuses_uncommitted_helper_without_sql(self):
        self.assertTrue(hasattr(replay_cache,'ReplayRuntime'),'bound runtime not implemented')
        from paperbot_r2 import test_protocol
        from paperbot_r2.test_issue_journal import TestIssueJournal
        with tempfile.TemporaryDirectory() as temp:
            book=TestIssueJournal(Path(temp)/'unused.sqlite');runtime=replay_cache.ReplayRuntime()
            # Negative manifest mutation only: omission must still refuse after
            # this complete helper has joined the actual positive commitment.
            without_helper=tuple(name for name in test_protocol.PROTOCOL_FILES if name!='replay_cache.py')
            with (patch.object(test_protocol,'PROTOCOL_FILES',without_helper),
                  patch('sqlite3.connect',side_effect=AssertionError('uncommitted helper before SQL')) as sql):
                with self.assertRaises(replay_cache.ReplayCacheError):
                    runtime.call(Tick(aware('2026-12-07T16:45:01+00:00')),book.load_issued,
                        'AAPL|2026-12-07T16:45:00+00:00',timezone_file=TZ)
                sql.assert_not_called()

    def test_runtime_refuses_nonphase_methods_and_clock_override(self):
        self.assertTrue(hasattr(replay_cache,'ReplayRuntime'),'bound runtime not implemented')
        from paperbot_r2.test_issue_journal import TestIssueJournal
        runtime=replay_cache.ReplayRuntime();book=TestIssueJournal(Path('no-file.sqlite'))
        clock=Tick(aware('2026-12-07T16:45:01+00:00'))
        for method,extra in ((book.snapshot,{}),(book.load_issued,dict(clock=clock))):
            with self.assertRaises(replay_cache.ReplayCacheError):runtime.call(clock,method,**extra)


if __name__=='__main__':unittest.main()
