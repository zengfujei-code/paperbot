"""Historical training JSON values only, never current training authority."""
from pathlib import Path
import json
import os
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch
import zoneinfo

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[3]),
              str(Path(__file__).resolve().parents[1])]
from issue_fixture import IssueWorld,Tick
from test_training_data import ROOT
import test_replay_cache as memo_fixture
from paperbot_r2 import replay_cache,training_codec,training_data,model_training,forecast_inputs,candidate_forecasts
from paperbot_r2.contracts import aware


class TrainingDecodeMemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prior=zoneinfo.TZPATH;zoneinfo.reset_tzpath([str(ROOT/'outputs/zoneinfo')]);zoneinfo.ZoneInfo.clear_cache()
        cls.addClassCleanup(zoneinfo.reset_tzpath,prior);cls.addClassCleanup(zoneinfo.ZoneInfo.clear_cache)
        cls.temp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        with patch.dict(os.environ,{},clear=True),patch.object(socket,'socket',side_effect=AssertionError('no network')):
            world=IssueWorld(Path(cls.temp.name))
        cls.payloads={'training_snapshot':training_codec.encode(world.training_receipt.training),
                      'training_fits':training_codec.encode(world.training_receipt.fits)}
        cls.expected={'training_snapshot':training_data.TrainingSnapshot,'training_fits':model_training.TrainingFits}

    def setUp(self):
        self.memo=replay_cache._DecodeMemo()
        self.assertTrue(set(self.payloads)<=set(self.memo._decoders),'historical training codecs not memoized')

    def functions(self):
        return (forecast_inputs.decode_input,candidate_forecasts.decode_panel,
                training_codec.decode_snapshot,training_codec.decode_fits)

    def scope_parts(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        path=Path(directory.name)/'original-code.py';path.write_bytes(b'ALPHA')
        return path,memo_fixture.ActualFileGuard(path),Tick(aware('2026-12-07T16:45:01+00:00'))

    def test_two_original_training_types_seals_and_exact_serialization(self):
        for kind,value in self.payloads.items():
            for _ in range(3):
                decoded=self.memo.decode(kind,value)
                self.assertIs(type(decoded),self.expected[kind]);self.assertEqual(training_codec.encode(decoded),value)
                decoded.__post_init__()
        stats=self.memo.stats();self.assertEqual(stats['hits'],4);self.assertEqual(stats['misses'],2)
        self.assertFalse(stats['phase_authority'])
        for kind in self.payloads:self.assertEqual(stats['kind_hits'][kind],2)

    def test_nested_mutations_cannot_poison_either_private_graph(self):
        snapshot=self.memo.decode('training_snapshot',self.payloads['training_snapshot'])
        object.__setattr__(snapshot.symbols[0].returns[0],'return_fraction',.75)
        fits=self.memo.decode('training_fits',self.payloads['training_fits'])
        object.__setattr__(fits.attempts[0],'family','changed')
        for kind,value in self.payloads.items():
            decoded=self.memo.decode(kind,value);self.assertEqual(training_codec.encode(decoded),value)
            decoded.__post_init__()

    def test_changed_invalid_json_cannot_reuse_good_records(self):
        for kind,value in self.payloads.items():
            self.memo.decode(kind,value);bad=json.loads(value);bad['source_sha256']='0'*64
            with self.assertRaises(ValueError):self.memo.decode(kind,json.dumps(bad))
            with self.assertRaises(ValueError):self.memo.decode(kind,'{"seal":')
            self.assertEqual(training_codec.encode(self.memo.decode(kind,value)),value)
        self.assertEqual(self.memo.stats()['entries'],2)

    def test_nonstring_and_oversized_values_still_use_original_validation(self):
        for kind,value in self.payloads.items():
            raw=json.loads(value);raw['available_at']=aware(raw['available_at'])
            self.assertIs(type(self.memo.decode(kind,raw)),self.expected[kind])
            self.assertEqual(self.memo.stats()['entries'],0)
            raw['source_sha256']='0'*64
            with self.assertRaises(ValueError):self.memo.decode(kind,raw)
            tiny=replay_cache._DecodeMemo(max_bytes=1)
            for _ in range(2):self.assertEqual(training_codec.encode(tiny.decode(kind,value)),value)
            self.assertEqual(tiny.stats()['entries'],0);self.assertEqual(tiny.stats()['hits'],0)

    def test_real_ledger_object_payloads_use_complete_canonical_json_keys(self):
        for kind,value in self.payloads.items():
            a=self.memo.decode(kind,json.loads(value));b=self.memo.decode(kind,json.loads(value))
            self.assertIsNot(a,b);self.assertEqual(training_codec.encode(a),value)
            self.assertEqual(training_codec.encode(b),value)
            self.assertGreater(self.memo.stats()['kind_hits'][kind],0)
        self.assertEqual(self.memo.stats()['entries'],2)
        changed=json.loads(self.payloads['training_snapshot']);changed['ready']=1
        before=self.memo.stats()['hits']
        with self.assertRaises(ValueError):self.memo.decode('training_snapshot',changed)
        self.assertEqual(self.memo.stats()['hits'],before)

    def test_non_json_key_types_and_cycles_cannot_launder_a_cached_record(self):
        class Text(str):pass
        for kind,value in self.payloads.items():
            self.memo.decode(kind,json.loads(value));before=self.memo.stats()
            unusual=json.loads(value);unusual[Text('source_sha256')]=unusual.pop('source_sha256')
            with self.assertRaises(ValueError):self.memo.decode(kind,unusual)
            after=self.memo.stats();self.assertEqual(after['hits'],before['hits'])
            self.assertEqual(after['nonstring_fallbacks'],before['nonstring_fallbacks']+1)
            cycle=json.loads(value);cycle['cycle']=cycle
            with self.assertRaises(ValueError):self.memo.decode(kind,cycle)
            bad=json.loads(value);bad[1]='unexpected'
            with self.assertRaises(ValueError):self.memo.decode(kind,bad)
            self.assertEqual(self.memo.stats()['hits'],before['hits'])

    def test_key_scan_is_bounded_and_does_not_coerce_non_json_types(self):
        self.assertTrue(replay_cache._literal_json({'number':1,'float':1.,'flag':True,'rows':[None,'text']}))
        for value in ({'row':()}, {'at':aware('2026-11-06T00:00:00+00:00')}, {'x':float('inf')}, {'x':float('nan')}):
            self.assertFalse(replay_cache._literal_json(value))
        nested=None
        for _ in range(66):nested=[nested]
        self.assertFalse(replay_cache._literal_json(nested))
        self.assertFalse(replay_cache._literal_json([None]*262145))

    def test_eviction_clear_and_charged_bounds_are_shared_with_original_kinds(self):
        memo=replay_cache._DecodeMemo(max_items=1)
        for kind in ('training_snapshot','training_fits','training_snapshot'):
            self.assertEqual(training_codec.encode(memo.decode(kind,self.payloads[kind])),self.payloads[kind])
        stats=memo.stats();self.assertEqual(stats['misses'],3);self.assertEqual(stats['evictions'],2)
        self.assertEqual(stats['entries'],1);self.assertLessEqual(stats['charged_bytes'],stats['byte_budget'])
        memo.clear();self.assertEqual(memo.stats()['entries'],0);self.assertEqual(memo.stats()['charged_bytes'],0)

    def test_scope_owns_four_codecs_and_delegates_without_recursion(self):
        _,guard,clock=self.scope_parts();original=self.functions()
        with replay_cache._CodecScope(self.memo,guard,clock) as scope:
            for before,after in zip(original,self.functions()):self.assertIsNot(before,after)
            for _ in range(2):
                self.assertEqual(training_codec.encode(training_codec.decode_snapshot(self.payloads['training_snapshot'])),
                                 self.payloads['training_snapshot'])
                self.assertEqual(training_codec.encode(training_codec.decode_fits(self.payloads['training_fits'])),
                                 self.payloads['training_fits'])
            scope.clock()
        self.assertEqual(self.functions(),original)
        for kind in self.payloads:self.assertGreater(self.memo.stats()['kind_hits'][kind],0)

    def test_scope_exception_clears_both_new_records_and_restores_four_functions(self):
        _,guard,clock=self.scope_parts();original=self.functions()
        with self.assertRaisesRegex(RuntimeError,'private test failure'):
            with replay_cache._CodecScope(self.memo,guard,clock):
                training_codec.decode_snapshot(self.payloads['training_snapshot'])
                training_codec.decode_fits(self.payloads['training_fits']);raise RuntimeError('private test failure')
        self.assertEqual(self.functions(),original);self.assertEqual(self.memo.stats()['entries'],0)

    def test_code_epoch_change_refuses_training_hits_and_restores_all(self):
        path,guard,clock=self.scope_parts();original=self.functions()
        with self.assertRaises(replay_cache.ReplayCacheError):
            with replay_cache._CodecScope(self.memo,guard,clock) as scope:
                training_codec.decode_snapshot(self.payloads['training_snapshot']);path.write_bytes(b'BRAVO');scope.clock()
        self.assertEqual(self.functions(),original);self.assertEqual(self.memo.stats()['entries'],0)

    def test_replaced_training_decoder_is_refused_and_original_restored(self):
        _,guard,clock=self.scope_parts();original=self.functions()
        with self.assertRaises(replay_cache.ReplayCacheError):
            with replay_cache._CodecScope(self.memo,guard,clock) as scope:
                training_codec.decode_fits=lambda value:None;scope.clock()
        self.assertEqual(self.functions(),original);self.assertEqual(self.memo.stats()['entries'],0)

    def test_replaced_encoder_cannot_launder_a_changed_object_key(self):
        _,guard,clock=self.scope_parts();original=self.functions();encoder=training_codec.encode
        self.addCleanup(setattr,training_codec,'encode',encoder)
        with self.assertRaises(replay_cache.ReplayCacheError):
            with replay_cache._CodecScope(self.memo,guard,clock):
                training_codec.decode_snapshot(json.loads(self.payloads['training_snapshot']))
                training_codec.encode=lambda value:self.payloads['training_snapshot']
                changed=json.loads(self.payloads['training_snapshot']);changed['ready']=1
                training_codec.decode_snapshot(changed)
        self.assertEqual(self.functions(),original);self.assertIs(training_codec.encode,encoder)
        self.assertEqual(self.memo.stats()['entries'],0)


if __name__=='__main__':unittest.main()
