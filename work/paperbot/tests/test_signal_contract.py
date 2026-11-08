"""Exercise real strategy methods without pretending to load Freqtrade.

Only imports/annotations and the IStrategy base are replaced by this AST harness.
Engine behavior and pandas integration are deliberately not simulated here.
"""
import ast
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


def contract_strategy():
    source = Path(__file__).resolve().parents[1] / 'DeepSeekPaperStrategy.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    cls.bases = [ast.Name(id='object', ctx=ast.Load())]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    ast.fix_missing_locations(module)
    import math
    namespace = dict(json=json, os=os, time=time, Path=Path, math=math)
    exec(compile(module, str(source), 'exec'), namespace)
    return namespace['DeepSeekPaperStrategy']()


class SignalContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'signal.json'
        self.tuning = Path(self.tmp.name) / 'tuning.json'
        self.env = patch.dict(os.environ, AI_SIGNAL_PATH=str(self.path), AI_TUNING_PATH=str(self.tuning))
        self.env.start()
        self.addCleanup(self.env.stop)
        self.strategy = contract_strategy()
        self.now = time.time()
        self.valid = dict(action='buy', confidence=0.72, generated_at=self.now,
                          expires_at=self.now+1200, pair='BTC/USDT', timeframe='15m', status='ok',schema=2,
                          data_closed_at=self.now-120,data_received_at=self.now-1,decision_sha256='a'*64)

    def read(self, changes=None):
        value = {**self.valid, **(changes or {})}
        self.path.write_text(json.dumps(value), encoding='utf-8')
        # metadata is supplied by the caller, never trusted from the signal.
        return self.strategy._read_signal('BTC/USDT')

    def test_valid_signal_is_accepted(self):
        self.assertEqual(self.read()['action'], 'buy')

    def test_wrong_identity_and_degraded_status_are_rejected(self):
        for changes in ({'pair':'ETH/USDT'}, {'timeframe':'1h'}, {'status':'degraded'},
                        {'pair':None}, {'status':None}):
            with self.subTest(changes=changes):
                self.assertIsNone(self.read(changes))

    def test_invalid_numbers_and_lifetimes_are_rejected(self):
        cases = []
        for field in ('confidence','generated_at','expires_at'):
            cases.extend({field:value} for value in (True, False, '0.72', float('nan'), float('inf'), -float('inf'),10**400))
        cases += [{'generated_at':self.now+120}, {'generated_at':self.now-1800},
                  {'expires_at':self.now-1}, {'expires_at':self.now-1,'generated_at':self.now},
                  {'expires_at':self.now+100000}, {'generated_at':0}]
        for changes in cases:
            with self.subTest(changes=changes):
                self.assertIsNone(self.read(changes))

    def test_missing_context_is_rejected(self):
        self.path.write_text(json.dumps(self.valid), encoding='utf-8')
        self.assertIsNone(self.strategy._read_signal())

    def test_missing_audit_identity_and_stale_source_cannot_enter(self):
        for changes in ({'schema':1},{'decision_sha256':None},{'decision_sha256':'xyz'},
                        {'data_closed_at':self.now-2400},{'data_received_at':True},
                        {'data_closed_at':self.now+1},{'data_received_at':self.now+1},
                        {'data_closed_at':self.now+20,'data_received_at':self.now+25,'generated_at':self.now+30}):
            with self.subTest(changes=changes): self.assertIsNone(self.read(changes))

    def test_duplicate_action_fields_cannot_change_the_execution_rule(self):
        payload=json.dumps(self.valid).replace('"action": "buy"','"action":"hold","action":"buy"')
        self.path.write_text(payload,encoding='utf-8')
        self.assertIsNone(self.strategy._read_signal('BTC/USDT'))

    def test_invalid_tuning_defaults_safely(self):
        for value in ([], True, {'buy_confidence':True}, {'buy_confidence':'0.7'},
                      {'buy_confidence':float('nan')}, {'buy_confidence':float('inf')}, {'buy_confidence':10**400}):
            with self.subTest(value=value):
                self.tuning.write_text(json.dumps(value), encoding='utf-8')
                self.assertEqual(self.strategy._threshold('buy_confidence',0.68,0.64,0.82),0.68)
