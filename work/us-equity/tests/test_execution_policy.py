import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


class ExecutionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.execution_policy'), 'pinned execution policy required')
        self.m=importlib.import_module('paperbot_us.execution_policy')
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'policy.json'
        self.payload=dict(schema=1,capital_usd=10000,execution_enabled=False,strategy_sha256=None,
                          acceptance_sha256=None,risk_limits_sha256='a'*64,max_daily_entries=10,require_protection=True)

    def write(self, **changes):
        raw=json.dumps({**self.payload,**changes},sort_keys=True).encode()
        self.path.write_bytes(raw)
        return self.m.PolicyFile(self.path,hashlib.sha256(raw).hexdigest())

    def test_disabled_policy_is_explicit_and_enabling_requires_acceptance_references(self):
        snapshot=self.write().read()
        self.assertFalse(snapshot.policy.execution_enabled)
        self.assertTrue(snapshot.policy.require_protection)
        with self.assertRaises(self.m.PolicyError): self.write(execution_enabled=True).read()
        enabled=self.write(execution_enabled=True,strategy_sha256='b'*64,acceptance_sha256='c'*64).read()
        self.assertTrue(enabled.policy.execution_enabled)
        self.assertFalse(enabled.public_summary()['execution_enabled'],'policy alone never grants an order permission')

    def test_changed_content_unsafe_fields_and_secrets_are_not_a_policy(self):
        source=self.write();self.path.write_text('{}')
        with self.assertRaises(self.m.PolicyError): source.read()
        for values in (dict(require_protection=False),dict(capital_usd=40000),dict(max_daily_entries=True),
                       dict(max_daily_entries=21),dict(execution_enabled='true'),dict(api_key='PRIVATE_DO_NOT_STORE')):
            with self.subTest(values=values),self.assertRaises(self.m.PolicyError): self.write(**values).read()

    def test_duplicate_json_and_missing_or_linked_file_cannot_load(self):
        raw=json.dumps(self.payload)[:-1]+',"execution_enabled":true}'
        self.path.write_text(raw)
        with self.assertRaises(self.m.PolicyError):
            self.m.PolicyFile(self.path,hashlib.sha256(raw.encode()).hexdigest()).read()
        self.path.unlink()
        with self.assertRaises(self.m.PolicyError): self.m.PolicyFile(self.path,'a'*64).read()
        self.assertFalse(self.path.exists())
        if os.name!='nt':
            target=self.path.with_name('target.json');target.write_text(json.dumps(self.payload));self.path.symlink_to(target)
            with self.assertRaises(self.m.PolicyError):
                self.m.PolicyFile(self.path,hashlib.sha256(target.read_bytes()).hexdigest()).read()


if __name__=='__main__': unittest.main()
