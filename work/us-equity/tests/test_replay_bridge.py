from contextlib import closing
from dataclasses import replace
import hashlib
from copy import deepcopy
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

import test_model_runner

CORE=('__init__','alpaca','data','scanner','audit','runner','risk','monitor',
      'quant','research','backtest','shadow','llm','model_runner')


class ReplayBridgeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('paperbot_us.replay_bridge'),
                             'original-source offline replay bridge required')
        self.m=importlib.import_module('paperbot_us.replay_bridge')
        self.f=test_model_runner.ModelRunnerTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.f.run_once();self.root=self.f.root;self.release=self.root/'legacy'
        package=self.release/'paperbot_us';package.mkdir(parents=True)
        live=Path(self.m.__file__).parent;files={}
        for name in CORE:
            raw=(live/(name+'.py')).read_bytes();(package/(name+'.py')).write_bytes(raw)
            files['paperbot_us/'+name+'.py']=hashlib.sha256(raw).hexdigest()
        self.pinned=self.m.PinnedRelease(self.release,files=files,package_sha256='a'*64)
        tzpath=os.environ.get('PYTHONTZPATH')
        self.bridge=self.m.ReplayBridge(self.pinned,timezone_root=tzpath or None)
        self.review=next(r for r in self.f.records() if r['kind']=='model_review')
        self.quant_id=self.review['source_identity'];self.review_id=self.review['identity']

    def test_original_release_replays_even_when_current_process_version_rejects_the_record(self):
        current=importlib.import_module('paperbot_us.shadow')
        with patch.object(current,'_version',return_value='f'*64):
            with self.assertRaises(ValueError):
                current.replay_decision(self.f.m.source_decision(self.root/'decisions.sqlite',self.quant_id))
            evidence=self.bridge.quant(self.root/'decisions.sqlite',self.quant_id)
        value=evidence.read()
        self.assertEqual(value['kind'],'quant')
        self.assertEqual(value['quant']['record']['identity'],self.quant_id)
        self.assertEqual(value['quant']['record']['final_action'],'hold')
        self.assertEqual(value['release_sha256'],self.pinned.digest)
        self.assertEqual(evidence.digest,hashlib.sha256(evidence.payload.encode()).hexdigest())

    def test_model_replay_binds_the_exact_source_attempt_request_and_receipt(self):
        evidence=self.bridge.review(self.root/'model.sqlite',self.review_id,source_archive=self.root/'decisions.sqlite')
        value=evidence.read()
        self.assertEqual(value['kind'],'model')
        self.assertEqual(value['model']['record']['source_sha256'],value['quant']['digest'])
        self.assertEqual(value['attempt']['record']['identity'],value['model']['record']['attempt_identity'])
        self.assertEqual(value['attempt']['record']['request'],value['model']['record']['request'])
        self.assertEqual(value['model']['record']['receipt']['action'],'hold')
        self.assertEqual(value['model']['record']['receipt']['reason'],'趋势偏强，等待量价确认。')
        self.assertEqual(value['summary']['final_action'],'hold')
        self.assertFalse(value['summary']['signal_usable'])
        value['model']['record']['receipt']['action']='buy'
        self.assertEqual(evidence.read()['model']['record']['receipt']['action'],'hold')

    def test_replay_does_not_change_archives_source_or_create_bytecode(self):
        paths=[self.root/'model.sqlite',self.root/'decisions.sqlite',
               *sorted((self.release/'paperbot_us').glob('*.py'))]
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        self.bridge.review(self.root/'model.sqlite',self.review_id,source_archive=self.root/'decisions.sqlite')
        self.assertEqual({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},before)
        self.assertEqual(list(self.release.rglob('__pycache__')),[])

    def test_changed_source_shadow_modules_and_bytecode_are_refused_before_replay(self):
        path=self.release/'paperbot_us/risk.py';original=path.read_bytes();path.write_bytes(original+b'\n# changed\n')
        with self.assertRaises(self.m.ReplayError): self.bridge.quant(self.root/'decisions.sqlite',self.quant_id)
        path.write_bytes(original)
        shadow=self.release/'json.py';shadow.write_text('raise RuntimeError("UNTRUSTED")')
        with self.assertRaises(self.m.ReplayError): self.bridge.quant(self.root/'decisions.sqlite',self.quant_id)
        shadow.unlink()
        cache=self.release/'paperbot_us/__pycache__';cache.mkdir()
        with self.assertRaises(self.m.ReplayError): self.bridge.quant(self.root/'decisions.sqlite',self.quant_id)

    def test_corrupt_archive_missing_record_or_wal_mode_is_never_normal_empty_evidence(self):
        source=self.root/'decisions.sqlite';original=source.read_bytes()
        with self.assertRaises(self.m.ReplayError): self.bridge.quant(self.root/'decisions.sqlite','f'*64)
        with closing(sqlite3.connect(self.root/'decisions.sqlite')) as db:
            db.execute("UPDATE observations SET digest='broken' WHERE seq=1");db.commit()
        with self.assertRaises(self.m.ReplayError): self.bridge.quant(self.root/'decisions.sqlite',self.quant_id)
        source.write_bytes(original)
        self.assertEqual(self.bridge.quant(source,self.quant_id).read()['kind'],'quant')
        with closing(sqlite3.connect(self.root/'model.sqlite')) as db:
            db.execute('PRAGMA journal_mode=WAL')
        with self.assertRaises(self.m.ReplayError):
            self.bridge.review(self.root/'model.sqlite',self.review_id,source_archive=self.root/'decisions.sqlite')

    def test_sensitive_extra_fields_are_rejected_and_never_returned_in_errors(self):
        from paperbot_us.monitor import ObservationArchive
        alternate=ObservationArchive(self.root/'extra.sqlite')
        attempt=next(r for r in self.f.records() if r['kind']=='model_attempt')
        alternate.append(attempt)
        changed={**self.review,'secret':'PRIVATE_MARKER'}
        alternate.append(changed)
        with self.assertRaises(self.m.ReplayError) as caught:
            self.bridge.review(alternate.path,self.review_id,source_archive=self.root/'decisions.sqlite')
        self.assertNotIn('PRIVATE',str(caught.exception))

    def test_child_worker_has_no_network_write_or_environment_credentials(self):
        # A hash-pinned diagnostic release deliberately attempts side effects;
        # the worker must deny these before any evidence can be returned.
        path=self.release/'paperbot_us/__init__.py'
        for extra in ('\nimport socket\nsocket.socket()\n',
                      '\nfrom pathlib import Path\nPath(__file__).with_name("forbidden.txt").write_text("x")\n',
                      '\nimport os\nassert "DEEPSEEK_API_KEY" in os.environ\n'):
            with self.subTest(operation=extra.splitlines()[1]):
                path.write_text(extra)
                files={str(p.relative_to(self.release)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (self.release/'paperbot_us').glob('*.py')}
                bridge=self.m.ReplayBridge(self.m.PinnedRelease(self.release,files=files,package_sha256='a'*64),
                    timezone_root=os.environ.get('PYTHONTZPATH') or None)
                with patch.dict(os.environ,{'DEEPSEEK_API_KEY':'PRIVATE_MARKER'}):
                    with self.assertRaises(self.m.ReplayError): bridge.quant(self.root/'decisions.sqlite',self.quant_id)
                self.assertFalse(path.with_name('forbidden.txt').exists())

    def test_nested_calendar_and_session_extra_fields_never_escape_in_raw_witnesses(self):
        from paperbot_us.monitor import ObservationArchive
        source=self.f.m.source_decision(self.root/'decisions.sqlite',self.quant_id)
        for index,level in enumerate(('calendar','session')):
            with self.subTest(level=level):
                changed={k:deepcopy(v) for k,v in source.items() if k!='audit_sha256'}
                changed['identity']='e'*64
                target=changed['calendar'] if level=='calendar' else changed['calendar']['sessions'][0]
                target['secret']='PRIVATE_SYNTHETIC_MARKER'
                archive=ObservationArchive(self.root/('nested'+str(index)+'.sqlite'));archive.append(changed)
                with self.assertRaises(self.m.ReplayError) as caught: self.bridge.quant(archive.path,'e'*64)
                self.assertNotIn('PRIVATE',str(caught.exception))

    def test_timezone_bytes_are_bound_and_cannot_change_between_bridge_creation_and_replay(self):
        import io,zoneinfo
        roots=[Path(os.environ['PYTHONTZPATH'])] if os.environ.get('PYTHONTZPATH') else [Path(p) for p in zoneinfo.TZPATH]
        source=next(p/'America/New_York' for p in roots if (p/'America/New_York').is_file())
        target=self.root/'tz/America/New_York';target.parent.mkdir(parents=True);raw=source.read_bytes();target.write_bytes(raw)
        bridge=self.m.ReplayBridge(self.pinned,timezone_root=target.parent.parent)
        changed=raw.replace(b'EDT',b'EQT');self.assertNotEqual(changed,raw)
        zoneinfo.ZoneInfo.from_file(io.BytesIO(changed),key='America/New_York')
        target.write_bytes(changed)
        with self.assertRaises(self.m.ReplayError): bridge.quant(self.root/'decisions.sqlite',self.quant_id)

    def test_proof_cannot_be_labelled_as_a_different_bridge_implementation(self):
        proof=self.bridge.quant(self.root/'decisions.sqlite',self.quant_id).read()
        self.assertTrue('bridge_sha256' in proof,'the validator version belongs in its evidence')
        with patch.object(self.m,'bridge_version',return_value='f'*64):
            with self.assertRaises(self.m.ReplayError): self.bridge.quant(self.root/'decisions.sqlite',self.quant_id)

    def test_readonly_sqlite_audit_uri_accepts_python310_bytes_without_allowing_writes(self):
        worker=importlib.import_module('paperbot_us.replay_worker')
        try: worker.deny_side_effects('sqlite3.connect',(b'file:///synthetic/archive.sqlite?mode=ro',))
        except PermissionError: self.fail('Python 3.10 readonly URI bytes must not be mistaken for write access')
        worker.deny_side_effects('sqlite3.connect',('file:///synthetic/archive.sqlite?mode=ro',))
        for database in (b'file:///synthetic/archive.sqlite?mode=rw',b'/synthetic/archive.sqlite',
                         'file:///synthetic/archive.sqlite?mode=rwc'):
            with self.subTest(database_type=type(database).__name__),self.assertRaises(PermissionError):
                worker.deny_side_effects('sqlite3.connect',(database,))

    def test_latest_review_selection_uses_the_guarded_read_without_wal_side_files(self):
        self.assertTrue(hasattr(self.bridge,'latest_review'),'record selection must share the guarded worker')
        result=self.bridge.latest_review(self.root/'model.sqlite',source_archive=self.root/'decisions.sqlite')
        self.assertEqual(result.read()['model']['record']['identity'],self.review_id)
        with closing(sqlite3.connect(self.root/'model.sqlite')) as db: db.execute('PRAGMA journal_mode=WAL')
        companions=[self.root/('model.sqlite'+suffix) for suffix in ('-wal','-shm')]
        self.assertTrue(all(not p.exists() for p in companions))
        with self.assertRaises(self.m.ReplayError):
            self.bridge.latest_review(self.root/'model.sqlite',source_archive=self.root/'decisions.sqlite')
        self.assertTrue(all(not p.exists() for p in companions),'a rejected read must not create WAL/SHM companions')


if __name__=='__main__': unittest.main()
