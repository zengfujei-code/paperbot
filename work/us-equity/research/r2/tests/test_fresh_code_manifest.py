"""Actual-file code-vector checks, no phase/admission or market authority."""
from contextlib import ExitStack
import os
from pathlib import Path
import socket
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).resolve().parents[3]),str(Path(__file__).resolve().parents[1])]
from paperbot_r2 import replay_cache,test_protocol


class FreshCodeManifestTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(replay_cache,'_FreshCodeManifest'),'fresh complete code vector not implemented')
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.paths=tuple(Path(self.temp.name)/name for name in ('a.py','b.py'))
        for path in self.paths:path.write_bytes(b'ALPHA')
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ,{},clear=True))
        self.stack.enter_context(patch.object(socket,'socket',side_effect=AssertionError('no network')))
        self.stack.enter_context(patch.object(sqlite3,'connect',side_effect=AssertionError('no SQL')))

    def test_changed_bytes_with_restored_mtime_poison_the_manifest(self):
        vector=replay_cache._FreshCodeManifest(self.paths);before=self.paths[0].stat()
        self.paths[0].write_bytes(b'BRAVO');os.utime(self.paths[0],ns=(before.st_atime_ns,before.st_mtime_ns))
        with self.assertRaises(replay_cache.ReplayCacheError):vector.check(self.paths)
        self.paths[0].write_bytes(b'ALPHA')
        with self.assertRaises(replay_cache.ReplayCacheError):vector.check(self.paths)

    def test_edit_of_already_read_file_during_later_read_is_refused(self):
        vector=replay_cache._FreshCodeManifest(self.paths);original=Path.read_bytes;changed=[]
        def concurrent(path):
            value=original(path)
            if path==self.paths[-1] and not changed:
                before=self.paths[0].stat();self.paths[0].write_bytes(b'BRAVO')
                os.utime(self.paths[0],ns=(before.st_atime_ns,before.st_mtime_ns));changed.append(True)
            return value
        with patch.object(Path,'read_bytes',concurrent):
            with self.assertRaises(replay_cache.ReplayCacheError):vector.check(self.paths)
        self.assertEqual(changed,[True])

    def test_path_replacement_missing_and_oversized_code_are_refused(self):
        for change in ('replace','missing','oversize'):
            with self.subTest(change=change):
                for path in self.paths:path.write_bytes(b'ALPHA')
                vector=replay_cache._FreshCodeManifest(self.paths)
                if change=='replace':
                    replacement=self.paths[0].with_suffix('.new');replacement.write_bytes(b'ALPHA');os.replace(replacement,self.paths[0])
                elif change=='missing':self.paths[0].unlink()
                else:self.paths[0].write_bytes(b'x'*524289)
                with self.assertRaises(replay_cache.ReplayCacheError):vector.check(self.paths)

    def test_incomplete_or_changed_dependency_set_is_refused(self):
        vector=replay_cache._FreshCodeManifest(self.paths);extra=Path(self.temp.name)/'added.py';extra.write_bytes(b'ALPHA')
        with self.assertRaises(replay_cache.ReplayCacheError):vector.check(self.paths+(extra,))
        for values in ((),(self.paths[0],self.paths[0]),('relative.py',)):
            with self.assertRaises(replay_cache.ReplayCacheError):replay_cache._FreshCodeManifest(values)

    def test_every_original_transitive_byte_input_is_freshly_checked(self):
        original=Path.read_bytes;dependencies=set()
        def traced(path):dependencies.add(path.absolute());return original(path)
        with patch.object(Path,'read_bytes',traced):expected=test_protocol.source_sha256()
        guard=replay_cache._ProtocolCodeGuard();guard.check();reads=set()
        def observed(path):reads.add(path.absolute());return original(path)
        with patch.object(Path,'read_bytes',observed):guard.check()
        self.assertTrue(dependencies);self.assertTrue(dependencies<=reads)
        self.assertEqual(guard.expected,expected)

    def test_runtime_guard_refuses_actual_code_edit_and_original_function_replacement(self):
        guard=replay_cache._ProtocolCodeGuard();guard.check()
        path=Path(test_protocol.__file__);before=path.stat();original=path.read_bytes()
        try:
            path.write_bytes(original.replace(b'Fixed prospective',b'Faked prospective',1))
            self.assertNotEqual(path.read_bytes(),original)
            os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
            with self.assertRaises(replay_cache.ReplayCacheError):guard.check()
        finally:
            path.write_bytes(original);os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
        with self.assertRaises(replay_cache.ReplayCacheError):guard.check()
        guard=replay_cache._ProtocolCodeGuard();guard.check()
        with patch.object(test_protocol,'source_sha256',lambda:guard.expected):
            with self.assertRaises(replay_cache.ReplayCacheError):guard.check()
        guard=replay_cache._ProtocolCodeGuard();guard.check()
        with patch.object(test_protocol,'PROTOCOL_FILES',test_protocol.PROTOCOL_FILES+('missing.py',)):
            with self.assertRaises(replay_cache.ReplayCacheError):guard.check()


if __name__=='__main__':unittest.main()
