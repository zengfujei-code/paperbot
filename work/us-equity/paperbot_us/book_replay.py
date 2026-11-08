"""Bounded offline proof of an original, never-traded USD 10,000 book."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

from .replay_bridge import (PinnedRelease, ReplayBridge, ReplayEvidence, canonical,
                            identity, strict_json, timezone_digest)


class BookReplayError(RuntimeError): pass


def replay_version():
    digest=hashlib.sha256()
    for name in ('book_replay.py','book_worker.py','replay_bridge.py','replay_worker.py'):
        digest.update(name.encode()+b'\0'+Path(__file__).with_name(name).read_bytes()+b'\0')
    return digest.hexdigest()


class BookReplay:
    def __init__(self,release,*,book,baseline,ledger,timezone_root=None):
        try:
            if type(release) is not PinnedRelease: raise ValueError('pinned source required')
            self.release=release
            self.paths=tuple(Path(p).absolute() for p in (book,baseline,ledger))
            if len(set(self.paths))!=3: raise ValueError('separate original evidence required')
            tz=ReplayBridge(release,timezone_root=timezone_root)
            self.timezone_root=tz.timezone_root; self.timezone_sha256=tz.timezone_sha256
        except Exception: raise BookReplayError('invalid_original_book_replay') from None

    def replay(self):
        try:
            self.release.verify(); version=replay_version()
            if timezone_digest(self.timezone_root)!=self.timezone_sha256: raise ValueError('timezone changed')
            request=dict(schema=1,root=str(self.release.root),manifest=self.release.manifest(),
                release_sha256=self.release.digest,validator_sha256=version,
                book=str(self.paths[0]),baseline=str(self.paths[1]),ledger=str(self.paths[2]),
                timezone_root=self.timezone_root,timezone_sha256=self.timezone_sha256)
            worker=Path(__file__).with_name('book_worker.py')
            env={k:os.environ[k] for k in ('PATH','SystemRoot','WINDIR','LANG') if k in os.environ}
            result=subprocess.run([sys.executable,'-I','-B',str(worker)],input=canonical(request).encode(),
                capture_output=True,timeout=20,env=env,cwd=str(worker.parent))
            if result.returncode or result.stderr or len(result.stdout)>16384: raise ValueError('worker unavailable')
            value=strict_json(result.stdout)
            if (set(value)!={'schema','kind','release_sha256','validator_sha256','timezone_sha256',
                            'anchor','book','baseline','ledger','summary'}
                    or type(value['schema']) is not int or value['schema']!=1 or value['kind']!='pristine_book'
                    or value['release_sha256']!=self.release.digest or value['validator_sha256']!=version
                    or value['timezone_sha256']!=self.timezone_sha256): raise ValueError('proof binding')
            self.release.verify()
            if replay_version()!=version or timezone_digest(self.timezone_root)!=self.timezone_sha256:
                raise ValueError('validator changed')
            return ReplayEvidence(canonical(value))
        except Exception: raise BookReplayError('original_pristine_book_unavailable') from None
