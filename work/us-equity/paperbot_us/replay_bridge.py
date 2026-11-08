"""Replay archived quant/model evidence under its hash-pinned original source.

The worker has no inherited credentials, runs isolated from this process's
imports and denies Python network/write/process operations. Only trusted,
reviewed source manifests are accepted in deployment; this is not a sandbox for
untrusted Python or an execution capability. Original version checks stay strict.
"""
from dataclasses import dataclass,field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import zoneinfo

CORE=frozenset(('__init__','alpaca','data','scanner','audit','runner','risk','monitor',
                'quant','research','backtest','shadow'))


class ReplayError(RuntimeError): pass


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


def strict_json(raw):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result: raise ValueError('duplicate json field')
            result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite json')))


def identity(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value): raise ValueError('digest required')
    return value


def regular_bytes(path,maximum):
    info=path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_size>maximum:
        raise ValueError('regular bounded file required')
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
    with os.fdopen(fd,'rb') as handle:
        opened=os.fstat(handle.fileno())
        if (opened.st_dev,opened.st_ino,opened.st_size)!=(info.st_dev,info.st_ino,info.st_size):
            raise ValueError('file changed during open')
        value=handle.read(maximum+1)
    if len(value)>maximum: raise ValueError('file budget exceeded')
    return value


def timezone_digest(root):
    path=Path(root)/'America/New_York'
    if not path.is_file() or path.stat().st_size>65536: raise ValueError('bounded timezone file required')
    raw=path.read_bytes()
    if not 32<=len(raw)<=65536 or not raw.startswith(b'TZif'): raise ValueError('IANA timezone bytes required')
    return hashlib.sha256(raw).hexdigest()


def bridge_version():
    digest=hashlib.sha256()
    for name in ('replay_bridge.py','replay_worker.py'):
        digest.update(name.encode()+b'\0'+Path(__file__).with_name(name).read_bytes()+b'\0')
    return digest.hexdigest()


@dataclass(frozen=True,init=False)
class PinnedRelease:
    root: Path=field(repr=False)
    files_json: str=field(repr=False)
    package_sha256: str

    def __init__(self,root,*,files,package_sha256):
        try:
            identity(package_sha256)
            if type(files) is not dict or not 12<=len(files)<=64: raise ValueError('source manifest required')
            for name,digest in files.items():
                if not isinstance(name,str) or not re.fullmatch(r'paperbot_us/[a-z_]{1,40}\.py',name):
                    raise ValueError('flat package source path required')
                identity(digest)
            if not {f'paperbot_us/{name}.py' for name in CORE}<=set(files): raise ValueError('replay dependency missing')
            object.__setattr__(self,'root',Path(root).absolute())
            object.__setattr__(self,'files_json',canonical(files))
            object.__setattr__(self,'package_sha256',package_sha256)
        except Exception: raise ReplayError('invalid_pinned_replay_release') from None

    def manifest(self):
        return dict(schema=1,package_sha256=self.package_sha256,files=strict_json(self.files_json))

    @property
    def digest(self): return hashlib.sha256(canonical(self.manifest()).encode()).hexdigest()

    def verify(self):
        try:
            root=self.root
            if (not root.is_dir() or root.is_symlink()
                    or os.path.normcase(str(root.resolve()))!=os.path.normcase(str(root))):
                raise ValueError('canonical source directory required')
            for entry in root.iterdir():
                if entry.is_symlink(): raise ValueError('root link refused')
                if entry.is_dir() and entry.name not in ('paperbot_us','tests','docs'):
                    raise ValueError('shadow import directory refused')
                if entry.is_file() and entry.suffix.lower() in ('.py','.pyc','.pyd','.so','.zip'):
                    raise ValueError('shadow import file refused')
            package=root/'paperbot_us'
            actual=set()
            for entry in package.iterdir():
                if entry.is_symlink() or not entry.is_file() or entry.suffix!='.py':
                    raise ValueError('bytecode/subpackage/link refused')
                actual.add('paperbot_us/'+entry.name)
            files=strict_json(self.files_json)
            if actual!=set(files): raise ValueError('source inventory changed')
            for name,digest in files.items():
                if hashlib.sha256(regular_bytes(root/name,262144)).hexdigest()!=digest:
                    raise ValueError('source hash mismatch')
            return self.digest
        except Exception: raise ReplayError('pinned_replay_release_unavailable') from None


@dataclass(frozen=True)
class ReplayEvidence:
    payload: str=field(repr=False)

    @property
    def digest(self): return hashlib.sha256(self.payload.encode()).hexdigest()

    def read(self): return strict_json(self.payload)


class ReplayBridge:
    def __init__(self,release,*,timezone_root=None):
        if type(release) is not PinnedRelease: raise ReplayError('pinned_replay_release_required')
        self.release=release
        try:
            roots=[Path(timezone_root).absolute()] if timezone_root is not None else [Path(p) for p in zoneinfo.TZPATH]
            root=next(p for p in roots if (p/'America/New_York').is_file())
            self.timezone_root=str(root)
            self.timezone_sha256=timezone_digest(root)
        except Exception: raise ReplayError('pinned_replay_timezone_unavailable') from None

    def _run(self,kind,archive,record_id,source_archive=None,*,expected_heads=None):
        try:
            if record_id is not None: identity(record_id)
            elif kind!='model': raise ValueError('record identity required')
            release=self.release
            heads={} if expected_heads is None else expected_heads
            if type(heads) is not dict or not set(heads)<=({'quant'} if kind=='quant' else {'quant','model'}):
                raise ValueError('invalid expected source heads')
            for head in heads.values():
                if type(head) is not dict or set(head)!={'sequence','digest'} or type(head['sequence']) is not int or not 1<=head['sequence']<=100000:
                    raise ValueError('invalid source prefix witness')
                identity(head['digest'])
            release.verify()
            if timezone_digest(self.timezone_root)!=self.timezone_sha256: raise ValueError('timezone bytes changed')
            request=dict(schema=1,kind=kind,root=str(release.root),manifest=release.manifest(),
                release_sha256=release.digest,bridge_sha256=bridge_version(),archive=str(Path(archive).absolute()),identity=record_id,
                source_archive=str(Path(source_archive).absolute()) if source_archive is not None else None,
                timezone_root=self.timezone_root,timezone_sha256=self.timezone_sha256,expected_heads=heads)
            worker=Path(__file__).with_name('replay_worker.py')
            raw=canonical(request).encode()
            if len(raw)>32768: raise ValueError('replay request budget')
            env={name:os.environ[name] for name in ('PATH','SystemRoot','WINDIR','LANG') if name in os.environ}
            run=subprocess.run([sys.executable,'-I','-B',str(worker)],input=raw,capture_output=True,
                timeout=20,env=env,cwd=str(worker.parent))
            if run.returncode!=0 or run.stderr or len(run.stdout)>131072: raise ValueError('replay worker unavailable')
            response=strict_json(run.stdout)
            if (set(response)!={'schema','kind','release_sha256','timezone_sha256','bridge_sha256','quant','model','attempt','summary'}
                    or response['schema']!=1 or response['kind']!=kind or response['release_sha256']!=release.digest
                    or response['timezone_sha256']!=self.timezone_sha256 or response['bridge_sha256']!=request['bridge_sha256']
                    or bridge_version()!=request['bridge_sha256']):
                raise ValueError('replay response binding mismatch')
            # A release must remain pinned through completion, not just startup.
            release.verify()
            if timezone_digest(self.timezone_root)!=self.timezone_sha256: raise ValueError('timezone changed during replay')
            return ReplayEvidence(canonical(response))
        except Exception: raise ReplayError('original_source_replay_unavailable') from None

    def quant(self,archive,identity,*,expected_head=None):
        return self._run('quant',archive,identity,expected_heads={'quant':expected_head} if expected_head is not None else None)

    def review(self,archive,identity,*,source_archive,expected_quant_head=None,expected_model_head=None):
        if identity is None: raise ReplayError('explicit_record_identity_required')
        heads={}
        if expected_quant_head is not None:heads['quant']=expected_quant_head
        if expected_model_head is not None:heads['model']=expected_model_head
        return self._run('model',archive,identity,source_archive,expected_heads=heads)

    def latest_review(self,archive,*,source_archive):
        return self._run('model',archive,None,source_archive)
