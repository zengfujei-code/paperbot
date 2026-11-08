"""Bounded immutable decoder DATA memo; no current-source/phase authority.

The runtime below delegates original journal APIs under fresh code and phase
checks. This module's actual bytes are part of the complete protocol; callers
must register that exact commitment before use. No SQL or model computation is
cached, and no trading authority is created.
"""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import fields,is_dataclass
from datetime import date,datetime
import contextlib
import hashlib
import math
import os
from pathlib import Path
import stat
import sys
import threading
from types import FunctionType,MethodType
import paperbot_us

from . import forecast_inputs,candidate_forecasts,training_codec,training_data,model_training

MAX_BYTES=33554432
MAX_ITEMS=2048


class ReplayCacheError(ValueError):pass


def _need(value):
    if not value:raise ReplayCacheError('immutable_replay_decode_cache_refused')


def _data_size(value,seen=None):
    """Conservative retained object charge, accepting frozen data only."""
    if seen is None:seen=set()
    identity=id(value)
    if identity in seen:return 0
    seen.add(identity);size=sys.getsizeof(value)
    if value is None or type(value) in (str,bytes,int,float,bool,date,datetime):return size
    if type(value) is tuple:return size+sum(_data_size(v,seen) for v in value)
    _need(is_dataclass(value) and not isinstance(value,type) and value.__dataclass_params__.frozen is True)
    if hasattr(value,'__dict__'):size+=sys.getsizeof(value.__dict__)
    return size+sum(_data_size(getattr(value,f.name),seen) for f in fields(value))


def _codec_targets():
    return {'input':(forecast_inputs,'decode_input',forecast_inputs.IssueInput),
            'panel':(candidate_forecasts,'decode_panel',candidate_forecasts.CandidatePanel),
            'training_snapshot':(training_codec,'decode_snapshot',training_data.TrainingSnapshot),
            'training_fits':(training_codec,'decode_fits',model_training.TrainingFits)}


def _literal_json(value):
    """Only exact JSON types; unsupported valid caller objects use originals."""
    active=set();remaining=262144
    def visit(item,depth):
        nonlocal remaining
        remaining-=1
        if remaining<0 or depth>64:return False
        if item is None or type(item) in (str,int,bool):return True
        if type(item) is float:return math.isfinite(item)
        if type(item) not in (dict,list) or id(item) in active:return False
        active.add(id(item))
        try:
            if type(item) is dict:
                return all(type(key) is str and visit(child,depth+1) for key,child in item.items())
            return all(visit(child,depth+1) for child in item)
        finally:active.remove(id(item))
    return visit(value,0)


def _original_encoder(function):
    return (type(function) is FunctionType and function.__module__==training_codec.__name__
            and function.__name__=='encode'
            and Path(function.__code__.co_filename).absolute()==Path(training_codec.__file__).absolute())


class _DecodeMemo:
    """Historical mathematical values only; every consumer owns its copy."""
    def __init__(self,*,max_bytes=MAX_BYTES,max_items=MAX_ITEMS):
        _need(type(max_bytes) is int and 0<max_bytes<=MAX_BYTES
              and type(max_items) is int and 0<max_items<=MAX_ITEMS)
        self._max_bytes=max_bytes;self._max_items=max_items;self._values=OrderedDict()
        self._charged=0;self._hits=0;self._misses=0;self._evictions=0;self._fallbacks=0
        self._decoders={kind:getattr(module,name) for kind,(module,name,_) in _codec_targets().items()}
        self._kind_hits={kind:0 for kind in self._decoders}
        self._encoder=training_codec.encode;_need(_original_encoder(self._encoder))

    def decode(self,kind,value):
        _need(kind in self._decoders)
        decoder=self._decoders[kind]
        expected=_codec_targets()[kind][2]
        if type(value) is not str:
            if kind in ('training_snapshot','training_fits') and type(value) is dict and _literal_json(value):
                # Original ledger bodies are dictionaries. The ORIGINAL strict
                # encoder supplies a complete bounded, type-faithful JSON key.
                # Decode THAT exact key on a miss, never a later mutable input.
                value=self._encoder(value)
            else:
                self._fallbacks+=1
                return decoder(value)
        key=(kind,value)
        if key in self._values:
            self._hits+=1;self._kind_hits[kind]+=1;self._values.move_to_end(key)
            return deepcopy(self._values[key][0])
        self._misses+=1;result=decoder(value);_need(type(result) is expected)
        # The public result must never alias the cache's retained graph. A
        # caller can bypass Python's frozen setter, so a frozen type alone is
        # insufficient ownership protection for later historical replays.
        private=deepcopy(result)
        charge=sys.getsizeof(key)+sys.getsizeof(kind)+sys.getsizeof(value)+_data_size(private)+256
        if charge>self._max_bytes:return result
        while self._values and (self._charged+charge>self._max_bytes or len(self._values)>=self._max_items):
            _,(_,old_charge)=self._values.popitem(last=False)
            self._charged-=old_charge;self._evictions+=1
        self._values[key]=(private,charge);self._charged+=charge
        _need(self._charged<=self._max_bytes and len(self._values)<=self._max_items)
        return result

    def clear(self):
        self._values.clear();self._charged=0

    def stats(self):
        return dict(hits=self._hits,misses=self._misses,evictions=self._evictions,nonstring_fallbacks=self._fallbacks,
                    kind_hits=dict(self._kind_hits),
                    entries=len(self._values),charged_bytes=self._charged,byte_budget=self._max_bytes,
                    item_limit=self._max_items,phase_authority=False)


_ACTIVE_SCOPE=None


def _original_codec(kind,function):
    target=_codec_targets().get(kind)
    if target is None:return False
    module,name,_=target
    return (type(function) is FunctionType and function.__module__==module.__name__
            and function.__name__==name
            and Path(function.__code__.co_filename).absolute()==Path(module.__file__).absolute())


class _FreshCodeManifest:
    """Expected code seals only; two complete ACTUAL byte passes per check."""
    def __init__(self,paths):
        self.failed=False;self.paths=self._paths(paths)
        try:
            self.expected=self._read();self.check(self.paths)
        except Exception:
            self.failed=True;raise ReplayCacheError('fresh_code_vector_refused') from None

    @staticmethod
    def _paths(paths):
        _need(type(paths) is tuple and 0<len(paths)<=256
              and all(isinstance(p,Path) and p.is_absolute() for p in paths)
              and len(set(paths))==len(paths))
        return tuple(sorted(paths))

    @staticmethod
    def _identity(path):
        value=os.stat(path,follow_symlinks=False)
        _need(stat.S_ISREG(value.st_mode) and 0<=value.st_size<=524288)
        return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns

    def _read(self):
        before=tuple(self._identity(path) for path in self.paths)
        _need(sum(value[2] for value in before)<=16777216)
        values=[]
        for path,identity in zip(self.paths,before):
            raw=path.read_bytes();_need(len(raw)==identity[2])
            values.append((str(path),identity,hashlib.sha256(raw).hexdigest()))
        _need(tuple(self._identity(path) for path in self.paths)==before)
        return tuple(values)

    def check(self,paths):
        try:
            _need(not self.failed and self._paths(paths)==self.paths)
            # A second fresh pass catches a same-size/restored-mtime edit of
            # an already-read file while a later dependency was being read.
            for _ in range(2):_need(self._read()==self.expected)
        except Exception:
            self.failed=True;raise ReplayCacheError('fresh_code_vector_refused') from None


def _runtime_code_paths():
    roots=(Path(__file__).absolute().parent,Path(paperbot_us.__file__).absolute().parent)
    _need(all(stat.S_ISDIR(os.stat(root,follow_symlinks=False).st_mode) and not root.is_symlink() for root in roots))
    paths={path.absolute() for root in roots for path in root.glob('*.py')}
    paths.update((Path(contextlib.__file__).absolute(),Path(sys.modules['pathlib'].__file__).absolute()))
    return tuple(sorted(paths))


def _source_pins():
    modules=[];functions=[]
    for name,module in tuple(sys.modules.items()):
        if name not in ('paperbot_us','paperbot_r2') and not name.startswith(('paperbot_us.','paperbot_r2.')):continue
        filename=getattr(module,'__file__',None);_need(type(filename) is str)
        modules.append((module,filename))
        for key in ('source_sha256','_source','_version','math_source_sha256'):
            function=getattr(module,key,None)
            if function is not None:
                _need(type(function) is FunctionType);functions.append((module,key,function,function.__code__))
    return tuple(modules),tuple(functions)


def _check_source_pins(pins):
    modules,functions=pins
    _need(all(module.__file__==filename for module,filename in modules)
          and all(getattr(module,key,None) is function and function.__code__ is code for module,key,function,code in functions))


class _ProtocolCodeGuard:
    """Bound actual root fingerprint and freshly verified complete code vector.

No source-hash/byte reader is replaced or current code bytes retained. The
vector is a superset of the original transitive byte inputs, checked in tests.
Original source/journal verification continues to use the original functions.
"""
    def __init__(self):
        self.expected=None;self.failed=False;self.manifest=None;self.pins=None;self.declared=None
    def check(self):
        try:
            from . import test_protocol
            _need(not self.failed and Path(__file__).name in test_protocol.PROTOCOL_FILES)
            if self.expected is None:
                declared=test_protocol.PROTOCOL_FILES;pins=_source_pins()
                manifest=_FreshCodeManifest(_runtime_code_paths())
                current=test_protocol.source_sha256()  # ORIGINAL complete root.
                manifest.check(_runtime_code_paths());_check_source_pins(pins)
                _need(type(current) is str and len(current)==64
                      and all(c in '0123456789abcdef' for c in current)
                      and test_protocol.PROTOCOL_FILES==declared)
                self.manifest=manifest;self.pins=_source_pins();self.declared=declared;self.expected=current
            else:
                _need(test_protocol.PROTOCOL_FILES==self.declared);_check_source_pins(self.pins)
                self.manifest.check(_runtime_code_paths());_check_source_pins(self.pins)
                _need(test_protocol.PROTOCOL_FILES==self.declared)
        except Exception:
            self.failed=True
            raise ReplayCacheError('replay_runtime_code_epoch_refused') from None


class _CodecScope:
    """Private single-owner codec scope; it grants no phase permission."""
    def __init__(self,memo,code_guard,clock,*,owner=None):
        _need(type(memo) is _DecodeMemo and callable(clock) and callable(code_guard.check))
        self.memo=memo;self.code_guard=code_guard;self.base_clock=clock
        self.owner=threading.get_ident() if owner is None else owner
        self.failed=False;self.installed=False
        self.originals=dict(memo._decoders)
        self.original_codes={k:getattr(v,'__code__',None) for k,v in self.originals.items()}
        self.original_encoder=memo._encoder;self.original_encoder_code=self.original_encoder.__code__
        def decode_input(value):return self._decode('input',value)
        def decode_panel(value):return self._decode('panel',value)
        def decode_snapshot(value):return self._decode('training_snapshot',value)
        def decode_fits(value):return self._decode('training_fits',value)
        self.wrappers={'input':decode_input,'panel':decode_panel,
                       'training_snapshot':decode_snapshot,'training_fits':decode_fits}
        self.wrapper_codes={k:v.__code__ for k,v in self.wrappers.items()}

    def _owned(self):
        _need(not self.failed and self.installed and _ACTIVE_SCOPE is self
              and threading.get_ident()==self.owner and threading.active_count()==1
              and all(getattr(module,name) is self.wrappers[kind]
                      for kind,(module,name,_) in _codec_targets().items())
              and training_codec.encode is self.original_encoder
              and self.original_encoder.__code__ is self.original_encoder_code
              and all(v.__code__ is self.wrapper_codes[k] for k,v in self.wrappers.items())
              and all(v.__code__ is self.original_codes[k] for k,v in self.originals.items()))

    def _fail(self):self.failed=True;self.memo.clear()

    def _decode(self,kind,value):
        try:self._owned();return self.memo.decode(kind,value)
        except Exception:self._fail();raise

    def clock(self):
        try:
            self._owned()
            self.base_clock()  # Phase veto before code/source work.
            self.code_guard.check()  # Real bytes, including restored-mtime edits.
            # Availability must reflect time AFTER that potentially expensive
            # read. A pre-check timestamp must not backdate a COMMIT witness.
            at=self.base_clock();self._owned();return at
        except Exception:self._fail();raise

    def _restore(self):
        global _ACTIVE_SCOPE
        if self.installed:
            for kind,(module,name,_) in _codec_targets().items():setattr(module,name,self.originals[kind])
            training_codec.encode=self.original_encoder
            self.installed=False
            if _ACTIVE_SCOPE is self:_ACTIVE_SCOPE=None

    def __enter__(self):
        global _ACTIVE_SCOPE
        # A refused nested attempt must never clear/restore its outer owner.
        _need(_ACTIVE_SCOPE is None and threading.get_ident()==self.owner and threading.active_count()==1)
        _need(all(_original_codec(k,v) for k,v in self.originals.items())
              and _original_encoder(self.original_encoder) and training_codec.encode is self.original_encoder
              and all(getattr(module,name) is self.originals[kind]
                      for kind,(module,name,_) in _codec_targets().items()))
        try:
            self.base_clock();self.code_guard.check();self.base_clock()
            _ACTIVE_SCOPE=self;self.installed=True
            for kind,(module,name,_) in _codec_targets().items():setattr(module,name,self.wrappers[kind])
            self.clock();return self
        except Exception:self._fail();self._restore();raise

    def __exit__(self,kind,value,traceback):
        try:
            if kind is None:
                self._owned();self.code_guard.check();self._owned()
            else:self._fail()
        except Exception:self._fail();raise
        finally:self._restore()
        return False


def _phase_method(method):
    """Only original journal entry points; never append/create/broker calls."""
    from . import validation_issue_journal,validation_settlements,validation_scores,selection_journal
    from . import test_issue_journal,test_settlements,test_scores
    allowed={
        validation_issue_journal.ValidationIssueJournal:('initialize','issue_once','load_issued'),
        validation_settlements.ValidationSettlementJournal:('initialize','settle_once'),
        validation_scores.ValidationScoreJournal:('initialize','freeze_once','load_frozen'),
        selection_journal.SelectionJournal:('initialize','select_once','load_current'),
        test_issue_journal.TestIssueJournal:('initialize','issue_once','load_issued'),
        test_settlements.TestSettlementJournal:('initialize','settle_once'),
        test_scores.TestScoreJournal:('initialize','freeze_once','load_frozen'),
    }
    _need(type(method) is MethodType and type(method.__self__) in allowed)
    owner=type(method.__self__);name=method.__name__;function=method.__func__
    _need(name in allowed[owner] and getattr(owner,name) is function and type(function) is FunctionType
          and function.__module__==owner.__module__
          and Path(function.__code__.co_filename).absolute()==Path(sys.modules[owner.__module__].__file__).absolute())


class ReplayRuntime:
    """Adapter to original phase APIs, with fresh code-aware clock boundaries.

It never retains admissions, receipts, snapshots, model results or SQL proofs.
It refuses every operation until THIS real module joins the complete protocol
commitment. Actual original policies still decide whether any phase is allowed.
"""
    def __init__(self,*,max_bytes=MAX_BYTES,max_items=MAX_ITEMS):
        self.owner=threading.get_ident();self.memo=_DecodeMemo(max_bytes=max_bytes,max_items=max_items)
        _need(all(_original_codec(k,v) for k,v in self.memo._decoders.items()))
        self.code_guard=_ProtocolCodeGuard()

    def call(self,preflight,method,/,*args,**kwargs):
        _need(callable(preflight) and 'clock' not in kwargs and threading.get_ident()==self.owner)
        preflight()  # Before importing phase factories, code reads, locks or SQL.
        _phase_method(method)
        with _CodecScope(self.memo,self.code_guard,preflight,owner=self.owner) as scope:
            return method(*args,clock=scope.clock,**kwargs)

    def stats(self):
        return dict(self.memo.stats(),runtime_code_sha256=self.code_guard.expected,code_epoch_failed=self.code_guard.failed)
