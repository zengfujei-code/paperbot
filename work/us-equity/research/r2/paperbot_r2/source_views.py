"""One-cycle immutable view of a deeply verified feature journal.

An open read-only SQLite connection supplies data_version; inode, size and
nanosecond file timestamps additionally detect replacement/in-place rollback.
Any source commit/change invalidates this view rather than refreshing it behind
the caller's back. No source write, long-held read transaction or network I/O.
"""
from types import MappingProxyType
import sqlite3

from . import feature_pipeline


class SourceViewError(ValueError):
    pass


def _need(value):
    if not value:
        raise SourceViewError('verified_feature_view_unavailable')


def _freeze(value):
    if type(value) is dict:
        return MappingProxyType({k:_freeze(v) for k,v in value.items()})
    if type(value) in (list,tuple):
        return tuple(_freeze(v) for v in value)
    return value


class VerifiedFeatureView:
    def __init__(self, journal, *, read_guard):
        self._db = None
        try:
            _need(type(journal) is feature_pipeline.FeatureJournal and callable(read_guard))
            self.path = journal.path
            self._guard = read_guard
            self._guard()
            journal._path()
            self._guard()
            self._db = sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True,isolation_level=None)
            self._db.execute('PRAGMA query_only=ON')
            before = self._identity()
            self._guard()
            state = journal.snapshot(deep=True)
            _need(self._identity()==before and state['source_sha256']==feature_pipeline._source())
            self._identity_at_read = before
            self._source = state['source_sha256']
            self._snapshot = _freeze(state)
        except Exception:
            self.close()
            raise SourceViewError('verified_feature_view_unavailable') from None

    def _identity(self):
        _need(self._db is not None)
        self._guard()
        _need(self.path.is_file() and not self.path.is_symlink() and not self.path.parent.is_symlink())
        stat = self.path.stat()
        self._guard()
        version = self._db.execute('PRAGMA data_version').fetchone()[0]
        return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,version)

    def snapshot(self, *, deep=False):
        try:
            _need(self._identity()==self._identity_at_read
                  and feature_pipeline._source()==self._source)
            return self._snapshot
        except Exception:
            raise SourceViewError('verified_feature_view_changed_or_closed') from None

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None

    def __enter__(self):
        self.snapshot()
        return self

    def __exit__(self, *_):
        self.close()
