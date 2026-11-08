"""Pinned, read-only paper execution configuration; no order authority.

Deployment must provide the expected digest through its administrator-owned
release/unit. Hash identity does not establish strategy profitability or verify
the acceptance artifact; that artifact remains a separate coordinator gate.
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat

from .alpaca import decode_json
from .risk import _identity


class PolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionPolicy:
    schema: int
    capital_usd: int
    execution_enabled: bool
    strategy_sha256: str | None
    acceptance_sha256: str | None
    risk_limits_sha256: str
    max_daily_entries: int
    require_protection: bool

    def __post_init__(self):
        if (type(self.schema) is not int or self.schema != 1 or type(self.capital_usd) is not int
                or self.capital_usd != 10000 or type(self.execution_enabled) is not bool
                or self.require_protection is not True or type(self.max_daily_entries) is not int
                or not 1 <= self.max_daily_entries <= 20):
            raise PolicyError('invalid_paper_execution_policy')
        _identity(self.risk_limits_sha256)
        for value in (self.strategy_sha256, self.acceptance_sha256):
            if value is not None:
                _identity(value)
        if self.execution_enabled and (self.strategy_sha256 is None or self.acceptance_sha256 is None):
            raise PolicyError('strategy_acceptance_reference_required')


@dataclass(frozen=True)
class PolicySnapshot:
    policy: ExecutionPolicy
    digest: str

    def public_summary(self):
        return dict(mode='paper', execution_enabled=False, policy_enabled=self.policy.execution_enabled,
                    protection_required=True, daily_entry_limit=self.policy.max_daily_entries)


class PolicyFile:
    def __init__(self, path, expected_sha256):
        _identity(expected_sha256)
        self.path = Path(path).absolute()
        self.expected_sha256 = expected_sha256

    def read(self):
        try:
            if self.path.is_symlink():
                raise ValueError('policy link refused')
            fd = os.open(self.path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            with os.fdopen(fd, 'rb') as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 8192:
                    raise ValueError('policy file contract')
                raw = handle.read(8193)
            if len(raw) > 8192 or hashlib.sha256(raw).hexdigest() != self.expected_sha256:
                raise ValueError('policy content changed')
            payload = decode_json(raw)
            if set(payload) != set(ExecutionPolicy.__dataclass_fields__):
                raise ValueError('policy fields invalid')
            return PolicySnapshot(ExecutionPolicy(**payload), self.expected_sha256)
        except Exception:
            raise PolicyError('paper_execution_policy_unavailable') from None
