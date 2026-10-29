"""API-key auth, per-key rate limiting, and role -> retrieval scope for FastAPI.

The scope is the real access-control boundary: it is applied at retrieval time, to
BOTH halves of hybrid search, before any chunk reaches the LLM.

  scope = build_scope(principal, company, doc_type)   # 403 if the role may not see it
  where = scope_to_chroma_where(scope)                # dense side: Chroma `where=`
  scope_allows(scope, chunk_meta)                     # sparse side: filter BM25 hits

Config (env):
  AUTH_MODE           apikey (default) | off   (off = local dev only, logs a warning)
  API_KEYS            "key1:admin,key2:analyst,key3:public"
  RATE_LIMIT_PER_MIN  per-key requests per minute (default 30)
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Optional

from fastapi import Depends, Header, HTTPException

log = logging.getLogger("finance_rag.security")

# role -> allowed metadata values. {} = unrestricted. Every list must be non-empty.
ROLE_SCOPES: dict[str, dict[str, list[str]]] = {
    "admin": {},
    "analyst": {"doc_type": ["rbi", "company"]},
    "public": {"doc_type": ["rbi"]},
}
assert all(v for scope in ROLE_SCOPES.values() for v in scope.values())

Scope = dict[str, list[str]]
_warned_auth_off = False


@dataclass(frozen=True)
class Principal:
    key_id: str  # first 8 hex chars of sha256(key); the raw key is never stored or logged
    role: str


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def parse_api_keys(raw: str) -> dict[str, str]:
    """'k1:admin,k2:analyst' -> {sha256(k1): 'admin', sha256(k2): 'analyst'}."""
    out: dict[str, str] = {}
    for item in (x.strip() for x in raw.split(",")):
        if not item:
            continue
        key, sep, role = item.partition(":")
        if not sep or not key or role not in ROLE_SCOPES:
            raise ValueError(
                f"bad API_KEYS entry: expected key:role, role in {sorted(ROLE_SCOPES)}"
            )
        out[_hash(key)] = role
    return out


def get_principal(x_api_key: Optional[str] = Header(default=None)) -> Principal:
    global _warned_auth_off
    if os.getenv("AUTH_MODE", "apikey").lower() == "off":
        if not _warned_auth_off:
            log.warning("AUTH_MODE=off: every request runs as admin. Local dev only.")
            _warned_auth_off = True
        return Principal("anon", "admin")
    keys = parse_api_keys(os.getenv("API_KEYS", ""))
    if not keys:
        raise HTTPException(503, "auth not configured: set API_KEYS")
    digest = _hash(x_api_key) if x_api_key else None
    role = keys.get(digest) if digest else None
    if role is None:
        raise HTTPException(401, "invalid or missing API key")
    return Principal(digest[:8], role)


def build_scope(
    principal: Principal, company: Optional[str] = None, doc_type: Optional[str] = None
) -> Scope:
    """Role scope merged with the caller's requested filters. Fails closed."""
    if principal.role not in ROLE_SCOPES:
        raise HTTPException(403, "unknown role")
    base = {k: list(v) for k, v in ROLE_SCOPES[principal.role].items()}
    scope = dict(base)
    for field, requested in (("company", company), ("doc_type", doc_type)):
        if requested is None:
            continue
        allowed = base.get(field)
        if allowed is not None and requested not in allowed:
            raise HTTPException(
                403, f"role '{principal.role}' cannot query {field}='{requested}'"
            )
        scope[field] = [requested]
    return scope


def scope_to_chroma_where(scope: Scope) -> Optional[dict]:
    clauses = [{f: {"$in": vals}} for f, vals in scope.items()]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def scope_allows(scope: Scope, meta: dict) -> bool:
    return all(meta.get(f) in vals for f, vals in scope.items())


class SlidingWindowLimiter:
    """In-memory, per-process. Behind multiple workers, move the counters to Redis."""

    def __init__(
        self,
        max_calls: int,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.max_calls, self.window_s, self.clock = max_calls, window_s, clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds)."""
        with self._lock:
            now, q = self.clock(), self._hits[key]
            while q and now - q[0] >= self.window_s:
                q.popleft()
            if len(q) >= self.max_calls:
                return False, self.window_s - (now - q[0])
            q.append(now)
            return True, 0.0


_limiter = SlidingWindowLimiter(int(os.getenv("RATE_LIMIT_PER_MIN", "30")))


def enforce_rate_limit(principal: Principal = Depends(get_principal)) -> Principal:
    """Single dependency for the route: authenticates, then rate-limits."""
    allowed, retry_after = _limiter.hit(principal.key_id)
    if not allowed:
        raise HTTPException(
            429,
            "rate limit exceeded",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )
    return principal
