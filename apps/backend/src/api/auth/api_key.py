"""API key authentication via the ``X-API-Key`` header.

Uses a two-step verification pattern (same as Stripe / GitHub):

1. Compute SHA-256 of the incoming key and query the indexed
   ``key_sha256`` column - O(1) database lookup.
2. Run ``bcrypt.checkpw`` once on the single matching row - O(1).

This keeps auth at constant time regardless of user count.

This module exposes two entry points:
- ``lookup_user_by_api_key``: pure helper returning ``User | None``.
  Used by the new ``optional_user`` dependency, which needs to
  silently fall through to anonymous on a bad key.
- ``require_api_key``: legacy FastAPI dependency that raises 401
  / 503 on failure. Kept for backward compatibility with the
  existing Streamlit frontend, CLI scripts, and unit tests.
"""

from __future__ import annotations

import hashlib
import logging

import bcrypt
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db import get_db
from api.db.models import User

logger = logging.getLogger(__name__)


def _sha256_hex(key: str) -> str:
    """Compute the SHA-256 hex digest of a plaintext API key.

    Args:
        key: The plaintext API key string.

    Returns:
        A 64-character lowercase hex string.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


async def lookup_user_by_api_key(
    db: AsyncSession,
    api_key: str,
) -> User | None:
    """Return the User matching *api_key*, or None on any failure.

    This is the non-raising sibling of ``require_api_key``. It is
    used by ``optional_user`` so that a missing or bad key on a
    public endpoint (e.g. ``/infer``) silently falls through to
    anonymous rather than 401-ing the request.

    Args:
        db: Active async DB session.
        api_key: The raw value of the X-API-Key header.

    Returns:
        The User row if the key is valid, None otherwise. Does not
        raise. Does not distinguish "empty table" from "wrong key" —
        that distinction is only useful for the legacy dependency.
    """
    key_digest = _sha256_hex(api_key)
    result = await db.execute(
        select(User).where(User.key_sha256 == key_digest),
    )
    user = result.scalar_one_or_none()
    if user is None or user.api_key_hash is None:
        return None

    try:
        if bcrypt.checkpw(api_key.encode("utf-8"), user.api_key_hash.encode("utf-8")):
            return user
    except (ValueError, TypeError):
        # Malformed hash row — treat as a non-match.
        return None
    return None


async def require_api_key(
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
        description="Per-user API key checked against the users table.",
    ),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Legacy dependency: 401 on missing/bad key, 503 on empty table.

    Kept intact for the Streamlit frontend, CLI scripts, and tests
    that still rely on the X-API-Key-only flow. New routes should
    prefer ``require_user`` from ``api.auth.dependencies``, which
    accepts both session cookies and X-API-Key.

    Raises:
        HTTPException: 401 if the header is missing or invalid,
            503 if the users table is empty (misconfiguration).
    """
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": "UNAUTHORIZED",
                "message": "Missing X-API-Key header.",
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )

    user = await lookup_user_by_api_key(db, x_api_key)
    if user is not None:
        logger.debug(
            "Authenticated user '%s' (role=%s) via X-API-Key.",
            user.name,
            user.role,
        )
        return user

    # Distinguish empty table (misconfiguration) from wrong key so
    # operators see SERVER_MISCONFIGURED in logs instead of a
    # flood of UNAUTHORIZED noise during a botched first deploy.
    count_result = await db.scalars(select(User.id).limit(1))
    if count_result.first() is None:
        logger.error("Users table is empty - no users have been seeded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "SERVER_MISCONFIGURED",
                "message": "No users configured. Run the seed script.",
            },
        )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "error_code": "UNAUTHORIZED",
            "message": "Invalid API key.",
        },
        headers={"WWW-Authenticate": "ApiKey"},
    )
