"""Mint and verify backend-issued HS256 JWTs.

Why HS256 and not RS256?
    We have a single backend that both issues and verifies tokens — there is
    no second service that needs to verify without being able to issue. HS256
    is simpler and faster. If a second service ever needs to verify tokens,
    switch to RS256 (two-line change here, key rotation needed everywhere).

Claims are intentionally minimal: ``sub`` (user id UUID), ``role``, ``iat``,
``exp``. Name and email are fetched from the DB on demand via ``resolve_jwt``.
This keeps tokens short and avoids the "stale claims" problem: a role change
takes effect on the next token issuance, not after the old one expires.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import jwt as pyjwt  # pyjwt[crypto] — ``pip install pyjwt[crypto]``
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import User

_ALGORITHM = "HS256"


def _signing_key() -> str:
    """Read the JWT signing key from the environment.

    Returns:
        The JWT_SIGNING_KEY environment variable value.

    Raises:
        RuntimeError: If ``JWT_SIGNING_KEY`` is not set or empty.
    """
    key = os.getenv("JWT_SIGNING_KEY", "")
    if not key:
        raise RuntimeError(
            "JWT_SIGNING_KEY is not configured. Generate one with: openssl rand -hex 32"
        )
    return key


def _ttl_minutes() -> int:
    """Read the JWT time-to-live from the environment.

    Returns:
        ``JWT_TTL_MIN`` as an integer, defaulting to 60 minutes.

    Raises:
        RuntimeError: If ``JWT_TTL_MIN`` is not a valid positive integer.
    """
    ttl_str = os.getenv("JWT_TTL_MIN", "60")
    try:
        ttl = int(ttl_str)
    except ValueError:
        raise RuntimeError(f"JWT_TTL_MIN must be a valid integer, got: {ttl_str!r}")
    if ttl <= 0:
        raise RuntimeError(f"JWT_TTL_MIN must be positive, got: {ttl}")
    return ttl


def mint_backend_jwt(user: User) -> str:
    """Create a signed JWT for the given user.

    The returned string should be sent to the client via the OAuth redirect
    query parameter. The client stores it and sends it as
    ``Authorization: Bearer <token>`` on subsequent requests.

    Args:
        user: The authenticated ``User`` row to encode into the token.

    Returns:
        A signed JWT string.

    Raises:
        RuntimeError: If ``JWT_SIGNING_KEY`` is not configured.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user.id),
        "role": user.role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=_ttl_minutes())).timestamp()),
    }
    return pyjwt.encode(payload, _signing_key(), algorithm=_ALGORITHM)


async def resolve_jwt(token: str, db: AsyncSession) -> User | None:
    """Verify a token and return the associated user row.

    Returns ``None`` for any verification failure: expired token, bad
    signature, malformed token, or ``sub`` not found in the database. The
    caller decides what to do — raise 401 or fall through to another auth
    path.

    Why ``None`` instead of raising an exception?
        The dependency that calls this function (``get_current_user_optional``)
        needs to distinguish "valid JWT" from "invalid JWT" and raise 401 for
        the latter. Returning None gives the caller that control without
        coupling this module to FastAPI's exception model.

    Args:
        token: The JWT string from the ``Authorization: Bearer`` header.
        db: Async database session.

    Returns:
        The ``User`` row if verification succeeds, otherwise ``None``.
    """
    try:
        payload = pyjwt.decode(
            token,
            _signing_key(),
            algorithms=[_ALGORITHM],
            options={"require": ["sub", "iat", "exp"]},
        )
    except pyjwt.PyJWTError:
        # Covers: expired, bad signature, malformed, or missing required claims.
        return None

    sub = payload.get("sub")
    if not sub:
        return None

    try:
        user_id = uuid.UUID(sub)
    except (ValueError, TypeError):
        # sub is not a valid UUID string, or not a string at all.
        return None

    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()
