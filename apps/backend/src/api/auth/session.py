"""Server-side session management.

A session row is the source of truth for "is this cookie valid".
The cookie value sent to the browser is the session UUID; on every
authenticated request we look it up, check expiry, and either
return the linked User or treat the request as anonymous.

Why not JWT:
- Revocation. DELETE FROM sessions WHERE id = ... is instant. JWT
  needs a blacklist table you have to consult on every request,
  at which point you have the same DB hit but with extra signing
  cryptography on top.
- Simplicity. No JWT secret to rotate, no signing algorithm choice
  to second-guess, no clock-skew edge cases.

Default session TTL is 7 days. Sliding expiry (extending the
session on every request) is deliberately NOT implemented in this
first cut: it adds a write per request and makes test fixtures
flaky. Easy to add later if needed.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import Session, User

logger = logging.getLogger(__name__)

DEFAULT_SESSION_TTL = timedelta(days=7)


async def create_session(
    db: AsyncSession,
    user_id: uuid.UUID,
    ttl: timedelta = DEFAULT_SESSION_TTL,
) -> Session:
    """Create and persist a new session for *user_id*.

    Also opportunistically deletes any expired sessions for the
    same user so the table does not accumulate dead rows. This is
    cheap because the cleanup is scoped to one user and uses the
    ix_sessions_expires_at index.

    Args:
        db: Active async DB session.
        user_id: UUID of the user the session belongs to.
        ttl: How long the session is valid. Default 7 days.

    Returns:
        The newly inserted Session row (refreshed, so .id is set).
    """
    now = datetime.now(UTC)

    # Opportunistic cleanup. We do not delete OTHER users' expired
    # sessions here on purpose: a periodic job is the right tool
    # for that. This keeps login latency bounded.
    await db.execute(
        delete(Session)
        .where(Session.user_id == user_id)
        .where(Session.expires_at < now)
    )

    session_row = Session(
        user_id=user_id,
        expires_at=now + ttl,
    )
    db.add(session_row)
    await db.commit()
    await db.refresh(session_row)
    return session_row


async def get_user_from_session(
    db: AsyncSession,
    session_id: str,
) -> User | None:
    """Return the User attached to *session_id*, or None if invalid.

    "Invalid" covers four cases, all treated identically:
    - Malformed UUID (the cookie was tampered with or never valid).
    - No matching row (session was deleted, i.e. logged out).
    - Row exists but expired (cookie outlived the row's TTL).
    - DB lookup failed (logged but not raised — we degrade to
      anonymous rather than 500 the request).

    Args:
        db: Active async DB session.
        session_id: The raw cookie value the client sent.

    Returns:
        The User if the session is valid, otherwise None.
    """
    try:
        session_uuid = uuid.UUID(session_id)
    except (ValueError, TypeError):
        return None

    try:
        result = await db.execute(
            select(Session, User)
            .join(User, Session.user_id == User.id)
            .where(Session.id == session_uuid)
        )
        row = result.first()
    except Exception:
        # Defensive: a DB hiccup should not log everyone out with
        # a 500. Treat it as "anonymous" and let the endpoint
        # decide whether that is allowed.
        logger.exception("Session lookup failed for id=%s", session_uuid)
        return None

    if row is None:
        return None

    session_row, user = row
    if session_row.expires_at < datetime.now(UTC):
        return None

    return user


async def delete_session(db: AsyncSession, session_id: str) -> None:
    """Delete the session row for *session_id* if it exists.

    Idempotent — deleting a non-existent or malformed session is
    a no-op. The logout endpoint relies on this so it never 404s.
    """
    try:
        session_uuid = uuid.UUID(session_id)
    except (ValueError, TypeError):
        return

    await db.execute(delete(Session).where(Session.id == session_uuid))
    await db.commit()
