"""Upsert a User row from an OAuth provider's userinfo response.

Identity key: ``(oauth_provider, oauth_subject)``. The subject is the
provider's stable opaque identifier (GitHub user id). Email is stored
for display but is NOT the identity key — emails can change, subjects
cannot.

This module has no HTTP knowledge. The auth router calls it after
successfully exchanging the OAuth code for a token and fetching the
user's profile.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import User

logger = logging.getLogger(__name__)


async def upsert_oauth_user(
    db: AsyncSession,
    *,
    provider: str,
    subject: str,
    email: str | None,
    name: str,
) -> User:
    """Create or update a User row for the given OAuth identity.

    If a row with the same ``(provider, subject)`` already exists, its
    ``name``, ``email``, and ``last_login_at`` are refreshed. If not, a
    new row is created with ``role='researcher'`` and no API key columns
    set (they are now nullable).

    Args:
        db: Async database session (transaction will be committed here).
        provider: OAuth provider name, e.g. ``"github"``.
        subject: Stable opaque user identifier from the provider.
        email: User's email, or ``None`` if the provider withheld it.
        name: Display name from the provider's profile.

    Returns:
        The freshly committed ``User`` row.
    """
    result = await db.execute(
        select(User).where(
            User.oauth_provider == provider,
            User.oauth_subject == subject,
        )
    )
    user = result.scalar_one_or_none()
    now = datetime.now(timezone.utc)

    if user is None:
        logger.info(
            "Creating new user via OAuth: provider=%s subject=%s", provider, subject
        )
        user = User(
            name=name,
            email=email,
            oauth_provider=provider,
            oauth_subject=subject,
            role="researcher",
            last_login_at=now,
        )
        db.add(user)
    else:
        logger.debug("Refreshing existing user id=%s via OAuth login.", user.id)
        # Refresh display fields — they can change on the provider's side.
        user.name = name
        user.email = email
        user.last_login_at = now

    await db.commit()
    await db.refresh(user)
    return user
