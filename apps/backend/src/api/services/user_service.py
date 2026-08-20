"""User management service.

Generates API keys, hashes them with bcrypt, and persists user rows.
The plaintext key is returned to the caller once and never stored.
"""

from __future__ import annotations

import hashlib
import logging
import secrets

import bcrypt
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import User

logger = logging.getLogger(__name__)


def generate_api_key() -> str:
    """Generate a random 32-character hex API key.

    Returns:
        A cryptographically secure random hex string.
    """
    return secrets.token_hex(16)


def _sha256_hex(key: str) -> str:
    """Compute the SHA-256 hex digest of a plaintext API key.

    Args:
        key: The plaintext API key string.

    Returns:
        A 64-character lowercase hex string.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


async def create_user(
    db: AsyncSession,
    name: str,
    role: str,
) -> tuple[User, str]:
    """Create a new user with a generated API key.

    Args:
        db: Active async database session.
        name: Display name for the user.
        role: Either ``'researcher'`` or ``'admin'``.

    Returns:
        A tuple of ``(User, plaintext_key)``. The plaintext key is
        returned once and never stored — the database only holds
        the bcrypt hash and a SHA-256 digest for fast lookup.
    """
    plaintext_key = generate_api_key()
    hashed = bcrypt.hashpw(
        plaintext_key.encode("utf-8"),
        bcrypt.gensalt(),
    )

    user = User(
        name=name,
        api_key_hash=hashed.decode("utf-8"),
        key_sha256=_sha256_hex(plaintext_key),
        role=role,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    logger.info(
        "Created user '%s' (role=%s, id=%s).",
        name,
        role,
        user.id,
    )
    return user, plaintext_key
