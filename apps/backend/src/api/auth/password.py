"""Password hashing helpers.

Uses bcrypt (already a project dependency for API key hashes), so
we avoid pulling in an additional library. Argon2id is the modern
state of the art, but bcrypt with a sensible cost factor is still
secure and the difference is academic for a school project.

Module-level functions instead of a class: there is no state to
carry. Functional style here is simpler and easier to test.
"""

from __future__ import annotations

import bcrypt


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of *plain* as a UTF-8 string.

    bcrypt.gensalt() defaults to cost=12 which is the current sweet
    spot for interactive logins (roughly 200ms on modern hardware).
    Increase if you ever benchmark and find it cheap.

    Args:
        plain: The plaintext password.

    Returns:
        The bcrypt hash, UTF-8 decoded for storage as Text.
    """
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Return True iff *plain* hashes to *hashed*.

    bcrypt.checkpw is constant-time with respect to the plaintext,
    so it does not leak timing information about password length
    or matching prefixes.

    Args:
        plain: The plaintext password the user just typed.
        hashed: The stored hash from users.password_hash.

    Returns:
        True if they match, False otherwise. Returns False instead
        of raising on malformed hashes so a corrupt row cannot 500
        the login endpoint.
    """
    # Guard against None being passed directly (e.g. OAuth-only user with no
    # password_hash). AttributeError from .encode() would not be caught by
    # (ValueError, TypeError), so we reject non-strings up front.
    if not isinstance(hashed, str):
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False
