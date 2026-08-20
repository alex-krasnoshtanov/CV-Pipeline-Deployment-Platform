"""Schemas for the /auth endpoints.

Register and login take an email + password. The MeResponse here
supersedes the smaller one previously inlined in routers/auth.py:
it includes ``email`` so the frontend can show "logged in as
<email>" without an extra round-trip.

Email validation here is a simple structural check (one ``@``,
something either side, a dot in the domain). We deliberately
avoid the ``email-validator`` package — RFC-strict validation
adds a dependency for very little upside on a school project.
The cost of accepting a string the SMTP would later reject is
near-zero since we do not send mail.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Single quote-free, anchored regex. Intentionally lax: we allow
# anything that *looks* like an email (one local-part, one @, a
# domain with a dot). Use a real validator if you wire up SMTP.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class RegisterRequest(BaseModel):
    """POST /auth/register body."""

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=200)
    name: str = Field(min_length=1, max_length=200)

    @field_validator("email")
    @classmethod
    def _check_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL_RE.match(v):
            raise ValueError("Email is not in a valid format.")
        return v


class LoginRequest(BaseModel):
    """POST /auth/login body."""

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=200)

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, v: str) -> str:
        # Lowercase + strip so "Alice@Example.com  " matches
        # "alice@example.com" at lookup time.
        return v.strip().lower()


class UserResponse(BaseModel):
    """Subset of User exposed to authenticated callers.

    Used by /auth/me and by the register/login success responses.
    Never includes password_hash, api_key_hash, or key_sha256.

    auth_method describes which credential the caller is using on
    this request — useful for the frontend to decide which UI to
    show.
    """

    id: str
    name: str
    role: str
    email: str | None = None
    auth_method: Literal["oauth", "password", "api_key"] | None = None
