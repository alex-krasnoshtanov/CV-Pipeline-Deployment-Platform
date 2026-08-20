"""Schemas for the /users endpoint.

Pydantic models for user creation request validation and response
serialisation. The response includes the plaintext API key — this
is the only time it is ever exposed.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CreateUserRequest(BaseModel):
    """POST /users request body."""

    name: str = Field(min_length=1, max_length=200)
    role: str = Field(default="researcher", pattern="^(researcher|admin)$")


class CreateUserResponse(BaseModel):
    """POST /users success response.

    The ``api_key`` field contains the plaintext key — shown once and never stored.
    """

    user_id: str
    name: str
    role: str
    api_key: str
    created_at: datetime
