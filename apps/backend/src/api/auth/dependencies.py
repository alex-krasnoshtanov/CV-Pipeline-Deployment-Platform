"""Unified auth dependencies.

These are the dependencies new routes should use. They accept
BOTH a ``session_id`` cookie (the web flow), an
``Authorization: Bearer <jwt>`` header (the OAuth/JWT web flow),
and an ``X-API-Key`` header (the legacy CLI / Streamlit /
service-account flow). The caller picks one tier:

- ``optional_user`` -> ``User | None``. For routes that work
  for anonymous callers (e.g. ``/infer``) but want to know who
  the user is when they ARE logged in (e.g. to attach
  ``user_id`` to a prediction row, or to apply a generous rate
  limit).

- ``require_user`` -> ``User``. For routes that require an
  authenticated user (e.g. ``/feedback``, ``/predictions``).
  Raises 401 if both credentials are missing or invalid.

- ``require_admin`` -> ``User``. For admin-only routes
  (``/users``). Raises 403 if the user is not an admin.

Why a single dep instead of two parallel paths (one for cookie,
one for X-API-Key)? The route handler only cares about the User
identity. Multiplexing here means routers stay clean and the
auth decision is made in one place that is easy to audit.
"""

from __future__ import annotations

import logging

from fastapi import Cookie, Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.api_key import lookup_user_by_api_key
from api.auth.jwt import resolve_jwt
from api.auth.session import get_user_from_session
from api.db import get_db
from api.db.models import User

logger = logging.getLogger(__name__)

# Single source of truth for the cookie name. Kept here (not in
# session.py) because both the dependency and the auth router
# need it, and depending router-on-session is fine but session
# should not depend on Cookie() semantics.
SESSION_COOKIE_NAME = "session_id"


async def optional_user(
    authorization: str | None = Header(default=None),
    session_id: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """Return the authenticated User, or None for anonymous callers.

    Resolution order:

    1. Authorization: Bearer <jwt> - the OAuth/JWT web flow
       If present but invalid, raises 401 (fail-closed for security).
    2. Session cookie              - the email+password web flow
       If invalid, silently falls through to next branch.
    3. X-API-Key                   - the legacy CLI / service flow
       If invalid, silently falls through to next branch.
    4. None                        - truly anonymous

    Only JWT raises 401 on invalid credentials. Cookies and API keys
    that are present but invalid fall through to the next branch
    (eventually anonymous). This design allows callers to use a
    convenience mechanism (persistent cookie) without hard-failing
    if it becomes invalid (e.g., expired or malformed).

    The route handler is responsible for deciding what None means.
    ``/infer`` treats it as "save the prediction with user_id=NULL
    and apply the lower rate limit". ``require_user`` raises 401.
    """
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        user = await resolve_jwt(token, db)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error_code": "UNAUTHORIZED",
                    "message": "Invalid or expired token.",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user

    if session_id is not None:
        user = await get_user_from_session(db, session_id)
        if user is not None:
            return user

    if x_api_key is not None:
        user = await lookup_user_by_api_key(db, x_api_key)
        if user is not None:
            return user

    return None


async def require_user(
    user: User | None = Depends(optional_user),
) -> User:
    """Return the authenticated User or raise 401.

    Use on any route where anonymous access is not acceptable
    (anything that mutates user-owned data, reveals private data,
    or grants privileged access).
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": "UNAUTHORIZED",
                "message": "Authentication required. Log in or send X-API-Key.",
            },
            headers={"WWW-Authenticate": "Cookie"},
        )
    return user


async def require_admin(
    user: User = Depends(require_user),
) -> User:
    """Return the authenticated User or raise 403 if not admin.

    Layered on ``require_user`` so a missing credential still
    returns 401 (not 403) — that distinction matters for the
    frontend, which redirects to login on 401 but shows a
    "forbidden" page on 403.
    """
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error_code": "FORBIDDEN",
                "message": "Admin role required.",
            },
        )
    return user
