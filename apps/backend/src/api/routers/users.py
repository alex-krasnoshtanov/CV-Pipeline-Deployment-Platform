"""POST /users - create a new user (admin only).

This is the LEGACY user-provisioning path: an admin mints a new
account, the response carries a plaintext API key, the admin hands
the key over out-of-band. Still used for service accounts (the
robotic platform, CLI scripts).

For human users, POST /auth/register is the preferred path: it
takes an email + password, sets a session cookie, and never
emits an API key. Both paths coexist on the same users table.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import require_admin
from api.db import get_db
from api.db.models import User
from api.schemas.users import CreateUserRequest, CreateUserResponse
from api.services.user_service import create_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])


@router.post(
    "",
    response_model=CreateUserResponse,
    summary="Create a new API-key user (admin only).",
)
async def add_user(
    body: CreateUserRequest,
    current_admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Create a new user and return their generated API key.

    Admin-only. The plaintext API key is included in the response
    and never stored - the admin must copy it immediately and
    deliver it to the new user out-of-band.

    Args:
        body: User creation payload with name and role.
        current_admin: Authenticated admin (enforced by dep).
        db: Async database session.

    Returns:
        A dict serialised through ``CreateUserResponse``.

    Raises:
        HTTPException: 401 if not authenticated, 403 if not admin.
    """
    _ = current_admin  # auth gate only; the new user's identity comes from body

    user, plaintext_key = await create_user(
        db=db,
        name=body.name,
        role=body.role,
    )

    return {
        "user_id": str(user.id),
        "name": user.name,
        "role": user.role,
        "api_key": plaintext_key,
        "created_at": user.created_at,
    }
