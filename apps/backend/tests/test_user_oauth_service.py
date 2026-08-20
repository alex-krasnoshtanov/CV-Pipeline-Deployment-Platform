"""Tests for the OAuth user upsert service (Card A7)."""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from api.db import get_db  # noqa: E402
from api.db.models import User  # noqa: E402
from api.services.user_oauth_service import upsert_oauth_user  # noqa: E402

_test_app = FastAPI()


@_test_app.post("/upsert")
async def _upsert_route(
    provider: str,
    subject: str,
    name: str,
    email: str | None = None,
    db=Depends(get_db),
):
    user = await upsert_oauth_user(
        db, provider=provider, subject=subject, email=email, name=name
    )
    return {
        "user_id": str(user.id),
        "name": user.name,
        "oauth_provider": user.oauth_provider,
        "email": user.email,
    }


def _db_for_upsert(existing_user=None):
    """Return a get_db override that simulates the select + commit cycle."""
    exec_result = MagicMock()
    exec_result.scalar_one_or_none.return_value = existing_user
    created_user = []  # Capture for later assertion in new-user path

    async def _get_db():
        session = AsyncMock()
        session.execute.return_value = exec_result
        # For the new-user path, add() receives an unsaved User object.
        # We give it an id so the route can return it.
        if existing_user is None:

            def _fake_add(obj):
                obj.id = uuid.uuid4()
                created_user.append(obj)  # Capture for assertion

            session.add = MagicMock(side_effect=_fake_add)
        session.refresh = AsyncMock()
        yield session

    _get_db.created_user = created_user  # Expose for test access
    return _get_db


@pytest.mark.unit
def test_upsert_creates_new_user_when_not_found():
    """When no matching row exists, a new User is added to the session."""
    db_getter = _db_for_upsert(existing_user=None)
    _test_app.dependency_overrides[get_db] = db_getter
    try:
        with TestClient(_test_app) as c:
            r = c.post(
                "/upsert",
                params={
                    "provider": "github",
                    "subject": "12345",
                    "name": "GH User",
                    "email": "user@example.com",
                },
            )
        assert r.status_code == 200
        assert r.json()["oauth_provider"] == "github"
        assert r.json()["name"] == "GH User"
        # Verify last_login_at is set on new user creation (P3-06)
        assert len(db_getter.created_user) == 1
        assert db_getter.created_user[0].last_login_at is not None
    finally:
        _test_app.dependency_overrides.clear()


@pytest.mark.unit
def test_upsert_updates_name_and_email_for_existing_user():
    """When a matching row exists, name + email are refreshed in-place."""
    existing = MagicMock(spec=User)
    existing.id = uuid.uuid4()
    existing.name = "Old Name"
    existing.email = "old@example.com"
    existing.oauth_provider = "github"
    existing.oauth_subject = "12345"
    existing.role = "researcher"
    existing.last_login_at = datetime(2020, 1, 1, tzinfo=timezone.utc)

    _test_app.dependency_overrides[get_db] = _db_for_upsert(existing_user=existing)
    try:
        with TestClient(_test_app) as c:
            r = c.post(
                "/upsert",
                params={
                    "provider": "github",
                    "subject": "12345",
                    "name": "New Name",
                    "email": "new@example.com",
                },
            )
        assert r.status_code == 200
        # The existing object's fields were mutated (not a new row).
        assert existing.name == "New Name"
        assert existing.email == "new@example.com"
        # Verify last_login_at is advanced on re-login (P3-06)
        assert existing.last_login_at > datetime(2020, 1, 1, tzinfo=timezone.utc)
    finally:
        _test_app.dependency_overrides.clear()


@pytest.mark.unit
def test_upsert_keys_on_provider_subject_not_email():
    """Changing email on an existing subject updates it — no new row is created."""
    existing = MagicMock(spec=User)
    existing.id = uuid.uuid4()
    existing.name = "User"
    existing.email = "original@example.com"
    existing.oauth_provider = "github"
    existing.oauth_subject = "99999"
    existing.role = "researcher"
    existing.last_login_at = None

    _test_app.dependency_overrides[get_db] = _db_for_upsert(existing_user=existing)
    try:
        with TestClient(_test_app) as c:
            r = c.post(
                "/upsert",
                params={
                    "provider": "github",
                    "subject": "99999",
                    "name": "User",
                    "email": "changed@example.com",
                },
            )
        assert r.status_code == 200
        assert r.json()["user_id"] == str(existing.id)
        assert existing.email == "changed@example.com"
    finally:
        _test_app.dependency_overrides.clear()
