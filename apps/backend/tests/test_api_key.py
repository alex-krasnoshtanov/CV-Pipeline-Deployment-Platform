"""Unit tests for the API-key authentication module.

Covers the two entry points in api.auth.api_key:
- lookup_user_by_api_key: non-raising helper
- require_api_key: legacy FastAPI dependency that raises on failure

These tests use a minimal in-process FastAPI app with a DB override so they
run without a live database or the cv_pipeline package.
"""

from __future__ import annotations

import hashlib
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import bcrypt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from api.auth.api_key import lookup_user_by_api_key, require_api_key  # noqa: E402
from api.db import get_db  # noqa: E402
from api.db.models import User  # noqa: E402

_VALID_KEY = "test-api-key-plaintext"
_VALID_HASH = bcrypt.hashpw(_VALID_KEY.encode(), bcrypt.gensalt()).decode()
_VALID_DIGEST = hashlib.sha256(_VALID_KEY.encode()).hexdigest()


def _make_user(api_key_hash: str | None = _VALID_HASH) -> MagicMock:
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.name = "Test User"
    user.role = "researcher"
    user.api_key_hash = api_key_hash
    user.key_sha256 = _VALID_DIGEST
    return user


def _db_returns(user: MagicMock | None, has_any_user: bool = True):
    """Return a get_db override yielding a session pre-wired with *user*."""

    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = user

    scalars_result = MagicMock()
    scalars_result.first.return_value = MagicMock() if has_any_user else None

    async def _override():
        session = AsyncMock()
        session.execute.return_value = execute_result
        session.scalars.return_value = scalars_result
        yield session

    return _override


_app = FastAPI()


@_app.get("/protected")
async def _protected(user: User = Depends(require_api_key)):
    return {"user_id": str(user.id), "role": user.role}


# ---------------------------------------------------------------------------
# lookup_user_by_api_key
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.anyio
async def test_lookup_returns_user_for_valid_key():
    user = _make_user()
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = user
    session.execute.return_value = result

    found = await lookup_user_by_api_key(session, _VALID_KEY)
    assert found is user


@pytest.mark.unit
@pytest.mark.anyio
async def test_lookup_returns_none_when_no_row():
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute.return_value = result

    found = await lookup_user_by_api_key(session, _VALID_KEY)
    assert found is None


@pytest.mark.unit
@pytest.mark.anyio
async def test_lookup_returns_none_when_api_key_hash_is_none():
    user = _make_user(api_key_hash=None)
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = user
    session.execute.return_value = result

    found = await lookup_user_by_api_key(session, _VALID_KEY)
    assert found is None


@pytest.mark.unit
@pytest.mark.anyio
async def test_lookup_returns_none_for_malformed_hash():
    """The except (ValueError, TypeError) branch: bcrypt raises on bad hash."""
    user = _make_user(api_key_hash="not-a-bcrypt-hash")
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = user
    session.execute.return_value = result

    found = await lookup_user_by_api_key(session, _VALID_KEY)
    assert found is None


@pytest.mark.unit
@pytest.mark.anyio
async def test_lookup_returns_none_for_wrong_key():
    """Correct hash in DB but different key should not match."""
    user = _make_user()
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = user
    session.execute.return_value = result

    found = await lookup_user_by_api_key(session, "wrong-key")
    assert found is None


# ---------------------------------------------------------------------------
# require_api_key (legacy FastAPI dependency)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_require_missing_header_returns_401():
    _app.dependency_overrides[get_db] = _db_returns(_make_user())
    try:
        with TestClient(_app) as client:
            response = client.get("/protected")
        assert response.status_code == 401
        assert response.json()["detail"]["error_code"] == "UNAUTHORIZED"
        assert "Missing" in response.json()["detail"]["message"]
    finally:
        _app.dependency_overrides.pop(get_db, None)


@pytest.mark.unit
def test_require_valid_key_returns_200():
    _app.dependency_overrides[get_db] = _db_returns(_make_user())
    try:
        with TestClient(_app) as client:
            response = client.get("/protected", headers={"X-API-Key": _VALID_KEY})
        assert response.status_code == 200
        assert response.json()["role"] == "researcher"
    finally:
        _app.dependency_overrides.pop(get_db, None)


@pytest.mark.unit
def test_require_invalid_key_nonempty_table_returns_401():
    """Wrong key but users exist → 401 UNAUTHORIZED (not 503)."""
    _app.dependency_overrides[get_db] = _db_returns(None, has_any_user=True)
    try:
        with TestClient(_app) as client:
            response = client.get("/protected", headers={"X-API-Key": "bad-key"})
        assert response.status_code == 401
        assert response.json()["detail"]["error_code"] == "UNAUTHORIZED"
        assert "Invalid" in response.json()["detail"]["message"]
    finally:
        _app.dependency_overrides.pop(get_db, None)


@pytest.mark.unit
def test_require_invalid_key_empty_table_returns_503():
    """Wrong key and empty users table → 503 SERVER_MISCONFIGURED."""
    _app.dependency_overrides[get_db] = _db_returns(None, has_any_user=False)
    try:
        with TestClient(_app) as client:
            response = client.get("/protected", headers={"X-API-Key": "bad-key"})
        assert response.status_code == 503
        assert response.json()["detail"]["error_code"] == "SERVER_MISCONFIGURED"
    finally:
        _app.dependency_overrides.pop(get_db, None)
