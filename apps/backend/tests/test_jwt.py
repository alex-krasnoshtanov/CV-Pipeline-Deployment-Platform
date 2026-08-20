"""Unit tests for the JWT mint + resolve functions (Card A5).

All tests run synchronously via a minimal FastAPI test app rather than
pytest-asyncio, keeping the test stack consistent with the rest of the
backend test suite.

The test app exposes two routes:
  GET /mint  → mints a JWT for the mock user, returns {token}
  GET /verify?token=<jwt>  → resolves the token, returns {user_id} or null
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import jwt as pyjwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from api.auth.jwt import mint_backend_jwt, resolve_jwt  # noqa: E402
from api.db import get_db  # noqa: E402
from api.db.models import User  # noqa: E402

# ---------------------------------------------------------------------------
# Test app + fixtures
# ---------------------------------------------------------------------------

_TEST_KEY = "test-signing-key-32-chars-exactly!!"
_ALGORITHM = "HS256"


def _make_user(role: str = "researcher") -> MagicMock:
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.role = role
    user.name = "Test User"
    user.email = "test@example.com"
    return user


_test_app = FastAPI()
_test_user = _make_user()


@_test_app.get("/mint")
async def _mint_route():
    return {"token": mint_backend_jwt(_test_user)}


@_test_app.get("/verify")
async def _verify_route(token: str, db=Depends(get_db)):
    user = await resolve_jwt(token, db)
    return {"user_id": str(user.id) if user else None}


def _db_returning(user):
    """Return a get_db override that makes the DB yield ``user``."""
    exec_result = MagicMock()
    exec_result.scalar_one_or_none.return_value = user

    async def _override():
        session = AsyncMock()
        session.execute.return_value = exec_result
        yield session

    return _override


@pytest.fixture(autouse=True)
def _set_jwt_key(monkeypatch):
    monkeypatch.setenv("JWT_SIGNING_KEY", _TEST_KEY)
    monkeypatch.setenv("JWT_TTL_MIN", "60")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mint_and_resolve_roundtrip():
    """Mint a JWT, resolve it against the DB, get the same user back."""
    _test_app.dependency_overrides[get_db] = _db_returning(_test_user)
    try:
        with TestClient(_test_app) as c:
            token = c.get("/mint").json()["token"]
            r = c.get("/verify", params={"token": token})
        assert r.status_code == 200
        assert r.json()["user_id"] == str(_test_user.id)
    finally:
        _test_app.dependency_overrides.clear()


@pytest.mark.unit
def test_resolve_expired_jwt_returns_none():
    """A token whose ``exp`` is in the past returns None."""
    expired_payload = {
        "sub": str(_test_user.id),
        "role": "researcher",
        "iat": 1_000_000,
        "exp": 1_000_060,  # 1970 — definitely expired
    }
    token = pyjwt.encode(expired_payload, _TEST_KEY, algorithm=_ALGORITHM)
    _test_app.dependency_overrides[get_db] = _db_returning(_test_user)
    try:
        with TestClient(_test_app) as c:
            r = c.get("/verify", params={"token": token})
        assert r.json()["user_id"] is None
    finally:
        _test_app.dependency_overrides.clear()


@pytest.mark.unit
def test_resolve_tampered_signature_returns_none():
    """A token with a replaced signature segment returns None.

    We replace the whole signature rather than flipping one character because
    the last base64url character may only carry 2-4 significant bits (the
    rest are ignored padding). Flipping it can leave the decoded byte
    sequence identical, causing the HMAC to still verify. Replacing the
    entire segment avoids this edge case entirely.
    """
    _test_app.dependency_overrides[get_db] = _db_returning(_test_user)
    try:
        with TestClient(_test_app) as c:
            good_token = c.get("/mint").json()["token"]
        header, payload, _ = good_token.split(".")
        # "thisiswrongsignature" in base64url — always invalid
        bad_token = f"{header}.{payload}.dGhpc2lzd3JvbmdzaWduYXR1cmU"
        with TestClient(_test_app) as c:
            r = c.get("/verify", params={"token": bad_token})
        assert r.json()["user_id"] is None
    finally:
        _test_app.dependency_overrides.clear()


@pytest.mark.unit
def test_resolve_wrong_signing_key_returns_none(monkeypatch):
    """A token signed with a different key returns None."""
    other_key = "completely-different-signing-key-!!"
    payload = {
        "sub": str(_test_user.id),
        "role": "researcher",
        "iat": int(datetime.now(timezone.utc).timestamp()),
        "exp": int(datetime.now(timezone.utc).timestamp()) + 3600,
    }
    token = pyjwt.encode(payload, other_key, algorithm=_ALGORITHM)
    _test_app.dependency_overrides[get_db] = _db_returning(_test_user)
    try:
        with TestClient(_test_app) as c:
            r = c.get("/verify", params={"token": token})
        assert r.json()["user_id"] is None
    finally:
        _test_app.dependency_overrides.clear()


@pytest.mark.unit
def test_resolve_malformed_token_returns_none():
    """A string that is not a JWT at all returns None — no exception raised."""
    _test_app.dependency_overrides[get_db] = _db_returning(_test_user)
    try:
        with TestClient(_test_app) as c:
            r = c.get("/verify", params={"token": "not.a.jwt"})
        assert r.json()["user_id"] is None
    finally:
        _test_app.dependency_overrides.clear()


@pytest.mark.unit
def test_resolve_unknown_user_returns_none():
    """Valid signature but ``sub`` UUID not in the database returns None."""
    payload = {
        "sub": str(uuid.uuid4()),  # random UUID, not in DB
        "role": "researcher",
        "iat": int(datetime.now(timezone.utc).timestamp()),
        "exp": int(datetime.now(timezone.utc).timestamp()) + 3600,
    }
    token = pyjwt.encode(payload, _TEST_KEY, algorithm=_ALGORITHM)
    # DB returns None — user not found.
    _test_app.dependency_overrides[get_db] = _db_returning(None)
    try:
        with TestClient(_test_app) as c:
            r = c.get("/verify", params={"token": token})
        assert r.json()["user_id"] is None
    finally:
        _test_app.dependency_overrides.clear()


@pytest.mark.unit
def test_resolve_non_uuid_sub_returns_none():
    """A token where ``sub`` is not a UUID string returns None."""
    payload = {
        "sub": "hello-i-am-not-a-uuid",
        "role": "researcher",
        "iat": int(datetime.now(timezone.utc).timestamp()),
        "exp": int(datetime.now(timezone.utc).timestamp()) + 3600,
    }
    token = pyjwt.encode(payload, _TEST_KEY, algorithm=_ALGORITHM)
    _test_app.dependency_overrides[get_db] = _db_returning(_test_user)
    try:
        with TestClient(_test_app) as c:
            r = c.get("/verify", params={"token": token})
        assert r.json()["user_id"] is None
    finally:
        _test_app.dependency_overrides.clear()


@pytest.mark.unit
def test_mint_raises_when_signing_key_unset(monkeypatch):
    """``mint_backend_jwt`` raises RuntimeError when JWT_SIGNING_KEY is empty."""
    monkeypatch.setenv("JWT_SIGNING_KEY", "")
    with pytest.raises(RuntimeError, match="JWT_SIGNING_KEY"):
        mint_backend_jwt(_test_user)


@pytest.mark.unit
def test_resolve_token_missing_exp_returns_none():
    """A token without the exp claim is rejected (P3-07 fix)."""
    # Payload missing "exp" — pyjwt.decode requires it now.
    payload = {
        "sub": str(_test_user.id),
        "role": "researcher",
        "iat": int(datetime.now(timezone.utc).timestamp()),
        # "exp" is intentionally omitted
    }
    token = pyjwt.encode(payload, _TEST_KEY, algorithm=_ALGORITHM)
    _test_app.dependency_overrides[get_db] = _db_returning(_test_user)
    try:
        with TestClient(_test_app) as c:
            r = c.get("/verify", params={"token": token})
        assert r.json()["user_id"] is None
    finally:
        _test_app.dependency_overrides.clear()


@pytest.mark.unit
def test_resolve_token_with_integer_sub_returns_none():
    """A token where sub is an integer (not string) returns None (P3-07 fix)."""
    # Sub is an integer, which will raise TypeError in uuid.UUID(int_value).
    payload = {
        "sub": 12345,  # Integer, not string
        "role": "researcher",
        "iat": int(datetime.now(timezone.utc).timestamp()),
        "exp": int(datetime.now(timezone.utc).timestamp()) + 3600,
    }
    token = pyjwt.encode(payload, _TEST_KEY, algorithm=_ALGORITHM)
    _test_app.dependency_overrides[get_db] = _db_returning(_test_user)
    try:
        with TestClient(_test_app) as c:
            r = c.get("/verify", params={"token": token})
        assert r.json()["user_id"] is None
    finally:
        _test_app.dependency_overrides.clear()


@pytest.mark.unit
def test_mint_raises_when_ttl_is_zero(monkeypatch):
    """``mint_backend_jwt`` raises RuntimeError when JWT_TTL_MIN is 0."""
    monkeypatch.setenv("JWT_TTL_MIN", "0")
    with pytest.raises(RuntimeError, match="JWT_TTL_MIN must be positive"):
        mint_backend_jwt(_test_user)


@pytest.mark.unit
def test_mint_raises_when_ttl_is_invalid(monkeypatch):
    """``mint_backend_jwt`` raises RuntimeError when JWT_TTL_MIN is not an integer."""
    monkeypatch.setenv("JWT_TTL_MIN", "not-a-number")
    with pytest.raises(RuntimeError, match="JWT_TTL_MIN must be a valid integer"):
        mint_backend_jwt(_test_user)
