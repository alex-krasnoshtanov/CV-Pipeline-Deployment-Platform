"""Tests for the auth dependency hierarchy.

These tests exercise the dependency layer directly through a tiny
FastAPI app so the router code stays untouched while we validate
credential precedence and failure semantics.
"""

from __future__ import annotations

import hashlib
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import bcrypt
import jwt as pyjwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from api.auth import optional_user, require_admin, require_user  # noqa: E402
from api.auth.dependencies import SESSION_COOKIE_NAME  # noqa: E402
from api.auth.jwt import mint_backend_jwt  # noqa: E402
from api.db import get_db  # noqa: E402
from api.db.models import Session, User  # noqa: E402

_TEST_KEY = "test-signing-key-32-chars-exactly!!"
_TEST_API_KEY = "test-key"
_ALGORITHM = "HS256"
_TEST_API_KEY_HASH = bcrypt.hashpw(
    _TEST_API_KEY.encode("utf-8"), bcrypt.gensalt()
).decode(
    "utf-8",
)
_TEST_API_KEY_DIGEST = hashlib.sha256(_TEST_API_KEY.encode("utf-8")).hexdigest()

_app = FastAPI()


def _make_user(role: str = "researcher") -> MagicMock:
    """Build a mock user object with the attributes auth depends on."""
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.role = role
    user.name = "Test User"
    user.email = "test@example.com"
    user.api_key_hash = _TEST_API_KEY_HASH
    user.key_sha256 = _TEST_API_KEY_DIGEST
    return user


def _db_override(user: MagicMock, session_user: MagicMock | None = None):
    """Return a get_db override that resolves to *user* for auth lookups."""
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = user
    execute_result.first.return_value = (
        session_user or MagicMock(spec=Session),
        user,
    )

    async def _override():
        session = AsyncMock()
        session.execute.return_value = execute_result
        yield session

    return _override


def _db_override_with_precedence(
    jwt_user: MagicMock, other_user: MagicMock, session_user: MagicMock | None = None
):
    """Return a get_db override with different users per auth path.

    JWT path resolves to jwt_user, others to other_user. This allows
    testing that JWT precedence actually runs the JWT branch, not just
    that multiple credentials don't crash.
    """

    async def _override():
        session = AsyncMock()

        async def _execute_impl(query):
            result = MagicMock()
            query_str = str(query)

            # Session path (join with User) -> resolve to other_user
            if "sessions" in query_str.lower() or "join" in query_str.lower():
                result.scalar_one_or_none.return_value = other_user
                result.first.return_value = (
                    session_user or MagicMock(spec=Session),
                    other_user,
                )
            # API-key path filters on key_sha256 -> resolve to other_user
            elif "where users.key_sha256" in query_str.lower():
                result.scalar_one_or_none.return_value = other_user
                result.first.return_value = (
                    session_user or MagicMock(spec=Session),
                    other_user,
                )
            # JWT path queries User by id -> resolve to jwt_user
            elif "where users.id" in query_str.lower():
                result.scalar_one_or_none.return_value = jwt_user
                result.first.return_value = (
                    session_user or MagicMock(spec=Session),
                    jwt_user,
                )
            else:
                result.scalar_one_or_none.return_value = jwt_user
                result.first.return_value = (
                    session_user or MagicMock(spec=Session),
                    jwt_user,
                )

            return result

        session.execute.side_effect = _execute_impl
        yield session

    return _override


@_app.get("/optional")
async def _optional_route(current_user: User | None = Depends(optional_user)):
    """Expose the optional dependency result for assertions."""
    return {
        "user_id": str(current_user.id) if current_user else None,
        "role": current_user.role if current_user else None,
    }


@_app.get("/required")
async def _required_route(current_user: User = Depends(require_user)):
    """Expose the required dependency result for assertions."""
    return {
        "user_id": str(current_user.id),
        "role": current_user.role,
    }


@_app.get("/admin")
async def _admin_route(current_user: User = Depends(require_admin)):
    """Expose the admin dependency result for assertions."""
    return {
        "user_id": str(current_user.id),
        "role": current_user.role,
    }


@pytest.fixture(autouse=True)
def _set_jwt_key(monkeypatch: pytest.MonkeyPatch):
    """Configure the JWT signing environment for every test."""
    monkeypatch.setenv("JWT_SIGNING_KEY", _TEST_KEY)
    monkeypatch.setenv("JWT_TTL_MIN", "60")


@pytest.fixture(autouse=True)
def _default_db_override():
    """Provide a DB session mock for all tests unless they replace it."""
    default_user = _make_user()
    _app.dependency_overrides[get_db] = _db_override(default_user)
    yield default_user
    _app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def user(_default_db_override: MagicMock) -> MagicMock:
    """Return the default mock user used by the shared DB override."""
    return _default_db_override


@pytest.fixture
def admin_user() -> MagicMock:
    """Return a mock admin user for admin-success tests."""
    return _make_user(role="admin")


@pytest.fixture
def client() -> TestClient:
    """Return a TestClient bound to the local dependency test app."""
    with TestClient(_app) as test_client:
        yield test_client


@pytest.fixture
def admin_client(admin_user: MagicMock):
    """Return a TestClient whose DB override resolves an admin user."""
    _app.dependency_overrides[get_db] = _db_override(admin_user)
    with TestClient(_app) as test_client:
        yield test_client
    _app.dependency_overrides[get_db] = _db_override(_make_user())


@pytest.mark.unit
def test_optional_returns_none_when_no_credentials(client: TestClient):
    """No headers or cookies should resolve to anonymous."""
    response = client.get("/optional")
    assert response.status_code == 200
    assert response.json() == {"user_id": None, "role": None}


@pytest.mark.unit
def test_optional_returns_user_for_valid_jwt(client: TestClient, user: MagicMock):
    """A valid Bearer token should return the authenticated user."""
    token = mint_backend_jwt(user)
    response = client.get("/optional", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["user_id"] == str(user.id)


@pytest.mark.unit
def test_optional_raises_401_for_invalid_jwt(client: TestClient):
    """A malformed Bearer token should fail closed with 401."""
    response = client.get(
        "/optional",
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert response.status_code == 401
    assert response.json()["detail"]["error_code"] == "UNAUTHORIZED"


@pytest.mark.unit
def test_optional_raises_401_for_expired_jwt(client: TestClient, user: MagicMock):
    """An expired JWT should fail closed with 401."""
    expired_payload = {
        "sub": str(user.id),
        "role": user.role,
        "iat": int(datetime.now(timezone.utc).timestamp()) - 7200,
        "exp": int((datetime.now(timezone.utc) - timedelta(minutes=1)).timestamp()),
    }
    token = pyjwt.encode(expired_payload, _TEST_KEY, algorithm=_ALGORITHM)
    response = client.get("/optional", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


@pytest.mark.unit
def test_optional_returns_user_for_valid_cookie(client: TestClient, user: MagicMock):
    """A valid session cookie should still authenticate successfully."""
    session_row = MagicMock(spec=Session)
    session_row.expires_at = datetime.now(timezone.utc) + timedelta(days=1)
    _app.dependency_overrides[get_db] = _db_override(user, session_user=session_row)
    try:
        response = client.get(
            "/optional", cookies={SESSION_COOKIE_NAME: str(uuid.uuid4())}
        )
        assert response.status_code == 200
        assert response.json()["user_id"] == str(user.id)
    finally:
        _app.dependency_overrides[get_db] = _db_override(_make_user())


@pytest.mark.unit
def test_optional_returns_user_for_valid_api_key(client: TestClient, user: MagicMock):
    """A valid X-API-Key should still authenticate successfully."""
    response = client.get("/optional", headers={"X-API-Key": _TEST_API_KEY})
    assert response.status_code == 200
    assert response.json()["user_id"] == str(user.id)


@pytest.mark.unit
def test_optional_jwt_takes_precedence_over_cookie(client: TestClient):
    """When both JWT and cookie are present, the JWT branch runs (not cookie).

    Verify by having JWT and cookie paths resolve to different users,
    then asserting that the JWT user is returned.
    """
    jwt_user = _make_user()
    cookie_user = _make_user()
    session_row = MagicMock(spec=Session)
    session_row.expires_at = datetime.now(timezone.utc) + timedelta(days=1)
    _app.dependency_overrides[get_db] = _db_override_with_precedence(
        jwt_user, cookie_user, session_user=session_row
    )
    try:
        token = mint_backend_jwt(jwt_user)
        response = client.get(
            "/optional",
            headers={"Authorization": f"Bearer {token}"},
            cookies={SESSION_COOKIE_NAME: str(uuid.uuid4())},
        )
        assert response.status_code == 200
        # JWT branch ran: returned jwt_user, not cookie_user
        assert response.json()["user_id"] == str(jwt_user.id)
        assert response.json()["user_id"] != str(cookie_user.id)
    finally:
        _app.dependency_overrides[get_db] = _db_override(_make_user())


@pytest.mark.unit
def test_optional_jwt_takes_precedence_over_api_key(client: TestClient):
    """When both JWT and API key are present, the JWT branch runs (not API-key).

    Verify by having JWT and API-key paths resolve to different users,
    then asserting that the JWT user is returned.
    """
    jwt_user = _make_user()
    api_key_user = _make_user()
    _app.dependency_overrides[get_db] = _db_override_with_precedence(
        jwt_user, api_key_user
    )
    try:
        token = mint_backend_jwt(jwt_user)
        response = client.get(
            "/optional",
            headers={"Authorization": f"Bearer {token}", "X-API-Key": _TEST_API_KEY},
        )
        assert response.status_code == 200
        # JWT branch ran: returned jwt_user, not api_key_user
        assert response.json()["user_id"] == str(jwt_user.id)
        assert response.json()["user_id"] != str(api_key_user.id)
    finally:
        _app.dependency_overrides[get_db] = _db_override(_make_user())


@pytest.mark.unit
def test_optional_bearer_is_case_insensitive(client: TestClient, user: MagicMock):
    """Lowercase bearer should be accepted the same as uppercase Bearer."""
    token = mint_backend_jwt(user)
    response = client.get("/optional", headers={"Authorization": f"bearer {token}"})
    assert response.status_code == 200
    assert response.json()["user_id"] == str(user.id)


@pytest.mark.unit
def test_required_raises_401_when_optional_returns_none(client: TestClient):
    """Anonymous callers should still get a 401 from require_user."""
    response = client.get("/required")
    assert response.status_code == 401
    assert response.json()["detail"]["error_code"] == "UNAUTHORIZED"


@pytest.mark.unit
def test_required_returns_user_when_authenticated(client: TestClient, user: MagicMock):
    """require_user should pass through the authenticated user unchanged."""
    token = mint_backend_jwt(user)
    response = client.get("/required", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["user_id"] == str(user.id)


@pytest.mark.unit
def test_admin_returns_user_for_admin_role(
    admin_client: TestClient, admin_user: MagicMock
):
    """An admin caller should pass through require_admin."""
    token = mint_backend_jwt(admin_user)
    response = admin_client.get("/admin", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["role"] == "admin"


@pytest.mark.unit
def test_admin_returns_403_for_researcher_role(client: TestClient):
    """A non-admin caller should get a 403 from require_admin."""
    token = mint_backend_jwt(_make_user(role="researcher"))
    response = client.get("/admin", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403
    assert response.json()["detail"]["error_code"] == "FORBIDDEN"


@pytest.mark.unit
def test_admin_returns_401_for_anonymous(client: TestClient):
    """Anonymous callers should get a 401 before the admin check runs."""
    response = client.get("/admin")
    assert response.status_code == 401
    assert response.json()["detail"]["error_code"] == "UNAUTHORIZED"
