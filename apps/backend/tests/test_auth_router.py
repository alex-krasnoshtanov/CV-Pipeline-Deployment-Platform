"""Tests for the auth router."""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlparse

import bcrypt as _bcrypt
import jwt as pyjwt
import pytest
from fastapi.responses import RedirectResponse

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import api.routers.auth as auth_router  # noqa: E402
from api.auth.dependencies import SESSION_COOKIE_NAME  # noqa: E402
from api.auth.jwt import mint_backend_jwt  # noqa: E402
from api.db import get_db  # noqa: E402
from api.db.models import User  # noqa: E402
from api.main import app  # noqa: E402

_TEST_JWT_KEY = "test-signing-key-32-chars-exactly!!"
_TEST_FRONTEND_URL = "http://frontend.local"
_TEST_CALLBACK_URL = "http://localhost:8501/auth/github/callback"


class _JsonResponse:
    """Small helper that mimics the subset of an OAuth response we need."""

    def __init__(self, payload: object):
        self._payload = payload

    def json(self) -> object:
        return self._payload


@pytest.fixture(autouse=True)
def _set_jwt_env(monkeypatch: pytest.MonkeyPatch):
    """Configure JWT signing for the tests that mint or verify tokens."""
    monkeypatch.setenv("JWT_SIGNING_KEY", _TEST_JWT_KEY)
    monkeypatch.setenv("JWT_TTL_MIN", "60")


def _make_user(
    *,
    name: str = "Test User",
    role: str = "researcher",
    email: str | None = "test@example.com",
) -> MagicMock:
    """Build a mock user with the attributes the router reads."""
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.name = name
    user.role = role
    user.email = email
    user.api_key_hash = None
    user.key_sha256 = None
    user.password_hash = None
    user.oauth_provider = None
    user.oauth_subject = None
    user.last_login_at = None
    return user


def _override_db_with_user(user: MagicMock):
    """Return a DB override that resolves every lookup to *user*."""
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = user

    async def _override():
        session = AsyncMock()
        session.execute.return_value = execute_result
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        yield session

    return _override


def _override_db_for_oauth(existing_user: MagicMock | None = None):
    """Return a DB override for the OAuth callback and track created rows."""
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = existing_user
    created_users: list[MagicMock] = []

    async def _override():
        session = AsyncMock()
        session.execute.return_value = execute_result
        session.commit = AsyncMock()
        session.refresh = AsyncMock()

        def _add(obj):
            if getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()
            created_users.append(obj)

        session.add = MagicMock(side_effect=_add)
        yield session

    return _override, created_users


async def _mock_authorize_redirect(request, redirect_uri: str):
    """Return a redirect response that looks like Authlib's login redirect."""
    _ = request
    assert redirect_uri == _TEST_CALLBACK_URL
    return RedirectResponse(
        "https://github.com/login/oauth/authorize?state=test-state",
        status_code=302,
    )


@pytest.mark.unit
def test_me_returns_user_when_key_valid(client_with_user, valid_api_key):
    """A valid X-API-Key returns the user's identity."""
    response = client_with_user.get(
        "/auth/me",
        headers={"X-API-Key": valid_api_key},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Test User"
    assert body["role"] in {"researcher", "admin"}
    assert body["auth_method"] == "api_key"
    assert "api_key_hash" not in body


@pytest.mark.unit
def test_me_returns_401_when_key_missing(client_with_user):
    """No X-API-Key header returns 401."""
    response = client_with_user.get("/auth/me")
    assert response.status_code == 401


@pytest.mark.unit
def test_me_returns_401_when_key_invalid(client_with_user):
    """A bad X-API-Key returns 401."""
    response = client_with_user.get(
        "/auth/me",
        headers={"X-API-Key": "not-a-real-key"},
    )
    assert response.status_code == 401


@pytest.mark.unit
def test_github_login_redirects_to_github(client_with_user, monkeypatch):
    """The login endpoint should send the browser to GitHub's OAuth page."""
    monkeypatch.setenv("OAUTH_REDIRECT_URI", _TEST_CALLBACK_URL)
    monkeypatch.setattr(
        auth_router.oauth.github,
        "authorize_redirect",
        AsyncMock(side_effect=_mock_authorize_redirect),
    )

    response = client_with_user.get("/auth/github/login", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"].startswith(
        "https://github.com/login/oauth/authorize"
    )


@pytest.mark.unit
def test_github_callback_creates_new_user(client_with_user, monkeypatch):
    """A first-time GitHub login should create a new OAuth user row."""
    monkeypatch.setenv("FRONTEND_URL", _TEST_FRONTEND_URL)
    override, created_users = _override_db_for_oauth()
    app.dependency_overrides[get_db] = override
    monkeypatch.setattr(
        auth_router.oauth.github,
        "authorize_access_token",
        AsyncMock(return_value={"access_token": "token"}),
    )
    monkeypatch.setattr(
        auth_router.oauth.github,
        "get",
        AsyncMock(
            side_effect=[
                _JsonResponse(
                    {
                        "id": 12345,
                        "login": "gh-user",
                        "name": "GitHub User",
                        "email": "user@example.com",
                    }
                ),
            ],
        ),
    )

    try:
        response = client_with_user.get(
            "/auth/github/callback?code=test", follow_redirects=False
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 302
    assert created_users
    created_user = created_users[0]
    assert created_user.oauth_provider == "github"
    assert created_user.oauth_subject == "12345"
    assert created_user.email == "user@example.com"
    assert created_user.last_login_at is not None


@pytest.mark.unit
def test_github_callback_updates_existing_user(client_with_user, monkeypatch):
    """A repeat GitHub login should refresh the existing row in place."""
    existing_user = _make_user(name="Old Name", email="old@example.com")
    existing_user.oauth_provider = "github"
    existing_user.oauth_subject = "12345"
    old_login = datetime(2020, 1, 1, tzinfo=timezone.utc)
    existing_user.last_login_at = old_login
    override, created_users = _override_db_for_oauth(existing_user)
    app.dependency_overrides[get_db] = override
    monkeypatch.setattr(
        auth_router.oauth.github,
        "authorize_access_token",
        AsyncMock(return_value={"access_token": "token"}),
    )
    monkeypatch.setattr(
        auth_router.oauth.github,
        "get",
        AsyncMock(
            side_effect=[
                _JsonResponse(
                    {
                        "id": 12345,
                        "login": "gh-user",
                        "name": "New Name",
                        "email": "new@example.com",
                    }
                ),
            ],
        ),
    )

    try:
        response = client_with_user.get(
            "/auth/github/callback?code=test", follow_redirects=False
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 302
    assert not created_users
    assert existing_user.name == "New Name"
    assert existing_user.email == "new@example.com"
    assert existing_user.last_login_at is not None
    assert existing_user.last_login_at > old_login


@pytest.mark.unit
def test_github_callback_fallback_to_user_emails(client_with_user, monkeypatch):
    """Private GitHub email should fall back to /user/emails."""
    override, created_users = _override_db_for_oauth()
    app.dependency_overrides[get_db] = override
    monkeypatch.setattr(
        auth_router.oauth.github,
        "authorize_access_token",
        AsyncMock(return_value={"access_token": "token"}),
    )
    monkeypatch.setattr(
        auth_router.oauth.github,
        "get",
        AsyncMock(
            side_effect=[
                _JsonResponse(
                    {
                        "id": 12345,
                        "login": "gh-user",
                        "name": "GitHub User",
                        "email": None,
                    }
                ),
                _JsonResponse(
                    [
                        {
                            "email": "primary@example.com",
                            "primary": True,
                            "verified": True,
                        },
                        {
                            "email": "other@example.com",
                            "primary": False,
                            "verified": True,
                        },
                    ]
                ),
            ],
        ),
    )

    try:
        response = client_with_user.get(
            "/auth/github/callback?code=test", follow_redirects=False
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 302
    assert created_users[0].email == "primary@example.com"


@pytest.mark.unit
def test_github_callback_redirects_to_frontend_with_token(
    client_with_user, monkeypatch
):
    """The callback should return a frontend redirect with a verifiable JWT."""
    monkeypatch.setenv("FRONTEND_URL", _TEST_FRONTEND_URL)
    override, created_users = _override_db_for_oauth()
    app.dependency_overrides[get_db] = override
    monkeypatch.setattr(
        auth_router.oauth.github,
        "authorize_access_token",
        AsyncMock(return_value={"access_token": "token"}),
    )
    monkeypatch.setattr(
        auth_router.oauth.github,
        "get",
        AsyncMock(
            side_effect=[
                _JsonResponse(
                    {
                        "id": 54321,
                        "login": "gh-user",
                        "name": "GitHub User",
                        "email": "user@example.com",
                    }
                ),
            ],
        ),
    )

    try:
        response = client_with_user.get(
            "/auth/github/callback?code=test", follow_redirects=False
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 302
    location = response.headers["location"]
    assert location.startswith(f"{_TEST_FRONTEND_URL}/?token=")
    token = parse_qs(urlparse(location).query)["token"][0]
    payload = pyjwt.decode(token, _TEST_JWT_KEY, algorithms=["HS256"])
    assert payload["sub"] == str(created_users[0].id)


@pytest.mark.unit
def test_github_callback_400_on_oauth_failure(client_with_user, monkeypatch):
    """GitHub callback failures should become a clean 400 response."""
    monkeypatch.setattr(
        auth_router.oauth.github,
        "authorize_access_token",
        AsyncMock(side_effect=RuntimeError("bad code")),
    )

    response = client_with_user.get(
        "/auth/github/callback?code=test", follow_redirects=False
    )

    assert response.status_code == 400
    assert response.json()["detail"]["error_code"] == "OAUTH_FAILED"


@pytest.mark.unit
def test_me_auth_method_oauth_for_jwt_request(client_with_user):
    """Bearer JWT requests should be classified as oauth."""
    user = _make_user(email="oauth@example.com")
    override = _override_db_with_user(user)
    app.dependency_overrides[get_db] = override
    token = mint_backend_jwt(user)

    try:
        response = client_with_user.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    body = response.json()
    assert body["auth_method"] == "oauth"
    assert body["name"] == user.name


@pytest.mark.unit
def test_me_auth_method_password_for_cookie_request(client_with_user, monkeypatch):
    """Session-cookie requests should be classified as password."""
    user = _make_user(email="cookie@example.com")
    override = _override_db_with_user(user)
    app.dependency_overrides[get_db] = override
    monkeypatch.setattr(
        "api.auth.dependencies.get_user_from_session",
        AsyncMock(return_value=user),
    )

    try:
        response = client_with_user.get(
            "/auth/me",
            cookies={SESSION_COOKIE_NAME: str(uuid.uuid4())},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    assert response.json()["auth_method"] == "password"


@pytest.mark.unit
def test_me_auth_method_api_key_for_header_request(client_with_user, valid_api_key):
    """X-API-Key requests should be classified as api_key."""
    response = client_with_user.get(
        "/auth/me",
        headers={"X-API-Key": valid_api_key},
    )
    assert response.status_code == 200
    assert response.json()["auth_method"] == "api_key"


@pytest.mark.unit
def test_logout_with_jwt_returns_204_no_db_change(client_with_user, monkeypatch):
    """JWT callers should be able to hit logout without a session row."""
    user = _make_user(email="jwt@example.com")
    override = _override_db_with_user(user)
    app.dependency_overrides[get_db] = override
    delete_session = AsyncMock()
    monkeypatch.setattr(auth_router, "delete_session", delete_session)
    token = mint_backend_jwt(user)

    try:
        response = client_with_user.post(
            "/auth/logout",
            headers={"Authorization": f"Bearer {token}"},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 204
    assert delete_session.await_count == 0


# ---------------------------------------------------------------------------
# Helpers for register / login tests
# ---------------------------------------------------------------------------

_LOGIN_TEST_PASS = "login_test_password_ok"
_LOGIN_TEST_HASH = _bcrypt.hashpw(
    _LOGIN_TEST_PASS.encode(), _bcrypt.gensalt(rounds=4)
).decode()


def _make_user_with_password(
    *,
    email: str = "user@example.com",
    password_hash: str = _LOGIN_TEST_HASH,
) -> MagicMock:
    """Build a mock user whose password_hash is set (email+password flow)."""
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.name = "Test User"
    user.role = "researcher"
    user.email = email
    user.password_hash = password_hash
    user.api_key_hash = None
    user.key_sha256 = None
    user.oauth_provider = None
    user.oauth_subject = None
    user.last_login_at = None
    return user


def _override_db_register(existing_user: MagicMock | None = None):
    """DB override for /auth/register.

    The first execute() is the email-existence check; it returns
    existing_user (or None). refresh() gives the new User row an id,
    simulating what Postgres would do after INSERT.
    create_session is mocked at the router level in each test, so no
    further execute() calls occur from that path.
    """

    async def _override():
        session = AsyncMock()
        # db.add() is synchronous in SQLAlchemy — override so the router
        # calling db.add(user) doesn't produce a "coroutine never awaited" warning.
        session.add = MagicMock()

        result = MagicMock()
        result.scalar_one_or_none.return_value = existing_user
        session.execute.return_value = result

        async def _refresh(obj):
            # Simulate the DB assigning a primary key after commit.
            obj.id = uuid.uuid4()

        session.refresh.side_effect = _refresh
        yield session

    return _override


def _override_db_login(user_row: MagicMock | None = None):
    """DB override for /auth/login: SELECT user by email."""

    async def _override():
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = user_row
        session.execute.return_value = result
        yield session

    return _override


# ---------------------------------------------------------------------------
# POST /auth/register tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_register_returns_201_with_user_body_and_cookie(client_with_user, monkeypatch):
    """Happy path: register creates an account and sets a session cookie."""
    fake_session = MagicMock()
    fake_session.id = uuid.uuid4()
    monkeypatch.setattr(
        auth_router, "create_session", AsyncMock(return_value=fake_session)
    )
    app.dependency_overrides[get_db] = _override_db_register()

    try:
        response = client_with_user.post(
            "/auth/register",
            json={
                "email": "new@example.com",
                "password": "secure1234",
                "name": "New User",
            },
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "new@example.com"
    assert body["name"] == "New User"
    assert body["role"] == "researcher"
    # Sensitive fields must never appear in the response.
    assert "password_hash" not in body
    assert "api_key_hash" not in body
    # Browser must receive a session cookie.
    assert SESSION_COOKIE_NAME in response.cookies


@pytest.mark.unit
def test_register_returns_409_for_duplicate_email(client_with_user):
    """Registering with an already-taken email returns 409 EMAIL_TAKEN."""
    existing = _make_user(email="taken@example.com")
    app.dependency_overrides[get_db] = _override_db_register(existing_user=existing)

    try:
        response = client_with_user.post(
            "/auth/register",
            json={
                "email": "taken@example.com",
                "password": "secure1234",
                "name": "Someone",
            },
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 409
    assert response.json()["detail"]["error_code"] == "EMAIL_TAKEN"


@pytest.mark.unit
def test_register_rejects_short_password(client_with_user):
    """Pydantic min_length=8 on password should return 422, not 500."""
    response = client_with_user.post(
        "/auth/register",
        json={"email": "user@example.com", "password": "short", "name": "User"},
    )
    assert response.status_code == 422


@pytest.mark.unit
def test_register_rejects_invalid_email_format(client_with_user):
    """An email without an @ sign should fail Pydantic validation (422)."""
    response = client_with_user.post(
        "/auth/register",
        json={"email": "not-an-email", "password": "secure1234", "name": "User"},
    )
    assert response.status_code == 422


@pytest.mark.unit
def test_register_normalises_email_to_lowercase(client_with_user, monkeypatch):
    """Email should be stored lowercase regardless of how it was submitted."""
    fake_session = MagicMock()
    fake_session.id = uuid.uuid4()
    monkeypatch.setattr(
        auth_router, "create_session", AsyncMock(return_value=fake_session)
    )
    app.dependency_overrides[get_db] = _override_db_register()

    try:
        response = client_with_user.post(
            "/auth/register",
            json={
                "email": "Mixed.Case@Example.COM",
                "password": "secure1234",
                "name": "User",
            },
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 201
    assert response.json()["email"] == "mixed.case@example.com"


# ---------------------------------------------------------------------------
# POST /auth/login tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_login_returns_200_and_sets_cookie_for_correct_credentials(
    client_with_user, monkeypatch
):
    """Happy path: correct email + password returns 200 and a session cookie."""
    user = _make_user_with_password()
    fake_session = MagicMock()
    fake_session.id = uuid.uuid4()
    monkeypatch.setattr(
        auth_router, "create_session", AsyncMock(return_value=fake_session)
    )
    app.dependency_overrides[get_db] = _override_db_login(user_row=user)

    try:
        response = client_with_user.post(
            "/auth/login",
            json={"email": "user@example.com", "password": _LOGIN_TEST_PASS},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    body = response.json()
    assert body["email"] == "user@example.com"
    assert body["role"] == "researcher"
    assert "password_hash" not in body
    assert SESSION_COOKIE_NAME in response.cookies


@pytest.mark.unit
def test_login_returns_401_for_wrong_password(client_with_user):
    """Wrong password returns 401 INVALID_CREDENTIALS."""
    user = _make_user_with_password()
    app.dependency_overrides[get_db] = _override_db_login(user_row=user)

    try:
        response = client_with_user.post(
            "/auth/login",
            json={"email": "user@example.com", "password": "totally_wrong"},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 401
    assert response.json()["detail"]["error_code"] == "INVALID_CREDENTIALS"


@pytest.mark.unit
def test_login_returns_401_for_unknown_email(client_with_user):
    """Non-existent email returns the same 401 as a wrong password.

    The identical error message for both cases prevents email enumeration:
    an attacker cannot tell whether the email exists.
    """
    app.dependency_overrides[get_db] = _override_db_login(user_row=None)

    try:
        response = client_with_user.post(
            "/auth/login",
            json={"email": "ghost@example.com", "password": "password123"},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 401
    assert response.json()["detail"]["error_code"] == "INVALID_CREDENTIALS"


@pytest.mark.unit
def test_login_returns_401_for_oauth_only_user(client_with_user):
    """An OAuth-only account (no password_hash) cannot log in via email+password."""
    user = _make_user(email="oauth@example.com")
    user.password_hash = None  # OAuth-only — password login is not set up
    app.dependency_overrides[get_db] = _override_db_login(user_row=user)

    try:
        response = client_with_user.post(
            "/auth/login",
            json={"email": "oauth@example.com", "password": "anything"},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 401
    assert response.json()["detail"]["error_code"] == "INVALID_CREDENTIALS"


# ---------------------------------------------------------------------------
# POST /auth/logout — session cookie path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_logout_with_session_cookie_calls_delete_session(client_with_user, monkeypatch):
    """Cookie-based logout must call delete_session with the session id."""
    user = _make_user(email="cookie@example.com")
    app.dependency_overrides[get_db] = _override_db_with_user(user)

    delete_session_mock = AsyncMock()
    monkeypatch.setattr(auth_router, "delete_session", delete_session_mock)
    # Make the session cookie resolve to a valid user so require_user passes.
    monkeypatch.setattr(
        "api.auth.dependencies.get_user_from_session",
        AsyncMock(return_value=user),
    )

    session_id = str(uuid.uuid4())
    try:
        response = client_with_user.post(
            "/auth/logout",
            cookies={SESSION_COOKIE_NAME: session_id},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 204
    # delete_session must be called with the exact cookie value.
    delete_session_mock.assert_awaited_once()
    call_args = delete_session_mock.await_args
    assert call_args.args[1] == session_id or session_id in str(call_args)


# ---------------------------------------------------------------------------
# GET /auth/github/login — missing env var
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_github_login_returns_500_when_oauth_redirect_uri_unset(
    client_with_user, monkeypatch
):
    """Missing OAUTH_REDIRECT_URI env var must return 500 MISCONFIGURED.

    This prevents a silent misconfiguration from starting an OAuth flow
    that would immediately fail at the callback step.
    """
    monkeypatch.delenv("OAUTH_REDIRECT_URI", raising=False)

    response = client_with_user.get("/auth/github/login", follow_redirects=False)

    assert response.status_code == 500
    assert response.json()["detail"]["error_code"] == "MISCONFIGURED"
