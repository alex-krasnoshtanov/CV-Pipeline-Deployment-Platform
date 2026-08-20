"""Unit tests for the /users router."""

from __future__ import annotations

import hashlib
import uuid
from unittest.mock import AsyncMock, MagicMock

import bcrypt
import pytest
from api.db import get_db
from api.main import app
from fastapi.testclient import TestClient

_TEST_KEY = "test-key-for-unit-tests"
_HASHED_KEY = bcrypt.hashpw(
    _TEST_KEY.encode("utf-8"),
    bcrypt.gensalt(),
).decode("utf-8")
_SHA256_KEY = hashlib.sha256(
    _TEST_KEY.encode("utf-8"),
).hexdigest()


def _build_mock_user(role: str = "admin") -> MagicMock:
    """Build a fresh mock user with the given role."""
    mock_user = MagicMock()
    mock_user.name = "Test Admin"
    mock_user.role = role
    mock_user.id = uuid.uuid4()
    mock_user.api_key_hash = _HASHED_KEY
    mock_user.key_sha256 = _SHA256_KEY
    return mock_user


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set API_KEY env var for all tests in this module."""
    monkeypatch.setenv("API_KEY", _TEST_KEY)


def _override_db_with_user(role: str = "admin"):
    """Create a DB override that returns a user with the given role."""
    mock_user = _build_mock_user(role=role)

    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = mock_user

    scalars_result = MagicMock()
    scalars_result.first.return_value = mock_user.id
    scalars_result.all.return_value = [mock_user]

    async def _mock_get_db():
        session = AsyncMock()
        session.execute.return_value = execute_result
        session.scalars.return_value = scalars_result
        yield session

    return _mock_get_db


@pytest.fixture
def admin_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient authenticated as an admin user."""
    mock_model = MagicMock()
    mock_model.model_version = "unet-v1"

    if hasattr(app.state, "model"):
        delattr(app.state, "model")

    monkeypatch.setattr("api.main.load_model", lambda: mock_model)
    app.dependency_overrides[get_db] = _override_db_with_user("admin")

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def researcher_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient authenticated as a researcher."""
    mock_model = MagicMock()
    mock_model.model_version = "unet-v1"

    if hasattr(app.state, "model"):
        delattr(app.state, "model")

    monkeypatch.setattr("api.main.load_model", lambda: mock_model)
    app.dependency_overrides[get_db] = _override_db_with_user(
        "researcher",
    )

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.pop(get_db, None)


# ---- authentication and authorisation --------------------------------


@pytest.mark.unit
class TestUsersAuth:
    """Tests for /users endpoint auth and role checks."""

    def test_returns_401_without_api_key(
        self,
        admin_client: TestClient,
    ) -> None:
        """POST /users without auth should return 401."""
        response = admin_client.post(
            "/users",
            json={"name": "New User", "role": "researcher"},
        )

        assert response.status_code == 401

    def test_returns_403_for_non_admin(
        self,
        researcher_client: TestClient,
    ) -> None:
        """A researcher should not be able to create users."""
        response = researcher_client.post(
            "/users",
            json={"name": "New User", "role": "researcher"},
            headers={"X-API-Key": _TEST_KEY},
        )

        assert response.status_code == 403


# ---- happy path ------------------------------------------------------


@pytest.mark.unit
class TestUsersHappyPath:
    """Tests for successful user creation."""

    def test_admin_creates_user_returns_200(
        self,
        admin_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An admin should be able to create a new user."""
        fake_user = MagicMock()
        fake_user.id = uuid.uuid4()
        fake_user.name = "Alice"
        fake_user.role = "researcher"
        fake_user.created_at = "2026-05-01T10:00:00Z"

        monkeypatch.setattr(
            "api.routers.users.create_user",
            AsyncMock(
                return_value=(
                    fake_user,
                    "abcdef1234567890abcdef1234567890",
                ),
            ),
        )

        response = admin_client.post(
            "/users",
            json={"name": "Alice", "role": "researcher"},
            headers={"X-API-Key": _TEST_KEY},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "Alice"
        assert body["role"] == "researcher"
        assert "api_key" in body
