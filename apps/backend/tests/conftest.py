"""Shared test configuration for the backend test suite."""

from __future__ import annotations

import hashlib
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import bcrypt
import pytest
from fastapi.testclient import TestClient

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_TEST_KEY = "test-key-for-unit-tests"
_HASHED_KEY = bcrypt.hashpw(
    _TEST_KEY.encode("utf-8"),
    bcrypt.gensalt(),
).decode("utf-8")
_SHA256_KEY = hashlib.sha256(
    _TEST_KEY.encode("utf-8"),
).hexdigest()


def _build_mock_user() -> MagicMock:
    """Build a fresh mock user with valid hashes.

    All new-schema fields (email, password_hash) are explicitly set
    to None so Pydantic serialisation in /auth/me does not stumble
    over MagicMock auto-attributes. Without this, accessing
    mock_user.email returns a MagicMock that Pydantic refuses to
    coerce to str | None.
    """
    mock_user = MagicMock()
    mock_user.name = "Test User"
    mock_user.role = "researcher"
    mock_user.id = uuid.uuid4()
    mock_user.api_key_hash = _HASHED_KEY
    mock_user.key_sha256 = _SHA256_KEY
    mock_user.email = None
    mock_user.password_hash = None
    mock_user.oauth_provider = None
    mock_user.oauth_subject = None
    mock_user.last_login_at = None
    return mock_user


@pytest.fixture(autouse=True)
def _stub_init_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent lifespan from requiring a live DB_URL in unit tests."""
    monkeypatch.setenv("DB_URL", "postgresql+asyncpg://test:test@localhost:5432/testdb")
    monkeypatch.setattr("api.main.init_db", lambda url: None)


@pytest.fixture(autouse=True)
def _override_db():
    """Replace the real DB session with a structured mock."""
    # noqa: E402 - sys.path was modified above; local imports safe here
    from api.db import get_db  # noqa: E402
    from api.main import app  # noqa: E402

    mock_user = _build_mock_user()

    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = mock_user

    scalars_result = MagicMock()
    scalars_result.first.return_value = mock_user.id

    async def _mock_get_db():
        session = AsyncMock()
        session.execute.return_value = execute_result
        session.scalars.return_value = scalars_result
        yield session

    app.dependency_overrides[get_db] = _mock_get_db
    yield
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def valid_api_key() -> str:
    """Return a valid test API key."""
    return _TEST_KEY


@pytest.fixture
def client_with_user(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient with a mocked user and model."""
    # noqa: E402 - sys.path was modified above; local imports safe here
    from api.main import app  # noqa: E402

    monkeypatch.setenv("API_KEY", _TEST_KEY)
    mock_model = MagicMock()
    mock_model.model_version = "unet-v1"

    if hasattr(app.state, "model"):
        delattr(app.state, "model")

    monkeypatch.setattr("api.main.load_model", lambda: mock_model)

    with TestClient(app) as client:
        yield client
