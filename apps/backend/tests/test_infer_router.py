"""Unit tests for the /infer router.

/infer is anonymous-allowed: a request with no credentials returns
the inference result and saves the prediction with user_id=NULL.
Authenticated callers (cookie OR X-API-Key) get a looser rate
limit and the prediction is bound to their user row.
"""

from __future__ import annotations

import base64
import hashlib
import json
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import bcrypt
import pytest
from api.db import get_db
from api.main import _rate_limit_handler, app, limiter
from api.middleware.request_id import request_id_ctx
from cv_pipeline.schema import InferenceResult, Landmark, Metadata
from cv_pipeline.validation import ValidationError
from fastapi.testclient import TestClient

_TEST_KEY = "test-key-for-unit-tests"

_HASHED_KEY = bcrypt.hashpw(
    _TEST_KEY.encode("utf-8"),
    bcrypt.gensalt(),
).decode("utf-8")

_SHA256_KEY = hashlib.sha256(
    _TEST_KEY.encode("utf-8"),
).hexdigest()


def _png_bytes() -> bytes:
    """Return a tiny valid PNG payload for multipart upload tests."""
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
        "ASsJTYQAAAAASUVORK5CYII="
    )


def _fake_result() -> InferenceResult:
    """Build a deterministic fake inference result."""
    return InferenceResult(
        pipeline_version="0.1.0",
        model_version="unet-v1",
        timestamp="2026-04-20T12:00:00Z",
        image_filename="test.png",
        image_width_px=256,
        image_height_px=256,
        metadata=Metadata(
            plate_id="PL-001",
            experiment_id="EXP-001",
            timestamp="2026-04-20T11:59:00Z",
        ),
        mask_b64=base64.b64encode(b"fake-mask").decode("ascii"),
        mask_confidence=0.91,
        landmark_count=1,
        landmarks=[Landmark(id=0, x=10, y=20, confidence=0.88)],
    )


@pytest.mark.unit
def test_rate_limit_configuration_and_envelope() -> None:
    """The app limiter and 429 handler should match the API contract."""
    assert limiter._default_limits[0]._LimitGroup__limit_provider == "20/minute"

    token = request_id_ctx.set("req-rate-limit-test")
    try:
        response = _rate_limit_handler(
            object(),
            MagicMock(detail="Too many requests"),
        )
    finally:
        request_id_ctx.reset(token)

    assert response.status_code == 429
    body = json.loads(response.body)
    assert body["error_code"] == "RATE_LIMITED"
    assert body["message"] == "Rate limit exceeded: Too many requests"
    assert body["pipeline_version"]
    assert body["timestamp"]
    assert body["request_id"] == "req-rate-limit-test"


def _build_mock_user() -> MagicMock:
    """Build a fresh mock user with valid hashes for each test."""
    mock_user = MagicMock()
    mock_user.name = "Test User"
    mock_user.role = "admin"
    mock_user.id = uuid.uuid4()
    mock_user.api_key_hash = _HASHED_KEY
    mock_user.key_sha256 = _SHA256_KEY
    mock_user.email = None
    mock_user.password_hash = None
    return mock_user


def _mock_db_session_with_user(user: MagicMock | None = None):
    """Build a mock DB session for the SHA-256 auth flow."""
    mock_user = user or _build_mock_user()

    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = mock_user

    scalars_result = MagicMock()
    scalars_result.first.return_value = mock_user.id
    scalars_result.all.return_value = [mock_user]

    session = AsyncMock()
    session.execute.return_value = execute_result
    session.scalars.return_value = scalars_result
    return session


@pytest.fixture(autouse=True)
def _override_db():
    """Replace the real DB session with a structured mock."""

    async def _mock_get_db():
        yield _mock_db_session_with_user()

    app.dependency_overrides[get_db] = _mock_get_db
    yield
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def mocked_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient with model and inference pipeline mocked.

    The inference pipeline mock is set up here (instead of per-test)
    because the new anonymous-allowed /infer would otherwise call
    the real cv_pipeline on a fake 1x1 PNG and raise IMAGE_TOO_SMALL.
    Tests that want to assert validation behaviour override this
    mock with their own.
    """
    mock_model = MagicMock()
    mock_model.model_version = "unet-v1"

    if hasattr(app.state, "model"):
        delattr(app.state, "model")

    monkeypatch.setattr("api.main.load_model", lambda: mock_model)

    async def _default_fake_pipeline(**_: object) -> InferenceResult:
        return _fake_result()

    monkeypatch.setattr(
        "api.services.inference_service.run_pipeline_inference",
        _default_fake_pipeline,
    )
    monkeypatch.setattr(
        "api.routers.infer.save_prediction",
        AsyncMock(return_value=None),
    )

    with TestClient(app) as client:
        yield client


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set API_KEY env var for all tests in this module."""
    monkeypatch.setenv("API_KEY", _TEST_KEY)


# ---- anonymous + authenticated paths --------------------------------


@pytest.mark.unit
def test_infer_anonymous_returns_200(
    mocked_client: TestClient,
) -> None:
    """No credentials should still return a prediction (anonymous)."""
    response = mocked_client.post(
        "/infer",
        files={"image": ("test.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["pipeline_version"] == "0.1.0"
    # Anonymous calls do not get a prediction_id when save fails or
    # is mocked to return None.
    assert body.get("prediction_id") is None


@pytest.mark.unit
def test_infer_with_invalid_key_falls_through_to_anonymous(
    mocked_client: TestClient,
) -> None:
    """An invalid X-API-Key is treated as no credentials (anonymous).

    This is a deliberate design choice in optional_user: bad creds
    do not raise, they degrade. Service clients sending a stale key
    will notice via the lower rate limit, not via 401.
    """
    # The mock DB returns mock_user regardless of WHERE clause, but
    # the bcrypt verification step inside lookup_user_by_api_key
    # will fail because mock_user.api_key_hash was built for
    # _TEST_KEY, not "wrong-key". Hence: anonymous fallback.
    response = mocked_client.post(
        "/infer",
        files={"image": ("test.png", _png_bytes(), "image/png")},
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 200


@pytest.mark.unit
def test_infer_returns_200_with_valid_api_key(
    mocked_client: TestClient,
) -> None:
    """A valid X-API-Key returns 200 and the expected schema."""
    response = mocked_client.post(
        "/infer",
        files={"image": ("test.png", _png_bytes(), "image/png")},
        data={"plate_id": "PL-001", "experiment_id": "EXP-001"},
        headers={"X-API-Key": _TEST_KEY},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["pipeline_version"] == "0.1.0"
    assert body["model_version"] == "unet-v1"
    assert body["image_filename"] == "test.png"
    assert body["image_width_px"] == 256
    assert body["image_height_px"] == 256
    assert body["landmark_count"] == 1
    assert isinstance(body["landmarks"], list)
    assert body["landmarks"][0]["id"] == 0
    assert 0.0 <= body["mask_confidence"] <= 1.0


# ---- error handling --------------------------------------------------


@pytest.mark.unit
def test_infer_returns_503_when_model_not_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If startup fails to load a model, /infer returns MODEL_NOT_READY."""
    if hasattr(app.state, "model"):
        delattr(app.state, "model")

    def _raise() -> MagicMock:
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("api.main.load_model", _raise)

    with TestClient(app) as client:
        response = client.post(
            "/infer",
            files={"image": ("test.png", _png_bytes(), "image/png")},
            headers={"X-API-Key": _TEST_KEY},
        )

    assert response.status_code == 503
    body = response.json()
    assert body["error_code"] == "MODEL_NOT_READY"


@pytest.mark.unit
def test_infer_maps_validation_error_to_correct_status(
    mocked_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ValidationError from pipeline should map to correct HTTP status."""

    async def _raise_validation(**_: object) -> InferenceResult:
        raise ValidationError("IMAGE_TOO_SMALL", "image too small")

    monkeypatch.setattr(
        "api.services.inference_service.run_pipeline_inference",
        _raise_validation,
    )

    response = mocked_client.post(
        "/infer",
        files={"image": ("test.png", _png_bytes(), "image/png")},
        headers={"X-API-Key": _TEST_KEY},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "IMAGE_TOO_SMALL"


@pytest.mark.unit
def test_infer_cleans_up_temp_file(
    mocked_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Temp upload file should be unlinked even on error."""

    unlinked: list[Path] = []

    def _tracking_unlink(
        self: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        unlinked.append(self)

    async def _raise_validation(**_: object) -> InferenceResult:
        raise ValidationError("IMAGE_TOO_SMALL", "image too small")

    monkeypatch.setattr(
        "api.services.inference_service.Path.unlink",
        _tracking_unlink,
    )
    monkeypatch.setattr(
        "api.services.inference_service.run_pipeline_inference",
        _raise_validation,
    )

    response = mocked_client.post(
        "/infer",
        files={"image": ("test.png", _png_bytes(), "image/png")},
        headers={"X-API-Key": _TEST_KEY},
    )

    assert response.status_code == 422
    assert len(unlinked) >= 1


@pytest.mark.unit
def test_infer_anonymous_works_with_empty_users_table(
    mocked_client: TestClient,
) -> None:
    """Empty users table no longer blocks /infer (was 503 in old auth).

    Anonymous inference does not require any user to exist. Only
    the legacy /auth/me path still raises SERVER_MISCONFIGURED on
    an empty table.
    """
    # SHA-256 lookup returns None (no user matches the bad key)
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = None

    scalars_result = MagicMock()
    scalars_result.first.return_value = None

    async def _mock_get_empty_db():
        session = AsyncMock()
        session.execute.return_value = execute_result
        session.scalars.return_value = scalars_result
        yield session

    app.dependency_overrides[get_db] = _mock_get_empty_db

    response = mocked_client.post(
        "/infer",
        files={"image": ("test.png", _png_bytes(), "image/png")},
        headers={"X-API-Key": _TEST_KEY},
    )

    assert response.status_code == 200


# ---- resilience ------------------------------------------------------


@pytest.mark.unit
def test_infer_returns_200_when_prediction_save_fails(
    mocked_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The API should still return 200 even if saving to the DB fails."""

    async def _mock_save_error(
        *args: object,
        **kwargs: object,
    ) -> None:
        raise Exception("Database connection lost during save")

    monkeypatch.setattr(
        "api.routers.infer.save_prediction",
        _mock_save_error,
    )

    response = mocked_client.post(
        "/infer",
        files={"image": ("test.png", _png_bytes(), "image/png")},
        headers={"X-API-Key": _TEST_KEY},
    )

    assert response.status_code == 200
    assert response.json().get("prediction_id") is None
