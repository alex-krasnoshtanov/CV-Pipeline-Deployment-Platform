"""Unit tests for the /feedback router."""

from __future__ import annotations

import hashlib
import uuid
from unittest.mock import AsyncMock, MagicMock

import bcrypt
import pytest
from api.auth.dependencies import require_admin
from api.db import get_db
from api.main import app
from api.services.feedback_service import PredictionNotFoundError
from api.services.mask_validation import MaskValidationError
from fastapi.testclient import TestClient

_TEST_KEY = "test-key-for-unit-tests"
_HASHED_KEY = bcrypt.hashpw(
    _TEST_KEY.encode("utf-8"),
    bcrypt.gensalt(),
).decode("utf-8")
_SHA256_KEY = hashlib.sha256(
    _TEST_KEY.encode("utf-8"),
).hexdigest()


def _build_mock_user() -> MagicMock:
    """Build a fresh mock user with valid hashes."""
    mock_user = MagicMock()
    mock_user.name = "Test User"
    mock_user.role = "researcher"
    mock_user.id = uuid.uuid4()
    mock_user.api_key_hash = _HASHED_KEY
    mock_user.key_sha256 = _SHA256_KEY
    return mock_user


def _build_admin_user() -> MagicMock:
    """Build a fresh mock admin user."""
    admin = MagicMock()
    admin.name = "Admin User"
    admin.role = "admin"
    admin.id = uuid.uuid4()
    admin.api_key_hash = _HASHED_KEY
    admin.key_sha256 = _SHA256_KEY
    return admin


@pytest.fixture(autouse=True)
def _override_db():
    """Replace the real DB session with a structured mock."""
    mock_user = _build_mock_user()

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

    app.dependency_overrides[get_db] = _mock_get_db
    yield
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set API_KEY env var for all tests in this module."""
    monkeypatch.setenv("API_KEY", _TEST_KEY)


@pytest.fixture
def mocked_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient with model loading mocked."""
    mock_model = MagicMock()
    mock_model.model_version = "unet-v1"

    if hasattr(app.state, "model"):
        delattr(app.state, "model")

    monkeypatch.setattr("api.main.load_model", lambda: mock_model)

    with TestClient(app) as client:
        yield client


def _valid_feedback_body() -> dict:
    """Return a minimal valid feedback request body."""
    return {
        "prediction_id": str(uuid.uuid4()),
        "flag": "bad",
        "notes": "Root tip 2 is misplaced.",
    }


# ---- authentication --------------------------------------------------


@pytest.mark.unit
class TestFeedbackAuth:
    """Tests for feedback endpoint authentication."""

    def test_returns_401_without_api_key(
        self,
        mocked_client: TestClient,
    ) -> None:
        """POST /feedback without X-API-Key should return 401."""
        response = mocked_client.post(
            "/feedback",
            json=_valid_feedback_body(),
        )

        assert response.status_code == 401

    def test_returns_401_with_wrong_key(
        self,
        mocked_client: TestClient,
    ) -> None:
        """POST /feedback with an invalid key should return 401."""
        response = mocked_client.post(
            "/feedback",
            json=_valid_feedback_body(),
            headers={"X-API-Key": "wrong-key"},
        )

        assert response.status_code == 401


# ---- happy path ------------------------------------------------------


@pytest.mark.unit
class TestFeedbackHappyPath:
    """Tests for successful feedback submission."""

    def test_returns_200_with_valid_feedback(
        self,
        mocked_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A valid feedback request should return 200."""
        fake_feedback = MagicMock()
        fake_feedback.id = uuid.uuid4()
        fake_feedback.prediction_id = uuid.uuid4()
        fake_feedback.flag = "bad"
        fake_feedback.created_at = "2026-05-01T10:30:00Z"

        monkeypatch.setattr(
            "api.routers.feedback.save_feedback",
            AsyncMock(return_value=fake_feedback),
        )

        response = mocked_client.post(
            "/feedback",
            json=_valid_feedback_body(),
            headers={"X-API-Key": _TEST_KEY},
        )

        assert response.status_code == 200
        body = response.json()
        assert "feedback_id" in body or "id" in body


# ---- validation errors -----------------------------------------------


@pytest.mark.unit
class TestFeedbackValidation:
    """Tests for feedback request validation."""

    def test_invalid_uuid_returns_422(
        self,
        mocked_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-UUID prediction_id should return 422."""
        monkeypatch.setattr(
            "api.routers.feedback.save_feedback",
            AsyncMock(
                side_effect=ValueError("is not a valid UUID"),
            ),
        )

        body = _valid_feedback_body()
        body["prediction_id"] = "not-a-uuid"

        response = mocked_client.post(
            "/feedback",
            json=body,
            headers={"X-API-Key": _TEST_KEY},
        )

        assert response.status_code == 422

    def test_missing_prediction_returns_404(
        self,
        mocked_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A valid UUID with no matching row should return 404."""
        pred_id = str(uuid.uuid4())
        monkeypatch.setattr(
            "api.routers.feedback.save_feedback",
            AsyncMock(
                side_effect=PredictionNotFoundError(
                    f"Prediction '{pred_id}' not found",
                ),
            ),
        )

        body = _valid_feedback_body()
        body["prediction_id"] = pred_id

        response = mocked_client.post(
            "/feedback",
            json=body,
            headers={"X-API-Key": _TEST_KEY},
        )

        assert response.status_code == 404

    def test_corrected_mask_on_public_endpoint_returns_422(
        self,
        mocked_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A mask sent to public POST /feedback should be rejected.

        Corrections must go through the admin-only relabel endpoint,
        which validates the mask. The public body forbids extra
        fields, so an unvalidated mask is rejected before it can be
        stored or silently resolve the prediction out of the queue.
        """
        called = AsyncMock()
        monkeypatch.setattr("api.routers.feedback.save_feedback", called)

        body = _valid_feedback_body()
        body["corrected_mask_b64"] = "aW52YWxpZA=="

        response = mocked_client.post(
            "/feedback",
            json=body,
            headers={"X-API-Key": _TEST_KEY},
        )

        assert response.status_code == 422
        called.assert_not_awaited()


# ---- review queue (admin) --------------------------------------------


@pytest.mark.unit
class TestReviewQueue:
    """Tests for GET /feedback/review-queue."""

    def test_returns_401_for_anonymous(
        self,
        mocked_client: TestClient,
    ) -> None:
        """Anonymous callers should get 401."""
        response = mocked_client.get("/feedback/review-queue")

        assert response.status_code == 401

    def test_returns_403_for_non_admin(
        self,
        mocked_client: TestClient,
    ) -> None:
        """An authenticated non-admin should get 403."""
        response = mocked_client.get(
            "/feedback/review-queue",
            headers={"X-API-Key": _TEST_KEY},
        )

        assert response.status_code == 403

    def test_returns_queue_for_admin(
        self,
        mocked_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An admin should get the list of predictions to review."""
        admin = _build_admin_user()
        app.dependency_overrides[require_admin] = lambda: admin
        try:
            item = {
                "prediction_id": str(uuid.uuid4()),
                "image_filename": "plate.png",
                "image_width_px": 32,
                "image_height_px": 24,
                "image_uri": "/data/feedback/raw/u/p.png",
                "mask_b64": "abc",
                "mask_confidence": 0.42,
                "flag": "bad",
                "notes": "wrong tip",
                "created_at": "2026-05-01T10:30:00Z",
            }
            monkeypatch.setattr(
                "api.routers.feedback.get_review_queue",
                AsyncMock(return_value=[item]),
            )

            response = mocked_client.get(
                "/feedback/review-queue",
                headers={"X-API-Key": _TEST_KEY},
            )

            assert response.status_code == 200
            body = response.json()
            assert isinstance(body, list)
            assert len(body) == 1
            assert body[0]["prediction_id"] == item["prediction_id"]
        finally:
            app.dependency_overrides.pop(require_admin, None)


# ---- relabel (admin) -------------------------------------------------


@pytest.mark.unit
class TestRelabel:
    """Tests for POST /feedback/relabel."""

    def test_returns_401_for_anonymous(
        self,
        mocked_client: TestClient,
    ) -> None:
        """Anonymous callers should get 401."""
        response = mocked_client.post(
            "/feedback/relabel",
            json={"prediction_id": str(uuid.uuid4()), "flag": "good"},
        )

        assert response.status_code == 401

    def test_returns_403_for_non_admin(
        self,
        mocked_client: TestClient,
    ) -> None:
        """An authenticated non-admin should get 403."""
        response = mocked_client.post(
            "/feedback/relabel",
            json={"prediction_id": str(uuid.uuid4()), "flag": "good"},
            headers={"X-API-Key": _TEST_KEY},
        )

        assert response.status_code == 403

    def test_returns_422_when_neither_mask_nor_flag(
        self,
        mocked_client: TestClient,
    ) -> None:
        """A relabel with no mask and no flag should return 422."""
        admin = _build_admin_user()
        app.dependency_overrides[require_admin] = lambda: admin
        try:
            response = mocked_client.post(
                "/feedback/relabel",
                json={"prediction_id": str(uuid.uuid4())},
                headers={"X-API-Key": _TEST_KEY},
            )

            assert response.status_code == 422
        finally:
            app.dependency_overrides.pop(require_admin, None)

    def test_returns_200_for_admin_correction(
        self,
        mocked_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A valid admin correction should return 200."""
        admin = _build_admin_user()
        app.dependency_overrides[require_admin] = lambda: admin
        try:
            fake_row = MagicMock()
            fake_row.id = uuid.uuid4()
            fake_row.prediction_id = uuid.uuid4()
            fake_row.flag = "bad"
            fake_row.created_at = "2026-05-01T10:30:00Z"

            monkeypatch.setattr(
                "api.routers.feedback.save_correction",
                AsyncMock(return_value=fake_row),
            )

            response = mocked_client.post(
                "/feedback/relabel",
                json={
                    "prediction_id": str(uuid.uuid4()),
                    "flag": "good",
                },
                headers={"X-API-Key": _TEST_KEY},
            )

            assert response.status_code == 200
            assert "feedback_id" in response.json()
        finally:
            app.dependency_overrides.pop(require_admin, None)

    def test_returns_404_for_missing_prediction(
        self,
        mocked_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A correction for an unknown prediction should return 404."""
        admin = _build_admin_user()
        app.dependency_overrides[require_admin] = lambda: admin
        try:
            monkeypatch.setattr(
                "api.routers.feedback.save_correction",
                AsyncMock(
                    side_effect=PredictionNotFoundError("not found"),
                ),
            )

            response = mocked_client.post(
                "/feedback/relabel",
                json={"prediction_id": str(uuid.uuid4()), "flag": "good"},
                headers={"X-API-Key": _TEST_KEY},
            )

            assert response.status_code == 404
        finally:
            app.dependency_overrides.pop(require_admin, None)

    def test_returns_422_for_invalid_mask(
        self,
        mocked_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A correction with an invalid mask should return 422."""
        admin = _build_admin_user()
        app.dependency_overrides[require_admin] = lambda: admin
        try:
            monkeypatch.setattr(
                "api.routers.feedback.save_correction",
                AsyncMock(
                    side_effect=MaskValidationError(
                        "MASK_DIMENSION_MISMATCH",
                        "wrong size",
                    ),
                ),
            )

            response = mocked_client.post(
                "/feedback/relabel",
                json={
                    "prediction_id": str(uuid.uuid4()),
                    "corrected_mask_b64": "abc",
                },
                headers={"X-API-Key": _TEST_KEY},
            )

            assert response.status_code == 422
            assert response.json()["detail"]["error_code"] == (
                "MASK_DIMENSION_MISMATCH"
            )
        finally:
            app.dependency_overrides.pop(require_admin, None)
