"""Unit tests for the /health router."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from api.main import app
from fastapi.testclient import TestClient


@pytest.fixture
def _loaded_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient where the model has loaded successfully."""
    mock_model = MagicMock()
    mock_model.model_version = "unet-v1"

    if hasattr(app.state, "model"):
        delattr(app.state, "model")

    monkeypatch.setattr("api.main.load_model", lambda: mock_model)

    with TestClient(app) as client:
        yield client


@pytest.fixture
def _unloaded_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient where the model failed to load."""
    if hasattr(app.state, "model"):
        delattr(app.state, "model")

    def _raise() -> None:
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("api.main.load_model", _raise)

    with TestClient(app) as client:
        yield client


# ---- healthy ---------------------------------------------------------


@pytest.mark.unit
class TestHealthHappyPath:
    """Tests for a healthy /health response."""

    def test_returns_200_when_model_loaded(
        self,
        _loaded_client: TestClient,
    ) -> None:
        """GET /health should return 200 when the model is ready."""
        response = _loaded_client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    def test_response_contains_version_fields(
        self,
        _loaded_client: TestClient,
    ) -> None:
        """Response should include pipeline_version and model_version."""
        body = _loaded_client.get("/health").json()

        assert "pipeline_version" in body
        assert body["model_version"] == "unet-v1"

    def test_response_contains_serving_mode(
        self,
        _loaded_client: TestClient,
    ) -> None:
        """Response should include the current serving mode."""
        body = _loaded_client.get("/health").json()

        assert "serving_mode" in body

    def test_no_auth_required(
        self,
        _loaded_client: TestClient,
    ) -> None:
        """GET /health must be accessible without an API key."""
        response = _loaded_client.get("/health")

        assert response.status_code == 200


# ---- unhealthy -------------------------------------------------------


@pytest.mark.unit
class TestHealthUnhealthy:
    """Tests for /health when the system is not ready."""

    def test_returns_503_when_model_not_loaded(
        self,
        _unloaded_client: TestClient,
    ) -> None:
        """GET /health should return 503 when model loading failed."""
        response = _unloaded_client.get("/health")

        assert response.status_code == 503
        body = response.json()
        assert body["model_loaded"] is False
