"""Unit tests for the /explain router.

Mirrors test_infer_router: /explain is anonymous-allowed and maps cv_pipeline
ValidationError to the same HTTP statuses. The heavy Grad-CAM call
(run_explanation) is mocked so these stay fast and torch-light; the actual
Seg-Grad-CAM math is covered in packages/cv-pipeline/tests/unit/test_explain.py.
"""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from api.main import app
from cv_pipeline.schema import ExplanationResult, Metadata
from cv_pipeline.validation import ValidationError
from fastapi.testclient import TestClient

_TEST_KEY = "test-key-for-unit-tests"


def _png_bytes() -> bytes:
    """Return a tiny valid PNG payload for multipart upload tests."""
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
        "ASsJTYQAAAAASUVORK5CYII="
    )


def _fake_explanation() -> ExplanationResult:
    """Build a deterministic fake explanation result."""
    return ExplanationResult(
        pipeline_version="0.1.0",
        model_version="unet-v1",
        timestamp="2026-04-20T12:00:00Z",
        image_filename="test.png",
        image_width_px=256,
        image_height_px=256,
        metadata=Metadata(plate_id="PL-001"),
        method="seg-grad-cam",
        target_layer="decoder.blocks[-1]",
        downscaled=False,
        heatmap_peak=0.42,
        heatmap_b64=base64.b64encode(b"fake-heatmap").decode("ascii"),
    )


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set API_KEY env var for all tests in this module."""
    monkeypatch.setenv("API_KEY", _TEST_KEY)


@pytest.fixture
def mocked_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with a mocked model and a mocked Grad-CAM call."""
    mock_model = MagicMock()
    mock_model.model_version = "unet-v1"

    if hasattr(app.state, "model"):
        delattr(app.state, "model")

    monkeypatch.setattr("api.main.load_model", lambda: mock_model)

    async def _default_fake_explanation(**_: object) -> ExplanationResult:
        return _fake_explanation()

    # The router imports run_explanation into its own namespace.
    monkeypatch.setattr(
        "api.routers.explain.run_explanation",
        _default_fake_explanation,
    )

    with TestClient(app) as client:
        yield client


# ---- happy paths -----------------------------------------------------


@pytest.mark.unit
def test_explain_anonymous_returns_200(mocked_client: TestClient) -> None:
    """No credentials should still return a heatmap (anonymous allowed)."""
    response = mocked_client.post(
        "/explain",
        files={"image": ("test.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "seg-grad-cam"
    assert body["target_layer"] == "decoder.blocks[-1]"
    assert body["heatmap_b64"]
    assert body["image_width_px"] == 256


@pytest.mark.unit
def test_explain_returns_200_with_api_key(mocked_client: TestClient) -> None:
    """A valid X-API-Key returns 200 (authenticated logging branch)."""
    response = mocked_client.post(
        "/explain",
        files={"image": ("test.png", _png_bytes(), "image/png")},
        data={"plate_id": "PL-001"},
        headers={"X-API-Key": _TEST_KEY},
    )
    assert response.status_code == 200
    assert response.json()["method"] == "seg-grad-cam"


# ---- error handling --------------------------------------------------


@pytest.mark.unit
def test_explain_returns_503_when_model_not_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If startup fails to load a model, /explain returns MODEL_NOT_READY."""
    if hasattr(app.state, "model"):
        delattr(app.state, "model")

    def _raise() -> MagicMock:
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("api.main.load_model", _raise)

    with TestClient(app) as client:
        response = client.post(
            "/explain",
            files={"image": ("test.png", _png_bytes(), "image/png")},
        )

    assert response.status_code == 503
    assert response.json()["error_code"] == "MODEL_NOT_READY"


@pytest.mark.unit
def test_explain_maps_validation_error_to_correct_status(
    mocked_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ValidationError from the pipeline maps to the correct HTTP status."""

    async def _raise_validation(**_: object) -> ExplanationResult:
        raise ValidationError("IMAGE_TOO_SMALL", "image too small")

    monkeypatch.setattr(
        "api.routers.explain.run_explanation",
        _raise_validation,
    )

    response = mocked_client.post(
        "/explain",
        files={"image": ("test.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 422
    assert response.json()["error_code"] == "IMAGE_TOO_SMALL"


@pytest.mark.unit
def test_explain_cleans_up_temp_file(
    mocked_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Temp upload file should be unlinked even when explanation fails."""
    unlinked: list[Path] = []

    def _tracking_unlink(self: Path, *args: object, **kwargs: object) -> None:
        unlinked.append(self)

    async def _raise_validation(**_: object) -> ExplanationResult:
        raise ValidationError("IMAGE_TOO_SMALL", "image too small")

    monkeypatch.setattr(
        "api.services.inference_service.Path.unlink",
        _tracking_unlink,
    )
    monkeypatch.setattr(
        "api.routers.explain.run_explanation",
        _raise_validation,
    )

    response = mocked_client.post(
        "/explain",
        files={"image": ("test.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 422
    assert len(unlinked) >= 1


@pytest.mark.unit
def test_explain_maps_unexpected_error_to_500(
    mocked_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-ValidationError failure returns a structured EXPLAIN_FAILED 500."""

    async def _raise_runtime(**_: object) -> ExplanationResult:
        raise RuntimeError("grad-cam hook never fired")

    monkeypatch.setattr("api.routers.explain.run_explanation", _raise_runtime)

    response = mocked_client.post(
        "/explain",
        files={"image": ("test.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 500
    assert response.json()["error_code"] == "EXPLAIN_FAILED"


@pytest.mark.unit
def test_explain_maps_timeout_to_504(
    mocked_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out explanation returns a structured EXPLAIN_TIMEOUT 504."""

    async def _raise_timeout(**_: object) -> ExplanationResult:
        raise TimeoutError

    monkeypatch.setattr("api.routers.explain.run_explanation", _raise_timeout)

    response = mocked_client.post(
        "/explain",
        files={"image": ("test.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 504
    assert response.json()["error_code"] == "EXPLAIN_TIMEOUT"
