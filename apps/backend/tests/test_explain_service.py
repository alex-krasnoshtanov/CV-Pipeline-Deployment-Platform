"""Unit tests for the explain service (serving-mode dispatch).

Cloud (``azure_ml``) explanation is delegated to the model endpoint; local /
on-prem reuses the in-memory model and runs Grad-CAM in a worker thread. These
tests monkeypatch the heavy calls so the suite stays fast and torch-free.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from api.services import explain_service as svc
from cv_pipeline.schema import Metadata


def _app(serving_mode: str, model: object | None = None) -> SimpleNamespace:
    """Build a minimal stand-in for the FastAPI app with the needed state.

    SimpleNamespace is used (not MagicMock) so a missing ``model`` attribute
    resolves to None via getattr, which MagicMock would not.
    """
    state = SimpleNamespace(service_state=SimpleNamespace(serving_mode=serving_mode))
    if model is not None:
        state.model = model
    return SimpleNamespace(state=state)


# ---- local model resolution ------------------------------------------


@pytest.mark.unit
def test_local_mode_reuses_in_memory_model() -> None:
    """local/on-prem: the model already on app.state is returned as-is."""
    sentinel = object()
    app = _app("local", model=sentinel)
    assert svc.get_explain_model(app) is sentinel


@pytest.mark.unit
def test_local_mode_raises_when_model_missing() -> None:
    """local mode with no loaded model is an explicit error, not a lazy load."""
    app = _app("local")  # no model attribute
    with pytest.raises(RuntimeError, match="No in-memory model"):
        svc.get_explain_model(app)


# ---- run_explanation dispatch ----------------------------------------


@pytest.mark.unit
def test_local_run_explanation_invokes_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """local: run_explanation resolves the model and calls cv_pipeline explain."""
    sentinel_model = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(svc, "get_explain_model", lambda app: sentinel_model)

    def _fake_pipeline_explain(image_path, model, metadata):
        captured["image_path"] = image_path
        captured["model"] = model
        captured["metadata"] = metadata
        return "LOCAL_RESULT"

    monkeypatch.setattr(svc, "_pipeline_explain", _fake_pipeline_explain)

    app = _app("local", model=sentinel_model)
    image_path = tmp_path / "plate.png"
    metadata = Metadata(plate_id="PL-1")

    result = asyncio.run(
        svc.run_explanation(app=app, image_path=image_path, metadata=metadata)
    )

    assert result == "LOCAL_RESULT"
    assert captured["model"] is sentinel_model
    assert captured["image_path"] == image_path
    assert captured["metadata"] is metadata


@pytest.mark.unit
def test_cloud_mode_delegates_to_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """azure_ml: run_explanation POSTs to the endpoint instead of loading a model."""
    captured: dict[str, object] = {}

    async def _fake_endpoint_explanation(image_path, metadata):
        captured["image_path"] = image_path
        captured["metadata"] = metadata
        return "CLOUD_RESULT"

    # run_explanation does `from api.services.endpoint_client import
    # run_endpoint_explanation` at call time, so patching the module attribute
    # is what the lazy import picks up.
    import api.services.endpoint_client as ec

    monkeypatch.setattr(ec, "run_endpoint_explanation", _fake_endpoint_explanation)

    app = _app("azure_ml")  # deliberately no in-memory model
    image_path = tmp_path / "plate.png"
    metadata = Metadata(plate_id="PL-2")

    result = asyncio.run(
        svc.run_explanation(app=app, image_path=image_path, metadata=metadata)
    )

    assert result == "CLOUD_RESULT"
    assert captured["image_path"] == image_path
    assert captured["metadata"] is metadata
