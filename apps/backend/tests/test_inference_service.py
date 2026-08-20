"""Unit tests for api.services.inference_service."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from cv_pipeline.schema import InferenceResult, Landmark, Metadata
from cv_pipeline.validation import ValidationError
from PIL import Image


def _make_test_image(tmp_path: Path) -> Path:
    """Create a minimal valid test image and return its path."""
    img = Image.fromarray(np.zeros((300, 300), dtype=np.uint8))
    path = tmp_path / "test.png"
    img.save(path)
    return path


def _fake_inference_result() -> InferenceResult:
    """Build a deterministic InferenceResult for assertions."""
    return InferenceResult(
        pipeline_version="0.1.0",
        model_version="unet-v1",
        timestamp="2026-05-01T12:00:00Z",
        image_filename="test.png",
        image_width_px=300,
        image_height_px=300,
        metadata=Metadata(),
        mask_b64=base64.b64encode(b"fake-mask").decode("ascii"),
        mask_confidence=0.85,
        landmark_count=1,
        landmarks=[Landmark(id=0, x=150, y=200, confidence=0.9)],
    )


@pytest.mark.unit
class TestRunPipelineInference:
    """Tests for the run_pipeline_inference service function."""

    @pytest.mark.anyio
    async def test_returns_inference_result(self, tmp_path: Path) -> None:
        """The service should return an InferenceResult from the pipeline."""
        from api.services.inference_service import run_pipeline_inference

        image_path = _make_test_image(tmp_path)
        mock_model = MagicMock()
        mock_model.model_version = "unet-v1"
        metadata = Metadata()
        expected = _fake_inference_result()

        with patch(
            "api.services.inference_service._pipeline_infer",
            return_value=expected,
        ):
            result = await run_pipeline_inference(
                image_path=image_path,
                model=mock_model,
                metadata=metadata,
            )

        assert isinstance(result, InferenceResult)
        assert result.image_filename == "test.png"

    @pytest.mark.anyio
    async def test_passes_model_to_pipeline(self, tmp_path: Path) -> None:
        """The service should forward the model object to the pipeline."""
        from api.services.inference_service import run_pipeline_inference

        image_path = _make_test_image(tmp_path)
        mock_model = MagicMock()
        mock_model.model_version = "unet-v1"
        metadata = Metadata(plate_id="PL-001")

        with patch(
            "api.services.inference_service._pipeline_infer",
            return_value=_fake_inference_result(),
        ) as mock_infer:
            await run_pipeline_inference(
                image_path=image_path,
                model=mock_model,
                metadata=metadata,
            )

        mock_infer.assert_called_once_with(
            image_path=image_path,
            model=mock_model,
            metadata=metadata,
        )

    @pytest.mark.anyio
    async def test_propagates_validation_error(self, tmp_path: Path) -> None:
        """ValidationError from cv_pipeline should propagate unchanged."""
        from api.services.inference_service import run_pipeline_inference

        image_path = _make_test_image(tmp_path)
        mock_model = MagicMock()
        metadata = Metadata()

        with patch(
            "api.services.inference_service._pipeline_infer",
            side_effect=ValidationError("IMAGE_TOO_SMALL", "too small"),
        ):
            with pytest.raises(ValidationError) as exc_info:
                await run_pipeline_inference(
                    image_path=image_path,
                    model=mock_model,
                    metadata=metadata,
                )

        assert exc_info.value.error_code == "IMAGE_TOO_SMALL"
