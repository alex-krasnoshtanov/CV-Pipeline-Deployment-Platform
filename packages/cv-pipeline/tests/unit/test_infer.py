"""Unit tests for cv_pipeline.infer using mocked SegmentationModel."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from cv_pipeline.infer import _binarise, _encode_mask, infer
from cv_pipeline.schema import InferenceResult, Metadata
from cv_pipeline.validation import ValidationError
from PIL import Image


def _make_test_image(tmp_path: Path, name: str = "test.png", size: int = 300) -> Path:
    """Create a valid grayscale test image on disk."""
    img = Image.fromarray(np.zeros((size, size), dtype=np.uint8))
    path = tmp_path / name
    img.save(path)
    return path


def _make_mock_model(h: int = 300, w: int = 300) -> MagicMock:
    """Create a mock SegmentationModel that returns a fake prob map."""
    model = MagicMock()
    model.model_version = "unet-v0"

    prob_map = np.random.rand(h, w).astype(np.float32) * 0.3
    prob_map[100:200, 100:200] = 0.9
    model.predict.return_value = prob_map
    return model


def _make_dynamic_mock_model() -> MagicMock:
    """Create a mock model whose predict() adapts to any input shape.

    This is needed for tests where the image gets cropped before inference,
    so the exact dimensions passed to predict() are not known in advance.
    """
    model = MagicMock()
    model.model_version = "unet-v0"

    def dynamic_predict(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        prob_map = np.ones((h, w), dtype=np.float32) * 0.8
        return prob_map

    model.predict.side_effect = dynamic_predict
    return model


# ---- infer() happy path ----------------------------------------------


@pytest.mark.unit
class TestInferHappyPath:
    """Tests for successful inference calls."""

    def test_returns_inference_result(self, tmp_path: Path) -> None:
        """infer should return an InferenceResult."""
        image_path = _make_test_image(tmp_path)
        model = _make_mock_model()

        result = infer(
            image_path=image_path,
            model=model,
            crop=False,
            num_plants=1,
            plant_start=150,
            plant_step=300,
            roi_width=150,
        )

        assert isinstance(result, InferenceResult)

    def test_result_has_correct_filename(self, tmp_path: Path) -> None:
        """Result should contain the input filename."""
        image_path = _make_test_image(tmp_path, name="plate_001.png")
        model = _make_mock_model()

        result = infer(
            image_path=image_path,
            model=model,
            crop=False,
            num_plants=1,
            plant_start=150,
            plant_step=300,
            roi_width=150,
        )

        assert result.image_filename == "plate_001.png"

    def test_result_has_correct_dimensions(self, tmp_path: Path) -> None:
        """Reported dimensions should match the image."""
        image_path = _make_test_image(tmp_path, size=400)
        model = _make_mock_model(h=400, w=400)

        result = infer(
            image_path=image_path,
            model=model,
            crop=False,
            num_plants=1,
            plant_start=200,
            plant_step=400,
            roi_width=200,
        )

        assert result.image_width_px == 400
        assert result.image_height_px == 400

    def test_result_has_mask_b64(self, tmp_path: Path) -> None:
        """Result should contain a non-empty base64 mask."""
        image_path = _make_test_image(tmp_path)
        model = _make_mock_model()

        result = infer(
            image_path=image_path,
            model=model,
            crop=False,
            num_plants=1,
            plant_start=150,
            plant_step=300,
            roi_width=150,
        )

        assert len(result.mask_b64) > 0

    def test_result_confidence_in_range(self, tmp_path: Path) -> None:
        """Mask confidence should be between 0 and 1."""
        image_path = _make_test_image(tmp_path)
        model = _make_mock_model()

        result = infer(
            image_path=image_path,
            model=model,
            crop=False,
            num_plants=1,
            plant_start=150,
            plant_step=300,
            roi_width=150,
        )

        assert 0.0 <= result.mask_confidence <= 1.0

    def test_default_metadata_is_empty(self, tmp_path: Path) -> None:
        """When no metadata is passed, defaults should be None."""
        image_path = _make_test_image(tmp_path)
        model = _make_mock_model()

        result = infer(
            image_path=image_path,
            model=model,
            crop=False,
            num_plants=1,
            plant_start=150,
            plant_step=300,
            roi_width=150,
        )

        assert result.metadata.plate_id is None

    def test_custom_metadata_passes_through(self, tmp_path: Path) -> None:
        """Provided metadata should appear in the result."""
        image_path = _make_test_image(tmp_path)
        model = _make_mock_model()
        meta = Metadata(plate_id="PL-001", experiment_id="EXP-42")

        result = infer(
            image_path=image_path,
            model=model,
            metadata=meta,
            crop=False,
            num_plants=1,
            plant_start=150,
            plant_step=300,
            roi_width=150,
        )

        assert result.metadata.plate_id == "PL-001"
        assert result.metadata.experiment_id == "EXP-42"

    def test_pipeline_version_matches_package(self, tmp_path: Path) -> None:
        """Result pipeline_version should match __version__."""
        from cv_pipeline._version import __version__

        image_path = _make_test_image(tmp_path)
        model = _make_mock_model()

        result = infer(
            image_path=image_path,
            model=model,
            crop=False,
            num_plants=1,
            plant_start=150,
            plant_step=300,
            roi_width=150,
        )

        assert result.pipeline_version == __version__

    def test_crop_true_returns_valid_result(self, tmp_path: Path) -> None:
        """infer with crop=True should crop then run inference successfully.

        Uses a bright-centred image so crop_to_dish has a region to detect,
        and a dynamic mock model that adapts to whatever cropped size it
        receives.
        """
        arr = np.zeros((500, 500), dtype=np.uint8)
        arr[100:400, 100:400] = 200
        img = Image.fromarray(arr)
        path = tmp_path / "dish.png"
        img.save(path)

        model = _make_dynamic_mock_model()

        result = infer(
            image_path=path,
            model=model,
            crop=True,
            num_plants=1,
            plant_start=150,
            plant_step=300,
            roi_width=150,
        )

        assert isinstance(result, InferenceResult)
        assert len(result.mask_b64) > 0
        assert 0.0 <= result.mask_confidence <= 1.0
        model.predict.assert_called_once()


# ---- infer() validation failures -------------------------------------


@pytest.mark.unit
class TestInferValidation:
    """Tests for validation errors during inference."""

    def test_rejects_invalid_extension(self, tmp_path: Path) -> None:
        """infer should raise ValidationError for bad file types."""
        path = tmp_path / "test.bmp"
        path.write_bytes(b"fake")
        model = _make_mock_model()

        with pytest.raises(ValidationError) as exc_info:
            infer(image_path=path, model=model, crop=False)

        assert exc_info.value.error_code == "UNSUPPORTED_FILE_TYPE"

    def test_rejects_missing_file(self, tmp_path: Path) -> None:
        """infer should raise ValidationError for missing files."""
        path = tmp_path / "missing.png"
        model = _make_mock_model()

        with pytest.raises(ValidationError) as exc_info:
            infer(image_path=path, model=model, crop=False)

        assert exc_info.value.error_code == "CORRUPT_FILE"


# ---- helper functions ------------------------------------------------


@pytest.mark.unit
class TestBinarise:
    """Tests for the _binarise helper."""

    def test_above_threshold_becomes_255(self) -> None:
        """Pixels above threshold should be 255."""
        prob_map = np.array([[0.8, 0.2], [0.6, 0.1]], dtype=np.float32)
        binary, confidence = _binarise(prob_map, threshold=0.5)

        assert binary[0, 0] == 255
        assert binary[0, 1] == 0
        assert binary[1, 0] == 255
        assert binary[1, 1] == 0

    def test_empty_mask_returns_zero_confidence(self) -> None:
        """All-zero prob map should give 0.0 confidence."""
        prob_map = np.zeros((10, 10), dtype=np.float32)
        binary, confidence = _binarise(prob_map, threshold=0.5)

        assert confidence == 0.0
        assert np.all(binary == 0)

    def test_confidence_is_mean_of_root_pixels(self) -> None:
        """Confidence should be mean of probs where mask is root."""
        prob_map = np.array([[0.9, 0.0], [0.7, 0.0]], dtype=np.float32)
        _, confidence = _binarise(prob_map, threshold=0.5)

        expected = (0.9 + 0.7) / 2
        assert abs(confidence - expected) < 0.01


@pytest.mark.unit
class TestEncodeMask:
    """Tests for the _encode_mask helper."""

    def test_returns_base64_string(self) -> None:
        """Encoded mask should be a non-empty string."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        result = _encode_mask(mask)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_can_decode_back_to_image(self) -> None:
        """Base64 mask should decode to a valid PNG."""
        import base64
        import io

        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        b64 = _encode_mask(mask)

        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw))
        assert img.size == (50, 50)
