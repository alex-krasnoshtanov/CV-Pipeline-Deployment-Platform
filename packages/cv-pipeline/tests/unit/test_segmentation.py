"""Unit tests for cv_pipeline.segmentation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from cv_pipeline.segmentation import SegmentationModel

# ---- constructor validation ------------------------------------------


@pytest.mark.unit
class TestSegmentationValidation:
    """Tests for SegmentationModel constructor validation."""

    def test_rejects_zero_patch_size(self) -> None:
        """patch_size=0 should raise ValueError."""
        with pytest.raises(ValueError, match="patch_size must be greater than 0"):
            SegmentationModel("dummy.pth", patch_size=0)

    def test_rejects_negative_patch_size(self) -> None:
        """Negative patch_size should raise ValueError."""
        with pytest.raises(ValueError, match="patch_size must be greater than 0"):
            SegmentationModel("dummy.pth", patch_size=-10)

    def test_rejects_overlap_at_one(self) -> None:
        """overlap=1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="overlap must satisfy"):
            SegmentationModel("dummy.pth", overlap=1.0)

    def test_rejects_negative_overlap(self) -> None:
        """Negative overlap should raise ValueError."""
        with pytest.raises(ValueError, match="overlap must satisfy"):
            SegmentationModel("dummy.pth", overlap=-0.1)

    def test_rejects_overlap_above_one(self) -> None:
        """overlap > 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="overlap must satisfy"):
            SegmentationModel("dummy.pth", overlap=1.5)


# ---- model loading with mocked checkpoint ----------------------------


def _fake_checkpoint() -> dict:
    """Build a minimal fake checkpoint for testing."""
    state_dict = {
        "encoder.conv1.weight": torch.zeros(64, 1, 7, 7),
        "other.weight": torch.zeros(10),
    }
    return {
        "model_state_dict": state_dict,
        "model_version": "unet-test",
        "epoch": 1,
    }


@pytest.mark.unit
class TestSegmentationLoading:
    """Tests for model loading with mocked checkpoint."""

    @patch("cv_pipeline.segmentation.torch.load")
    @patch("cv_pipeline.segmentation.SegmentationModel._build_model")
    def test_detects_input_channels(
        self,
        mock_build: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        """Should detect 1 channel from checkpoint."""
        mock_load.return_value = _fake_checkpoint()

        model = SegmentationModel("fake.pth", device="cpu")

        assert model.in_channels == 1

    @patch("cv_pipeline.segmentation.torch.load")
    @patch("cv_pipeline.segmentation.SegmentationModel._build_model")
    def test_reads_model_version(
        self,
        mock_build: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        """Should read model_version from checkpoint."""
        mock_load.return_value = _fake_checkpoint()

        model = SegmentationModel("fake.pth", device="cpu")

        assert model.model_version == "unet-test"

    @patch("cv_pipeline.segmentation.torch.load")
    @patch("cv_pipeline.segmentation.SegmentationModel._build_model")
    def test_defaults_version_when_missing(
        self,
        mock_build: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        """Should default to unet-v0 when no version in checkpoint."""
        checkpoint = _fake_checkpoint()
        del checkpoint["model_version"]
        mock_load.return_value = checkpoint

        model = SegmentationModel("fake.pth", device="cpu")

        assert model.model_version == "unet-v0"

    @patch("cv_pipeline.weights.get_weights")
    @patch("cv_pipeline.segmentation.torch.load")
    @patch("cv_pipeline.segmentation.SegmentationModel._build_model")
    def test_registry_version_overrides_stale_checkpoint_metadata(
        self,
        mock_build: MagicMock,
        mock_load: MagicMock,
        mock_get_weights: MagicMock,
    ) -> None:
        """Requested registry version should win over stale checkpoint metadata."""
        checkpoint = _fake_checkpoint()
        checkpoint["model_version"] = "unet-v0"
        mock_load.return_value = checkpoint
        mock_get_weights.return_value = Path("/tmp/unet-v1.pth")

        with patch.dict("cv_pipeline.weights.REGISTRY", {"unet-v1": "https://x"}):
            model = SegmentationModel("unet-v1", device="cpu")

        assert model.model_version == "unet-v1"

    @patch("cv_pipeline.segmentation.torch.load")
    @patch("cv_pipeline.segmentation.SegmentationModel._build_model")
    def test_defaults_channels_when_key_missing(
        self,
        mock_build: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        """Should default to 1 channel when encoder key not found."""
        mock_load.return_value = {"model_state_dict": {"some.other.key": None}}

        model = SegmentationModel("fake.pth", device="cpu")

        assert model.in_channels == 1

    @patch("cv_pipeline.segmentation.torch.load")
    def test_load_checkpoint_failure_raises(
        self,
        mock_load: MagicMock,
    ) -> None:
        """Should raise RuntimeError if checkpoint cannot be loaded."""
        mock_load.side_effect = FileNotFoundError("not found")

        with pytest.raises(RuntimeError, match="Cannot load checkpoint"):
            SegmentationModel("missing.pth", device="cpu")

    @patch("cv_pipeline.segmentation.torch.load")
    @patch("cv_pipeline.segmentation.SegmentationModel._build_model")
    def test_step_calculation(
        self,
        mock_build: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        """Step should be patch_size * (1 - overlap)."""
        mock_load.return_value = _fake_checkpoint()

        model = SegmentationModel(
            "fake.pth",
            patch_size=256,
            overlap=0.5,
            device="cpu",
        )

        assert model.step == 128

    @patch("cv_pipeline.segmentation.torch.load")
    @patch("cv_pipeline.segmentation.SegmentationModel._build_model")
    def test_device_defaults_to_cpu_without_cuda(
        self,
        mock_build: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        """Device should default to cpu when CUDA unavailable."""
        mock_load.return_value = _fake_checkpoint()

        with patch(
            "cv_pipeline.segmentation.torch.cuda.is_available",
            return_value=False,
        ):
            model = SegmentationModel("fake.pth")

        assert model.device == "cpu"


# ---- patching and reconstruction helpers -----------------------------


@pytest.mark.unit
class TestSegmentationHelpers:
    """Tests for internal helper methods using a mocked model."""

    @staticmethod
    def _make_model() -> SegmentationModel:
        """Create a SegmentationModel with mocked loading."""
        with (
            patch("cv_pipeline.segmentation.torch.load") as mock_load,
            patch("cv_pipeline.segmentation.SegmentationModel._build_model"),
        ):
            mock_load.return_value = _fake_checkpoint()
            return SegmentationModel(
                "fake.pth",
                patch_size=256,
                overlap=0.5,
                device="cpu",
            )

    def test_calculate_padding_exact_fit(self) -> None:
        """No padding needed when image fits patches exactly."""
        model = self._make_model()
        padding = model._calculate_padding(256, 256)

        assert all(p == 0 for p in padding)

    def test_calculate_padding_adds_when_needed(self) -> None:
        """Padding should be added for non-fitting dimensions."""
        model = self._make_model()
        padding = model._calculate_padding(300, 300)

        top, bottom, left, right = padding
        assert (300 + top + bottom - 256) % model.step == 0

    def test_prepare_channels_grayscale(self) -> None:
        """Grayscale image should get channel dimension added."""
        model = self._make_model()
        model.in_channels = 1
        image = np.zeros((100, 100), dtype=np.uint8)

        result = model._prepare_channels(image)

        assert result.shape == (100, 100, 1)

    def test_prepare_channels_rgb_to_gray(self) -> None:
        """RGB image should be converted to grayscale for 1-channel model."""
        model = self._make_model()
        model.in_channels = 1
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = model._prepare_channels(image)

        assert result.shape == (100, 100, 1)

    def test_remove_padding_restores_size(self) -> None:
        """Removing padding should restore original dimensions."""
        model = self._make_model()
        padded = np.zeros((300, 300), dtype=np.float32)
        padding = (10, 10, 20, 20)

        result = model._remove_padding(padded, padding)

        assert result.shape == (280, 260)

    def test_pad_image_adds_border(self) -> None:
        """Padded image should be larger than original."""
        model = self._make_model()
        image = np.zeros((100, 100, 1), dtype=np.uint8)
        padding = (10, 10, 20, 20)

        result = model._pad_image(image, padding)

        assert result.shape[0] == 120
        assert result.shape[1] == 140

    def test_create_patches_correct_count(self) -> None:
        """Number of patches should match grid calculation."""
        model = self._make_model()
        image = np.zeros((256, 256, 1), dtype=np.uint8)

        patches = model._create_patches(image)

        assert patches.shape[0] == 1
        assert patches.shape[1] == 1
        assert patches.shape[2] == 256
        assert patches.shape[3] == 256

    def test_reconstruct_correct_shape(self) -> None:
        """Reconstructed map should match padded dimensions."""
        model = self._make_model()
        predictions = np.ones((1, 256, 256), dtype=np.float32) * 0.5

        result = model._reconstruct(predictions, 256, 256)

        assert result.shape == (256, 256)
        assert np.allclose(result, 0.5)


# ---- predict and predict_mask ----------------------------------------


@pytest.mark.unit
class TestSegmentationPredict:
    """Tests for the public predict() and predict_mask() methods."""

    @staticmethod
    def _make_model_with_mock_network() -> SegmentationModel:
        """Create a SegmentationModel whose internal torch model returns a constant.

        The internal self.model is replaced with a callable that returns
        a sigmoid-like output (constant 0.7) for any input patch. This
        avoids needing real weights while exercising the full predict flow:
        prepare channels -> pad -> patch -> forward pass -> reconstruct -> unpad.
        """
        with (
            patch("cv_pipeline.segmentation.torch.load") as mock_load,
            patch("cv_pipeline.segmentation.SegmentationModel._build_model"),
        ):
            mock_load.return_value = _fake_checkpoint()
            seg_model = SegmentationModel(
                "fake.pth",
                patch_size=256,
                overlap=0.5,
                device="cpu",
            )

        def fake_forward(tensor: torch.Tensor) -> torch.Tensor:
            """Return a constant probability for every pixel in the patch."""
            batch, channels, h, w = tensor.shape
            return torch.full((batch, 1, h, w), 0.7)

        seg_model.model = MagicMock(side_effect=fake_forward)
        seg_model.model.eval = MagicMock(return_value=seg_model.model)
        return seg_model

    def test_predict_returns_correct_shape(self) -> None:
        """predict() output should match input spatial dimensions."""
        model = self._make_model_with_mock_network()
        image = np.zeros((256, 256), dtype=np.uint8)

        prob_map = model.predict(image)

        assert prob_map.shape == (256, 256)

    def test_predict_returns_float32(self) -> None:
        """predict() output should be float32."""
        model = self._make_model_with_mock_network()
        image = np.zeros((256, 256), dtype=np.uint8)

        prob_map = model.predict(image)

        assert prob_map.dtype == np.float32

    def test_predict_values_in_zero_one_range(self) -> None:
        """All values in the probability map should be between 0 and 1."""
        model = self._make_model_with_mock_network()
        image = np.zeros((256, 256), dtype=np.uint8)

        prob_map = model.predict(image)

        assert prob_map.min() >= 0.0
        assert prob_map.max() <= 1.0

    def test_predict_handles_non_square_image(self) -> None:
        """predict() should handle images that are not exact patch multiples."""
        model = self._make_model_with_mock_network()
        image = np.zeros((300, 400), dtype=np.uint8)

        prob_map = model.predict(image)

        assert prob_map.shape == (300, 400)

    def test_predict_mask_returns_binary_and_confidence(self) -> None:
        """predict_mask() should return a uint8 binary mask and a float confidence."""
        model = self._make_model_with_mock_network()
        image = np.zeros((256, 256), dtype=np.uint8)

        mask, confidence = model.predict_mask(image)

        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_predict_mask_confidence_reflects_constant_output(self) -> None:
        """With constant 0.7 output, all pixels should be root at threshold 0.5."""
        model = self._make_model_with_mock_network()
        image = np.zeros((256, 256), dtype=np.uint8)

        mask, confidence = model.predict_mask(image)

        assert np.all(mask == 255)
        assert abs(confidence - 0.7) < 0.05
