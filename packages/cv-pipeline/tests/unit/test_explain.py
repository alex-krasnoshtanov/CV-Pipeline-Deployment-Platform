"""Unit tests for cv_pipeline.explain (Seg-Grad-CAM).

These tests build a real (randomly-initialised) smp U-Net so the actual
Grad-CAM math runs, but keep it tiny (resnet18 encoder, single 256x256 patch)
so it stays fast on a CPU-only CI runner. The SegmentationModel constructor is
short-circuited with mocks so no checkpoint file is needed; we attach a real
torch module afterwards.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from cv_pipeline.explain import _resolve_target_layer, explain
from cv_pipeline.schema import ExplanationResult
from cv_pipeline.segmentation import SegmentationModel
from PIL import Image


@pytest.fixture
def tiny_model() -> SegmentationModel:
    """A SegmentationModel wrapping a small, real, untrained smp U-Net."""
    import segmentation_models_pytorch as smp

    checkpoint = {
        "model_state_dict": {"encoder.conv1.weight": torch.zeros(64, 3, 7, 7)},
        "model_version": "unet-test",
    }
    with (
        patch("cv_pipeline.segmentation.torch.load", return_value=checkpoint),
        patch.object(SegmentationModel, "_build_model"),
    ):
        model = SegmentationModel("fake.pth", patch_size=256, overlap=0.0, device="cpu")

    net = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    net.eval()
    model.model = net
    return model


def _make_image(tmp_path: Path, size: int = 256) -> Path:
    """Write a random RGB PNG of the given square size and return its path."""
    arr = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    path = tmp_path / "plate.png"
    Image.fromarray(arr).save(path)
    return path


@pytest.mark.unit
class TestExplain:
    """End-to-end behaviour of the explain() entry point."""

    def test_returns_explanation_result(self, tiny_model, tmp_path) -> None:
        """explain() should return a populated ExplanationResult."""
        image_path = _make_image(tmp_path)

        result = explain(image_path=image_path, model=tiny_model, crop=False)

        assert isinstance(result, ExplanationResult)
        assert result.method == "seg-grad-cam"
        assert result.model_version == "unet-test"
        assert result.image_width_px == 256
        assert result.image_height_px == 256
        assert result.target_layer == "decoder.blocks[-1]"

    def test_heatmap_matches_original_dimensions(self, tiny_model, tmp_path) -> None:
        """The decoded heatmap must be the same size as the input image."""
        image_path = _make_image(tmp_path, size=256)

        result = explain(image_path=image_path, model=tiny_model, crop=False)

        raw = base64.b64decode(result.heatmap_b64)
        heatmap = np.array(Image.open(io.BytesIO(raw)))
        assert heatmap.shape == (256, 256)
        assert heatmap.dtype == np.uint8
        assert heatmap.min() >= 0 and heatmap.max() <= 255

    def test_downscale_flag_set_for_large_image(self, tiny_model, tmp_path) -> None:
        """An image above max_side is processed downscaled but returned full-size."""
        image_path = _make_image(tmp_path, size=512)

        result = explain(
            image_path=image_path, model=tiny_model, crop=False, max_side=256
        )

        assert result.downscaled is True
        raw = base64.b64decode(result.heatmap_b64)
        heatmap = np.array(Image.open(io.BytesIO(raw)))
        # Heatmap is upsampled back to the original 512x512 frame.
        assert heatmap.shape == (512, 512)


@pytest.mark.unit
class TestResolveTargetLayer:
    """Target-layer resolution and its fallback."""

    def test_prefers_decoder_last_block(self, tiny_model) -> None:
        """A standard smp U-Net resolves to the last decoder block."""
        _, name = _resolve_target_layer(tiny_model.model)
        assert name == "decoder.blocks[-1]"

    def test_falls_back_to_last_conv(self) -> None:
        """A module without decoder.blocks falls back to the last Conv2d."""

        class Tiny(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.nn.Conv2d(3, 4, 3, padding=1)
                self.b = torch.nn.Conv2d(4, 1, 3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.b(self.a(x))

        layer, name = _resolve_target_layer(Tiny())
        assert name == "b"
        assert isinstance(layer, torch.nn.Conv2d)
