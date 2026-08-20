"""Unit tests for cv_pipeline.preprocessing."""

from __future__ import annotations

import numpy as np
import pytest
from cv_pipeline.preprocessing import (
    _fallback_box,
    _to_grayscale,
    crop_to_dish,
    detect_petri_dish,
)

# ---- grayscale conversion -------------------------------------------


@pytest.mark.unit
class TestToGrayscale:
    """Tests for the internal grayscale converter."""

    def test_rgb_to_grayscale(self) -> None:
        """RGB image should be converted to single channel."""
        rgb = np.zeros((300, 300, 3), dtype=np.uint8)
        gray = _to_grayscale(rgb)

        assert len(gray.shape) == 2
        assert gray.shape == (300, 300)

    def test_grayscale_passes_through(self) -> None:
        """Already grayscale image should not change shape."""
        gray_input = np.zeros((300, 300), dtype=np.uint8)
        gray_output = _to_grayscale(gray_input)

        assert gray_output.shape == (300, 300)


# ---- fallback box ----------------------------------------------------


@pytest.mark.unit
class TestFallbackBox:
    """Tests for the fallback bounding box."""

    def test_fallback_covers_80_percent(self) -> None:
        """Fallback should trim 10% from each side."""
        x1, y1, x2, y2 = _fallback_box(1000, 1000)

        assert x1 == 100
        assert y1 == 100
        assert x2 == 900
        assert y2 == 900

    def test_fallback_on_rectangular_image(self) -> None:
        """Margin should be based on the shortest side."""
        x1, y1, x2, y2 = _fallback_box(500, 1000)

        margin = int(500 * 0.1)
        assert x1 == margin
        assert y1 == margin


# ---- dish detection --------------------------------------------------


@pytest.mark.unit
class TestDetectPetriDish:
    """Tests for petri dish detection."""

    def test_returns_four_coordinates(self) -> None:
        """Result should be a tuple of four integers."""
        image = np.ones((500, 500), dtype=np.uint8) * 200
        x1, y1, x2, y2 = detect_petri_dish(image)

        assert isinstance(x1, int)
        assert isinstance(y1, int)
        assert isinstance(x2, int)
        assert isinstance(y2, int)

    def test_bounding_box_within_image(self) -> None:
        """Bounding box should not exceed image dimensions."""
        image = np.ones((500, 500), dtype=np.uint8) * 200
        x1, y1, x2, y2 = detect_petri_dish(image)

        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 500
        assert y2 <= 500

    def test_black_image_uses_fallback(self) -> None:
        """A fully black image should trigger the fallback box."""
        image = np.zeros((500, 500), dtype=np.uint8)
        x1, y1, x2, y2 = detect_petri_dish(image)

        assert x1 == 50
        assert y1 == 50
        assert x2 == 450
        assert y2 == 450


# ---- crop_to_dish ----------------------------------------------------


@pytest.mark.unit
class TestCropToDish:
    """Tests for the crop convenience wrapper."""

    def test_output_is_smaller_than_input(self) -> None:
        """Cropped image should be smaller or equal to original."""
        image = np.ones((500, 500), dtype=np.uint8) * 200
        cropped = crop_to_dish(image)

        assert cropped.shape[0] <= 500
        assert cropped.shape[1] <= 500

    def test_output_is_numpy_array(self) -> None:
        """Result should be a numpy array."""
        image = np.ones((500, 500), dtype=np.uint8) * 200
        cropped = crop_to_dish(image)

        assert isinstance(cropped, np.ndarray)

    def test_handles_rgb_input(self) -> None:
        """crop_to_dish should accept a 3-channel RGB image without error.

        The function internally converts to grayscale for dish detection,
        then crops the original image. This verifies the full path works
        with RGB input, not just grayscale.
        """
        image = np.ones((500, 500, 3), dtype=np.uint8) * 200
        cropped = crop_to_dish(image)

        assert isinstance(cropped, np.ndarray)
        assert cropped.shape[0] <= 500
        assert cropped.shape[1] <= 500
