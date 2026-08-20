"""Unit tests for cv_pipeline.landmarks."""

from __future__ import annotations

import numpy as np
import pytest
from cv_pipeline.landmarks import (
    _coords_in_bounds,
    _find_root_tip,
    detect_landmarks,
)
from cv_pipeline.schema import Landmark

# ---- coords_in_bounds ------------------------------------------------


@pytest.mark.unit
class TestCoordsInBounds:
    """Tests for the bounds-checking helper."""

    def test_valid_coordinates(self) -> None:
        """Coordinates inside the array should return True."""
        assert _coords_in_bounds(5, 5, (10, 10)) is True

    def test_origin_is_valid(self) -> None:
        """(0, 0) should be valid."""
        assert _coords_in_bounds(0, 0, (10, 10)) is True

    def test_negative_y_is_invalid(self) -> None:
        """Negative y should return False."""
        assert _coords_in_bounds(-1, 5, (10, 10)) is False

    def test_negative_x_is_invalid(self) -> None:
        """Negative x should return False."""
        assert _coords_in_bounds(5, -1, (10, 10)) is False

    def test_y_at_boundary_is_invalid(self) -> None:
        """y equal to height should return False."""
        assert _coords_in_bounds(10, 5, (10, 10)) is False

    def test_x_at_boundary_is_invalid(self) -> None:
        """x equal to width should return False."""
        assert _coords_in_bounds(5, 10, (10, 10)) is False


# ---- find_root_tip ---------------------------------------------------


@pytest.mark.unit
class TestFindRootTip:
    """Tests for root tip finding on a single plant mask."""

    def test_empty_mask_returns_none(self) -> None:
        """An all-zero mask should return None."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        assert _find_root_tip(mask) is None

    def test_finds_bottommost_point(self) -> None:
        """The tip should be at the lowest y coordinate with root pixels."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 40:60] = 255
        mask[70, 45:55] = 255

        tip = _find_root_tip(mask)
        assert tip is not None
        assert int(tip[1]) == 70

    def test_uses_median_x_for_wide_bottom(self) -> None:
        """When multiple pixels share max y, use median x."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[80, 30:70] = 255

        tip = _find_root_tip(mask)
        assert tip is not None
        assert 45 <= int(tip[0]) <= 55


# ---- detect_landmarks ------------------------------------------------


@pytest.mark.unit
class TestDetectLandmarks:
    """Tests for the full landmark detection pipeline."""

    def test_empty_mask_returns_empty_list(self) -> None:
        """No root pixels should produce no landmarks."""
        prob_map = np.zeros((100, 100), dtype=np.float32)
        binary_mask = np.zeros((100, 100), dtype=np.uint8)

        landmarks = detect_landmarks(
            prob_map,
            binary_mask,
            num_plants=5,
            plant_start=10,
            plant_step=20,
            roi_width=10,
        )

        assert landmarks == []

    def test_returns_landmark_objects(self) -> None:
        """Detected landmarks should be Landmark instances."""
        prob_map = np.zeros((200, 200), dtype=np.float32)
        binary_mask = np.zeros((200, 200), dtype=np.uint8)

        prob_map[50:150, 90:110] = 0.9
        binary_mask[50:150, 90:110] = 255

        landmarks = detect_landmarks(
            prob_map,
            binary_mask,
            num_plants=1,
            plant_start=100,
            plant_step=200,
            roi_width=100,
        )

        for lm in landmarks:
            assert isinstance(lm, Landmark)

    def test_landmark_has_valid_confidence(self) -> None:
        """Each landmark confidence should be between 0 and 1."""
        prob_map = np.zeros((200, 200), dtype=np.float32)
        binary_mask = np.zeros((200, 200), dtype=np.uint8)

        prob_map[50:150, 90:110] = 0.85
        binary_mask[50:150, 90:110] = 255

        landmarks = detect_landmarks(
            prob_map,
            binary_mask,
            num_plants=1,
            plant_start=100,
            plant_step=200,
            roi_width=100,
        )

        for lm in landmarks:
            assert 0.0 <= lm.confidence <= 1.0

    def test_multiple_plants_detected(self) -> None:
        """detect_landmarks should find tips across multiple plant ROIs.

        Creates a 300-tall, 600-wide image with 3 plant ROIs at x=100,
        x=300, x=500 (step=200, roi_width=100). Plants 0 and 2 have root
        pixels, plant 1 is empty. Should return exactly 2 landmarks.
        """
        h, w = 300, 600
        prob_map = np.zeros((h, w), dtype=np.float32)
        binary_mask = np.zeros((h, w), dtype=np.uint8)

        # Plant 0: root strip centred at x=100
        prob_map[100:250, 80:120] = 0.85
        binary_mask[100:250, 80:120] = 255

        # Plant 1: empty (no root pixels near x=300)

        # Plant 2: root strip centred at x=500
        prob_map[50:200, 480:520] = 0.9
        binary_mask[50:200, 480:520] = 255

        landmarks = detect_landmarks(
            prob_map,
            binary_mask,
            num_plants=3,
            plant_start=100,
            plant_step=200,
            roi_width=100,
        )

        assert len(landmarks) == 2
        for lm in landmarks:
            assert isinstance(lm, Landmark)
            assert 0.0 <= lm.confidence <= 1.0

    def test_landmark_ids_are_sequential(self) -> None:
        """Returned landmarks should have zero-based sequential IDs."""
        h, w = 300, 600
        prob_map = np.zeros((h, w), dtype=np.float32)
        binary_mask = np.zeros((h, w), dtype=np.uint8)

        prob_map[100:250, 80:120] = 0.85
        binary_mask[100:250, 80:120] = 255

        prob_map[50:200, 480:520] = 0.9
        binary_mask[50:200, 480:520] = 255

        landmarks = detect_landmarks(
            prob_map,
            binary_mask,
            num_plants=3,
            plant_start=100,
            plant_step=200,
            roi_width=100,
        )

        ids = [lm.id for lm in landmarks]
        assert ids == list(range(len(landmarks)))
