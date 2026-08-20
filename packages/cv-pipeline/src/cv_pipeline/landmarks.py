"""Root tip landmark detection for the cv-pipeline package.

Separates individual plants from a segmentation mask, finds the
bottommost point (root tip) of each plant, and returns pixel-space
coordinates with per-landmark confidence scores.

Adapted from the Block B root_tip_detection module.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from cv_pipeline.schema import Landmark

logger = logging.getLogger(__name__)

# Default layout for HADES Petri dishes with 5 Arabidopsis seedlings.
_DEFAULT_NUM_PLANTS: int = 5
_DEFAULT_PLANT_START: int = 350
_DEFAULT_PLANT_STEP: int = 500
_DEFAULT_ROI_WIDTH: int = 250

# Connected components smaller than this are treated as noise.
_MIN_COMPONENT_AREA: int = 50

# Root tip Y-coordinate is clamped to this fraction of image height
# to avoid detecting noise at the very bottom of the image.
_MAX_Y_PERCENTILE: float = 0.95


def detect_landmarks(
    prob_map: np.ndarray,
    binary_mask: np.ndarray,
    num_plants: int = _DEFAULT_NUM_PLANTS,
    plant_start: int = _DEFAULT_PLANT_START,
    plant_step: int = _DEFAULT_PLANT_STEP,
    roi_width: int = _DEFAULT_ROI_WIDTH,
) -> list[Landmark]:
    """Detect root tip landmarks from segmentation output.

    Separates the binary mask into individual plant regions using
    expected horizontal positions, finds the bottommost point of each
    plant, and reads the per-landmark confidence from the probability
    map at that coordinate.

    Args:
        prob_map: Segmentation probability map (H, W) as float32 in
            [0, 1]. Used to derive per-landmark confidence.
        binary_mask: Binary segmentation mask (H, W) as uint8 with
            0 for background and 255 for root.
        num_plants: Number of plants expected in the dish.
        plant_start: X pixel position of the first plant centre.
        plant_step: Horizontal distance between plant centres in pixels.
        roi_width: Half-width of the search region around each plant
            centre in pixels.

    Returns:
        A list of ``Landmark`` objects, one per detected root tip.
        The list may be empty if no tips are found.
    """
    plant_masks = _separate_plants(
        binary_mask,
        num_plants=num_plants,
        plant_start=plant_start,
        plant_step=plant_step,
        roi_width=roi_width,
    )

    landmarks: list[Landmark] = []
    landmark_id = 0

    for idx, plant_mask in enumerate(plant_masks):
        tip = _find_root_tip(plant_mask)
        if tip is None:
            logger.debug("Plant %d: no root tip found.", idx + 1)
            continue

        x, y = int(round(tip[0])), int(round(tip[1]))
        confidence = (
            float(prob_map[y, x]) if _coords_in_bounds(y, x, prob_map.shape) else 0.0
        )

        landmarks.append(Landmark(id=landmark_id, x=x, y=y, confidence=confidence))
        logger.debug(
            "Plant %d: tip at (%d, %d), confidence %.3f.",
            idx + 1,
            x,
            y,
            confidence,
        )
        landmark_id += 1

    logger.info("Detected %d root tip landmark(s).", len(landmarks))
    return landmarks


# ---- plant separation ------------------------------------------------


def _separate_plants(
    binary_mask: np.ndarray,
    num_plants: int,
    plant_start: int,
    plant_step: int,
    roi_width: int,
) -> list[np.ndarray]:
    """Separate the full root mask into individual plant masks.

    Each connected component in the mask is assigned to the nearest
    expected plant position based on horizontal overlap with the
    plant's region of interest.

    Args:
        binary_mask: Binary mask (0 or 255).
        num_plants: Number of expected plants.
        plant_start: X position of first plant centre.
        plant_step: Distance between plant centres.
        roi_width: Half-width of each plant's search region.

    Returns:
        A list of binary masks, one per plant that has root pixels.
        Plants with no pixels are omitted.
    """
    cleaned = _morphological_cleanup(binary_mask)
    h, w = cleaned.shape

    expected_positions = [plant_start + i * plant_step for i in range(num_plants)]
    logger.debug("Expected plant X positions: %s.", expected_positions)

    num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(
        cleaned,
        connectivity=8,
    )
    logger.debug("Found %d connected component(s).", num_labels - 1)

    plant_masks = [np.zeros((h, w), dtype=np.uint8) for _ in range(num_plants)]

    for comp_id in range(1, num_labels):
        area = stats[comp_id, cv2.CC_STAT_AREA]
        if area < _MIN_COMPONENT_AREA:
            continue

        comp_mask = labels_map == comp_id
        x_coords = np.where(comp_mask.any(axis=0))[0]
        if len(x_coords) == 0:
            continue

        comp_x_min = int(x_coords.min())
        comp_x_max = int(x_coords.max())

        best_plant, best_overlap = _best_plant_match(
            comp_x_min,
            comp_x_max,
            expected_positions,
            roi_width,
            w,
        )

        if best_plant is not None and best_overlap > 0:
            plant_masks[best_plant][comp_mask] = 255

    # Return only non-empty masks.
    result = [m for m in plant_masks if m.any()]
    logger.info("Separated mask into %d plant region(s).", len(result))
    return result


def _best_plant_match(
    comp_x_min: int,
    comp_x_max: int,
    expected_positions: list[int],
    roi_width: int,
    image_width: int,
) -> tuple[int | None, int]:
    """Find the plant whose ROI overlaps most with a connected component.

    Args:
        comp_x_min: Leftmost x of the component.
        comp_x_max: Rightmost x of the component.
        expected_positions: List of expected plant centre x-coordinates.
        roi_width: Half-width of each plant ROI.
        image_width: Image width (for clamping).

    Returns:
        A tuple (plant_index, overlap_pixels). ``plant_index`` is
        ``None`` if no overlap is found.
    """
    best_plant: int | None = None
    best_overlap = 0

    for plant_idx, x_centre in enumerate(expected_positions):
        roi_min = max(0, x_centre - roi_width)
        roi_max = min(image_width, x_centre + roi_width)
        overlap = max(0, min(comp_x_max, roi_max) - max(comp_x_min, roi_min))

        if overlap > best_overlap:
            best_overlap = overlap
            best_plant = plant_idx

    return best_plant, best_overlap


# ---- root tip finding ------------------------------------------------


def _find_root_tip(plant_mask: np.ndarray) -> np.ndarray | None:
    """Find the bottommost point (root tip) of a single plant mask.

    If multiple pixels share the maximum Y coordinate, the median X
    is used. A boundary constraint prevents detecting noise at the
    very bottom of the image.

    Args:
        plant_mask: Binary mask (0 or 255) of a single plant.

    Returns:
        A numpy array ``[x, y]`` in pixel coordinates, or ``None`` if
        the mask is empty.
    """
    y_coords, x_coords = np.where(plant_mask > 0)

    if len(y_coords) == 0:
        return None

    h = plant_mask.shape[0]
    max_allowed_y = int(h * _MAX_Y_PERCENTILE)

    valid = y_coords <= max_allowed_y
    if valid.any():
        y_coords = y_coords[valid]
        x_coords = x_coords[valid]

    max_y = int(y_coords.max())
    bottommost_x = x_coords[y_coords == max_y]
    tip_x = float(np.median(bottommost_x))

    return np.array([tip_x, float(max_y)])


# ---- helpers ---------------------------------------------------------


def _morphological_cleanup(mask: np.ndarray) -> np.ndarray:
    """Remove noise from a binary mask with morphological operations.

    Args:
        mask: Binary mask (0 or 255).

    Returns:
        Cleaned binary mask.
    """
    binary = (mask > 128).astype(np.uint8) * 255
    k_small = np.ones((3, 3), dtype=np.uint8)
    k_large = np.ones((5, 5), dtype=np.uint8)

    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_small)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_large)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_small)
    return cleaned


def _coords_in_bounds(y: int, x: int, shape: tuple[int, ...]) -> bool:
    """Check whether (y, x) is within the array bounds.

    Args:
        y: Row index.
        x: Column index.
        shape: Array shape (H, W, ...).

    Returns:
        ``True`` if the coordinates are valid.
    """
    return 0 <= y < shape[0] and 0 <= x < shape[1]
