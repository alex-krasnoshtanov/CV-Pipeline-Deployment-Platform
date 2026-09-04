"""Image preprocessing for the cv-pipeline package.

Handles petri dish detection and image cropping before the image
reaches the segmentation model. The core logic is adapted from
the team's earlier petri_detection prototype.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Morphological kernel size for noise removal during dish detection.
_MORPH_KERNEL_SIZE: int = 20

# If the detected dish area is below this fraction of the total image
# area, the detection is treated as unreliable and the fallback is used.
_MIN_AREA_FRACTION: float = 0.2

# Fallback margin: the percentage of the shortest side to trim when
# contour detection fails or produces an unreliable result.
_FALLBACK_MARGIN_FRACTION: float = 0.1


def detect_petri_dish(
    image: np.ndarray,
    shrink: int = 20,
) -> tuple[int, int, int, int]:
    """Detect the petri dish boundary and return a square bounding box.

    Uses Otsu thresholding and morphological operations to find the
    largest bright region in the image. Falls back to the centre 80%
    of the image if detection fails or the detected region is too small.

    Args:
        image: Input image as a numpy array (H, W) or (H, W, 3).
        shrink: Pixels to shrink inward from the detected boundary to
            remove edge artefacts.

    Returns:
        A tuple (x1, y1, x2, y2) defining the crop bounding box in
        pixel coordinates.
    """
    logger.info(
        "Detecting petri dish in image of shape %s.",
        image.shape,
    )

    gray = _to_grayscale(image)
    binary, threshold_value = _otsu_threshold(gray)
    binary = _morphological_cleanup(binary)
    x1, y1, x2, y2 = _find_bounding_box(binary, image.shape, shrink)

    logger.info(
        "Final bounding box: (%d, %d) to (%d, %d) — %dx%d px.",
        x1,
        y1,
        x2,
        y2,
        x2 - x1,
        y2 - y1,
    )
    return x1, y1, x2, y2


def crop_to_dish(
    image: np.ndarray,
    shrink: int = 20,
) -> np.ndarray:
    """Detect the petri dish and return the cropped region.

    Convenience wrapper that calls ``detect_petri_dish`` and applies
    the bounding box to the image in one step.

    Args:
        image: Input image as a numpy array (H, W) or (H, W, 3).
        shrink: Pixels to shrink inward from the detected boundary.

    Returns:
        The cropped image region as a numpy array.
    """
    x1, y1, x2, y2 = detect_petri_dish(image, shrink=shrink)
    return image[y1:y2, x1:x2].copy()


# ---- internal helpers ------------------------------------------------


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale if it has multiple channels.

    Args:
        image: Input image as a numpy array.

    Returns:
        Single-channel grayscale image.
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.copy()


def _otsu_threshold(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Apply Otsu thresholding to a grayscale image.

    Args:
        gray: Single-channel grayscale image.

    Returns:
        A tuple of (binary_mask, threshold_value).
    """
    threshold_value, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    logger.debug("Otsu threshold value: %.1f.", threshold_value)
    return binary, float(threshold_value)


def _morphological_cleanup(binary: np.ndarray) -> np.ndarray:
    """Apply morphological close and open to remove noise.

    Args:
        binary: Binary mask from thresholding.

    Returns:
        Cleaned binary mask.
    """
    kernel = np.ones(
        (_MORPH_KERNEL_SIZE, _MORPH_KERNEL_SIZE),
        dtype=np.uint8,
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


def _find_bounding_box(
    binary: np.ndarray,
    image_shape: tuple[int, ...],
    shrink: int,
) -> tuple[int, int, int, int]:
    """Find the dish bounding box from a binary mask.

    Returns a square bounding box centred on the largest contour.
    Falls back to the centre region of the image if no reliable
    contour is found.

    Args:
        binary: Cleaned binary mask.
        image_shape: Shape of the original image (H, W) or (H, W, C).
        shrink: Pixels to shrink inward from the detected boundary.

    Returns:
        A tuple (x1, y1, x2, y2).
    """
    img_h, img_w = image_shape[:2]
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        logger.warning("No contours found — falling back to centre region.")
        return _fallback_box(img_h, img_w)

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    detected_area = w * h
    img_area = img_h * img_w

    if detected_area < _MIN_AREA_FRACTION * img_area:
        logger.warning(
            "Detected area is %.1f%% of image — too small, using fallback.",
            100 * detected_area / img_area,
        )
        return _fallback_box(img_h, img_w)

    # Shrink boundary to remove edge artefacts.
    x += shrink
    y += shrink
    w -= 2 * shrink
    h -= 2 * shrink

    # Make the bounding box square, centred on the detected region.
    size = max(w, h)
    center_x = x + w // 2
    center_y = y + h // 2

    x1 = max(0, center_x - size // 2)
    y1 = max(0, center_y - size // 2)
    x2 = min(img_w, x1 + size)
    y2 = min(img_h, y1 + size)

    return x1, y1, x2, y2


def _fallback_box(
    img_h: int,
    img_w: int,
) -> tuple[int, int, int, int]:
    """Return a centred bounding box covering 80% of the image.

    Args:
        img_h: Image height in pixels.
        img_w: Image width in pixels.

    Returns:
        A tuple (x1, y1, x2, y2).
    """
    margin = int(min(img_h, img_w) * _FALLBACK_MARGIN_FRACTION)
    return margin, margin, img_w - margin, img_h - margin
