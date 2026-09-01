"""Top-level inference entry point for the cv-pipeline package.

Wires together validation, preprocessing, segmentation, and landmark
detection into a single ``infer`` call that implements the contract
defined in the CV Pipeline Specification.
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from cv_pipeline._version import __version__
from cv_pipeline.landmarks import detect_landmarks
from cv_pipeline.preprocessing import detect_petri_dish
from cv_pipeline.schema import (
    InferenceResult,
    Metadata,
)
from cv_pipeline.segmentation import SegmentationModel
from cv_pipeline.validation import validate_image

logger = logging.getLogger(__name__)


def infer(
    image_path: str | Path,
    model: SegmentationModel,
    metadata: Metadata | None = None,
    threshold: float = 0.5,
    crop: bool = True,
    num_plants: int = 5,
    plant_start: int = 350,
    plant_step: int = 500,
    roi_width: int = 250,
) -> InferenceResult:
    """Run the full inference pipeline on a single image.

    This is the main public function of the cv-pipeline package. It
    validates the input, optionally crops to the Petri dish, runs
    segmentation, detects root tip landmarks, and returns a complete
    ``InferenceResult`` matching the specification schema.

    Args:
        image_path: Path to the input image file.
        model: A loaded ``SegmentationModel`` instance. The caller is
            responsible for creating this once and reusing it across
            calls.
        metadata: Optional metadata to pass through to the response.
            If ``None``, an empty ``Metadata`` object is used.
        threshold: Probability threshold for binarising the
            segmentation mask.
        crop: Whether to detect and crop to the Petri dish before
            running segmentation.
        num_plants: Expected number of plants for landmark detection.
        plant_start: X pixel position of the first plant centre.
        plant_step: Horizontal distance between plant centres.
        roi_width: Half-width of the landmark search region per plant.

    Returns:
        An ``InferenceResult`` containing the segmentation mask,
        landmark coordinates, and confidence scores. When cropping
        is enabled, the mask and coordinates are mapped back to the
        original image coordinate space.

    Raises:
        ValidationError: If the input image fails any validation check.
    """
    image_path = Path(image_path)
    logger.info("Starting inference on '%s'.", image_path.name)

    if metadata is None:
        metadata = Metadata()

    # --- 1. Validate input -------------------------------------------
    image = validate_image(image_path)

    # --- 2. Preprocess (optional crop) --------------------------------
    original_h, original_w = image.shape[:2]
    crop_x1, crop_y1 = 0, 0

    if crop:
        crop_x1, crop_y1, crop_x2, crop_y2 = detect_petri_dish(image)
        image = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        logger.info(
            "Cropped to dish: %dx%d -> %dx%d (offset x=%d, y=%d).",
            original_w,
            original_h,
            image.shape[1],
            image.shape[0],
            crop_x1,
            crop_y1,
        )

    # --- 3. Segment ---------------------------------------------------
    prob_map = model.predict(image)
    binary_mask, mask_confidence = _binarise(prob_map, threshold)

    # --- 4. Detect landmarks ------------------------------------------
    landmarks = detect_landmarks(
        prob_map,
        binary_mask,
        num_plants=num_plants,
        plant_start=plant_start,
        plant_step=plant_step,
        roi_width=roi_width,
    )

    # --- 5. Map back to original coordinate space ---------------------
    if crop:
        full_mask = np.zeros((original_h, original_w), dtype=np.uint8)
        full_mask[
            crop_y1 : crop_y1 + binary_mask.shape[0],
            crop_x1 : crop_x1 + binary_mask.shape[1],
        ] = binary_mask
        binary_mask = full_mask

        for lm in landmarks:
            lm.x += crop_x1
            lm.y += crop_y1

    # --- 6. Encode mask -----------------------------------------------
    mask_b64 = _encode_mask(binary_mask)

    # --- 7. Build result ----------------------------------------------
    result = InferenceResult(
        pipeline_version=__version__,
        model_version=model.model_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        image_filename=image_path.name,
        image_width_px=original_w,
        image_height_px=original_h,
        metadata=metadata,
        mask_b64=mask_b64,
        mask_confidence=round(mask_confidence, 4),
        landmark_count=len(landmarks),
        landmarks=landmarks,
    )

    logger.info(
        "Inference complete — %d landmark(s), mask confidence %.3f.",
        result.landmark_count,
        result.mask_confidence,
    )
    return result


# ---- helpers ---------------------------------------------------------


def _binarise(
    prob_map: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, float]:
    """Binarise a probability map and compute mask confidence.

    Args:
        prob_map: (H, W) float32 in [0, 1].
        threshold: Probability threshold.

    Returns:
        A tuple of (binary_mask, mask_confidence). The mask is uint8
        with 0 / 255 values. Confidence is the mean probability of
        root-classified pixels, or 0.0 if none are classified as root.
    """
    binary = (prob_map >= threshold).astype(np.uint8) * 255

    root_pixels = prob_map[binary == 255]
    if len(root_pixels) == 0:
        return binary, 0.0

    return binary, float(np.mean(root_pixels))


def _encode_mask(mask: np.ndarray) -> str:
    """Encode a binary mask as a base64 PNG string.

    Args:
        mask: (H, W) uint8 array with 0 and 255 values.

    Returns:
        Base64-encoded PNG string.
    """
    img = Image.fromarray(mask, mode="L")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
