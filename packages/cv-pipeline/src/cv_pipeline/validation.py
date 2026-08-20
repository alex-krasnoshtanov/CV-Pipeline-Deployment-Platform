"""Input validation for the cv-pipeline package.

Implements all input checks from the CV Pipeline Specification section 5.
Images are validated before they reach the model. A ValidationError is
raised for any input that does not meet the specification constraints.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS: set[str] = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
MIN_DIMENSION_PX: int = 256
MAX_DIMENSION_PX: int = 8192


class ValidationError(Exception):
    """Raised when an input image fails a validation check.

    Args:
        error_code: Machine-readable code identifying the error type.
        message: Human-readable explanation with specific values.
    """

    def __init__(self, error_code: str, message: str) -> None:
        self.error_code = error_code
        self.message = message
        super().__init__(message)


def validate_image(path: Path) -> np.ndarray:
    """Validate an input image and return it as a numpy array.

    Runs all specification checks in order: file extension, file size,
    decodability, colour mode, minimum dimensions, maximum dimensions.
    Returns the image as a numpy array (8-bit, grayscale or RGB) ready
    for the segmentation pipeline.

    Args:
        path: Path to the image file on disk.

    Returns:
        The decoded image as a numpy array (H, W) or (H, W, 3).

    Raises:
        ValidationError: If any validation check fails. The error_code
            attribute contains the machine-readable code from the spec.
    """
    path = Path(path)

    _check_extension(path)
    _check_file_size(path)
    image_pil = _decode_image(path)
    image_pil = _check_colour_mode(image_pil)
    image_np = np.array(image_pil)
    _check_min_dimensions(image_np)
    _check_max_dimensions(image_np)

    return image_np


def _check_extension(path: Path) -> None:
    """Reject files with unsupported extensions.

    Args:
        path: Path to the image file.

    Raises:
        ValidationError: If the file extension is not in the allowed set.
    """
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            error_code="UNSUPPORTED_FILE_TYPE",
            message=(
                f"File extension '{suffix}' is not supported. "
                f"Accepted formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            ),
        )


def _check_file_size(path: Path) -> None:
    """Reject files that exceed the maximum size.

    Args:
        path: Path to the image file.

    Raises:
        ValidationError: If the file exceeds 50 MB.
    """
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        raise ValidationError(
            error_code="CORRUPT_FILE",
            message=f"Cannot read file at '{path}': {exc}.",
        ) from exc

    if size_bytes > MAX_FILE_SIZE_BYTES:
        size_mb = size_bytes / (1024 * 1024)
        raise ValidationError(
            error_code="FILE_TOO_LARGE",
            message=(
                f"The uploaded file is {size_mb:.1f} MB, "
                f"which exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB limit."
            ),
        )


def _decode_image(path: Path) -> Image.Image:
    """Attempt to open and decode the image file with Pillow.

    Args:
        path: Path to the image file.

    Returns:
        A Pillow Image object.

    Raises:
        ValidationError: If the file cannot be decoded as an image.
    """
    try:
        with Image.open(path) as img:
            img.load()
            return img.copy()
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(
            error_code="CORRUPT_FILE",
            message=f"Cannot decode image at '{path}': {exc}.",
        ) from exc


def _check_colour_mode(img: Image.Image) -> Image.Image:
    """Validate the colour mode and convert where necessary.

    CMYK images are rejected. RGBA images have their alpha channel
    dropped with a warning. 16-bit images are converted to 8-bit.
    Grayscale and RGB images pass through unchanged.

    Args:
        img: A decoded Pillow Image object.

    Returns:
        The image in a supported mode (L or RGB).

    Raises:
        ValidationError: If the image is CMYK.
    """
    mode = img.mode

    if mode == "CMYK":
        raise ValidationError(
            error_code="UNSUPPORTED_COLOR_MODE",
            message=(
                "CMYK colour mode is not supported. Provide a grayscale or RGB image."
            ),
        )

    if mode == "RGBA":
        logger.warning("RGBA image detected — dropping alpha channel.")
        img = img.convert("RGB")

    elif mode == "I;16":
        logger.info("16-bit image detected — normalising to 8-bit.")
        arr = np.array(img, dtype=np.uint16)
        arr = (arr / 256).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")

    elif mode not in ("L", "RGB"):
        img = img.convert("RGB")

    return img


def _check_min_dimensions(image: np.ndarray) -> None:
    """Reject images below the minimum resolution.

    Args:
        image: The decoded image as a numpy array.
        path: Path to the original file (for the error message).

    Raises:
        ValidationError: If either dimension is below 256 px.
    """
    h, w = image.shape[:2]
    if h < MIN_DIMENSION_PX or w < MIN_DIMENSION_PX:
        raise ValidationError(
            error_code="IMAGE_TOO_SMALL",
            message=(
                f"Image dimensions {w}x{h} px are below the "
                f"minimum {MIN_DIMENSION_PX}x{MIN_DIMENSION_PX} px."
            ),
        )


def _check_max_dimensions(image: np.ndarray) -> None:
    """Reject images above the maximum resolution.

    Args:
        image: The decoded image as a numpy array.
        path: Path to the original file (for the error message).

    Raises:
        ValidationError: If either dimension exceeds 8192 px.
    """
    h, w = image.shape[:2]
    if h > MAX_DIMENSION_PX or w > MAX_DIMENSION_PX:
        raise ValidationError(
            error_code="IMAGE_TOO_LARGE",
            message=(
                f"Image dimensions {w}x{h} px exceed the "
                f"maximum {MAX_DIMENSION_PX}x{MAX_DIMENSION_PX} px."
            ),
        )
