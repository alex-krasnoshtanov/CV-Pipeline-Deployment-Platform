"""Unit tests for cv_pipeline.validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from cv_pipeline.validation import (
    MAX_FILE_SIZE_BYTES,
    MIN_DIMENSION_PX,
    ValidationError,
    validate_image,
)
from PIL import Image

# ---- extension checks ------------------------------------------------


@pytest.mark.unit
class TestExtensionValidation:
    """Tests for file extension validation."""

    def test_accepts_png(self, tmp_path: Path) -> None:
        """PNG files should be accepted."""
        img = Image.fromarray(np.zeros((300, 300), dtype=np.uint8))
        path = tmp_path / "test.png"
        img.save(path)

        result = validate_image(path)
        assert isinstance(result, np.ndarray)

    def test_accepts_jpg(self, tmp_path: Path) -> None:
        """JPG files should be accepted."""
        img = Image.fromarray(np.zeros((300, 300), dtype=np.uint8))
        path = tmp_path / "test.jpg"
        img.save(path)

        result = validate_image(path)
        assert isinstance(result, np.ndarray)

    def test_accepts_tiff(self, tmp_path: Path) -> None:
        """TIFF files should be accepted."""
        img = Image.fromarray(np.zeros((300, 300), dtype=np.uint8))
        path = tmp_path / "test.tiff"
        img.save(path)

        result = validate_image(path)
        assert isinstance(result, np.ndarray)

    def test_rejects_unsupported_extension(self, tmp_path: Path) -> None:
        """Files with unsupported extensions should raise UNSUPPORTED_FILE_TYPE."""
        path = tmp_path / "test.bmp"
        path.write_bytes(b"fake data")

        with pytest.raises(ValidationError) as exc_info:
            validate_image(path)

        assert exc_info.value.error_code == "UNSUPPORTED_FILE_TYPE"

    def test_rejects_gif(self, tmp_path: Path) -> None:
        """GIF files should be rejected."""
        path = tmp_path / "test.gif"
        path.write_bytes(b"fake data")

        with pytest.raises(ValidationError) as exc_info:
            validate_image(path)

        assert exc_info.value.error_code == "UNSUPPORTED_FILE_TYPE"


# ---- file size checks ------------------------------------------------


@pytest.mark.unit
class TestFileSizeValidation:
    """Tests for file size validation."""

    def test_rejects_file_over_50mb(self, tmp_path: Path) -> None:
        """Files exceeding 50 MB should raise FILE_TOO_LARGE."""
        path = tmp_path / "huge.png"
        path.write_bytes(b"\x00" * (MAX_FILE_SIZE_BYTES + 1))

        with pytest.raises(ValidationError) as exc_info:
            validate_image(path)

        assert exc_info.value.error_code == "FILE_TOO_LARGE"

    def test_rejects_nonexistent_file(self, tmp_path: Path) -> None:
        """Missing files should raise CORRUPT_FILE."""
        path = tmp_path / "missing.png"

        with pytest.raises(ValidationError) as exc_info:
            validate_image(path)

        assert exc_info.value.error_code == "CORRUPT_FILE"


# ---- dimension checks ------------------------------------------------


@pytest.mark.unit
class TestDimensionValidation:
    """Tests for image dimension validation."""

    def test_rejects_image_too_small(self, tmp_path: Path) -> None:
        """Images below 256x256 should raise IMAGE_TOO_SMALL."""
        img = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
        path = tmp_path / "tiny.png"
        img.save(path)

        with pytest.raises(ValidationError) as exc_info:
            validate_image(path)

        assert exc_info.value.error_code == "IMAGE_TOO_SMALL"

    def test_rejects_image_too_small_one_dimension(
        self,
        tmp_path: Path,
    ) -> None:
        """Images with one dimension below 256 should be rejected."""
        img = Image.fromarray(np.zeros((100, 500), dtype=np.uint8))
        path = tmp_path / "narrow.png"
        img.save(path)

        with pytest.raises(ValidationError) as exc_info:
            validate_image(path)

        assert exc_info.value.error_code == "IMAGE_TOO_SMALL"

    def test_accepts_minimum_dimensions(self, tmp_path: Path) -> None:
        """Images exactly 256x256 should be accepted."""
        img = Image.fromarray(
            np.zeros((MIN_DIMENSION_PX, MIN_DIMENSION_PX), dtype=np.uint8),
        )
        path = tmp_path / "minimum.png"
        img.save(path)

        result = validate_image(path)
        assert result.shape[0] == MIN_DIMENSION_PX
        assert result.shape[1] == MIN_DIMENSION_PX

    def test_accepts_large_image_within_limit(self, tmp_path: Path) -> None:
        """Images up to 8192x8192 should be accepted."""
        img = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
        path = tmp_path / "large.png"
        img.save(path)

        result = validate_image(path)
        assert isinstance(result, np.ndarray)

    def test_rejects_image_too_large(self, tmp_path: Path) -> None:
        """Images exceeding 8192 on one dimension should raise IMAGE_TOO_LARGE.

        Uses Image.new instead of numpy to avoid allocating a huge array
        in memory. A 8193x256 image is just over the limit on width.
        """
        img = Image.new("L", (8193, 256))
        path = tmp_path / "oversized.tiff"
        img.save(path)

        with pytest.raises(ValidationError) as exc_info:
            validate_image(path)

        assert exc_info.value.error_code == "IMAGE_TOO_LARGE"


# ---- colour mode checks ---------------------------------------------


@pytest.mark.unit
class TestColourModeValidation:
    """Tests for colour mode validation."""

    def test_accepts_grayscale(self, tmp_path: Path) -> None:
        """Grayscale images should pass through unchanged."""
        img = Image.fromarray(np.zeros((300, 300), dtype=np.uint8), mode="L")
        path = tmp_path / "gray.png"
        img.save(path)

        result = validate_image(path)
        assert len(result.shape) == 2

    def test_accepts_rgb(self, tmp_path: Path) -> None:
        """RGB images should pass through unchanged."""
        img = Image.fromarray(
            np.zeros((300, 300, 3), dtype=np.uint8),
            mode="RGB",
        )
        path = tmp_path / "rgb.png"
        img.save(path)

        result = validate_image(path)
        assert result.shape[2] == 3

    def test_rgba_drops_alpha(self, tmp_path: Path) -> None:
        """RGBA images should have alpha dropped and convert to RGB."""
        img = Image.fromarray(
            np.zeros((300, 300, 4), dtype=np.uint8),
            mode="RGBA",
        )
        path = tmp_path / "rgba.png"
        img.save(path)

        result = validate_image(path)
        assert result.shape[2] == 3

    def test_rejects_cmyk(self, tmp_path: Path) -> None:
        """CMYK images should raise UNSUPPORTED_COLOR_MODE.

        CMYK is a print colour space that the pipeline does not support.
        Saved as TIFF because PNG does not support CMYK mode.
        """
        img = Image.new("CMYK", (300, 300))
        path = tmp_path / "cmyk.tiff"
        img.save(path)

        with pytest.raises(ValidationError) as exc_info:
            validate_image(path)

        assert exc_info.value.error_code == "UNSUPPORTED_COLOR_MODE"

    def test_corrupt_file_raises(self, tmp_path: Path) -> None:
        """Files that cannot be decoded should raise CORRUPT_FILE."""
        path = tmp_path / "corrupt.png"
        path.write_bytes(b"this is not an image")

        with pytest.raises(ValidationError) as exc_info:
            validate_image(path)

        assert exc_info.value.error_code == "CORRUPT_FILE"
