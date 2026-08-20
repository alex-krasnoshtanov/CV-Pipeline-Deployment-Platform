"""Unit tests for the corrected-mask validation module."""

from __future__ import annotations

import base64
import io

import pytest
from api.services.mask_validation import (
    MaskValidationError,
    validate_corrected_mask,
)
from PIL import Image

_W = 32
_H = 24


def _png_b64(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode(mask_b64: str) -> Image.Image:
    """Decode a base64 PNG string back into a PIL image."""
    return Image.open(io.BytesIO(base64.b64decode(mask_b64)))


@pytest.mark.unit
class TestValidMasks:
    """Masks that should pass and be normalised."""

    def test_binary_mask_is_normalised(self) -> None:
        """A correct-size binary mask returns single-channel 0/255."""
        image = Image.new("L", (_W, _H), 0)
        for x in range(10):
            for y in range(10):
                image.putpixel((x, y), 255)

        out = validate_corrected_mask(_png_b64(image), _W, _H)
        result = _decode(out)

        assert result.size == (_W, _H)
        assert result.mode == "L"
        assert set(result.getdata()) <= {0, 255}

    def test_rgb_input_flattened_to_single_channel(self) -> None:
        """An RGB mask is collapsed to a single-channel mask."""
        image = Image.new("RGB", (_W, _H), (255, 255, 255))

        out = validate_corrected_mask(_png_b64(image), _W, _H)

        assert _decode(out).mode == "L"

    def test_non_binary_grayscale_is_coerced(self) -> None:
        """Any non-zero grayscale value becomes root (255)."""
        image = Image.new("L", (_W, _H), 128)

        out = validate_corrected_mask(_png_b64(image), _W, _H)

        assert set(_decode(out).getdata()) == {255}

    def test_all_background_mask_is_valid(self) -> None:
        """An all-zero mask is valid (prediction had no roots)."""
        image = Image.new("L", (_W, _H), 0)

        out = validate_corrected_mask(_png_b64(image), _W, _H)

        assert set(_decode(out).getdata()) == {0}


@pytest.mark.unit
class TestInvalidMasks:
    """Masks that should be rejected with a specific error code."""

    def test_wrong_dimensions_rejected(self) -> None:
        """A mask whose size differs from the prediction is rejected."""
        image = Image.new("L", (_W + 1, _H), 0)

        with pytest.raises(MaskValidationError) as exc_info:
            validate_corrected_mask(_png_b64(image), _W, _H)

        assert exc_info.value.error_code == "MASK_DIMENSION_MISMATCH"

    def test_invalid_base64_rejected(self) -> None:
        """A non-base64 string is rejected before image decoding."""
        with pytest.raises(MaskValidationError) as exc_info:
            validate_corrected_mask("!!! not base64 !!!", _W, _H)

        assert exc_info.value.error_code == "MASK_NOT_BASE64"

    def test_valid_base64_non_image_rejected(self) -> None:
        """Valid base64 that is not an image is rejected."""
        not_an_image = base64.b64encode(b"this is not a png").decode("ascii")

        with pytest.raises(MaskValidationError) as exc_info:
            validate_corrected_mask(not_an_image, _W, _H)

        assert exc_info.value.error_code == "MASK_CORRUPT"
