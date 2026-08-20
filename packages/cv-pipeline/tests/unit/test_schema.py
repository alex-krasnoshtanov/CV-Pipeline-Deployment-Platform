"""Unit tests for cv_pipeline.schema."""

from __future__ import annotations

import pytest
from cv_pipeline.schema import (
    ErrorResponse,
    InferenceResult,
    Landmark,
    Metadata,
)

# ---- Landmark --------------------------------------------------------


@pytest.mark.unit
class TestLandmark:
    """Tests for the Landmark dataclass."""

    def test_to_dict_contains_all_fields(self) -> None:
        """to_dict should return all four fields."""
        lm = Landmark(id=0, x=100, y=200, confidence=0.95)
        d = lm.to_dict()

        assert d == {
            "id": 0,
            "x": 100,
            "y": 200,
            "confidence": 0.95,
        }

    def test_multiple_landmarks_have_unique_ids(self) -> None:
        """Each landmark should have a distinct id."""
        landmarks = [
            Landmark(id=i, x=i * 10, y=i * 20, confidence=0.8) for i in range(5)
        ]
        ids = [lm.id for lm in landmarks]
        assert len(set(ids)) == 5

    def test_from_dict_round_trip(self) -> None:
        """from_dict(to_dict(x)) must equal x — they are true inverses."""
        lm = Landmark(id=3, x=150, y=275, confidence=0.92)
        assert Landmark.from_dict(lm.to_dict()) == lm

    def test_from_dict_sets_all_fields(self) -> None:
        """from_dict must populate every field from the dict."""
        lm = Landmark.from_dict({"id": 7, "x": 42, "y": 88, "confidence": 0.5})
        assert lm.id == 7
        assert lm.x == 42
        assert lm.y == 88
        assert lm.confidence == 0.5

    def test_from_dict_missing_key_raises(self) -> None:
        """from_dict uses data[key] for all fields — a missing key must raise KeyError.

        NOTE for Alex: this asserts current behaviour. If the endpoint can send
        partial Landmark dicts, consider switching to .get() with a sentinel.
        """
        with pytest.raises(KeyError):
            Landmark.from_dict({"x": 1, "y": 2, "confidence": 0.5})  # no "id"


# ---- Metadata --------------------------------------------------------


@pytest.mark.unit
class TestMetadata:
    """Tests for the Metadata dataclass."""

    def test_defaults_are_none(self) -> None:
        """All fields should default to None."""
        m = Metadata()
        d = m.to_dict()

        assert d["plate_id"] is None
        assert d["experiment_id"] is None
        assert d["timestamp"] is None

    def test_to_dict_with_values(self) -> None:
        """Provided values should appear in the dict."""
        m = Metadata(
            plate_id="PL-001",
            experiment_id="EXP-42",
            timestamp="2026-04-17T12:00:00Z",
        )
        d = m.to_dict()

        assert d["plate_id"] == "PL-001"
        assert d["experiment_id"] == "EXP-42"
        assert d["timestamp"] == "2026-04-17T12:00:00Z"

    def test_from_dict_round_trip(self) -> None:
        """from_dict(to_dict(x)) must equal x — they are true inverses."""
        meta = Metadata(
            plate_id="PL-001",
            experiment_id="EXP-42",
            timestamp="2026-04-17T12:00:00Z",
        )
        assert Metadata.from_dict(meta.to_dict()) == meta

    def test_from_dict_none_input(self) -> None:
        """None is valid input — all fields must default to None."""
        m = Metadata.from_dict(None)
        assert m.plate_id is None
        assert m.experiment_id is None
        assert m.timestamp is None

    def test_from_dict_partial_dict(self) -> None:
        """Only present keys are set; absent keys default to None via .get()."""
        m = Metadata.from_dict({"plate_id": "PL-42"})
        assert m.plate_id == "PL-42"
        assert m.experiment_id is None
        assert m.timestamp is None

    def test_from_dict_full_dict(self) -> None:
        """All three fields are populated when the dict contains all keys."""
        m = Metadata.from_dict(
            {
                "plate_id": "PL-007",
                "experiment_id": "EXP-99",
                "timestamp": "2026-06-01T08:00:00Z",
            }
        )
        assert m.plate_id == "PL-007"
        assert m.experiment_id == "EXP-99"
        assert m.timestamp == "2026-06-01T08:00:00Z"


# ---- InferenceResult -------------------------------------------------


@pytest.mark.unit
class TestInferenceResult:
    """Tests for the InferenceResult dataclass."""

    def test_to_dict_structure_matches_spec(self) -> None:
        """to_dict should contain all fields from the spec schema."""
        result = InferenceResult(
            pipeline_version="0.1.0",
            model_version="unet-v0",
            timestamp="2026-04-17T12:00:00Z",
            image_filename="plate_001.tif",
            image_width_px=2048,
            image_height_px=2048,
            metadata=Metadata(),
            mask_b64="dGVzdA==",
            mask_confidence=0.87,
            landmark_count=2,
            landmarks=[
                Landmark(id=0, x=100, y=200, confidence=0.9),
                Landmark(id=1, x=300, y=400, confidence=0.8),
            ],
        )
        d = result.to_dict()

        assert d["pipeline_version"] == "0.1.0"
        assert d["model_version"] == "unet-v0"
        assert d["image_filename"] == "plate_001.tif"
        assert d["image_width_px"] == 2048
        assert d["landmark_count"] == 2
        assert len(d["landmarks"]) == 2
        assert d["landmarks"][0]["x"] == 100

    def test_empty_landmarks_is_valid(self) -> None:
        """Zero landmarks is a valid result per the spec."""
        result = InferenceResult(
            pipeline_version="0.1.0",
            model_version="unet-v0",
            timestamp="2026-04-17T12:00:00Z",
            image_filename="empty.tif",
            image_width_px=512,
            image_height_px=512,
            metadata=Metadata(),
            mask_b64="",
            mask_confidence=0.0,
            landmark_count=0,
            landmarks=[],
        )
        d = result.to_dict()

        assert d["landmark_count"] == 0
        assert d["landmarks"] == []
        assert d["mask_confidence"] == 0.0

    def test_metadata_nested_in_dict(self) -> None:
        """Metadata should be serialized as a nested dict."""
        result = InferenceResult(
            pipeline_version="0.1.0",
            model_version="unet-v0",
            timestamp="2026-04-17T12:00:00Z",
            image_filename="plate.tif",
            image_width_px=1024,
            image_height_px=1024,
            metadata=Metadata(plate_id="PL-001"),
            mask_b64="",
            mask_confidence=0.5,
            landmark_count=0,
        )
        d = result.to_dict()

        assert isinstance(d["metadata"], dict)
        assert d["metadata"]["plate_id"] == "PL-001"

    def test_from_dict_round_trip(self) -> None:
        """from_dict(to_dict(x)) must equal x — they are true inverses."""
        result = InferenceResult(
            pipeline_version="1.2.3",
            model_version="unet-v2",
            timestamp="2026-06-01T10:00:00Z",
            image_filename="plate_007.tif",
            image_width_px=2048,
            image_height_px=2048,
            metadata=Metadata(plate_id="PL-007", experiment_id="EXP-1"),
            mask_b64="dGVzdA==",
            mask_confidence=0.87,
            landmark_count=2,
            landmarks=[
                Landmark(id=0, x=100, y=200, confidence=0.9),
                Landmark(id=1, x=300, y=400, confidence=0.8),
            ],
        )
        assert InferenceResult.from_dict(result.to_dict()) == result

    def test_from_dict_metadata_is_metadata_instance(self) -> None:
        """Nested metadata dict must be reconstructed as a Metadata object."""
        result = InferenceResult(
            pipeline_version="0.1.0",
            model_version="unet-v0",
            timestamp="2026-04-17T12:00:00Z",
            image_filename="plate.tif",
            image_width_px=512,
            image_height_px=512,
            metadata=Metadata(plate_id="PL-001"),
            mask_b64="",
            mask_confidence=0.5,
            landmark_count=0,
        )
        reconstructed = InferenceResult.from_dict(result.to_dict())
        assert isinstance(reconstructed.metadata, Metadata)
        assert reconstructed.metadata.plate_id == "PL-001"

    def test_from_dict_landmarks_are_landmark_instances(self) -> None:
        """Each item in the landmarks list must be a Landmark object."""
        result = InferenceResult(
            pipeline_version="0.1.0",
            model_version="unet-v0",
            timestamp="2026-04-17T12:00:00Z",
            image_filename="plate.tif",
            image_width_px=512,
            image_height_px=512,
            metadata=Metadata(),
            mask_b64="",
            mask_confidence=0.7,
            landmark_count=3,
            landmarks=[
                Landmark(id=0, x=10, y=20, confidence=0.9),
                Landmark(id=1, x=30, y=40, confidence=0.85),
                Landmark(id=2, x=50, y=60, confidence=0.75),
            ],
        )
        reconstructed = InferenceResult.from_dict(result.to_dict())
        assert len(reconstructed.landmarks) == 3
        assert all(isinstance(lm, Landmark) for lm in reconstructed.landmarks)

    def test_from_dict_absent_landmarks_key(self) -> None:
        """A dict without a 'landmarks' key must produce an empty list, not an error."""
        d = {
            "pipeline_version": "0.1.0",
            "model_version": "unet-v0",
            "timestamp": "2026-04-17T12:00:00Z",
            "image_filename": "plate.tif",
            "image_width_px": 512,
            "image_height_px": 512,
            "metadata": None,
            "mask_b64": "",
            "mask_confidence": 0.0,
            "landmark_count": 0,
            # "landmarks" key intentionally absent
        }
        result = InferenceResult.from_dict(d)
        assert result.landmarks == []

    def test_from_dict_explicit_empty_landmarks(self) -> None:
        """An explicit empty landmarks list must round-trip to an empty list."""
        result = InferenceResult(
            pipeline_version="0.1.0",
            model_version="unet-v0",
            timestamp="2026-04-17T12:00:00Z",
            image_filename="empty.tif",
            image_width_px=512,
            image_height_px=512,
            metadata=Metadata(),
            mask_b64="",
            mask_confidence=0.0,
            landmark_count=0,
            landmarks=[],
        )
        reconstructed = InferenceResult.from_dict(result.to_dict())
        assert reconstructed.landmarks == []

    def test_from_dict_missing_required_key_raises(self) -> None:
        """from_dict uses data[key] for required fields — missing key raises KeyError.

        NOTE for Alex: this asserts current behaviour. If the endpoint may omit
        required fields, consider explicit validation before indexing.
        """
        d = {
            # "pipeline_version" intentionally absent
            "model_version": "unet-v0",
            "timestamp": "2026-04-17T12:00:00Z",
            "image_filename": "plate.tif",
            "image_width_px": 512,
            "image_height_px": 512,
            "metadata": None,
            "mask_b64": "",
            "mask_confidence": 0.0,
            "landmark_count": 0,
            "landmarks": [],
        }
        with pytest.raises(KeyError):
            InferenceResult.from_dict(d)


# ---- ErrorResponse ---------------------------------------------------


@pytest.mark.unit
class TestErrorResponse:
    """Tests for the ErrorResponse dataclass."""

    def test_to_dict_contains_all_fields(self) -> None:
        """to_dict should return all four error fields."""
        err = ErrorResponse(
            error_code="FILE_TOO_LARGE",
            message="File is 62.4 MB, exceeds 50 MB limit.",
            pipeline_version="0.1.0",
            timestamp="2026-04-17T12:00:00Z",
        )
        d = err.to_dict()

        assert d["error_code"] == "FILE_TOO_LARGE"
        assert "62.4" in d["message"]
        assert d["pipeline_version"] == "0.1.0"
