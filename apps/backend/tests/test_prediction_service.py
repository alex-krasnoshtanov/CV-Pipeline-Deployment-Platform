"""Unit tests for api.services.prediction_service."""

from __future__ import annotations

import base64
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from api.services.prediction_service import save_prediction
from cv_pipeline.schema import InferenceResult, Landmark, Metadata
from sqlalchemy.exc import OperationalError


def _fake_result() -> InferenceResult:
    """Build a deterministic InferenceResult for testing."""
    return InferenceResult(
        pipeline_version="0.1.0",
        model_version="unet-v1",
        timestamp="2026-05-01T12:00:00Z",
        image_filename="test.png",
        image_width_px=300,
        image_height_px=300,
        metadata=Metadata(plate_id="PL-001"),
        mask_b64=base64.b64encode(b"fake-mask").decode("ascii"),
        mask_confidence=0.85,
        landmark_count=1,
        landmarks=[Landmark(id=0, x=150, y=200, confidence=0.9)],
    )


async def _simulate_refresh(obj: object) -> None:
    """Simulate what the real DB does on refresh: assign a server-generated ID."""
    obj.id = uuid.uuid4()


@pytest.mark.unit
class TestSavePredictionHappyPath:
    """Tests for successful prediction persistence."""

    @pytest.mark.anyio
    async def test_returns_prediction_id(self) -> None:
        """save_prediction should return the new prediction's ID on success."""
        db = MagicMock()
        db.add = MagicMock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock(side_effect=_simulate_refresh)

        result = _fake_result()
        prediction_id = await save_prediction(db, result, user_id=None)

        assert prediction_id is not None
        db.add.assert_called_once()
        db.commit.assert_called_once()

    @pytest.mark.anyio
    async def test_accepts_user_id(self) -> None:
        """save_prediction should accept an optional user_id."""
        db = MagicMock()
        db.add = MagicMock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock(side_effect=_simulate_refresh)

        user_id = uuid.uuid4()
        result = _fake_result()
        prediction_id = await save_prediction(db, result, user_id=user_id)

        assert prediction_id is not None


@pytest.mark.unit
class TestSavePredictionErrors:
    """Tests for prediction persistence error handling."""

    @pytest.mark.anyio
    async def test_returns_none_on_commit_failure(self) -> None:
        """save_prediction should return None and roll back on DB errors."""
        db = MagicMock()
        db.commit = AsyncMock(
            side_effect=OperationalError("node down", {}, None),
        )
        db.rollback = AsyncMock()
        db.add = MagicMock()

        result = _fake_result()
        prediction_id = await save_prediction(db, result, user_id=None)

        assert prediction_id is None
        db.rollback.assert_called_once()
