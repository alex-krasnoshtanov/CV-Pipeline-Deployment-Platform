"""Unit tests for api.services.feedback_service."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from api.services.feedback_service import (
    PredictionNotFoundError,
    get_review_queue,
    save_correction,
    save_feedback,
)
from api.services.mask_validation import MaskValidationError


@pytest.mark.unit
class TestSaveFeedbackHappyPath:
    """Tests for successful feedback saving."""

    @pytest.mark.anyio
    async def test_saves_and_returns_feedback(self) -> None:
        """save_feedback should persist the feedback row and return it."""
        db = MagicMock()
        db.execute = AsyncMock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock()
        db.add = MagicMock()

        pred_id = str(uuid.uuid4())
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = MagicMock()
        db.execute.return_value = mock_result

        result = await save_feedback(
            db=db,
            prediction_id=pred_id,
            user_id=uuid.uuid4(),
            flag="bad",
            notes="Too much noise",
        )

        assert result.flag == "bad"
        db.add.assert_called_once()
        db.commit.assert_called_once()
        db.refresh.assert_called_once()

    @pytest.mark.anyio
    async def test_public_feedback_stores_no_mask(self) -> None:
        """Public save_feedback must never store a corrected mask.

        The public path records a verdict only; a corrected mask can
        only enter through the validated admin relabel path. The
        stored row therefore always has a null corrected_mask_b64,
        which keeps the prediction in the review queue.
        """
        db = MagicMock()
        db.execute = AsyncMock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock()
        db.add = MagicMock()

        pred_id = str(uuid.uuid4())
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = MagicMock()
        db.execute.return_value = mock_result

        result = await save_feedback(
            db=db,
            prediction_id=pred_id,
            user_id=uuid.uuid4(),
            flag="bad",
            notes="Corrected root tip position",
        )

        assert result.flag == "bad"
        assert result.corrected_mask_b64 is None
        assert result.notes == "Corrected root tip position"
        db.add.assert_called_once()

    @pytest.mark.anyio
    async def test_returned_row_has_correct_prediction_id(self) -> None:
        """The returned Feedback row should reference the right prediction."""
        db = MagicMock()
        db.execute = AsyncMock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock()
        db.add = MagicMock()

        pred_id = str(uuid.uuid4())
        user_id = uuid.uuid4()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = MagicMock()
        db.execute.return_value = mock_result

        result = await save_feedback(
            db=db,
            prediction_id=pred_id,
            user_id=user_id,
            flag="good",
            notes=None,
        )

        assert result.prediction_id == uuid.UUID(pred_id)
        assert result.user_id == user_id


@pytest.mark.unit
class TestSaveFeedbackErrors:
    """Tests for feedback validation and error handling."""

    @pytest.mark.anyio
    async def test_rejects_invalid_uuid(self) -> None:
        """An unparseable prediction_id should raise ValueError."""
        db = MagicMock()

        with pytest.raises(ValueError, match="is not a valid UUID"):
            await save_feedback(
                db,
                "invalid-uuid-string",
                uuid.uuid4(),
                "good",
                None,
            )

    @pytest.mark.anyio
    async def test_rejects_missing_prediction(self) -> None:
        """A valid UUID not in the DB should raise PredictionNotFoundError."""
        db = MagicMock()
        db.execute = AsyncMock()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        db.execute.return_value = mock_result

        pred_id = str(uuid.uuid4())

        with pytest.raises(
            PredictionNotFoundError,
            match=f"Prediction '{pred_id}' not found",
        ):
            await save_feedback(
                db,
                pred_id,
                uuid.uuid4(),
                "good",
                None,
            )


@pytest.mark.unit
class TestGetReviewQueue:
    """Tests for get_review_queue selection and assembly logic."""

    @pytest.mark.anyio
    async def test_empty_when_no_predictions(self) -> None:
        """An empty prediction result should return an empty list."""
        db = MagicMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(return_value=result)

        items = await get_review_queue(db, limit=20, offset=0)

        assert items == []

    @pytest.mark.anyio
    async def test_assembles_items_with_latest_feedback(self) -> None:
        """Each queued prediction carries its latest flag and notes."""
        pred = MagicMock()
        pred.id = uuid.uuid4()
        pred.image_filename = "plate.png"
        pred.image_width_px = 256
        pred.image_height_px = 256
        pred.image_uri = "/data/feedback/raw/u/p.png"
        pred.mask_b64 = "bWFzaw=="
        pred.mask_confidence = 0.8
        pred.created_at = "2026-05-01T10:00:00Z"

        # The service orders feedback created_at DESC, so the newest row
        # is seen first and wins the setdefault dedup.
        newer = MagicMock()
        newer.prediction_id = pred.id
        newer.flag = "bad"
        newer.notes = "newest"
        older = MagicMock()
        older.prediction_id = pred.id
        older.flag = "uncertain"
        older.notes = "oldest"

        pred_result = MagicMock()
        pred_result.scalars.return_value.all.return_value = [pred]
        fb_result = MagicMock()
        fb_result.scalars.return_value.all.return_value = [newer, older]

        db = MagicMock()
        db.execute = AsyncMock(side_effect=[pred_result, fb_result])

        items = await get_review_queue(db, limit=20, offset=0)

        assert len(items) == 1
        item = items[0]
        assert item["prediction_id"] == str(pred.id)
        assert item["image_uri"] == "/data/feedback/raw/u/p.png"
        assert item["flag"] == "bad"
        assert item["notes"] == "newest"


@pytest.mark.unit
class TestSaveCorrection:
    """Tests for save_correction (the admin relabel path)."""

    @staticmethod
    def _db_with_prediction() -> MagicMock:
        """Return a mock session whose lookup yields a prediction."""
        db = MagicMock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock()
        db.add = MagicMock()
        prediction = MagicMock()
        prediction.image_width_px = 256
        prediction.image_height_px = 256
        result = MagicMock()
        result.scalar_one_or_none.return_value = prediction
        db.execute = AsyncMock(return_value=result)
        return db

    @pytest.mark.anyio
    async def test_flag_only_correction_stores_no_mask(self) -> None:
        """A flag-only correction resolves with no mask stored."""
        db = self._db_with_prediction()

        row = await save_correction(
            db=db,
            prediction_id=str(uuid.uuid4()),
            admin_id=uuid.uuid4(),
            corrected_mask_b64=None,
            flag="good",
            notes=None,
        )

        assert row.flag == "good"
        assert row.corrected_mask_b64 is None
        db.add.assert_called_once()
        db.commit.assert_called_once()

    @pytest.mark.anyio
    async def test_mask_is_validated_and_stored(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A corrected mask is validated, normalised, and stored.

        A mask supplied without an explicit flag defaults to ``bad``.
        """
        db = self._db_with_prediction()
        monkeypatch.setattr(
            "api.services.feedback_service.validate_corrected_mask",
            lambda mask, width, height: "normalised",
        )

        row = await save_correction(
            db=db,
            prediction_id=str(uuid.uuid4()),
            admin_id=uuid.uuid4(),
            corrected_mask_b64="raw-mask",
            flag=None,
            notes="fixed tip",
        )

        assert row.corrected_mask_b64 == "normalised"
        assert row.flag == "bad"

    @pytest.mark.anyio
    async def test_invalid_mask_propagates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A mask validation failure should propagate to the caller."""
        db = self._db_with_prediction()

        def _raise(mask: str, width: int, height: int) -> str:
            raise MaskValidationError("MASK_CORRUPT", "bad mask")

        monkeypatch.setattr(
            "api.services.feedback_service.validate_corrected_mask",
            _raise,
        )

        with pytest.raises(MaskValidationError):
            await save_correction(
                db=db,
                prediction_id=str(uuid.uuid4()),
                admin_id=uuid.uuid4(),
                corrected_mask_b64="raw",
                flag=None,
                notes=None,
            )

    @pytest.mark.anyio
    async def test_rejects_invalid_uuid(self) -> None:
        """An unparseable prediction_id should raise ValueError."""
        with pytest.raises(ValueError, match="is not a valid UUID"):
            await save_correction(
                db=MagicMock(),
                prediction_id="not-a-uuid",
                admin_id=uuid.uuid4(),
                corrected_mask_b64=None,
                flag="good",
                notes=None,
            )

    @pytest.mark.anyio
    async def test_rejects_missing_prediction(self) -> None:
        """A valid UUID not in the DB should raise PredictionNotFoundError."""
        db = MagicMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        db.execute = AsyncMock(return_value=result)

        with pytest.raises(PredictionNotFoundError):
            await save_correction(
                db=db,
                prediction_id=str(uuid.uuid4()),
                admin_id=uuid.uuid4(),
                corrected_mask_b64=None,
                flag="good",
                notes=None,
            )
