"""Schemas for the /feedback endpoints.

Pydantic models for request validation and response serialisation.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FeedbackRequest(BaseModel):
    """POST /feedback request body.

    Public feedback records a verdict and optional notes only. A
    corrected mask is a training label and must enter through the
    admin-only, validated ``POST /feedback/relabel`` endpoint, so this
    model forbids unexpected fields: sending ``corrected_mask_b64``
    here returns 422 instead of silently storing an unvalidated mask.
    """

    model_config = ConfigDict(extra="forbid")

    prediction_id: str
    flag: str = Field(pattern="^(good|bad|uncertain)$")
    notes: str | None = Field(None, max_length=2000)


class FeedbackResponse(BaseModel):
    """POST /feedback success response."""

    feedback_id: str
    prediction_id: str
    flag: str
    created_at: datetime


class ReviewQueueItem(BaseModel):
    """One prediction awaiting reviewer correction.

    Carries the prediction's image locator and predicted mask so a
    reviewer can inspect it, plus the latest flag and notes for
    context.
    """

    prediction_id: str
    image_filename: str
    image_width_px: int
    image_height_px: int
    image_uri: str | None
    mask_b64: str
    mask_confidence: float
    flag: str | None
    notes: str | None
    created_at: datetime


class RelabelRequest(BaseModel):
    """POST /feedback/relabel request body.

    At least one of ``corrected_mask_b64`` or ``flag`` must be
    provided. Only a corrected mask, or a ``good`` flag, removes a
    prediction from the review queue; a ``bad`` or ``uncertain`` flag
    without a mask is recorded but leaves the prediction flagged for a
    later correction.
    """

    prediction_id: str
    corrected_mask_b64: str | None = Field(None, max_length=25_000_000)
    flag: str | None = Field(None, pattern="^(good|bad|uncertain)$")
    notes: str | None = Field(None, max_length=2000)

    @model_validator(mode="after")
    def _require_resolution(self) -> RelabelRequest:
        """Reject a relabel that neither corrects nor resolves.

        Returns:
            The validated model.

        Raises:
            ValueError: If both ``corrected_mask_b64`` and ``flag``
                are absent.
        """
        if self.corrected_mask_b64 is None and self.flag is None:
            raise ValueError(
                "Provide a corrected_mask_b64, a flag, or both.",
            )
        return self
