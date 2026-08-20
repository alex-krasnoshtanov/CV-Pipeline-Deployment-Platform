"""Response schemas for the /infer endpoint and error envelopes.

Pydantic mirror of cv_pipeline.schema.InferenceResult. Wrapping the
dataclass at the API boundary gives us automatic request validation
and OpenAPI schema generation without polluting the cv-pipeline
package with a web-framework dependency.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class LandmarkSchema(BaseModel):
    """A single detected root tip."""

    id: int
    x: int
    y: int
    confidence: float = Field(ge=0.0, le=1.0)


class MetadataSchema(BaseModel):
    """Optional metadata passed alongside the image."""

    plate_id: str | None = None
    experiment_id: str | None = None
    timestamp: str | None = None


class InferenceResponse(BaseModel):
    """Successful /infer response body

    The ``prediction_id`` field is added by the API layer after the
    prediction is persisted to the database. It is ``None`` if the
    database write failed (inference still succeeds).
    """

    pipeline_version: str
    model_version: str
    timestamp: str
    image_filename: str
    image_width_px: int
    image_height_px: int
    metadata: MetadataSchema
    mask_b64: str
    mask_confidence: float = Field(ge=0.0, le=1.0)
    landmark_count: int
    landmarks: list[LandmarkSchema]
    prediction_id: str | None = None


class ErrorDetail(BaseModel):
    """Machine-readable detail attached to raised HTTPException."""

    error_code: str
    message: str


class ErrorResponse(BaseModel):
    """Full error body emitted by the global exception handler.

    Matches CV Pipeline Specification section 6 plus a ``request_id``
    field added by the request-ID middleware for traceability.
    """

    error_code: str
    message: str
    pipeline_version: str
    timestamp: datetime
    request_id: str
