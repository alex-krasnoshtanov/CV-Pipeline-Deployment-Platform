"""Pydantic response model for POST /explain.

Mirrors :class:`cv_pipeline.schema.ExplanationResult.to_dict`. The route
returns the dataclass's dict directly; this model is what FastAPI validates
against and documents in the OpenAPI schema.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExplainMetadata(BaseModel):
    """Pass-through metadata echoed back in the explanation response."""

    plate_id: str | None = None
    experiment_id: str | None = None
    timestamp: str | None = None


class ExplainResponse(BaseModel):
    """Seg-Grad-CAM explanation for a single image."""

    pipeline_version: str
    model_version: str
    timestamp: str
    image_filename: str
    image_width_px: int
    image_height_px: int
    metadata: ExplainMetadata
    method: str = Field(description="Attribution method, e.g. 'seg-grad-cam'.")
    target_layer: str = Field(description="Layer the heatmap was computed from.")
    downscaled: bool = Field(
        description="Whether the image was downscaled before attribution."
    )
    heatmap_peak: float = Field(
        description="Pre-normalisation peak CAM value; 0.0 means no salient region."
    )
    heatmap_b64: str = Field(
        description="Base64 grayscale PNG heatmap at the original image size."
    )
