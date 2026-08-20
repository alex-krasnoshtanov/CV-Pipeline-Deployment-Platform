"""Response and error data structures for the cv-pipeline package.

Defines the JSON-serializable dataclasses that represent inference results
and error responses. These structures implement the contract from the
CV Pipeline Specification (task_328), sections 4 and 6.

All consumers of the pipeline — FastAPI, CLI, Azure ML jobs — receive
these same structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Landmark:
    """A single detected root tip location.

    Args:
        id: Zero-based index of this landmark.
        x: X coordinate in input pixel space.
        y: Y coordinate in input pixel space.
        confidence: Confidence that this point is a true root tip, range [0, 1].
    """

    id: int
    x: int
    y: int
    confidence: float

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for JSON serialization."""
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Landmark:
        """Reconstruct a Landmark from its ``to_dict`` representation.

        Args:
            data: Dict with ``id``, ``x``, ``y``, and ``confidence`` keys.

        Returns:
            The reconstructed Landmark.
        """
        return cls(
            id=data["id"],
            x=data["x"],
            y=data["y"],
            confidence=data["confidence"],
        )


@dataclass
class Metadata:
    """Optional metadata passed alongside the input image.

    Args:
        plate_id: Identifier for the Petri dish or plate.
        experiment_id: Identifier for the experiment this image belongs to.
        timestamp: ISO 8601 time the image was captured.
    """

    plate_id: str | None = None
    experiment_id: str | None = None
    timestamp: str | None = None

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for JSON serialization."""
        return {
            "plate_id": self.plate_id,
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> Metadata:
        """Reconstruct Metadata from its ``to_dict`` representation.

        Args:
            data: Dict with optional ``plate_id``, ``experiment_id``, and
                ``timestamp`` keys, or ``None``.

        Returns:
            The reconstructed Metadata (each field defaults to ``None``).
        """
        data = data or {}
        return cls(
            plate_id=data.get("plate_id"),
            experiment_id=data.get("experiment_id"),
            timestamp=data.get("timestamp"),
        )


@dataclass
class InferenceResult:
    """Complete inference response for a single image.

    Args:
        pipeline_version: Semantic version of the cv-pipeline package.
        model_version: Name and version of the registered model used.
        timestamp: ISO 8601 UTC time the inference completed.
        image_filename: Filename of the input image (not the full path).
        image_width_px: Width of the input image in pixels.
        image_height_px: Height of the input image in pixels.
        metadata: Optional metadata passed through from the input.
        mask_b64: Base64-encoded PNG of the binary segmentation mask.
        mask_confidence: Mean confidence across root-classified pixels, range [0, 1].
        landmark_count: Number of root tips detected. Zero is a valid value.
        landmarks: List of detected root tip landmarks. Empty list if none detected.
    """

    pipeline_version: str
    model_version: str
    timestamp: str
    image_filename: str
    image_width_px: int
    image_height_px: int
    metadata: Metadata
    mask_b64: str
    mask_confidence: float
    landmark_count: int
    landmarks: list[Landmark] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for JSON serialization."""
        return {
            "pipeline_version": self.pipeline_version,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "image_filename": self.image_filename,
            "image_width_px": self.image_width_px,
            "image_height_px": self.image_height_px,
            "metadata": self.metadata.to_dict(),
            "mask_b64": self.mask_b64,
            "mask_confidence": self.mask_confidence,
            "landmark_count": self.landmark_count,
            "landmarks": [lm.to_dict() for lm in self.landmarks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> InferenceResult:
        """Reconstruct an InferenceResult from its ``to_dict`` representation.

        Inverse of ``to_dict``. Used to parse responses from the remote model
        endpoint, which returns exactly that JSON shape (see the endpoint
        scoring script, which serialises with ``to_dict``).

        Args:
            data: Dict produced by ``to_dict`` (or an equivalent endpoint
                response), including the nested ``metadata`` object and the
                ``landmarks`` list.

        Returns:
            The reconstructed InferenceResult.
        """
        return cls(
            pipeline_version=data["pipeline_version"],
            model_version=data["model_version"],
            timestamp=data["timestamp"],
            image_filename=data["image_filename"],
            image_width_px=data["image_width_px"],
            image_height_px=data["image_height_px"],
            metadata=Metadata.from_dict(data.get("metadata")),
            mask_b64=data["mask_b64"],
            mask_confidence=data["mask_confidence"],
            landmark_count=data["landmark_count"],
            landmarks=[Landmark.from_dict(lm) for lm in data.get("landmarks", [])],
        )


@dataclass
class ErrorResponse:
    """Error response returned when the pipeline rejects an input.

    Args:
        error_code: Machine-readable code identifying the error type.
        message: Human-readable explanation with specific values where available.
        pipeline_version: Package version at the time of the error.
        timestamp: ISO 8601 UTC time the error occurred.
    """

    error_code: str
    message: str
    pipeline_version: str
    timestamp: str

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for JSON serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "pipeline_version": self.pipeline_version,
            "timestamp": self.timestamp,
        }


@dataclass
class ExplanationResult:
    """Seg-Grad-CAM explanation for a single image.

    Companion to :class:`InferenceResult`. Where ``InferenceResult`` answers
    "what did the model predict", this answers "which regions drove the root
    classification". The heatmap is sized to the original image so a frontend
    can overlay it exactly like the segmentation mask.

    Args:
        pipeline_version: Semantic version of the cv-pipeline package.
        model_version: Name and version of the model that was explained.
        timestamp: ISO 8601 UTC time the explanation completed.
        image_filename: Filename of the input image (not the full path).
        image_width_px: Width of the input image in pixels.
        image_height_px: Height of the input image in pixels.
        metadata: Optional metadata passed through from the input.
        method: Attribution method used (e.g. ``"seg-grad-cam"``).
        target_layer: Human-readable name of the layer attributed against.
        downscaled: Whether the image was downscaled before attribution.
        heatmap_peak: Pre-normalisation peak CAM value (diagnostic). A peak of
            0.0 means the model found no salient region.
        heatmap_b64: Base64-encoded grayscale PNG of the [0, 255] heatmap.
    """

    pipeline_version: str
    model_version: str
    timestamp: str
    image_filename: str
    image_width_px: int
    image_height_px: int
    metadata: Metadata
    method: str
    target_layer: str
    downscaled: bool
    heatmap_peak: float
    heatmap_b64: str

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for JSON serialization."""
        return {
            "pipeline_version": self.pipeline_version,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "image_filename": self.image_filename,
            "image_width_px": self.image_width_px,
            "image_height_px": self.image_height_px,
            "metadata": self.metadata.to_dict(),
            "method": self.method,
            "target_layer": self.target_layer,
            "downscaled": self.downscaled,
            "heatmap_peak": self.heatmap_peak,
            "heatmap_b64": self.heatmap_b64,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExplanationResult:
        """Reconstruct an ExplanationResult from its ``to_dict`` representation.

        Args:
            data: Dict produced by ``to_dict``.

        Returns:
            The reconstructed ExplanationResult.
        """
        return cls(
            pipeline_version=data["pipeline_version"],
            model_version=data["model_version"],
            timestamp=data["timestamp"],
            image_filename=data["image_filename"],
            image_width_px=data["image_width_px"],
            image_height_px=data["image_height_px"],
            metadata=Metadata.from_dict(data.get("metadata")),
            method=data["method"],
            target_layer=data["target_layer"],
            downscaled=data["downscaled"],
            heatmap_peak=data["heatmap_peak"],
            heatmap_b64=data["heatmap_b64"],
        )
