"""Inference service — business logic behind POST /infer."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO

from cv_pipeline import infer as _pipeline_infer
from cv_pipeline.schema import InferenceResult, Metadata
from cv_pipeline.segmentation import SegmentationModel
from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

_UPLOAD_CHUNK_BYTES = 1024 * 1024


async def save_upload_to_tempfile(
    upload_file: BinaryIO,
    filename: str | None,
) -> Path:
    """Stream an uploaded file to a local tempfile in 1 MB chunks.

    Starlette hands us an ``UploadFile`` whose ``.file`` is a
    ``SpooledTemporaryFile``. Reading the whole payload into memory
    with ``await upload.read()`` would buffer up to 50 MB per
    concurrent request; we stream to disk instead so memory stays
    bounded.

    Args:
        upload_file: The underlying file-like object from ``UploadFile.file``.
        filename: Original filename (used to preserve the suffix so
            Pillow can detect the format). ``None`` defaults to .png.

    Returns:
        Absolute path to the temp file on disk. The caller is
        responsible for deleting it in a ``finally`` block.
    """
    suffix = Path(filename or "upload").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        # copyfileobj is synchronous; threadpool keeps the event loop free.
        await run_in_threadpool(
            shutil.copyfileobj,
            upload_file,
            tmp,
            _UPLOAD_CHUNK_BYTES,
        )
    return tmp_path


async def run_pipeline_inference(
    image_path: Path,
    model: SegmentationModel,
    metadata: Metadata,
) -> InferenceResult:
    """Invoke the cv-pipeline on a temp file and return the full result.

    The torch forward pass is blocking, so we execute the pipeline
    inside a threadpool. The event loop stays free to serve /health
    and concurrent /infer calls.

    Any ``ValidationError`` raised by the pipeline propagates upward;
    the router maps it to an HTTP status via the specification §5 table.

    Args:
        image_path: Path to the validated image on disk.
        model: A loaded ``SegmentationModel`` instance.
        metadata: Optional pass-through metadata for the response.

    Returns:
        An ``InferenceResult`` matching CV Pipeline Spec §4.
    """
    return await run_in_threadpool(
        _pipeline_infer,
        image_path=image_path,
        model=model,
        metadata=metadata,
    )


def cleanup_tempfile(path: Path) -> None:
    """Remove a temp file, logging rather than raising on failure.

    Args:
        path: Temp file path returned by ``save_upload_to_tempfile``.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        # A stale temp file is a minor issue — log it but never let it
        # turn a successful inference into a failed request.
        logger.warning("Failed to remove temp file '%s': %s", path, exc)
