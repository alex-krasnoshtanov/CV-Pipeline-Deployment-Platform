"""Background persistence of inference images for the feedback loop.

Runs as a FastAPI ``BackgroundTask`` after the response is sent, so it
adds no latency to ``/infer``. It reads the already-written tempfile,
stores the bytes via the configured storage backend, records the
locator on the prediction row, and deletes the tempfile.

Failures are logged, never raised: a storage outage must not turn a
successful prediction into an error for the user. A row whose image
write failed keeps ``image_uri = NULL`` and is simply skipped by the
later export step.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from api.db import session as db_session
from api.db.models import Prediction
from api.services.storage import feedback_image_key, get_storage_backend

logger = logging.getLogger(__name__)


def _safe_unlink(path: Path) -> None:
    """Delete *path* if present, swallowing and logging any error."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.error("Failed to delete tempfile '%s': %s", path, exc)


async def persist_inference_image(
    prediction_id: uuid.UUID,
    user_id: uuid.UUID,
    src_path: Path,
    suffix: str,
) -> None:
    """Persist a prediction's input image and record its locator.

    Args:
        prediction_id: UUID of the committed prediction row.
        user_id: UUID of the authenticated owner (per-user folder).
        src_path: Tempfile holding the uploaded image bytes. This
            function takes ownership and deletes it when done.
        suffix: File suffix including the dot, e.g. ``.png``.
    """
    key = feedback_image_key(str(user_id), str(prediction_id), suffix)
    try:
        try:
            data = src_path.read_bytes()
        except OSError as exc:
            logger.error("Cannot read tempfile '%s': %s", src_path, exc)
            return

        backend = get_storage_backend()
        try:
            await run_in_threadpool(backend.write, key, data)
        except Exception:
            logger.exception("Failed to write feedback image for %s", prediction_id)
            return

        uri = backend.uri_for(key)

        # Read the session factory from the module at call time. It is
        # None at import and only set by init_db() during startup, so a
        # module-level ``from ... import SessionLocal`` would capture the
        # stale None. Accessing the attribute here sees the live value.
        session_factory = db_session.SessionLocal
        if session_factory is None:
            logger.error("SessionLocal unset; cannot record image_uri.")
            return

        try:
            async with session_factory() as db:
                await db.execute(
                    update(Prediction)
                    .where(Prediction.id == prediction_id)
                    .values(image_uri=uri)
                )
                await db.commit()
            logger.info("Stored feedback image for %s at %s", prediction_id, uri)
        except SQLAlchemyError:
            logger.exception("Failed to record image_uri for %s", prediction_id)
    finally:
        _safe_unlink(src_path)
