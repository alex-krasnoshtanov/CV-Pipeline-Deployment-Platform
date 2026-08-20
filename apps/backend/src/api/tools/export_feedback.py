"""Stage corrected feedback into a training-ready images/+masks/ folder.

This is the backend half of the feedback-to-retraining bridge. It reads
researcher-corrected predictions from the database, pairs each corrected
mask with the original input image, and writes them into the directory
layout the Azure ML incremental preprocessing job expects:

    <output-dir>/
        images/<prediction_id>.png   (the original input image)
        masks/<prediction_id>.png    (the researcher's corrected mask)

The Airflow side registers that folder as a new ``hades-raw-upload`` data
asset version and triggers the existing incremental pipeline; this script
does not touch Azure.

Rows are selected only when they carry a corrected mask and have not been
exported before (``exported_at IS NULL``). Every selected row is stamped
with ``exported_at`` in the same transaction, so a correction is never
ingested into more than one training run and a re-run is a safe no-op.

Run inside the backend container, where both the database and the feedback
image volume are reachable::

    python -m api.tools.export_feedback --output-dir /tmp/feedback_stage
"""

from __future__ import annotations

import argparse
import base64
import binascii
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session

from api.db.models import Feedback, Prediction

logger = logging.getLogger(__name__)


def _sync_url(db_url: str) -> str:
    """Return a synchronous SQLAlchemy URL.

    The app runs async (``+asyncpg``); this one-shot script runs
    synchronously via psycopg2, mirroring ``seed.py``.

    Args:
        db_url: The configured (possibly async) database URL.

    Returns:
        The same URL with the async driver suffix removed.
    """
    return db_url.replace("postgresql+asyncpg://", "postgresql://")


# Local/manual export only: corrected masks on local-disk images. The
# cloud bridge (dags/feedback_export.py) is the authoritative training-ready
# path (good OR corrected); this tool is intentionally corrections-only.
def _select_pending(session: Session) -> list[tuple]:
    """Fetch corrected, not-yet-exported feedback joined to its image URI.

    Args:
        session: An open database session.

    Returns:
        Rows of (feedback_id, prediction_id, corrected_mask_b64,
        created_at, image_uri), ordered oldest first.
    """
    stmt = (
        select(
            Feedback.id,
            Feedback.prediction_id,
            Feedback.corrected_mask_b64,
            Feedback.created_at,
            Prediction.image_uri,
        )
        .join(Prediction, Feedback.prediction_id == Prediction.id)
        .where(
            Feedback.corrected_mask_b64.is_not(None),
            Feedback.exported_at.is_(None),
        )
        .order_by(Feedback.created_at.asc())
    )
    return list(session.execute(stmt).all())


def _latest_per_prediction(rows: list[tuple]) -> dict:
    """Keep one correction per prediction — the most recent wins.

    The same prediction may be corrected more than once. Only the newest
    corrected mask should enter training; older rows are still stamped as
    exported so they do not linger in the queue.

    Args:
        rows: Output of ``_select_pending`` (ordered oldest first).

    Returns:
        Mapping of prediction_id -> (corrected_mask_b64, image_uri) for the
        latest correction of each prediction.
    """
    latest: dict = {}
    for _fid, prediction_id, mask_b64, _created, image_uri in rows:
        latest[prediction_id] = (mask_b64, image_uri)  # later row overwrites
    return latest


def _resolve_image_bytes(image_uri: str | None) -> bytes | None:
    """Read the original input image referenced by a prediction.

    Local storage records ``image_uri`` as a filesystem path on the mounted
    feedback volume. Blob storage records an ``azureml://`` URI; those
    images already live in the datastore and are staged on the cloud side,
    so this local exporter skips them rather than guessing.

    Args:
        image_uri: The prediction's stored image URI, or None.

    Returns:
        The image bytes, or None if the image cannot be read locally.
    """
    if not image_uri:
        logger.warning("Prediction has no image_uri; skipping.")
        return None
    if image_uri.startswith("azureml://"):
        logger.warning(
            "image_uri is a blob URI (%s); the local exporter does not fetch "
            "from blob. Stage blob-backed feedback on the cloud side.",
            image_uri,
        )
        return None
    path = Path(image_uri)
    if not path.is_file():
        logger.warning("Image file missing on volume: %s; skipping.", path)
        return None
    try:
        return path.read_bytes()
    except OSError as exc:
        logger.error("Failed to read image %s: %s", path, exc)
        return None


def _write_pair(
    output_dir: Path, prediction_id: object, image_bytes: bytes, mask_b64: str
) -> bool:
    """Write one image/mask pair under ``output_dir``.

    Both files share the prediction id as their stem so the preprocessing
    job pairs them. A mask that is not valid base64 is logged and skipped
    rather than aborting the batch.

    Args:
        output_dir: Staging root containing images/ and masks/.
        prediction_id: The prediction UUID, used as the file stem.
        image_bytes: Raw bytes of the original input image.
        mask_b64: Base64-encoded corrected mask PNG.

    Returns:
        True if both files were written, False otherwise.
    """
    try:
        mask_bytes = base64.b64decode(mask_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        logger.error("Bad base64 mask for %s: %s; skipping.", prediction_id, exc)
        return False
    stem = str(prediction_id)
    try:
        (output_dir / "images" / f"{stem}.png").write_bytes(image_bytes)
        (output_dir / "masks" / f"{stem}_root_mask.png").write_bytes(mask_bytes)
    except OSError as exc:
        logger.error("Failed to write pair for %s: %s", prediction_id, exc)
        return False
    return True


def export_feedback(session: Session, output_dir: Path) -> int:
    """Stage all pending corrected feedback into ``output_dir``.

    Selects corrected, not-yet-exported feedback; writes one image/mask
    pair per prediction (latest correction wins); and stamps every selected
    row's ``exported_at`` in a single transaction so the run is idempotent
    and corrections are consumed exactly once.

    Args:
        session: An open database session.
        output_dir: Staging root; ``images/`` and ``masks/`` are created.

    Returns:
        The number of image/mask pairs written.
    """
    rows = _select_pending(session)
    if not rows:
        logger.info("No pending corrected feedback to export.")
        return 0

    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)

    written = 0
    for prediction_id, (mask_b64, image_uri) in _latest_per_prediction(rows).items():
        image_bytes = _resolve_image_bytes(image_uri)
        if image_bytes is None:
            continue
        if _write_pair(output_dir, prediction_id, image_bytes, mask_b64):
            written += 1

    # Stamp every selected row — including superseded duplicates and rows
    # whose image could not be read — so the queue does not re-surface them
    # next run. A missing image is a data issue to investigate, not
    # something to retry blindly on every pass.
    selected_ids = [row[0] for row in rows]
    session.execute(
        update(Feedback)
        .where(Feedback.id.in_(selected_ids))
        .values(exported_at=datetime.now(timezone.utc))
    )
    session.commit()

    logger.info(
        "Exported %d image/mask pair(s) from %d feedback row(s) to %s.",
        written,
        len(selected_ids),
        output_dir,
    )
    return written


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit code: 0 on success (including an empty run), 2 if
        ``DB_URL`` is not configured.
    """
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Stage corrected feedback into an images/+masks/ folder."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Staging directory; images/ and masks/ are created inside it.",
    )
    args = parser.parse_args()

    db_url = os.getenv("DB_URL", "")
    if not db_url:
        logger.error("DB_URL is empty or unset; cannot export feedback.")
        return 2

    engine = create_engine(_sync_url(db_url), future=True)
    with Session(engine) as session:
        export_feedback(session, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
