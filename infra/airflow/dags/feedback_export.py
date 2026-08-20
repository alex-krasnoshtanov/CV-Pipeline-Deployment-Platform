"""Helpers for the feedback-to-retraining bridge DAG.

Kept free of Airflow and heavy Azure imports at module load so the logic
is unit-testable in isolation. Database connections and the blob
container client are injected by the caller (the DAG tasks build them
from environment variables); psycopg2 and azure-storage-blob are imported
lazily inside the DAG, never here.

The "good set" is the union the project agreed on:

- ``flag = 'good'``  -> the model's predicted mask was approved, so the
  training label is the prediction's own ``mask_b64``.
- a relabelled row  -> ``corrected_mask_b64`` is the human-drawn label.

Rows already consumed (``exported_at`` set) and rows without a stored
input image (``image_uri IS NULL``) are excluded. ``uncertain`` flags and
not-yet-relabelled ``bad`` rows stay in the review queue and never reach
training until a human turns them into one of the two cases above.
"""

from __future__ import annotations

import base64
import binascii
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Feedback that is ready to become training data: approved predictions or
# relabelled corrections, with a stored image, not yet exported.
_READY_WHERE = (
    "f.exported_at IS NULL "
    "AND p.image_uri IS NOT NULL "
    "AND (f.flag = 'good' OR f.corrected_mask_b64 IS NOT NULL)"
)

_SELECT_SQL = (
    "SELECT f.id AS feedback_id, f.prediction_id, p.image_uri, "
    "p.mask_b64 AS predicted_mask_b64, f.corrected_mask_b64, "
    "f.flag, f.created_at "
    "FROM feedback f JOIN predictions p ON f.prediction_id = p.id "
    f"WHERE {_READY_WHERE} ORDER BY f.created_at ASC"
)

_COUNT_SQL = (
    "SELECT COUNT(*) FROM feedback f "
    "JOIN predictions p ON f.prediction_id = p.id "
    f"WHERE {_READY_WHERE}"
)


def fetch_good_set(conn: Any) -> list[dict]:
    """Return every ready feedback row joined to its prediction.

    Args:
        conn: An open DB-API connection (psycopg2).

    Returns:
        A list of dict rows, oldest first, with keys feedback_id,
        prediction_id, image_uri, predicted_mask_b64, corrected_mask_b64,
        flag, and created_at.
    """
    with conn.cursor() as cur:
        cur.execute(_SELECT_SQL)
        columns = [col.name for col in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def count_pending(conn: Any) -> int:
    """Return how many ready feedback rows are waiting to be exported.

    Args:
        conn: An open DB-API connection (psycopg2).

    Returns:
        The count of rows matching the ready-set criteria.
    """
    with conn.cursor() as cur:
        cur.execute(_COUNT_SQL)
        return int(cur.fetchone()[0])


def select_labels(rows: list[dict]) -> tuple[dict, list]:
    """Reduce raw rows to one labelled pair per prediction.

    The same prediction can have several feedback rows; the most recent
    wins (rows arrive oldest first, so later entries overwrite). The label
    is the corrected mask when present, otherwise the approved prediction
    mask. Every row's feedback id is returned so all of them are stamped
    exported, including superseded duplicates.

    Args:
        rows: Output of ``fetch_good_set``.

    Returns:
        A tuple of (pairs, feedback_ids) where pairs maps prediction_id to
        (image_uri, label_mask_b64) and feedback_ids lists every row id.
    """
    pairs: dict = {}
    feedback_ids: list = []
    for row in rows:
        feedback_ids.append(row["feedback_id"])
        label = row["corrected_mask_b64"] or row["predicted_mask_b64"]
        pairs[row["prediction_id"]] = (row["image_uri"], label)
    return pairs, feedback_ids


def blob_key_from_uri(image_uri: str) -> str:
    """Extract the in-container blob name from an azureml datastore URI.

    ``azureml://datastores/<store>/paths/<key>`` -> ``<key>``. The key is
    the blob's name inside the container backing the datastore.

    Args:
        image_uri: A datastore URI for the stored input image.

    Returns:
        The blob key relative to the container.

    Raises:
        ValueError: If the URI is not a datastore path.
    """
    marker = "/paths/"
    index = image_uri.find(marker)
    if not image_uri.startswith("azureml://") or index == -1:
        raise ValueError(f"Not an azureml datastore URI: {image_uri}")
    return image_uri[index + len(marker) :]


def stage_pairs(container_client: Any, pairs: dict, dest_dir: Path) -> int:
    """Materialise image/mask pairs into a staging folder.

    Downloads each input image from blob and writes the decoded mask
    beside it, both stemmed by prediction id so the preprocessing job
    pairs them::

        dest_dir/images/<prediction_id>.png
        dest_dir/masks/<prediction_id>.png

    A pair whose image URI is unparseable, whose blob is unreadable, or
    whose mask is not valid base64 is logged and skipped rather than
    aborting the batch.

    Args:
        container_client: An Azure ``ContainerClient`` for the datastore
            container.
        pairs: Mapping of prediction_id to (image_uri, label_mask_b64).
        dest_dir: Staging root; ``images/`` and ``masks/`` are created.

    Returns:
        The number of pairs successfully staged.
    """
    (dest_dir / "images").mkdir(parents=True, exist_ok=True)
    (dest_dir / "masks").mkdir(parents=True, exist_ok=True)

    staged = 0
    for prediction_id, (image_uri, label_b64) in pairs.items():
        try:
            key = blob_key_from_uri(image_uri)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", prediction_id, exc)
            continue
        try:
            image_bytes = container_client.download_blob(key).readall()
        except Exception as exc:  # noqa: BLE001 - any SDK error is non-fatal here
            logger.error("Could not download %s: %s", key, exc)
            continue
        try:
            mask_bytes = base64.b64decode(label_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            logger.error("Bad base64 mask for %s: %s", prediction_id, exc)
            continue

        # The incremental preprocessor (cloud_preprocess_incremental.find_root_mask)
        # pairs an image with its mask by the "<image_stem>_root_mask" naming
        # convention, not a same-stem mask. Match it or the pair is silently
        # skipped at merge time.
        stem = str(prediction_id)
        (dest_dir / "images" / f"{stem}.png").write_bytes(image_bytes)
        (dest_dir / "masks" / f"{stem}_root_mask.png").write_bytes(mask_bytes)
        staged += 1

    return staged


def mark_exported(conn: Any, feedback_ids: list) -> None:
    """Stamp ``exported_at`` on the given feedback rows in one transaction.

    Args:
        conn: An open DB-API connection (psycopg2).
        feedback_ids: The feedback row ids consumed by this export.
    """
    if not feedback_ids:
        return
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE feedback SET exported_at = NOW() WHERE id = ANY(%s::uuid[])",
            (feedback_ids,),
        )
    conn.commit()
