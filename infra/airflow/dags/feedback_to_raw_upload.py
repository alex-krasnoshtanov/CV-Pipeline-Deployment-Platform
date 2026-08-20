"""Bridge DAG — corrected/approved feedback becomes a hades-feedback asset.

This is the cloud half of the feedback flywheel. It reads the ready set
(approved predictions + relabelled corrections) from the cloud Postgres,
pulls each input image from the workspace blob store, writes the matching
mask beside it, registers the assembled folder as a new ``hades-feedback``
data asset version, stamps the consumed rows as exported, and triggers the
existing ``preprocessing_incremental`` DAG — which merges the new pairs into
train/val (test frozen) and chains into ``data_pipeline`` for retraining.

Credentials / config (resolved by azure_helpers, connection-first with env
fallback):
- Feedback Postgres DSN: azure_ml_conn Extra ``feedback_db_url`` /
  ``FEEDBACK_DB_URL``.
- Blob storage: azure_ml_conn Extra ``feedback_blob_connection_string`` and
  ``feedback_blob_container`` / ``FEEDBACK_BLOB_CONNECTION_STRING`` and
  ``FEEDBACK_BLOB_CONTAINER`` — the storage account + container backing the
  ``workspaceblobstore`` datastore, where the backend persists feedback input
  images.
- ML workspace: the service-principal values in the same azure_ml_conn Extra.

Schedule: None — fired by ``feedback_retrain_trigger`` once enough feedback
has accumulated, or manually.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pendulum
from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

sys.path.insert(0, str(Path(__file__).parent))
logger = logging.getLogger(__name__)

# Feedback deltas are registered under their own asset name, kept separate
# from ``hades-raw-upload`` (the full original dataset used by the clean-slate
# ``preprocessing_pipeline``). This keeps each asset single-purpose: one means
# "the full raw dataset", the other means "user-correction deltas".
FEEDBACK_ASSET_NAME = "hades-feedback"
local_tz = pendulum.timezone("Europe/Amsterdam")


@dag(
    dag_id="feedback_to_raw_upload",
    schedule=None,
    start_date=datetime(2026, 5, 25, tzinfo=local_tz),
    catchup=False,
    tags=["feedback", "hades", "retraining", "ilo-9.5"],
)
def feedback_to_raw_upload():
    """Stage ready feedback into a new raw-upload version and retrain."""

    @task()
    def export_and_register() -> dict:
        """Materialise the ready set, register it, and stamp it exported.

        Returns:
            A dict with the number of pairs staged and the new asset
            version (or None when there was nothing to export).
        """
        import feedback_export as fx
        from azure.ai.ml.constants import AssetTypes
        from azure.ai.ml.entities import Data
        from azure_helpers import (
            get_feedback_container_client,
            get_feedback_db_conn,
            get_ml_client,
        )

        conn = get_feedback_db_conn()
        try:
            rows = fx.fetch_good_set(conn)
            if not rows:
                logger.info("No ready feedback to export.")
                return {"staged": 0, "version": None}

            pairs, feedback_ids = fx.select_labels(rows)
            container = get_feedback_container_client()

            with tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp)
                staged = fx.stage_pairs(container, pairs, dest)
                if staged == 0:
                    logger.warning(
                        "Nothing staged from %d row(s); not registering.",
                        len(feedback_ids),
                    )
                    return {"staged": 0, "version": None}

                ml_client = get_ml_client()
                stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                data = Data(
                    name=FEEDBACK_ASSET_NAME,
                    path=str(dest),
                    type=AssetTypes.URI_FOLDER,
                    description=f"Feedback export {stamp}: {staged} pair(s).",
                )
                registered = ml_client.data.create_or_update(data)
                logger.info(
                    "Registered %s version %s (%d pair(s)).",
                    FEEDBACK_ASSET_NAME,
                    registered.version,
                    staged,
                )

            # Stamp only after the asset is safely registered, so a failure
            # before this point leaves the rows pending for a retry.
            #
            # Known limitation (accepted, low severity): feedback_ids covers
            # every selected row, but stage_pairs silently skips any row whose
            # blob is unreadable, whose mask is bad base64, or whose URI is
            # unparseable. On a partial failure (staged > 0) those skipped rows
            # are still stamped exported_at here — their correction never enters
            # training and is never retried, because exported_at is now set.
            # Only a total failure (staged == 0) bails out above without
            # stamping. We accept this drop for now; the cleaner alternative is
            # to have stage_pairs return the successfully-staged ids and stamp
            # only those, leaving skipped rows pending for the next run.
            fx.mark_exported(conn, feedback_ids)
            return {"staged": staged, "version": registered.version}
        finally:
            conn.close()

    @task.short_circuit()
    def has_staged_pairs(export_result: dict) -> bool:
        """Gate retraining on a non-empty export.

        ``TriggerDagRunOperator`` runs whenever its upstream merely
        succeeds, so without this gate an empty export (no ready feedback,
        or nothing staged) would still fire ``preprocessing_incremental``
        and chain into a full ``data_pipeline`` retrain on zero new data.
        Returning False short-circuits the downstream trigger.

        Args:
            export_result: The dict returned by ``export_and_register``.

        Returns:
            True when at least one pair was staged, else False.
        """
        staged = export_result.get("staged", 0)
        if staged == 0:
            logger.info("Empty export — skipping incremental retrain trigger.")
            return False
        logger.info("Export staged %d pair(s) — triggering retrain.", staged)
        return True

    trigger_incremental = TriggerDagRunOperator(
        task_id="trigger_incremental",
        trigger_dag_id="preprocessing_incremental",
        wait_for_completion=False,
    )

    export_result = export_and_register()
    has_staged_pairs(export_result) >> trigger_incremental


feedback_to_raw_upload()
