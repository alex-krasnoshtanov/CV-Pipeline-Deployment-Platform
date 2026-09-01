"""Trigger DAG — fire the feedback bridge when enough feedback accumulates.

Runs daily, counts the ready set (approved + relabelled feedback not yet
exported) in the cloud Postgres, and triggers ``feedback_to_raw_upload``
once it reaches the retrain threshold. The bridge then stages the data,
registers a new ``hades-raw-upload`` version, and chains into
``preprocessing_incremental`` and ``data_pipeline`` for retraining.

This is the "triggered by new data or user feedback" half of the creative
brief's retraining requirement; the weekly half is owned by
``data_pipeline``'s schedule, so the model retrains on a weekly basis or
when enough new feedback accumulates, whichever comes first.

Credentials / config (resolved by azure_helpers, connection-first with env
fallback):
- Feedback Postgres DSN: ``get_feedback_db_conn`` (azure_ml_conn Extra
  ``FEEDBACK_DB_URL`` / env ``FEEDBACK_DB_URL``).
- Threshold: ``get_retrain_threshold`` (azure_ml_conn Extra
  ``RETRAIN_FEEDBACK_THRESHOLD`` / env of the same name / default 50).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pendulum
from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

sys.path.insert(0, str(Path(__file__).parent))
logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 50
local_tz = pendulum.timezone("Europe/Amsterdam")


@dag(
    dag_id="feedback_retrain_trigger",
    schedule="0 3 * * *",
    start_date=datetime(2026, 5, 25, tzinfo=local_tz),
    catchup=False,
    tags=["feedback", "hades", "retraining"],
)
def feedback_retrain_trigger():
    """Daily check that fires the bridge when feedback crosses threshold."""

    @task.short_circuit()
    def threshold_met() -> bool:
        """Return True when the ready set meets the configured threshold.

        Returning False short-circuits the DAG, so the bridge is not
        triggered until enough feedback has accumulated.
        """
        import feedback_export as fx
        from azure_helpers import get_feedback_db_conn, get_retrain_threshold

        threshold = get_retrain_threshold(default=DEFAULT_THRESHOLD)
        conn = get_feedback_db_conn()
        try:
            pending = fx.count_pending(conn)
        finally:
            conn.close()

        logger.info("Ready feedback: %d (threshold %d).", pending, threshold)
        return pending >= threshold

    fire_bridge = TriggerDagRunOperator(
        task_id="fire_bridge",
        trigger_dag_id="feedback_to_raw_upload",
        wait_for_completion=False,
    )

    threshold_met() >> fire_bridge


feedback_retrain_trigger()
