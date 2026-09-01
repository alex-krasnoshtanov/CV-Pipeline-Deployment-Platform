"""Preprocessing pipeline DAG — processes raw data into training patches.

Flow:
  1. User registers raw images as a data asset (hades-raw-upload)
  2. User triggers this DAG (manually or on schedule)
  3. DAG submits cloud_preprocess.py to Azure ML
  4. Registers output as new versioned hades-train/val/test assets
  5. Triggers the training pipeline

No direct blob storage access — everything goes through Azure ML SDK.

Job output is pinned to workspaceblobstore so the resulting data assets
are mountable as inputs to downstream training and sweep jobs. Writing
to the default artifact store produces ExperimentRun paths that cannot
be mounted as job inputs.

Trigger with config to specify a raw data version:
    {"raw_version": "3"}

If no config is provided, uses the latest hades-raw-upload version.

Schedule: None (triggered manually after uploading raw data).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pendulum
from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

sys.path.insert(0, str(Path(__file__).parent))
logger = logging.getLogger(__name__)

COMPUTE_NAME = "lambda-4"
ENVIRONMENT_NAME = "cv-pipeline-training"
RAW_ASSET_NAME = "hades-raw-upload"
TRAIN_ASSET = "hades-train"
VAL_ASSET = "hades-val"
TEST_ASSET = "hades-test"
TRAINING_CODE_DIR = str(Path(__file__).parent.parent / "training_code")

# Base datastore path where job outputs are written. Pinned to
# workspaceblobstore (mountable) rather than the default artifact store.
OUTPUT_BASE = "azureml://datastores/workspaceblobstore/paths/preprocessed/clean-slate"
local_tz = pendulum.timezone("Europe/Amsterdam")


@dag(
    dag_id="preprocessing_pipeline",
    schedule=None,
    start_date=datetime(2026, 5, 25, tzinfo=local_tz),
    catchup=False,
    tags=["preprocessing", "hades"],
    params={"raw_version": "latest"},
)
def preprocessing_pipeline():
    """Process raw HADES images into training patches."""

    @task()
    def check_connection() -> str:
        """Verify Azure ML workspace is reachable."""
        from azure_helpers import get_ml_client

        ml_client = get_ml_client()
        ws = ml_client.workspaces.get(ml_client.workspace_name)
        logger.info("Connected to workspace: %s", ws.name)
        return ws.name

    @task()
    def resolve_raw_asset(**context) -> str:
        """Resolve which raw data asset version to process.

        Uses the version from DAG config if provided,
        otherwise finds the latest hades-raw-upload version.

        Returns:
            Asset URI string (azureml:hades-raw-upload:N).
        """
        from azure_helpers import get_ml_client

        ml_client = get_ml_client()
        params = context["params"]
        raw_version = params.get("raw_version", "latest")

        if raw_version != "latest":
            uri = f"azureml:{RAW_ASSET_NAME}:{raw_version}"
            logger.info("Using specified raw asset: %s", uri)
            return uri

        all_versions = list(ml_client.data.list(name=RAW_ASSET_NAME))
        if not all_versions:
            raise RuntimeError(
                f"No versions found for '{RAW_ASSET_NAME}'. "
                "Register raw data first with register_raw_data.py."
            )
        latest = max(all_versions, key=lambda d: int(d.version))
        uri = f"azureml:{RAW_ASSET_NAME}:{latest.version}"
        logger.info("Resolved latest raw asset: %s", uri)
        return uri

    @task()
    def resolve_environment() -> str:
        """Get the latest training environment version."""
        from azure_helpers import get_ml_client

        ml_client = get_ml_client()
        all_envs = list(ml_client.environments.list(name=ENVIRONMENT_NAME))
        if not all_envs:
            raise RuntimeError(
                f"No versions found for environment '{ENVIRONMENT_NAME}'."
            )
        latest = max(all_envs, key=lambda e: int(e.version))
        uri = f"azureml:{ENVIRONMENT_NAME}:{latest.version}"
        logger.info("Using environment: %s", uri)
        return uri

    @task()
    def submit_preprocessing_job(raw_asset_uri: str, env_uri: str) -> str:
        """Submit cloud_preprocess.py to Azure ML.

        Processes raw images into 256x256 patches with 70/20/10 split.
        Does NOT merge with existing data — fresh split from raw only.
        Output is pinned to workspaceblobstore so it is mountable later.

        Args:
            raw_asset_uri: URI of the raw data asset.
            env_uri: URI of the environment.

        Returns:
            The Azure ML job name.
        """
        from azure.ai.ml import Input, Output, command
        from azure.ai.ml.constants import AssetTypes
        from azure_helpers import get_ml_client

        ml_client = get_ml_client()

        job = command(
            code=TRAINING_CODE_DIR,
            command=(
                "python cloud_preprocess.py "
                "--raw-dir ${{inputs.raw_data}} "
                "--output-dir ${{outputs.processed}}"
            ),
            inputs={
                "raw_data": Input(
                    type=AssetTypes.URI_FOLDER,
                    path=raw_asset_uri,
                ),
            },
            outputs={
                "processed": Output(
                    type="uri_folder",
                    path=f"{OUTPUT_BASE}/${{{{name}}}}/",
                ),
            },
            environment=env_uri,
            compute=COMPUTE_NAME,
            display_name="preprocess-hades-raw",
        )

        submitted = ml_client.jobs.create_or_update(job)
        logger.info("Preprocessing job submitted: %s", submitted.name)
        return submitted.name

    @task(execution_timeout=timedelta(hours=26), retries=0)
    def wait_for_completion(job_name: str) -> str:
        """Poll until the preprocessing job completes.

        Tolerates transient connection errors during the poll loop
        so a single network blip does not kill a multi-hour wait.

        Args:
            job_name: Name of the submitted Azure ML job.

        Returns:
            The job name once completed.

        Raises:
            RuntimeError: If the job ends in Failed or Canceled.
        """
        import time

        from azure_helpers import get_ml_client

        terminal_states = {"Completed", "Failed", "Canceled"}
        poll_interval = 120
        consecutive_errors = 0
        max_consecutive_errors = 10
        job = None

        while True:
            try:
                ml_client = get_ml_client()
                job = ml_client.jobs.get(job_name)
                consecutive_errors = 0
                logger.info("Job %s — status: %s", job_name, job.status)
                if job.status in terminal_states:
                    break
            except Exception as exc:
                consecutive_errors += 1
                logger.warning(
                    "Poll error (%d/%d) for %s: %s",
                    consecutive_errors,
                    max_consecutive_errors,
                    job_name,
                    exc,
                )
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"Lost connection to Azure ML for {job_name} "
                        f"after {max_consecutive_errors} attempts."
                    ) from exc

            time.sleep(poll_interval)

        if job.status != "Completed":
            raise RuntimeError(
                f"Preprocessing job {job_name} ended with status: {job.status}."
            )

        logger.info("Preprocessing job %s completed.", job_name)
        return job_name

    @task()
    def register_data_assets(job_name: str) -> dict:
        """Register output splits as new versioned data assets.

        The output path is reconstructed deterministically from the
        pinned datastore location plus the job name, rather than read
        from job.outputs (which is not reliably populated for finished
        jobs).

        Args:
            job_name: Completed preprocessing job name.

        Returns:
            Dict mapping asset names to their new version numbers.
        """
        from azure.ai.ml.constants import AssetTypes
        from azure.ai.ml.entities import Data
        from azure_helpers import get_ml_client

        ml_client = get_ml_client()

        base_path = f"{OUTPUT_BASE}/{job_name}"

        new_versions = {}
        for split in ("train", "val", "test"):
            asset_name = f"hades-{split}"
            data = Data(
                name=asset_name,
                path=f"{base_path}/{split}",
                type=AssetTypes.URI_FOLDER,
                description=f"Preprocessed by job {job_name} (clean slate).",
            )
            registered = ml_client.data.create_or_update(data)
            new_versions[asset_name] = registered.version
            logger.info("Registered %s version %s.", asset_name, registered.version)

        return new_versions

    trigger_training = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id="data_pipeline",
        wait_for_completion=False,
    )

    # Task chain
    conn = check_connection()
    raw_uri = resolve_raw_asset()
    env_uri = resolve_environment()
    job_name = submit_preprocessing_job(raw_uri, env_uri)
    completed = wait_for_completion(job_name)
    new_assets = register_data_assets(completed)

    (
        conn
        >> raw_uri
        >> env_uri
        >> job_name
        >> completed
        >> new_assets
        >> trigger_training
    )


preprocessing_pipeline()
