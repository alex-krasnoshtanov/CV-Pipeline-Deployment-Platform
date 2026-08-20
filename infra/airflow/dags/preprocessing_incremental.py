"""Incremental preprocessing DAG — merges new data into existing assets.

Unlike preprocessing_pipeline (clean-slate fresh split), this DAG
preserves a frozen test set so models stay comparable across versions.

Flow:
  1. New images arrive as the hades-feedback data asset (written by the
     feedback_to_raw_upload bridge from user corrections/approvals)
  2. The bridge triggers this DAG automatically after registering them
  3. DAG submits cloud_preprocess_incremental.py to Azure ML, passing
     the latest train/val/test assets to merge against
  4. Job: patches new images, keeps test frozen, re-splits train+val
  5. Registers output as new versioned hades-train/val/test assets
  6. Triggers the training pipeline

No direct blob storage access — everything goes through Azure ML SDK.

Job output is pinned to workspaceblobstore so the resulting data assets
are mountable as inputs to downstream training and sweep jobs. Writing
to the default artifact store produces ExperimentRun paths that cannot
be mounted as job inputs.

Trigger with config to pin specific versions:
    {"raw_version": "3", "train_version": "8", "val_version": "8", "test_version": "8"}

If no config is provided, uses the latest version of each asset.

Schedule: None (triggered by register_raw_data.py on upload).
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
# The incremental delta now comes from the dedicated feedback asset, not the
# overloaded hades-raw-upload (which remains the full original dataset used by
# the clean-slate preprocessing_pipeline).
RAW_ASSET_NAME = "hades-feedback"
TRAIN_ASSET = "hades-train"
VAL_ASSET = "hades-val"
TEST_ASSET = "hades-test"
TRAINING_CODE_DIR = str(Path(__file__).parent.parent / "training_code")

# Base datastore path where job outputs are written. Pinned to
# workspaceblobstore (mountable) rather than the default artifact store.
# Separate subfolder from clean-slate runs to avoid collisions.
OUTPUT_BASE = "azureml://datastores/workspaceblobstore/paths/preprocessed/incremental"
local_tz = pendulum.timezone("Europe/Amsterdam")


def _resolve_latest(ml_client, asset_name: str) -> str:
    """Return the highest version number for an asset as a string.

    Args:
        ml_client: Azure ML client.
        asset_name: Name of the data asset.

    Returns:
        The latest version string.

    Raises:
        RuntimeError: If no versions exist.
    """
    all_versions = list(ml_client.data.list(name=asset_name))
    if not all_versions:
        raise RuntimeError(
            f"No versions found for '{asset_name}'. "
            "The incremental pipeline requires existing assets to merge against. "
            "Run the clean-slate preprocessing_pipeline first."
        )
    latest = max(all_versions, key=lambda d: int(d.version))
    return latest.version


@dag(
    dag_id="preprocessing_incremental",
    schedule=None,
    start_date=datetime(2026, 5, 25, tzinfo=local_tz),
    catchup=False,
    tags=["preprocessing", "hades", "incremental", "ilo-9.4"],
    params={
        "raw_version": "latest",
        "train_version": "latest",
        "val_version": "latest",
        "test_version": "latest",
    },
)
def preprocessing_incremental():
    """Merge new HADES images into existing assets, keeping test frozen."""

    @task()
    def check_connection() -> str:
        """Verify Azure ML workspace is reachable."""
        from azure_helpers import get_ml_client

        ml_client = get_ml_client()
        ws = ml_client.workspaces.get(ml_client.workspace_name)
        logger.info("Connected to workspace: %s", ws.name)
        return ws.name

    @task()
    def resolve_assets(**context) -> dict:
        """Resolve raw, train, val, and test asset versions to use.

        Uses versions from DAG config if provided, otherwise the
        latest version of each asset.

        Returns:
            Dict of asset URIs keyed by role (raw, train, val, test).
        """
        from azure_helpers import get_ml_client

        ml_client = get_ml_client()
        params = context["params"]

        def pick(asset_name: str, param_key: str) -> str:
            requested = params.get(param_key, "latest")
            version = (
                _resolve_latest(ml_client, asset_name)
                if requested == "latest"
                else requested
            )
            uri = f"azureml:{asset_name}:{version}"
            logger.info("Using %s -> %s", param_key, uri)
            return uri

        uris = {
            "raw": pick(RAW_ASSET_NAME, "raw_version"),
            "train": pick(TRAIN_ASSET, "train_version"),
            "val": pick(VAL_ASSET, "val_version"),
            "test": pick(TEST_ASSET, "test_version"),
        }
        return uris

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
    def submit_preprocessing_job(uris: dict, env_uri: str) -> str:
        """Submit cloud_preprocess_incremental.py to Azure ML.

        Merges new raw images into the existing train+val pool,
        keeps the test set frozen, and re-splits train/val.
        Output is pinned to workspaceblobstore so it is mountable later.

        Args:
            uris: Dict of asset URIs (raw, train, val, test).
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
                "python cloud_preprocess_incremental.py "
                "--raw-dir ${{inputs.raw_data}} "
                "--existing-train-dir ${{inputs.existing_train}} "
                "--existing-val-dir ${{inputs.existing_val}} "
                "--existing-test-dir ${{inputs.existing_test}} "
                "--output-dir ${{outputs.processed}}"
            ),
            inputs={
                "raw_data": Input(type=AssetTypes.URI_FOLDER, path=uris["raw"]),
                "existing_train": Input(type=AssetTypes.URI_FOLDER, path=uris["train"]),
                "existing_val": Input(type=AssetTypes.URI_FOLDER, path=uris["val"]),
                "existing_test": Input(type=AssetTypes.URI_FOLDER, path=uris["test"]),
            },
            outputs={
                "processed": Output(
                    type="uri_folder",
                    path=f"{OUTPUT_BASE}/${{{{name}}}}/",
                ),
            },
            environment=env_uri,
            compute=COMPUTE_NAME,
            display_name="preprocess-hades-incremental",
        )

        submitted = ml_client.jobs.create_or_update(job)
        logger.info("Incremental preprocessing job submitted: %s", submitted.name)
        return submitted.name

    @task(execution_timeout=timedelta(hours=26), retries=0)
    def wait_for_completion(job_name: str) -> str:
        """Poll until the preprocessing job completes.

        Tolerates transient connection errors during the poll loop so
        a single network blip does not kill a multi-hour wait. Rebuilds
        the ML client each poll to refresh the auth token.

        Args:
            job_name: Name of the submitted Azure ML job.

        Returns:
            The job name once completed.

        Raises:
            RuntimeError: If the job ends in Failed or Canceled, or if
                connection is lost for too many consecutive polls.
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
                description=f"Incremental merge by job {job_name} (test frozen).",
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
    uris = resolve_assets()
    env_uri = resolve_environment()
    job_name = submit_preprocessing_job(uris, env_uri)
    completed = wait_for_completion(job_name)
    new_assets = register_data_assets(completed)

    conn >> uris >> env_uri >> job_name >> completed >> new_assets >> trigger_training


preprocessing_incremental()
