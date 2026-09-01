"""Training pipeline DAG — submits HADES model training to Azure ML.

Requirements covered:
  1. Ingest data from versioned data assets
  2. Preprocess the data
  3. Train a model
  4. Evaluate the model (on held-out test set)
  5. Register the model if it meets evaluation criteria

Hyperparameters: if the hyperparameter_tuning DAG has run, this DAG
reads the winning lr and batch_size from the Airflow Variable
'hades_best_hparams'. If the Variable is absent or malformed, it
falls back to the hardcoded defaults below.

Schedule: Every Monday at 2 AM.
Steps 1–5 execute on the Azure ML compute cluster (lambda-0).
Airflow is the orchestrator — it submits the job and monitors it.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pendulum
from airflow.decorators import dag, task
from airflow.models import Variable

sys.path.insert(0, str(Path(__file__).parent))
logger = logging.getLogger(__name__)

# --- Configuration ---
COMPUTE_NAME = "lambda-0"
ENVIRONMENT_NAME = "cv-pipeline-training"
TRAIN_ASSET_NAME = "hades-train"
VAL_ASSET_NAME = "hades-val"
TEST_ASSET_NAME = "hades-test"
MODEL_NAME = "hades-unet"
MIN_F1 = 0.75
EPOCHS = 50
TRAINING_CODE_DIR = str(Path(__file__).parent.parent / "training_code")

# Champion-challenger promotion gate (decide-only until ENABLE_AUTO_DEPLOY=True).
PROMOTE_MARGIN = 0.005  # candidate must beat champion F1 by at least this much
ENABLE_AUTO_DEPLOY = True

# Fallback hyperparameters — used only if the tuning Variable is absent.
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 3.929e-4
HPARAMS_VARIABLE = "hades_best_hparams"
local_tz = pendulum.timezone("Europe/Amsterdam")


@dag(
    dag_id="data_pipeline",
    schedule="0 2 * * 1",
    start_date=datetime(2026, 5, 25, tzinfo=local_tz),
    catchup=False,
    tags=["training", "hades", "data-pipeline"],
)
def data_pipeline():
    """Submit HADES training job to Azure ML and monitor completion."""

    @task()
    def check_connection() -> str:
        """Verify Azure ML workspace is reachable.

        Returns:
            Workspace name if connection succeeds.
        """
        from azure_helpers import get_ml_client

        ml_client = get_ml_client()
        ws = ml_client.workspaces.get(ml_client.workspace_name)
        logger.info("Connected to workspace: %s", ws.name)
        return ws.name

    @task()
    def resolve_hparams() -> dict:
        """Read tuned hyperparameters from the Airflow Variable.

        Falls back to the hardcoded defaults if the Variable is missing,
        empty, or cannot be parsed.

        Returns:
            Dict with 'lr' and 'batch_size'.
        """
        raw = Variable.get(HPARAMS_VARIABLE, default_var=None)

        if not raw:
            logger.info(
                "Variable '%s' not set — using defaults (lr=%.6g, batch_size=%d).",
                HPARAMS_VARIABLE,
                DEFAULT_LEARNING_RATE,
                DEFAULT_BATCH_SIZE,
            )
            return {"lr": DEFAULT_LEARNING_RATE, "batch_size": DEFAULT_BATCH_SIZE}

        try:
            data = json.loads(raw)
            lr = float(data["lr"])
            batch_size = int(data["batch_size"])
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "Could not parse Variable '%s' (%s) — using defaults.",
                HPARAMS_VARIABLE,
                exc,
            )
            return {"lr": DEFAULT_LEARNING_RATE, "batch_size": DEFAULT_BATCH_SIZE}

        logger.info(
            "Using tuned hyperparameters from '%s': lr=%.6g, batch_size=%d.",
            HPARAMS_VARIABLE,
            lr,
            batch_size,
        )
        return {"lr": lr, "batch_size": batch_size}

    @task()
    def resolve_versions() -> dict:
        """Query Azure ML for the latest version of each data asset and environment.

        Returns:
            Dict with asset/environment names mapped to their latest version numbers.
        """
        from azure_helpers import get_ml_client

        ml_client = get_ml_client()
        versions = {}

        for asset_name in [TRAIN_ASSET_NAME, VAL_ASSET_NAME, TEST_ASSET_NAME]:
            all_versions = list(ml_client.data.list(name=asset_name))
            if not all_versions:
                raise RuntimeError(
                    f"No versions found for data asset '{asset_name}' "
                    f"in workspace '{ml_client.workspace_name}'."
                )
            latest = max(all_versions, key=lambda d: int(d.version))
            versions[asset_name] = latest.version
            logger.info("Resolved %s to version %s.", asset_name, latest.version)

        all_envs = list(ml_client.environments.list(name=ENVIRONMENT_NAME))
        if not all_envs:
            raise RuntimeError(
                f"No versions found for environment '{ENVIRONMENT_NAME}'."
            )
        latest_env = max(all_envs, key=lambda e: int(e.version))
        versions["environment"] = latest_env.version
        logger.info("Resolved environment to version %s.", latest_env.version)

        return versions

    @task()
    def submit_training_job(resolved: dict, hparams: dict) -> str:
        """Submit a command job to Azure ML that trains and evaluates the model.

        Args:
            resolved: Dict mapping asset/environment names to version strings.
            hparams: Dict with 'lr' and 'batch_size'.

        Returns:
            The Azure ML job name (used to poll status).
        """
        from azure.ai.ml import Input, command
        from azure.ai.ml.constants import AssetTypes
        from azure_helpers import get_ml_client

        ml_client = get_ml_client()

        train_v = resolved[TRAIN_ASSET_NAME]
        val_v = resolved[VAL_ASSET_NAME]
        test_v = resolved[TEST_ASSET_NAME]
        env_v = resolved["environment"]

        lr = hparams["lr"]
        batch_size = hparams["batch_size"]

        logger.info(
            "Submitting job — train:%s, val:%s, test:%s, env:%s, "
            "lr:%.6g, batch_size:%d.",
            train_v,
            val_v,
            test_v,
            env_v,
            lr,
            batch_size,
        )

        job = command(
            code=TRAINING_CODE_DIR,
            command=(
                "python cloud_train.py "
                "--train-dir ${{inputs.train_data}} "
                "--val-dir ${{inputs.val_data}} "
                "--test-dir ${{inputs.test_data}} "
                "--output-dir ./outputs "
                f"--epochs {EPOCHS} "
                f"--batch-size {batch_size} "
                f"--lr {lr} "
                f"--min-f1 {MIN_F1} "
                f"--model-name {MODEL_NAME}"
            ),
            inputs={
                "train_data": Input(
                    type=AssetTypes.URI_FOLDER,
                    path=f"azureml:{TRAIN_ASSET_NAME}:{train_v}",
                ),
                "val_data": Input(
                    type=AssetTypes.URI_FOLDER,
                    path=f"azureml:{VAL_ASSET_NAME}:{val_v}",
                ),
                "test_data": Input(
                    type=AssetTypes.URI_FOLDER,
                    path=f"azureml:{TEST_ASSET_NAME}:{test_v}",
                ),
            },
            environment=f"azureml:{ENVIRONMENT_NAME}:{env_v}",
            compute=COMPUTE_NAME,
            resources={"instance_type": "gpu"},
            display_name="airflow-weekly-training",
        )

        submitted = ml_client.jobs.create_or_update(job)
        logger.info("Job submitted: %s (status: %s)", submitted.name, submitted.status)
        return submitted.name

    @task()
    def wait_for_completion(job_name: str) -> dict:
        """Poll the Azure ML job until it completes.

        Args:
            job_name: Name of the submitted job.

        Returns:
            Dict with job status and metrics URL.
        """
        import time

        from azure_helpers import get_ml_client

        ml_client = get_ml_client()
        terminal_states = {"Completed", "Failed", "Canceled"}
        poll_interval = 60

        while True:
            job = ml_client.jobs.get(job_name)
            logger.info("Job %s — status: %s", job_name, job.status)

            if job.status in terminal_states:
                break
            time.sleep(poll_interval)

        if job.status != "Completed":
            raise RuntimeError(
                f"Azure ML job {job_name} ended with status: {job.status}. "
                f"Check the Azure ML UI for details."
            )

        studio_url = job.studio_url or "N/A"
        logger.info("Job %s completed. Studio URL: %s", job_name, studio_url)

        return {
            "job_name": job_name,
            "status": job.status,
            "studio_url": studio_url,
        }

    @task()
    def resolve_promotion(train_result: dict) -> dict:
        """Resolve champion + candidate; skip if they are the same version."""
        from airflow.exceptions import AirflowSkipException
        from azure_helpers import get_ml_client
        from promotion import resolve_candidate_version, resolve_champion_version

        ml_client = get_ml_client()
        champion = resolve_champion_version(ml_client)
        candidate = resolve_candidate_version(ml_client)
        if candidate == champion:
            raise AirflowSkipException(
                f"Candidate v{candidate} is already the champion — nothing to do."
            )
        logger.info("Promotion candidate v%s vs champion v%s.", candidate, champion)
        return {"champion": champion, "candidate": candidate}

    @task()
    def run_smoke_eval(versions: dict) -> str:
        """Submit the smoke-eval job and return its job name."""
        from azure_helpers import get_ml_client
        from promotion import build_smoke_job

        ml_client = get_ml_client()
        job = build_smoke_job(
            ml_client,
            versions["champion"],
            versions["candidate"],
            margin=PROMOTE_MARGIN,
        )
        submitted = ml_client.jobs.create_or_update(job)
        logger.info("Smoke-eval job submitted: %s", submitted.name)
        return submitted.name

    @task()
    def evaluate_and_decide(job_name: str) -> dict:
        """Wait for the smoke job, read the verdict, and act on it.

        On a winning verdict with ENABLE_AUTO_DEPLOY set, deploys the candidate
        (creates the deployment and flips 100% traffic). Otherwise logs only.
        """
        from azure_helpers import get_ml_client
        from promotion import deploy_candidate, read_verdict, wait_for_job

        ml_client = get_ml_client()
        wait_for_job(ml_client, job_name)
        verdict = read_verdict(ml_client, job_name)

        if not verdict["promote"]:
            logger.info(
                "SKIP: candidate v%s F1=%.4f did not beat champion v%s F1=%.4f "
                "by margin %.3f.",
                verdict["candidate_version"],
                verdict["candidate_f1"],
                verdict["champion_version"],
                verdict["champion_f1"],
                verdict["margin"],
            )
            return verdict

        if not ENABLE_AUTO_DEPLOY:
            logger.info(
                "WOULD PROMOTE (auto-deploy disabled): candidate v%s F1=%.4f.",
                verdict["candidate_version"],
                verdict["candidate_f1"],
            )
            return verdict

        deployment = deploy_candidate(ml_client, verdict["candidate_version"])
        logger.info(
            "PROMOTED: 100%% traffic -> %s (candidate v%s F1=%.4f beat "
            "champion v%s F1=%.4f).",
            deployment,
            verdict["candidate_version"],
            verdict["candidate_f1"],
            verdict["champion_version"],
            verdict["champion_f1"],
        )
        return verdict

    connection = check_connection()
    hparams = resolve_hparams()
    versions = resolve_versions()
    job_name = submit_training_job(versions, hparams)
    result = wait_for_completion(job_name)

    connection >> hparams >> versions >> job_name >> result

    promotion_versions = resolve_promotion(result)
    smoke_job = run_smoke_eval(promotion_versions)
    evaluate_and_decide(smoke_job)


data_pipeline()
