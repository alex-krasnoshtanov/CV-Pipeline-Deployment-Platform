"""Hyperparameter tuning DAG — submits a sweep job to Azure ML.

Covers Creative Brief Sprint 3 requirement:
  "Create a pipeline for automated hyperparameter tuning." (ILO 8.9B)

After the sweep completes, the winning hyperparameters are extracted
and written to the Airflow Variable 'hades_best_hparams'. The training
DAG (data_pipeline) reads this Variable automatically, so no manual
copying of values is needed.

Trigger: Manual only. Run this when you want to find better
hyperparameters; data_pipeline picks them up on its next run.
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
MODEL_NAME = "hades-unet"
MIN_F1 = 0.75
EPOCHS = 30
NUM_WORKERS = 4
MAX_TOTAL_TRIALS = 10
MAX_CONCURRENT_TRIALS = 2
HPARAMS_VARIABLE = "hades_best_hparams"
PRIMARY_METRIC = "best_val_f1"
local_tz = pendulum.timezone("Europe/Amsterdam")
TRAINING_CODE_DIR = str(Path(__file__).parent.parent / "training_code")


@dag(
    dag_id="hyperparameter_tuning",
    schedule=None,
    start_date=datetime(2026, 5, 25, tzinfo=local_tz),
    catchup=False,
    tags=["tuning", "hades", "ilo-8.9"],
)
def hyperparameter_tuning():
    """Submit a sweep job to Azure ML to find optimal hyperparameters."""

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
    def resolve_versions() -> dict:
        """Query Azure ML for the latest version of each data asset and environment.

        Returns:
            Dict with asset/environment names mapped to their latest version numbers.
        """
        from azure_helpers import get_ml_client

        ml_client = get_ml_client()
        versions = {}

        for asset_name in [TRAIN_ASSET_NAME, VAL_ASSET_NAME]:
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
    def submit_sweep_job(resolved: dict) -> str:
        """Submit a sweep job that tunes learning rate and batch size.

        Args:
            resolved: Dict mapping asset/environment names to version strings.

        Returns:
            The Azure ML job name (used to poll status).
        """
        from azure.ai.ml import Input, command
        from azure.ai.ml.constants import AssetTypes
        from azure.ai.ml.sweep import BanditPolicy, Choice, LogUniform
        from azure_helpers import get_ml_client

        ml_client = get_ml_client()

        train_v = resolved[TRAIN_ASSET_NAME]
        val_v = resolved[VAL_ASSET_NAME]
        env_v = resolved["environment"]

        logger.info(
            "Submitting sweep — train:%s, val:%s, env:%s.",
            train_v,
            val_v,
            env_v,
        )

        command_job = command(
            code=TRAINING_CODE_DIR,
            command=(
                "python cloud_train.py "
                "--train-dir ${{inputs.train_data}} "
                "--val-dir ${{inputs.val_data}} "
                "--output-dir ./outputs "
                f"--epochs {EPOCHS} "
                "--batch-size ${{search_space.batch_size}} "
                "--lr ${{search_space.lr}} "
                f"--num-workers {NUM_WORKERS} "
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
            },
            environment=f"azureml:{ENVIRONMENT_NAME}:{env_v}",
            compute=COMPUTE_NAME,
            resources={"instance_type": "gpu"},
        )

        sweep_job = command_job.sweep(
            sampling_algorithm="random",
            primary_metric=PRIMARY_METRIC,
            goal="maximize",
        )

        sweep_job.search_space = {
            "lr": LogUniform(min_value=-11.5, max_value=-4.6),
            "batch_size": Choice(values=[8, 16, 32]),
        }

        sweep_job.limits.max_total_trials = MAX_TOTAL_TRIALS
        sweep_job.limits.max_concurrent_trials = MAX_CONCURRENT_TRIALS

        sweep_job.early_termination = BanditPolicy(
            slack_factor=0.15,
            evaluation_interval=1,
        )

        sweep_job.display_name = "airflow-hyperparameter-sweep"

        submitted = ml_client.jobs.create_or_update(sweep_job)
        logger.info(
            "Sweep submitted: %s (status: %s)", submitted.name, submitted.status
        )
        return submitted.name

    @task()
    def wait_for_completion(job_name: str) -> str:
        """Poll the Azure ML sweep job until it completes.

        Args:
            job_name: Name of the submitted sweep job.

        Returns:
            The sweep job name once completed.

        Raises:
            RuntimeError: If the sweep ends in Failed or Canceled.
        """
        import time

        from azure_helpers import get_ml_client

        ml_client = get_ml_client()
        terminal_states = {"Completed", "Failed", "Canceled"}
        poll_interval = 120

        while True:
            job = ml_client.jobs.get(job_name)
            logger.info("Sweep %s — status: %s", job_name, job.status)

            if job.status in terminal_states:
                break
            time.sleep(poll_interval)

        if job.status != "Completed":
            raise RuntimeError(
                f"Sweep job {job_name} ended with status: {job.status}. "
                f"Check the Azure ML UI for details."
            )

        logger.info("Sweep %s completed.", job_name)
        return job_name

    @task()
    def extract_and_store_best(sweep_job_name: str) -> dict:
        """Find the best trial via MLflow and store its hyperparameters.

        Searches the child runs of the completed sweep, selects the run
        with the highest primary metric, and writes the winning
        hyperparameters to the Airflow Variable 'hades_best_hparams'
        as a JSON string. The training DAG reads this automatically.

        Args:
            sweep_job_name: Name of the completed sweep (the parent run).

        Returns:
            Dict with the winning lr, batch_size, and metric value.

        Raises:
            RuntimeError: If no completed child runs with the metric are found.
        """
        import mlflow
        from azure_helpers import get_ml_client

        # Point MLflow at the Azure ML workspace tracking URI.
        ml_client = get_ml_client()
        tracking_uri = ml_client.workspaces.get(
            ml_client.workspace_name
        ).mlflow_tracking_uri
        mlflow.set_tracking_uri(tracking_uri)

        # Child runs of the sweep carry the sweep run id as parentRunId.
        filter_string = f"tags.mlflow.parentRunId = '{sweep_job_name}'"
        runs = mlflow.search_runs(
            experiment_names=None,
            filter_string=filter_string,
            search_all_experiments=True,
            order_by=[f"metrics.{PRIMARY_METRIC} DESC"],
        )

        if runs is None or len(runs) == 0:
            raise RuntimeError(
                f"No child runs found for sweep '{sweep_job_name}' "
                f"with parentRunId tag. Cannot extract best hyperparameters."
            )

        metric_col = f"metrics.{PRIMARY_METRIC}"
        if metric_col not in runs.columns:
            raise RuntimeError(
                f"Metric '{PRIMARY_METRIC}' not found in child runs. "
                f"Available columns: {list(runs.columns)}"
            )

        runs = runs.dropna(subset=[metric_col])
        if len(runs) == 0:
            raise RuntimeError(f"No child runs logged metric '{PRIMARY_METRIC}'.")

        best = runs.iloc[0]

        # Param columns logged by cloud_train.py via mlflow.log_params.
        lr_col = "params.learning_rate"
        bs_col = "params.batch_size"
        if lr_col not in runs.columns or bs_col not in runs.columns:
            raise RuntimeError(
                f"Expected param columns '{lr_col}' and '{bs_col}' not found. "
                f"Available columns: {list(runs.columns)}"
            )

        best_lr = float(best[lr_col])
        best_bs = int(float(best[bs_col]))
        best_metric = float(best[metric_col])

        payload = {
            "lr": best_lr,
            "batch_size": best_bs,
            PRIMARY_METRIC: best_metric,
            "sweep_job": sweep_job_name,
            "stored_at": datetime.now(tz=local_tz).isoformat(),
        }

        Variable.set(HPARAMS_VARIABLE, json.dumps(payload))
        logger.info(
            "Stored best hyperparameters to Variable '%s': lr=%.6g, "
            "batch_size=%d, %s=%.4f.",
            HPARAMS_VARIABLE,
            best_lr,
            best_bs,
            PRIMARY_METRIC,
            best_metric,
        )

        return payload

    connection = check_connection()
    versions = resolve_versions()
    job_name = submit_sweep_job(versions)
    completed = wait_for_completion(job_name)
    best = extract_and_store_best(completed)

    connection >> versions >> job_name >> completed >> best


hyperparameter_tuning()
