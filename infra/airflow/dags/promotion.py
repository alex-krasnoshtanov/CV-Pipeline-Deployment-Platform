"""Promotion helpers for the data_pipeline DAG (champion-challenger gate).

Plain functions kept out of the DAG file so they are unit-testable and the
DAG stays a thin wiring layer. They cover the full path:

- resolve the champion (model version currently serving the most traffic) and
  the candidate (latest registered version),
- build + submit the Azure ML smoke-eval job that scores both on hades-smoke,
- wait for it and read the verdict,
- deploy the candidate when promoted: create the deployment, health-check it
  at 0% traffic (blue-green), then flip traffic with a split that always sums
  to 100.

Lives in dags/ so the DAG imports it as a sibling (`from promotion import ...`),
the same pattern data_pipeline.py uses for azure_helpers. Azure SDK imports are
local to each function so this module imports cleanly (and stays testable)
without azure-ai-ml present. compute_traffic_split is pure and unit-tested.
"""

from __future__ import annotations

import glob
import json
import logging
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ENDPOINT_NAME = "hades-unet-endpoint"
MODEL_NAME = "hades-unet"
SMOKE_ASSET = "hades-smoke"
COMPUTE_NAME = "lambda-0"
TRAINING_ENV = "cv-pipeline-training"  # smoke job: has torch + cv_pipeline
INFERENCE_ENV = "cv-pipeline-inference"  # deployment: the endpoint's env

# smoke_eval.py lives in infra/airflow/smoke_code/, a sibling of dags/.
SMOKE_CODE_DIR = str(Path(__file__).parent.parent / "smoke_code")
# score.py + sample_request.json live in infra/cloud/endpoint/ (root = three up).
SCORE_DIR = str(Path(__file__).parent.parent.parent / "cloud" / "endpoint")

DEFAULT_MARGIN = 0.005


def resolve_champion_version(ml_client) -> str:
    """Return the model version currently serving the most traffic (champion).

    The endpoint's traffic dict maps deployment name -> percent. The
    deployment with the most traffic is the champion; its name is
    'unet-v{N}', so the version is the trailing number.

    Args:
        ml_client: An authenticated MLClient.

    Returns:
        The champion model version as a string, e.g. "9".

    Raises:
        RuntimeError: If no deployment is receiving traffic.
    """
    endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
    traffic = endpoint.traffic or {}
    if not traffic or max(traffic.values()) == 0:
        raise RuntimeError(
            f"Endpoint '{ENDPOINT_NAME}' has no live deployment (traffic={traffic})."
        )
    top_deployment = max(traffic, key=traffic.get)  # e.g. "unet-v9"
    version = top_deployment.split("-v")[-1]
    logger.info("Champion: deployment '%s' -> version %s.", top_deployment, version)
    return version


def resolve_candidate_version(ml_client) -> str:
    """Return the latest registered hades-unet version (the candidate).

    Args:
        ml_client: An authenticated MLClient.

    Returns:
        The highest registered model version as a string.

    Raises:
        RuntimeError: If no versions are registered.
    """
    versions = list(ml_client.models.list(name=MODEL_NAME))
    if not versions:
        raise RuntimeError(f"No registered versions of '{MODEL_NAME}'.")
    latest = max(versions, key=lambda m: int(m.version))
    logger.info("Candidate: latest registered version %s.", latest.version)
    return str(latest.version)


def _download_model(ml_client, version: str) -> str:
    """Download a registered model version locally and return the folder.

    Models here are registered from an MLflow run (``runs:/<run>/model`` in
    cloud_train.py). Such models expose an asset URI that the Azure ML
    command-job INPUT resolver rejects with "Invalid ResolvedAssetUri from
    model", so they cannot be passed as ``azureml:hades-unet:<v>`` job inputs.

    models.download uses the same resolution path that model DEPLOYMENT uses
    (which handles run-backed models correctly), and the resulting folder is
    then handed to the job as a plain uri_folder input -- the input type that
    already resolves reliably for the data assets. The download runs in
    Airflow, where credentials already work, so the job itself needs no Azure
    SDK or managed identity.

    Args:
        ml_client: An authenticated MLClient.
        version: The model version to download.

    Returns:
        Local path containing the downloaded model tree (including the .pth).
    """
    dst = tempfile.mkdtemp(prefix=f"model_v{version}_")
    ml_client.models.download(name=MODEL_NAME, version=version, download_path=dst)
    logger.info("Downloaded model v%s to %s", version, dst)
    return dst


def build_smoke_job(
    ml_client,
    champion_version: str,
    candidate_version: str,
    margin: float = DEFAULT_MARGIN,
):
    """Build (do not submit) the Azure ML command job that scores both models.

    Mounts both model versions and the latest hades-smoke data asset, runs
    smoke_eval.py on lambda-0, and writes verdict.json to the 'verdict'
    output.

    Args:
        ml_client: An authenticated MLClient.
        champion_version: Currently-deployed version.
        candidate_version: Newly-registered version to test.
        margin: F1 margin the candidate must beat the champion by.

    Returns:
        An unsubmitted azure.ai.ml command job.
    """
    from azure.ai.ml import Input, Output, command
    from azure.ai.ml.constants import AssetTypes

    smoke_versions = list(ml_client.data.list(name=SMOKE_ASSET))
    smoke_v = max(smoke_versions, key=lambda d: int(d.version)).version
    env_versions = list(ml_client.environments.list(name=TRAINING_ENV))
    env_v = max(env_versions, key=lambda e: int(e.version)).version

    job = command(
        code=SMOKE_CODE_DIR,
        command=(
            "python smoke_eval.py "
            "--champion-dir ${{inputs.champion}} "
            "--candidate-dir ${{inputs.candidate}} "
            "--eval-dir ${{inputs.eval_set}} "
            "--output-dir ${{outputs.verdict}} "
            f"--champion-version {champion_version} "
            f"--candidate-version {candidate_version} "
            f"--margin {margin}"
        ),
        inputs={
            "champion": Input(
                type=AssetTypes.URI_FOLDER,
                path=_download_model(ml_client, champion_version),
            ),
            "candidate": Input(
                type=AssetTypes.URI_FOLDER,
                path=_download_model(ml_client, candidate_version),
            ),
            "eval_set": Input(
                type=AssetTypes.URI_FOLDER,
                path=f"azureml:{SMOKE_ASSET}:{smoke_v}",
            ),
        },
        outputs={"verdict": Output(type=AssetTypes.URI_FOLDER)},
        environment=f"azureml:{TRAINING_ENV}:{env_v}",
        compute=COMPUTE_NAME,
        resources={"instance_type": "gpu"},
        display_name=f"smoke-eval-v{candidate_version}-vs-v{champion_version}",
    )
    return job


def wait_for_job(ml_client, job_name: str, poll_interval: int = 60) -> None:
    """Block until an Azure ML job reaches a terminal state.

    Args:
        ml_client: An authenticated MLClient.
        job_name: Name of the submitted job.
        poll_interval: Seconds between status checks.

    Raises:
        RuntimeError: If the job ends in any state other than Completed.
    """
    terminal = {"Completed", "Failed", "Canceled"}
    while True:
        job = ml_client.jobs.get(job_name)
        logger.info("Job %s — status: %s", job_name, job.status)
        if job.status in terminal:
            break
        time.sleep(poll_interval)
    if job.status != "Completed":
        url = getattr(job, "studio_url", None) or "(open the job in Azure ML Studio)"
        raise RuntimeError(
            f"Smoke-eval job {job_name} ended with status: {job.status}. Logs: {url}"
        )


def read_verdict(ml_client, job_name: str) -> dict:
    """Download the job's 'verdict' output and parse verdict.json.

    Args:
        ml_client: An authenticated MLClient.
        job_name: Name of the completed job.

    Returns:
        The parsed verdict dict.

    Raises:
        FileNotFoundError: If verdict.json is not found in the output.
    """
    tmp = tempfile.mkdtemp(prefix="smoke_verdict_")
    ml_client.jobs.download(name=job_name, download_path=tmp, output_name="verdict")
    matches = glob.glob(f"{tmp}/**/verdict.json", recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"No verdict.json in downloaded output of job '{job_name}'."
        )
    verdict = json.loads(Path(matches[0]).read_text())
    logger.info("Verdict: %s", verdict)
    return verdict


def compute_traffic_split(
    existing: dict | None, deployment_name: str, traffic: int
) -> dict:
    """Return a traffic split routing `traffic`% to deployment_name.

    The remaining deployments are scaled proportionally to share what is left,
    and the new deployment absorbs any integer-rounding remainder so the split
    sums to EXACTLY 100 (Azure ML rejects any other total). This is the part
    that made the old `{new: traffic}` assignment a trap -- it only summed to
    100 when traffic was 100.

    Pure function (no Azure calls) so the arithmetic is unit-tested directly.

    Args:
        existing: Current {deployment: percent} split (may be None/empty).
        deployment_name: The deployment to route `traffic`% to.
        traffic: Percent for the new deployment (1..100).

    Returns:
        A new {deployment: percent} dict summing to exactly 100. Note the new
        deployment may receive slightly more than `traffic` when rounding the
        others leaves a remainder.
    """
    if traffic >= 100:
        return {deployment_name: 100}

    others = {k: v for k, v in (existing or {}).items() if k != deployment_name}
    remaining = 100 - traffic
    total = sum(others.values())
    if others and total > 0:
        for name in others:
            others[name] = int(others[name] / total * remaining)
    others[deployment_name] = 100 - sum(others.values())
    return others


def _parse_invoke_response(raw):
    """Parse the body returned by online_endpoints.invoke into a dict.

    score.py's run() returns ``json.dumps(result.to_dict())`` -- a JSON string
    -- and the Azure ML scoring server serializes that return value again, so
    the HTTP body can be a JSON-encoded JSON string (double-encoded). A single
    json.loads then yields a *str*, not the dict (exactly the "no mask_b64):
    str" false negative we hit). Decode bytes and unwrap up to two layers so
    both single- and double-encoded responses resolve to the dict.

    Args:
        raw: The value invoke() returned (bytes or str).

    Returns:
        The decoded object (normally the InferenceResult dict).
    """
    value = raw
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    for _ in range(2):
        if isinstance(value, str):
            value = json.loads(value)
        else:
            break
    return value


def _health_check(ml_client, deployment_name: str) -> bool:
    """Send one real request to a specific deployment and verify it serves.

    Blue-green safety: this targets the named deployment directly (regardless
    of its traffic share) with the committed sample_request.json, then confirms
    the response parses to an InferenceResult. That proves score.py runs
    end-to-end -- not merely that the container provisioned -- before any live
    traffic is routed to it.

    Args:
        ml_client: An authenticated MLClient.
        deployment_name: The deployment to probe, e.g. "unet-v10".

    Returns:
        True if the deployment served a valid response. False (with a warning)
        if the sample fixture is missing, so a missing fixture does not block
        deploys.

    Raises:
        RuntimeError: If the deployment returns a non-JSON or unexpected
            payload (a real serving fault) -- the caller must NOT flip traffic.
    """
    sample = Path(SCORE_DIR) / "sample_request.json"
    if not sample.is_file():
        logger.warning(
            "No sample_request.json next to score.py — skipping health check. "
            "Commit infra/cloud/endpoint/sample_request.json to enable it."
        )
        return False

    logger.info("Health-checking deployment '%s' at 0%% traffic...", deployment_name)
    raw = ml_client.online_endpoints.invoke(
        endpoint_name=ENDPOINT_NAME,
        deployment_name=deployment_name,
        request_file=str(sample),
    )
    try:
        parsed = _parse_invoke_response(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Health check for '{deployment_name}' returned non-JSON: {raw!r}."
        ) from exc

    if not isinstance(parsed, dict) or "mask_b64" not in parsed:
        keys = list(parsed) if isinstance(parsed, dict) else type(parsed).__name__
        raise RuntimeError(
            f"Health check for '{deployment_name}' returned an unexpected "
            f"payload (no mask_b64): {keys}."
        )
    logger.info("Health check passed for '%s'.", deployment_name)
    return True


def deploy_candidate(ml_client, version: str, traffic: int = 100) -> str:
    """Deploy a model version blue-green: create, health-check, then flip.

    Mirrors scripts/azure/deploy_endpoint.py but uses the injected (Airflow)
    ml_client so auth is consistent with the rest of the DAG. The new
    deployment is created first and stays at 0% traffic (the champion keeps
    serving). It is then probed with a real request via _health_check, and only
    if that passes is traffic flipped using compute_traffic_split. Old
    deployments are left in place, so rollback is just flipping traffic back.

    Note: the Airflow service principal must have permission to create
    deployments and update the endpoint on the workspace (Contributor or the
    AzureML Data Scientist role). If it does not, this raises an auth error and
    the fix is to grant that role (or trigger deploy-endpoint.yml instead).

    Args:
        ml_client: An authenticated MLClient (Airflow's, via get_ml_client).
        version: Model version to deploy, e.g. "10".
        traffic: Percent of traffic to send to this deployment (default 100).

    Returns:
        The deployment name that now serves traffic, e.g. "unet-v10".

    Raises:
        RuntimeError: If the inference environment is missing or the health
            check fails (traffic is left on the champion in that case).
    """
    from azure.ai.ml.entities import (
        CodeConfiguration,
        KubernetesOnlineDeployment,
    )

    model = ml_client.models.get(name=MODEL_NAME, version=version)

    envs = list(ml_client.environments.list(name=INFERENCE_ENV))
    if not envs:
        raise RuntimeError(f"No '{INFERENCE_ENV}' environment registered.")
    latest_env = max(envs, key=lambda e: int(e.version))

    deployment_name = f"unet-v{version}"
    deployment = KubernetesOnlineDeployment(
        name=deployment_name,
        endpoint_name=ENDPOINT_NAME,
        model=model,
        environment=latest_env,
        code_configuration=CodeConfiguration(
            code=SCORE_DIR,
            scoring_script="score.py",
        ),
        instance_type="gpu",
        instance_count=1,
    )
    logger.info(
        "Creating/updating deployment '%s' at 0%% traffic (a few minutes)...",
        deployment_name,
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    # Validate the live deployment BEFORE routing any traffic to it. A raise
    # here leaves the champion serving 100% and the new deployment idle at 0%.
    _health_check(ml_client, deployment_name)

    endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
    endpoint.traffic = compute_traffic_split(endpoint.traffic, deployment_name, traffic)
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    logger.info("Traffic now: %s", endpoint.traffic)
    return deployment_name
