"""Deploy a model to a Kubernetes online endpoint on lambda-0.

Usage:
    python scripts/azure/deploy_endpoint.py --model-version 6
    python scripts/azure/deploy_endpoint.py --model-version 5 --traffic 10
"""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    CodeConfiguration,
    KubernetesOnlineDeployment,
    KubernetesOnlineEndpoint,
)
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ENDPOINT_NAME = "hades-unet-endpoint"
MODEL_NAME = "hades-unet"
COMPUTE_NAME = "lambda-0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", type=str, required=True)
    parser.add_argument(
        "--traffic",
        type=int,
        default=100,
        help="Traffic %% for this deployment. 100 = all traffic.",
    )
    return parser.parse_args()


def main():
    load_dotenv("configs/env/.env")
    args = parse_args()

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
    )
    logger.info("Connected to workspace: %s", ml_client.workspace_name)

    # Step 1: Create endpoint if it doesn't exist
    try:
        endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
        logger.info("Endpoint '%s' already exists.", ENDPOINT_NAME)
    except Exception:
        logger.info("Creating endpoint '%s'...", ENDPOINT_NAME)
        endpoint = KubernetesOnlineEndpoint(
            name=ENDPOINT_NAME,
            compute=COMPUTE_NAME,
            auth_mode="key",
            description="HADES plant segmentation inference.",
        )
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        logger.info("Endpoint created.")

    # Step 2: Get the model
    model = ml_client.models.get(MODEL_NAME, version=args.model_version)
    logger.info("Model: %s v%s", model.name, model.version)

    # Step 3: Get the environment (same one used for training)
    envs = list(ml_client.environments.list(name="cv-pipeline-inference"))
    latest_env = max(envs, key=lambda e: int(e.version))
    logger.info("Environment: %s v%s", latest_env.name, latest_env.version)

    # Step 4: Create the deployment
    deployment_name = f"unet-v{args.model_version}"
    score_dir = (
        Path(__file__).resolve().parent.parent.parent / "infra" / "cloud" / "endpoint"
    )

    deployment = KubernetesOnlineDeployment(
        name=deployment_name,
        endpoint_name=ENDPOINT_NAME,
        model=model,
        environment=latest_env,
        code_configuration=CodeConfiguration(
            code=str(score_dir),
            scoring_script="score.py",
        ),
        instance_type="gpu",
        instance_count=1,
    )

    logger.info(
        "Creating deployment '%s' (this takes a few minutes)...", deployment_name
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    logger.info("Deployment created.")

    # Step 5: Set traffic
    endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
    if args.traffic == 100:
        endpoint.traffic = {deployment_name: 100}
    else:
        existing = endpoint.traffic or {}
        remaining = 100 - args.traffic
        if existing:
            total = sum(existing.values())
            for name in existing:
                existing[name] = int(existing[name] / total * remaining)
        existing[deployment_name] = args.traffic
        endpoint.traffic = existing

    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    logger.info("Traffic: %s", endpoint.traffic)

    # Step 6: Print connection info
    keys = ml_client.online_endpoints.get_keys(ENDPOINT_NAME)
    print()
    print("=== Endpoint ready ===")
    print(f"URL:  {endpoint.scoring_uri}")
    print(f"Key:  {keys.primary_key[:20]}...")
    print()
    print("Add to your .env:")
    print(f"MODEL_ENDPOINT_URL={endpoint.scoring_uri}")
    print(f"MODEL_ENDPOINT_KEY={keys.primary_key}")


if __name__ == "__main__":
    main()
