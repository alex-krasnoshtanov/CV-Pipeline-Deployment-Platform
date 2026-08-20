"""Register the cv-pipeline training train_environment in Azure ML.

This version uses a build context (Dockerfile) instead of a bare
conda.yml, so that the cv-pipeline wheel is baked into the
train_environment image. Jobs no longer need 'pip install' at startup.

Before running:
  1. Build the wheel: cd packages/cv-pipeline && uv build
  2. Copy cv_pipeline-0.1.0-py3-none-any.whl into this directory
     (next to Dockerfile and conda.yml)

Usage:
    python env_registration.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import BuildContext, Environment
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Register a custom training train_environment in the Azure ML workspace."""
    load_dotenv()

    # Verify the wheel exists in this directory.
    script_dir = Path(__file__).parent
    wheel = script_dir / "cv_pipeline-0.1.0-py3-none-any.whl"
    if not wheel.exists():
        logger.error(
            "Wheel not found at %s. Build it with 'uv build' in "
            "packages/cv-pipeline and copy the .whl here.",
            wheel,
        )
        return

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
    )
    logger.info("Connected to workspace: %s", ml_client.workspace_name)

    env = Environment(
        name="cv-pipeline-training",
        description="Training train_environment for HADES plant CV pipeline",
        build=BuildContext(path=str(script_dir)),
        version="6",
    )

    registered = ml_client.environments.create_or_update(env)
    logger.info(
        "Registered train_environment: %s v%s",
        registered.name,
        registered.version,
    )


if __name__ == "__main__":
    main()
