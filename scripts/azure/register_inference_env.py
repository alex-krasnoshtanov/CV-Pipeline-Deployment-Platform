"""Register inference environment with azureml-inference-server-http."""

import os
from pathlib import Path

from dotenv import load_dotenv

from azure.ai.ml import MLClient
from azure.ai.ml.entities import BuildContext, Environment
from azure.identity import DefaultAzureCredential

load_dotenv("configs/env/.env")

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
)

env = Environment(
    name="cv-pipeline-inference",
    description="Inference environment for HADES endpoint (includes inference server)",
    build=BuildContext(path=str(Path("infra/cloud/endpoint"))),
)

result = ml_client.environments.create_or_update(env)
print(f"Registered: {result.name} v{result.version}")
