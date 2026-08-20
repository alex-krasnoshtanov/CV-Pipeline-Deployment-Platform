# Save this as upload_single.py
import os
import sys

from dotenv import load_dotenv

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential

load_dotenv()
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
)

split = sys.argv[1]  # train, val, or test
version = sys.argv[2]  # version number
data_dir = sys.argv[3]  # path to hades-patches

asset = Data(
    name=f"hades-{split}",
    version=version,
    path=f"{data_dir}/{split}",
    type=AssetTypes.URI_FOLDER,
    description=f"HADES patches: {split} v{version}",
)
result = ml_client.data.create_or_update(asset)
print(f"Registered: {result.name} v{result.version}")
