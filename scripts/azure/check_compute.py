import os

from dotenv import load_dotenv

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

load_dotenv()
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
)

k8s = ml_client.compute.get("lambda-4")
print("Type:", k8s.type)
print("Properties:", k8s.properties)
