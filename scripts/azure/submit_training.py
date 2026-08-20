"""Submit a training job to Azure ML.

Before running this script:
1. Build the cv-pipeline wheel:
   cd packages/cv-pipeline && uv build
2. Copy the .whl file to the same directory as this script and cloud_train.py

Usage:
    python job_submit.py --data-version 4 --epochs 5 --min-f1 0.0
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse arguments for job submission."""
    parser = argparse.ArgumentParser(description="Submit training job to Azure ML.")
    parser.add_argument(
        "--data-version",
        type=str,
        required=True,
        help="Version of hades-train and hades-val data assets.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--min-f1",
        type=float,
        default=0.75,
        help="Minimum val F1 to register model. Default 0.75.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="hades-unet",
        help="Name for the registered model in Azure ML.",
    )
    return parser.parse_args()


def main() -> None:
    """Build and submit the training command job."""
    load_dotenv()
    args = parse_args()

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
    )
    logger.info("Connected to workspace: %s", ml_client.workspace_name)

    # Verify the wheel file exists next to this script
    script_dir = Path(__file__).parent.parent / "training_jobs"
    wheels = list(script_dir.glob("cv_pipeline-*.whl"))
    if not wheels:
        logger.error(
            "No cv_pipeline wheel found in %s. "
            "Run 'uv build' in packages/cv-pipeline and copy the .whl here.",
            script_dir,
        )
        return
    wheel_name = wheels[0].name
    logger.info("Using wheel: %s", wheel_name)

    # Build the command string
    run_name_arg = f"--run-name {args.run_name}" if args.run_name else ""

    job = command(
        display_name=f"cv-pipeline-train-v{args.data_version}",
        description="Train U-Net segmentation model on HADES plant images.",
        code=str(script_dir),
        command=(
            f"pip install --no-deps {wheel_name} && "
            "python cloud_train.py "
            "--train-dir ${{inputs.train_data}} "
            "--val-dir ${{inputs.val_data}} "
            "--output-dir ${{outputs.model_output}} "
            f"--epochs {args.epochs} "
            f"--batch-size {args.batch_size} "
            f"--lr {args.lr} "
            f"--device {args.device} "
            f"--min-f1 {args.min_f1} "
            f"--model-name {args.model_name} "
            f"--num-workers {args.num_workers} "
            f"{run_name_arg}"
        ),
        inputs={
            "train_data": Input(
                type=AssetTypes.URI_FOLDER,
                path=f"azureml:hades-train:{args.data_version}",
            ),
            "val_data": Input(
                type=AssetTypes.URI_FOLDER,
                path=f"azureml:hades-val:{args.data_version}",
            ),
        },
        outputs={
            "model_output": {"type": "uri_folder"},
        },
        environment="azureml:cv-pipeline-training:4",
        compute="lambda-0",
        resources={"instance_type": "gpu"},
        experiment_name="cv-pipeline-training",
    )

    submitted = ml_client.jobs.create_or_update(job)
    logger.info("Job submitted: %s", submitted.name)
    logger.info("Studio URL: %s", submitted.studio_url)


if __name__ == "__main__":
    main()
