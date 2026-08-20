"""Upload all three HADES splits as versioned data assets with the same version.

Usage:
    python upload_all_splits.py --data-dir D:\\path\to\\hades-subset --version 1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def main() -> None:
    """Upload train, val, and test folders under a single version."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Upload all HADES splits to Azure ML with the same version.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Parent folder containing train/, val/, test/ subdirs.",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Version tag applied to all three assets.",
    )
    args = parser.parse_args()

    # Verify all split dirs exist before uploading anything
    for split in SPLITS:
        split_path = args.data_dir / split
        if not split_path.exists():
            logger.error("Missing split directory: %s", split_path)
            sys.exit(1)

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
    )
    logger.info("Connected to workspace: %s", ml_client.workspace_name)

    registered = []
    for split in SPLITS:
        asset = Data(
            name=f"hades-{split}",
            version=args.version,
            path=str(args.data_dir / split),
            type=AssetTypes.URI_FOLDER,
            description=f"HADES plant image dataset: {split} v{args.version}",
        )
        result = ml_client.data.create_or_update(asset)
        registered.append(result)
        logger.info(
            "  Registered: %s v%s -> %s",
            result.name,
            result.version,
            result.path,
        )

    print(f"\nAll 3 splits registered as v{args.version}.")


if __name__ == "__main__":
    main()
