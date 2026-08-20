"""Manually run the champion-challenger smoke eval for two model versions.

Use this to validate the whole gate end-to-end WITHOUT waiting for a retrain
-- for example while the latest registered version still equals the deployed
champion:

    python scripts/azure/run_smoke_eval.py --champion 9 --candidate 7

It submits the SAME Azure ML smoke-eval job the Airflow DAG submits (it reuses
build_smoke_job / wait_for_job / read_verdict from
infra/airflow/dags/promotion.py, so there is no duplicated logic that could
drift from what runs in production), waits, and prints the verdict. A green run
here means the DAG's promotion path will work when a genuinely new version
registers.

Auth: DefaultAzureCredential plus the AZURE_* env vars, same as
deploy_endpoint.py.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# promotion.py lives in the Airflow dags dir; add it to the path so this
# standalone runner reuses the exact same job-building logic as the DAG.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "infra" / "airflow" / "dags"))

from promotion import build_smoke_job, read_verdict, wait_for_job  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse champion/candidate versions and the promotion margin."""
    p = argparse.ArgumentParser(description="Manual champion-challenger smoke eval.")
    p.add_argument("--champion", required=True, help="Champion model version, e.g. 9.")
    p.add_argument(
        "--candidate", required=True, help="Candidate model version, e.g. 10."
    )
    p.add_argument("--margin", type=float, default=0.005)
    return p.parse_args()


def main() -> None:
    """Submit the smoke-eval job for the two versions and print the verdict."""
    load_dotenv("configs/env/.env")
    args = parse_args()

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
    )

    job = build_smoke_job(ml_client, args.champion, args.candidate, margin=args.margin)
    submitted = ml_client.jobs.create_or_update(job)
    logger.info("Submitted smoke-eval job: %s", submitted.name)
    logger.info("Studio: %s", submitted.studio_url or "N/A")

    wait_for_job(ml_client, submitted.name)
    verdict = read_verdict(ml_client, submitted.name)

    print()
    print("=== Smoke-eval verdict ===")
    for key, value in verdict.items():
        print(f"{key}: {value}")
    print(f"\nPROMOTE: {verdict['promote']}")


if __name__ == "__main__":
    main()
