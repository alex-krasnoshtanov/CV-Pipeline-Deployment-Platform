"""Demo seed script — inserts sample predictions and feedback for dashboard demos.

DEMO-ONLY: never runs automatically; never called from production code.
Run manually before a demo when the database has no data:

    uv run python scripts/seed_demo_stats.py --db-url postgresql+psycopg2://user:pass@localhost/db

The script inserts 90 prediction rows spanning 30 days (3 per day) across two
model versions, and one feedback row per prediction (alternating good/bad/uncertain).
Existing rows are not removed.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import random
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _b64_1x1_png() -> str:
    """Return a tiny 1×1 transparent PNG as base64, used as a dummy mask."""
    tiny = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
        b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return base64.b64encode(tiny).decode()


def seed(db_url: str, n_days: int = 30, per_day: int = 3) -> None:
    """Insert demo predictions and feedback into the database.

    Args:
        db_url: Sync SQLAlchemy connection URL (psycopg2 or similar).
        n_days: Number of calendar days to spread predictions over.
        per_day: Predictions inserted per day.
    """
    random.seed(42)
    engine = create_engine(db_url, future=True)
    mask_b64 = _b64_1x1_png()
    flags = ["good", "bad", "uncertain"]
    model_versions = ["unet-v1", "unet-v2"]
    now = datetime.now(UTC)

    with Session(engine) as session:
        for day_offset in range(n_days):
            day = now - timedelta(days=n_days - 1 - day_offset)
            for i in range(per_day):
                pred_id = uuid.uuid4()
                model_version = model_versions[i % len(model_versions)]
                confidence = round(random.uniform(0.5, 0.98), 4)
                landmark_count = random.randint(3, 8)
                landmarks = [
                    {
                        "id": j,
                        "x": random.randint(10, 200),
                        "y": random.randint(10, 200),
                        "confidence": round(random.uniform(0.6, 0.99), 4),
                    }
                    for j in range(landmark_count)
                ]
                session.execute(
                    text(
                        "INSERT INTO predictions "
                        "(id, created_at, image_filename, "
                        "image_width_px, image_height_px, "
                        "pipeline_version, model_version, "
                        "mask_b64, mask_confidence, "
                        "landmark_count, landmarks) "
                        "VALUES (:id, :ts, :fn, :w, :h, "
                        ":pv, :mv, :mb, :mc, :lc, :lm)"
                    ),
                    {
                        "id": pred_id,
                        "ts": day,
                        "fn": f"demo_plate_{day_offset}_{i}.tif",
                        "w": 512,
                        "h": 512,
                        "pv": "0.1.0",
                        "mv": model_version,
                        "mb": mask_b64,
                        "mc": confidence,
                        "lc": landmark_count,
                        "lm": json.dumps(landmarks),
                    },
                )
                fb_flag = flags[(day_offset * per_day + i) % len(flags)]
                session.execute(
                    text(
                        "INSERT INTO feedback (id, created_at, prediction_id, flag) "
                        "VALUES (:id, :ts, :pid, :flag)"
                    ),
                    {
                        "id": uuid.uuid4(),
                        "ts": day,
                        "pid": pred_id,
                        "flag": fb_flag,
                    },
                )
        session.commit()

    total = n_days * per_day
    logger.info("Seeded %d demo predictions and %d feedback rows.", total, total)
    logger.info("Run this script again to add another batch (rows are cumulative).")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-url",
        default=os.getenv("DB_URL", ""),
        help="Sync SQLAlchemy DB URL (e.g. postgresql+psycopg2://user:pass@localhost/db)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of calendar days to spread predictions over (default 30)",
    )
    parser.add_argument(
        "--per-day",
        type=int,
        default=3,
        help="Predictions per day (default 3)",
    )
    args = parser.parse_args()
    if not args.db_url:
        parser.error("--db-url or DB_URL env var is required")
    seed(args.db_url, n_days=args.days, per_day=args.per_day)
