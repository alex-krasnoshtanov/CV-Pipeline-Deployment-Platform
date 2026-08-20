#!/usr/bin/env bash
set -euo pipefail

# Run database schema migrations before accepting traffic.
cd /app/apps/backend
alembic upgrade head

# Seed default user if the users table is empty.
PYTHONPATH=/app/apps/backend/src python -c "from api.db.seed import seed_default_user; seed_default_user()"

# Pre-cache model weights when using registry-based download.
# This is a best-effort step: if the download fails (network outage,
# auth required, etc.) we log a warning and continue. uvicorn will
# start, /health returns 503 until load_model() succeeds on startup.
if [ -n "${MODEL_VERSION:-}" ] && [ -z "${MODEL_PATH:-}" ]; then
    python - <<'PYEOF' || echo "WARNING: weight pre-cache failed — uvicorn will retry on startup"
import logging
import os
from cv_pipeline.weights import get_weights

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
version = os.environ["MODEL_VERSION"]
logging.info("Pre-caching weights for version %s", version)
get_weights(version)
logging.info("Weight cache ready.")
PYEOF
fi

exec "$@"