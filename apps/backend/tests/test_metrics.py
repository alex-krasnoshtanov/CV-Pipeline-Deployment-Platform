"""Unit tests: Prometheus /metrics endpoint wiring.

Tests against the real app imported from api.main so there is no second
Instrumentator registration (a second one would collide in prometheus_client's
global REGISTRY and raise ValueError at collection time).

The conftest autouse fixtures (init_db stub + get_db override) make TestClient(app)
safe: lifespan catches model-load failure and yields, so the app starts on CI
even without model weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from api.main import app  # noqa: E402


@pytest.mark.unit
def test_metrics_endpoint_returns_200() -> None:
    """GET /metrics returns 200 with Prometheus text-format content-type."""
    with TestClient(app) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/plain")
