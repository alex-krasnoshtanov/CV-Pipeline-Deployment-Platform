"""Unit tests for the GET /stats router.

The service layer is mocked so we test only:
  - authentication gating (401 without credentials)
  - 200 + correct top-level response shape on valid auth
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import bcrypt
import pytest
from api.db import get_db
from api.main import app
from api.schemas.stats import (
    BusinessStats,
    FeedbackBreakdown,
    OperationalStats,
    RequestMetrics,
    StatsResponse,
)
from fastapi.testclient import TestClient

_TEST_KEY = "stats-test-key"
_HASHED_KEY = bcrypt.hashpw(_TEST_KEY.encode(), bcrypt.gensalt()).decode()
_SHA256_KEY = hashlib.sha256(_TEST_KEY.encode()).hexdigest()


def _mock_user() -> MagicMock:
    u = MagicMock()
    u.id = "user-uuid"
    u.role = "researcher"
    u.name = "Tester"
    u.email = None
    u.api_key_hash = _HASHED_KEY
    u.key_sha256 = _SHA256_KEY
    u.password_hash = None
    u.oauth_provider = None
    u.oauth_subject = None
    u.last_login_at = None
    return u


def _fake_stats() -> StatsResponse:
    fb = FeedbackBreakdown(good=2, bad=1, uncertain=0, total=3, bad_rate=1 / 3)
    biz = BusinessStats(
        total_predictions=10,
        avg_confidence=0.8,
        confidence_histogram=[],
        predictions_over_time=[],
        landmark_distribution=[],
        feedback=fb,
        per_model_version=[],
    )
    ops = OperationalStats(
        serving_model_version="unet-v1",
        prometheus_metrics_available=True,
        request_metrics=RequestMetrics(
            total_requests=0, error_rate=0.0, avg_latency_ms=None
        ),
    )
    return StatsResponse(business=biz, operational=ops)


@pytest.fixture(autouse=True)
def _override_db(monkeypatch: pytest.MonkeyPatch):
    mock_user = _mock_user()
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = mock_user

    async def _mock_get_db():
        session = AsyncMock()
        session.execute.return_value = execute_result
        yield session

    app.dependency_overrides[get_db] = _mock_get_db
    monkeypatch.setenv("API_KEY", _TEST_KEY)
    yield
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    mock_model = MagicMock()
    mock_model.model_version = "unet-v1"
    if hasattr(app.state, "model"):
        delattr(app.state, "model")
    monkeypatch.setattr("api.main.load_model", lambda: mock_model)
    with TestClient(app) as c:
        yield c


@pytest.mark.unit
class TestStatsAuth:
    def test_no_auth_returns_401(self, client: TestClient) -> None:
        res = client.get("/stats")
        assert res.status_code == 401

    def test_wrong_key_returns_401(self, client: TestClient) -> None:
        res = client.get("/stats", headers={"X-API-Key": "wrong"})
        assert res.status_code == 401


@pytest.mark.unit
class TestStatsShape:
    def test_returns_200_with_valid_key(self, client: TestClient) -> None:
        with patch(
            "api.routers.stats.get_stats",
            new=AsyncMock(return_value=_fake_stats()),
        ):
            res = client.get("/stats", headers={"X-API-Key": _TEST_KEY})
        assert res.status_code == 200

    def test_response_has_business_and_operational(self, client: TestClient) -> None:
        with patch(
            "api.routers.stats.get_stats",
            new=AsyncMock(return_value=_fake_stats()),
        ):
            body = client.get("/stats", headers={"X-API-Key": _TEST_KEY}).json()
        assert "business" in body
        assert "operational" in body

    def test_business_has_required_fields(self, client: TestClient) -> None:
        with patch(
            "api.routers.stats.get_stats",
            new=AsyncMock(return_value=_fake_stats()),
        ):
            body = client.get("/stats", headers={"X-API-Key": _TEST_KEY}).json()
        biz = body["business"]
        assert biz["total_predictions"] == 10
        assert biz["avg_confidence"] == pytest.approx(0.8)
        assert "confidence_histogram" in biz
        assert "feedback" in biz

    def test_operational_has_serving_version(self, client: TestClient) -> None:
        with patch(
            "api.routers.stats.get_stats",
            new=AsyncMock(return_value=_fake_stats()),
        ):
            body = client.get("/stats", headers={"X-API-Key": _TEST_KEY}).json()
        assert body["operational"]["serving_model_version"] == "unet-v1"
        assert body["operational"]["prometheus_metrics_available"] is True

    def test_operational_has_request_metrics(self, client: TestClient) -> None:
        with patch(
            "api.routers.stats.get_stats",
            new=AsyncMock(return_value=_fake_stats()),
        ):
            body = client.get("/stats", headers={"X-API-Key": _TEST_KEY}).json()
        rm = body["operational"]["request_metrics"]
        assert "total_requests" in rm
        assert "error_rate" in rm
        assert "avg_latency_ms" in rm
