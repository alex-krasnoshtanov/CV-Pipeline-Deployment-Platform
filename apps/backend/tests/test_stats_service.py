"""Unit tests for the stats aggregation service.

The DB is replaced by a structured AsyncMock whose execute() side_effect
returns one pre-built result per query in the same order that get_stats()
issues them:
  1. totals (count, avg_confidence)
  2. confidence histogram
  3. time-series (predictions per day)
  4. feedback breakdown
  5. per-model-version stats
  6. landmark distribution

_collect_request_metrics() is patched to a fixed zero-state in TestGetStats
so those tests are independent of the global Prometheus registry state.
The real aggregation logic is tested separately in TestPrometheusAggregation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from api.schemas.stats import RequestMetrics
from api.services.stats_service import _collect_request_metrics, get_stats

_ZERO_REQUEST_METRICS = RequestMetrics(
    total_requests=0, error_rate=0.0, avg_latency_ms=None
)


def _make_row(**kwargs) -> MagicMock:
    """Return a MagicMock with attributes set from kwargs."""
    row = MagicMock()
    for k, v in kwargs.items():
        setattr(row, k, v)
    return row


def _make_result(rows: list | None = None, one=None) -> MagicMock:
    """Build an execute() return value with .all() and .one() methods."""
    result = MagicMock()
    result.all.return_value = rows or []
    result.one.return_value = one or MagicMock()
    return result


def _make_sample(name: str, labels: dict, value: float) -> MagicMock:
    s = MagicMock()
    s.name = name
    s.labels = labels
    s.value = value
    return s


def _make_metric_family(name: str, samples: list) -> MagicMock:
    mf = MagicMock()
    mf.name = name
    mf.samples = samples
    return mf


@pytest.fixture
def mock_db():
    """AsyncMock session whose execute() side_effect drives get_stats()."""
    # Query 1 — totals
    totals = _make_result(one=_make_row(total=42, avg_confidence=0.72))
    # Query 2 — confidence histogram
    hist = _make_result(
        rows=[
            _make_row(bucket_idx=5, count=10),
            _make_row(bucket_idx=7, count=25),
            _make_row(bucket_idx=9, count=7),
        ]
    )
    # Query 3 — time series
    day1 = MagicMock()
    day1.strftime.return_value = "2026-05-01"
    day2 = MagicMock()
    day2.strftime.return_value = "2026-05-02"
    timeseries = _make_result(
        rows=[
            _make_row(day=day1, count=3),
            _make_row(day=day2, count=5),
        ]
    )
    # Query 4 — feedback
    feedback = _make_result(
        rows=[
            _make_row(flag="good", count=8),
            _make_row(flag="bad", count=4),
            _make_row(flag="uncertain", count=2),
        ]
    )
    # Query 5 — per-model-version
    mv = _make_result(
        rows=[
            _make_row(model_version="unet-v1", count=20, avg_conf=0.68),
            _make_row(model_version="unet-v2", count=22, avg_conf=0.76),
        ]
    )
    # Query 6 — landmark distribution
    lm = _make_result(
        rows=[
            _make_row(landmark_count=4, count=15),
            _make_row(landmark_count=5, count=27),
        ]
    )

    session = AsyncMock()
    session.execute = AsyncMock(
        side_effect=[totals, hist, timeseries, feedback, mv, lm]
    )
    return session


@pytest.mark.unit
class TestGetStats:
    """Tests for get_stats().

    _collect_request_metrics is patched to a fixed zero-state so these
    tests are independent of the global Prometheus registry (which
    accumulates real samples from TestClient-driven router tests running
    in the same pytest process).
    """

    @pytest.fixture(autouse=True)
    def _patch_prometheus(self):
        with patch(
            "api.services.stats_service._collect_request_metrics",
            return_value=_ZERO_REQUEST_METRICS,
        ):
            yield

    @pytest.mark.anyio
    async def test_total_predictions(self, mock_db) -> None:
        stats = await get_stats(
            mock_db, serving_version="unet-v2", prometheus_available=True
        )
        assert stats.business.total_predictions == 42

    @pytest.mark.anyio
    async def test_avg_confidence(self, mock_db) -> None:
        stats = await get_stats(
            mock_db, serving_version="unet-v2", prometheus_available=True
        )
        assert stats.business.avg_confidence == pytest.approx(0.72)

    @pytest.mark.anyio
    async def test_confidence_histogram_has_10_buckets(self, mock_db) -> None:
        stats = await get_stats(
            mock_db, serving_version="unet-v2", prometheus_available=True
        )
        assert len(stats.business.confidence_histogram) == 10

    @pytest.mark.anyio
    async def test_confidence_histogram_counts(self, mock_db) -> None:
        stats = await get_stats(
            mock_db, serving_version="unet-v2", prometheus_available=True
        )
        hist = {b.bucket_label: b.count for b in stats.business.confidence_histogram}
        assert hist["0.5–0.6"] == 10
        assert hist["0.7–0.8"] == 25
        assert hist["0.9–1.0"] == 7
        assert hist["0.0–0.1"] == 0

    @pytest.mark.anyio
    async def test_time_series_length(self, mock_db) -> None:
        stats = await get_stats(
            mock_db, serving_version="unet-v2", prometheus_available=True
        )
        assert len(stats.business.predictions_over_time) == 2

    @pytest.mark.anyio
    async def test_time_series_dates(self, mock_db) -> None:
        stats = await get_stats(
            mock_db, serving_version="unet-v2", prometheus_available=True
        )
        dates = [p.date for p in stats.business.predictions_over_time]
        assert "2026-05-01" in dates
        assert "2026-05-02" in dates

    @pytest.mark.anyio
    async def test_feedback_breakdown(self, mock_db) -> None:
        stats = await get_stats(
            mock_db, serving_version="unet-v2", prometheus_available=True
        )
        fb = stats.business.feedback
        assert fb.good == 8
        assert fb.bad == 4
        assert fb.uncertain == 2
        assert fb.total == 14
        assert fb.bad_rate == pytest.approx(4 / 14)

    @pytest.mark.anyio
    async def test_per_model_version_count(self, mock_db) -> None:
        stats = await get_stats(
            mock_db, serving_version="unet-v2", prometheus_available=True
        )
        assert len(stats.business.per_model_version) == 2

    @pytest.mark.anyio
    async def test_per_model_version_numeric_sort(self, mock_db) -> None:
        stats = await get_stats(
            mock_db, serving_version="unet-v2", prometheus_available=True
        )
        versions = [mv.model_version for mv in stats.business.per_model_version]
        assert versions == ["unet-v1", "unet-v2"]

    @pytest.mark.anyio
    async def test_serving_model_version(self, mock_db) -> None:
        stats = await get_stats(
            mock_db, serving_version="unet-v2", prometheus_available=True
        )
        assert stats.operational.serving_model_version == "unet-v2"

    @pytest.mark.anyio
    async def test_prometheus_available(self, mock_db) -> None:
        stats = await get_stats(
            mock_db, serving_version="unet-v2", prometheus_available=True
        )
        assert stats.operational.prometheus_metrics_available is True

    @pytest.mark.anyio
    async def test_prometheus_unavailable(self, mock_db) -> None:
        stats = await get_stats(
            mock_db, serving_version="unet-v2", prometheus_available=False
        )
        assert stats.operational.prometheus_metrics_available is False

    @pytest.mark.anyio
    async def test_request_metrics_present(self, mock_db) -> None:
        stats = await get_stats(
            mock_db, serving_version="unet-v2", prometheus_available=True
        )
        rm = stats.operational.request_metrics
        assert isinstance(rm.total_requests, int)
        assert 0.0 <= rm.error_rate <= 1.0

    @pytest.mark.anyio
    async def test_landmark_distribution(self, mock_db) -> None:
        stats = await get_stats(
            mock_db, serving_version="unet-v2", prometheus_available=True
        )
        ld = stats.business.landmark_distribution
        assert len(ld) == 2
        assert ld[0].landmark_count == 4
        assert ld[0].count == 15

    @pytest.mark.anyio
    async def test_empty_db_graceful(self) -> None:
        """All-zero counts when the DB is empty — no division errors."""
        totals = _make_result(one=_make_row(total=0, avg_confidence=None))
        empty = _make_result(rows=[])
        session = AsyncMock()
        session.execute = AsyncMock(
            side_effect=[totals, empty, empty, empty, empty, empty]
        )
        stats = await get_stats(
            session, serving_version=None, prometheus_available=False
        )
        assert stats.business.total_predictions == 0
        assert stats.business.avg_confidence is None
        assert stats.business.feedback.bad_rate == 0.0
        assert stats.operational.serving_model_version is None

    @pytest.mark.anyio
    async def test_confidence_clamp_at_1_0(self) -> None:
        """Confidence == 1.0 must be clamped from bucket_idx=10 into bucket 9."""
        totals = _make_result(one=_make_row(total=3, avg_confidence=1.0))
        # floor(1.0 * 10) = 10 — the service must clamp this to index 9
        hist = _make_result(rows=[_make_row(bucket_idx=10, count=3)])
        empty = _make_result(rows=[])
        session = AsyncMock()
        session.execute = AsyncMock(
            side_effect=[totals, hist, empty, empty, empty, empty]
        )
        stats = await get_stats(
            session, serving_version=None, prometheus_available=False
        )
        assert stats.business.confidence_histogram[9].count == 3
        assert stats.business.confidence_histogram[9].bucket_label == "0.9–1.0"


@pytest.mark.unit
class TestPrometheusAggregation:
    """Tests for _collect_request_metrics() with a mocked REGISTRY."""

    def _mock_registry(self, families: list) -> MagicMock:
        reg = MagicMock()
        reg.collect.return_value = families
        return reg

    def test_zero_traffic_returns_zero_state(self) -> None:
        reg = self._mock_registry([])
        with patch("api.services.stats_service.REGISTRY", reg):
            result = _collect_request_metrics()
        assert result.total_requests == 0
        assert result.error_rate == 0.0
        assert result.avg_latency_ms is None

    def test_normal_traffic_computes_error_rate(self) -> None:
        families = [
            _make_metric_family(
                "http_requests",
                [
                    _make_sample(
                        "http_requests_total",
                        {"method": "GET", "status": "2xx", "handler": "/infer"},
                        10.0,
                    ),
                    _make_sample(
                        "http_requests_total",
                        {"method": "POST", "status": "5xx", "handler": "/infer"},
                        2.0,
                    ),
                    # _created samples must be ignored
                    _make_sample(
                        "http_requests_created",
                        {"method": "GET", "status": "2xx", "handler": "/infer"},
                        99999.0,
                    ),
                ],
            ),
            _make_metric_family(
                "http_request_duration_highr_seconds",
                [
                    _make_sample("http_request_duration_highr_seconds_sum", {}, 1.2),
                    _make_sample("http_request_duration_highr_seconds_count", {}, 12.0),
                ],
            ),
        ]
        reg = self._mock_registry(families)
        with patch("api.services.stats_service.REGISTRY", reg):
            result = _collect_request_metrics()

        assert result.total_requests == 12
        assert result.error_rate == pytest.approx(2 / 12)
        assert result.avg_latency_ms == pytest.approx(1.2 / 12 * 1000, rel=1e-3)

    def test_4xx_counted_in_error_rate(self) -> None:
        families = [
            _make_metric_family(
                "http_requests",
                [
                    _make_sample(
                        "http_requests_total",
                        {"method": "GET", "status": "2xx", "handler": "/health"},
                        8.0,
                    ),
                    _make_sample(
                        "http_requests_total",
                        {"method": "GET", "status": "4xx", "handler": "/stats"},
                        1.0,
                    ),
                ],
            ),
        ]
        reg = self._mock_registry(families)
        with patch("api.services.stats_service.REGISTRY", reg):
            result = _collect_request_metrics()

        assert result.total_requests == 9
        assert result.error_rate == pytest.approx(1 / 9)
        assert result.avg_latency_ms is None  # no histogram family present

    def test_all_5xx_gives_error_rate_1(self) -> None:
        families = [
            _make_metric_family(
                "http_requests",
                [
                    _make_sample(
                        "http_requests_total",
                        {"method": "GET", "status": "5xx", "handler": "/infer"},
                        5.0,
                    ),
                ],
            ),
        ]
        reg = self._mock_registry(families)
        with patch("api.services.stats_service.REGISTRY", reg):
            result = _collect_request_metrics()

        assert result.total_requests == 5
        assert result.error_rate == pytest.approx(1.0)
