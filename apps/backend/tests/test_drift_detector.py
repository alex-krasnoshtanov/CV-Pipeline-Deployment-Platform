"""Unit tests for the rolling-confidence drift detector.

The DB session and Prometheus metrics are mocked so these tests run
without a live database or Prometheus scrape.  They cover:
  - empty predictions table: no-data status, gauges zeroed
  - healthy window (fraction below alert threshold): no alert
  - degraded window (fraction above alert threshold): alert + warning log
  - exactly at the alert boundary (fraction == threshold): alert fires
  - Teams webhook called when alert fires and URL is configured
  - Teams webhook not called when confidence is healthy
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from api.config import Settings
from api.services.drift_detector import check_confidence_drift


def _settings(**kwargs) -> Settings:
    defaults = dict(
        db_url="postgresql+asyncpg://test:test@localhost/test",
        alert_confidence_min=0.60,
        alert_low_conf_fraction=0.20,
        drift_window_size=100,
    )
    defaults.update(kwargs)
    return Settings(**defaults)


def _mock_db(confidences: list[float]) -> AsyncMock:
    """Return an AsyncMock session that yields the given confidence scores."""
    rows = [MagicMock(mask_confidence=c) for c in confidences]
    execute_result = MagicMock()
    execute_result.all.return_value = rows
    session = AsyncMock()
    session.execute = AsyncMock(return_value=execute_result)
    return session


@pytest.mark.unit
class TestDriftDetector:
    @pytest.mark.anyio
    async def test_empty_table_returns_no_data(self) -> None:
        db = _mock_db([])
        settings = _settings()
        with (
            patch("api.services.drift_detector.low_confidence_fraction") as mock_frac,
            patch("api.services.drift_detector.low_confidence_alert") as mock_alert,
        ):
            result = await check_confidence_drift(db, settings)

        assert result["status"] == "no_data"
        mock_frac.set.assert_called_once_with(0.0)
        mock_alert.set.assert_called_once_with(0.0)

    @pytest.mark.anyio
    async def test_healthy_window_no_alert(self) -> None:
        confidences = [0.90] * 90 + [0.50] * 10  # 10% below 0.60 → ok
        db = _mock_db(confidences)
        settings = _settings()
        with (
            patch("api.services.drift_detector.low_confidence_fraction") as mock_frac,
            patch("api.services.drift_detector.low_confidence_alert") as mock_alert,
        ):
            result = await check_confidence_drift(db, settings)

        assert result["alert"] is False
        assert result["low_confidence_fraction"] == pytest.approx(0.10)
        assert result["window"] == 100
        mock_frac.set.assert_called_once_with(pytest.approx(0.10))
        mock_alert.set.assert_called_once_with(0.0)

    @pytest.mark.anyio
    async def test_degraded_window_raises_alert(self) -> None:
        confidences = [0.90] * 70 + [0.40] * 30  # 30% below 0.60 → alert
        db = _mock_db(confidences)
        settings = _settings()
        with (
            patch("api.services.drift_detector.low_confidence_fraction") as mock_frac,
            patch("api.services.drift_detector.low_confidence_alert") as mock_alert,
        ):
            result = await check_confidence_drift(db, settings)

        assert result["alert"] is True
        assert result["low_confidence_fraction"] == pytest.approx(0.30)
        mock_frac.set.assert_called_once_with(pytest.approx(0.30))
        mock_alert.set.assert_called_once_with(1.0)

    @pytest.mark.anyio
    async def test_exactly_at_threshold_fires_alert(self) -> None:
        confidences = [0.90] * 80 + [0.40] * 20  # exactly 20% → alert
        db = _mock_db(confidences)
        settings = _settings(alert_low_conf_fraction=0.20)
        with (
            patch("api.services.drift_detector.low_confidence_fraction"),
            patch("api.services.drift_detector.low_confidence_alert") as mock_alert,
        ):
            result = await check_confidence_drift(db, settings)

        assert result["alert"] is True
        mock_alert.set.assert_called_once_with(1.0)

    @pytest.mark.anyio
    async def test_warning_logged_on_alert(self, caplog) -> None:
        confidences = [0.40] * 100  # all bad
        db = _mock_db(confidences)
        settings = _settings()
        with (
            patch("api.services.drift_detector.low_confidence_fraction"),
            patch("api.services.drift_detector.low_confidence_alert"),
        ):
            import logging

            with caplog.at_level(logging.WARNING, logger="api.services.drift_detector"):
                result = await check_confidence_drift(db, settings)

        assert result["alert"] is True
        assert any("DRIFT ALERT" in r.message for r in caplog.records)

    @pytest.mark.anyio
    async def test_smaller_window_than_configured(self) -> None:
        confidences = [0.70, 0.80, 0.90]  # only 3 rows, window=100
        db = _mock_db(confidences)
        settings = _settings(drift_window_size=100)
        with (
            patch("api.services.drift_detector.low_confidence_fraction"),
            patch("api.services.drift_detector.low_confidence_alert"),
        ):
            result = await check_confidence_drift(db, settings)

        assert result["window"] == 3
        assert result["alert"] is False

    @pytest.mark.anyio
    async def test_webhook_called_on_alert_when_url_set(self) -> None:
        confidences = [0.40] * 30 + [0.90] * 70  # 30% bad → alert
        db = _mock_db(confidences)
        settings = _settings(teams_webhook_url="https://teams.example.com/hook")
        with (
            patch("api.services.drift_detector.low_confidence_fraction"),
            patch("api.services.drift_detector.low_confidence_alert"),
            patch("api.services.drift_detector._post_teams_alert") as mock_webhook,
        ):
            mock_webhook.return_value = None
            result = await check_confidence_drift(db, settings)

        assert result["alert"] is True
        mock_webhook.assert_awaited_once()
        assert mock_webhook.await_args[0][0] == "https://teams.example.com/hook"

    @pytest.mark.anyio
    async def test_webhook_not_called_when_healthy(self) -> None:
        confidences = [0.90] * 100  # 0% bad → no alert
        db = _mock_db(confidences)
        settings = _settings(teams_webhook_url="https://teams.example.com/hook")
        with (
            patch("api.services.drift_detector.low_confidence_fraction"),
            patch("api.services.drift_detector.low_confidence_alert"),
            patch("api.services.drift_detector._post_teams_alert") as mock_webhook,
        ):
            result = await check_confidence_drift(db, settings)

        assert result["alert"] is False
        mock_webhook.assert_not_awaited()
