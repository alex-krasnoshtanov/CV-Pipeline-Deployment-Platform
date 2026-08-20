"""Rolling-confidence drift detector for the monitoring pipeline.

Reads the most recent N prediction confidence scores from the database
and checks whether the fraction of low-confidence predictions exceeds
the configured alert threshold.

When drift is detected:
  - A WARNING is logged (visible in cloud logs).
  - The ``cv_low_confidence_fraction`` Prometheus Gauge is updated so
    it appears on the /metrics scrape.
  - The binary ``cv_low_confidence_alert`` Gauge is set to 1.0.
  - If ``TEAMS_WEBHOOK_URL`` is set, an Adaptive Card is posted to the
    configured Microsoft Teams channel (failure is logged, never raised).

Config (from Settings, see api.config):
    drift_window_size        — how many recent predictions to read (default 100)
    alert_confidence_min     — per-prediction threshold (default 0.60)
    alert_low_conf_fraction  — fraction above which to alert (default 0.20)
    teams_webhook_url        — optional Teams Incoming Webhook URL
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import Settings
from api.db.models import Prediction
from api.metrics import low_confidence_alert, low_confidence_fraction

logger = logging.getLogger(__name__)


async def _post_teams_alert(
    webhook_url: str, fraction: float, window: int, settings: Settings
) -> None:
    payload = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": "CV Pipeline — Confidence Drift Alert",
                            "weight": "Bolder",
                            "size": "Medium",
                        },
                        {
                            "type": "FactSet",
                            "facts": [
                                {
                                    "title": "Low-confidence fraction",
                                    "value": f"{fraction:.1%}",
                                },
                                {
                                    "title": "Alert threshold",
                                    "value": f"{settings.alert_low_conf_fraction:.1%}",
                                },
                                {
                                    "title": "Confidence floor",
                                    "value": f"{settings.alert_confidence_min:.2f}",
                                },
                                {"title": "Window (predictions)", "value": str(window)},
                                {
                                    "title": "Detected at",
                                    "value": datetime.now(UTC).isoformat(
                                        timespec="seconds"
                                    ),
                                },
                            ],
                        },
                    ],
                },
            }
        ],
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(webhook_url, json=payload)
        if resp.status_code not in (200, 202):
            logger.warning(
                "Teams webhook returned unexpected status %s: %s",
                resp.status_code,
                resp.text[:200],
            )


async def check_confidence_drift(
    db: AsyncSession,
    settings: Settings,
) -> dict[str, object]:
    """Compute rolling low-confidence fraction and emit metrics/alerts.

    Args:
        db: Async database session.
        settings: Application settings (window size + alert thresholds).

    Returns:
        Dict with keys ``window`` (int), ``low_confidence_fraction``
        (float), and ``alert`` (bool).  Returns ``{"status": "no_data"}``
        when the predictions table is empty.
    """
    window = settings.drift_window_size
    rows = await db.execute(
        select(Prediction.mask_confidence)
        .order_by(Prediction.created_at.desc())
        .limit(window)
    )
    confidences: list[float] = [row.mask_confidence for row in rows.all()]

    if not confidences:
        logger.debug("Drift check: predictions table is empty — skipping.")
        low_confidence_fraction.set(0.0)
        low_confidence_alert.set(0.0)
        return {"status": "no_data", "window": window}

    low_count = sum(1 for c in confidences if c < settings.alert_confidence_min)
    fraction = low_count / len(confidences)
    is_alert = fraction >= settings.alert_low_conf_fraction

    low_confidence_fraction.set(fraction)
    low_confidence_alert.set(1.0 if is_alert else 0.0)

    if is_alert:
        logger.warning(
            "DRIFT ALERT: %.1f%% of the last %d predictions had confidence "
            "< %.2f (alert threshold %.1f%%). Possible model degradation.",
            fraction * 100,
            len(confidences),
            settings.alert_confidence_min,
            settings.alert_low_conf_fraction * 100,
        )
        if settings.teams_webhook_url:
            try:
                await _post_teams_alert(
                    settings.teams_webhook_url, fraction, len(confidences), settings
                )
            except Exception:  # noqa: BLE001
                logger.exception("Failed to post Teams drift alert.")
    else:
        logger.debug(
            "Drift check: %.1f%% low-confidence in last %d predictions (ok).",
            fraction * 100,
            len(confidences),
        )

    return {
        "window": len(confidences),
        "low_confidence_fraction": fraction,
        "alert": is_alert,
    }
