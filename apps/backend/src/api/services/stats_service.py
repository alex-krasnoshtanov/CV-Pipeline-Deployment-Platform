"""Aggregation service for the monitoring dashboard.

All heavy computation runs in SQL via SQLAlchemy core expressions.
Python post-processing is limited to assembling the typed response from
raw rows (dict lookups, list comprehensions, one division for bad_rate).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

from prometheus_client import REGISTRY
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import Feedback, Prediction
from api.schemas.stats import (
    BusinessStats,
    ConfidenceBucket,
    FeedbackBreakdown,
    LandmarkBucket,
    ModelVersionStat,
    OperationalStats,
    RequestMetrics,
    StatsResponse,
    TimePoint,
)


def _version_numeric_key(stat: ModelVersionStat) -> int:
    """Extract trailing digits from a version string for numeric sort.

    "unet-v12" → 12, "unet-v3" → 3, "model" (no digits) → 0.
    Prevents lexicographic mis-ordering (e.g. v3 > v12 as text).
    """
    m = re.search(r"(\d+)$", stat.model_version)
    return int(m.group(1)) if m else 0


def _collect_request_metrics() -> RequestMetrics:
    """Read HTTP request counts and latency from the in-process Prometheus registry.

    Aggregates across all handlers and methods.  Sources:
    - ``http_requests_total`` Counter (labels: method, status, handler) — from
      prometheus-fastapi-instrumentator (merged in PR #83).  ``status`` is a
      string prefix like ``"2xx"``, ``"4xx"``, ``"5xx"``.
    - ``http_request_duration_highr_seconds`` Histogram (no labels) — high-
      resolution buckets for global latency stats.

    Both total and error counts come from the same ``http_requests_total``
    counter so the error_rate denominator cannot exceed 1.

    In a multi-worker deployment these reflect one worker's counters only.
    Returns a zero-state when no requests have been processed yet.
    """
    total_requests: float = 0.0
    error_requests: float = 0.0
    latency_sum: float = 0.0
    latency_count: float = 0.0

    for mf in REGISTRY.collect():
        if mf.name == "http_requests":
            for sample in mf.samples:
                if sample.name == "http_requests_total":
                    total_requests += sample.value
                    status = sample.labels.get("status", "")
                    if status.startswith("4") or status.startswith("5"):
                        error_requests += sample.value

        elif mf.name == "http_request_duration_highr_seconds":
            for sample in mf.samples:
                if sample.name == "http_request_duration_highr_seconds_sum":
                    latency_sum += sample.value
                elif sample.name == "http_request_duration_highr_seconds_count":
                    latency_count += sample.value

    error_rate = error_requests / total_requests if total_requests > 0 else 0.0
    avg_latency_ms: float | None = (
        round(latency_sum / latency_count * 1000, 2) if latency_count > 0 else None
    )

    return RequestMetrics(
        total_requests=int(total_requests),
        error_rate=error_rate,
        avg_latency_ms=avg_latency_ms,
    )


async def get_stats(
    db: AsyncSession,
    *,
    serving_version: str | None,
    prometheus_available: bool,
) -> StatsResponse:
    """Compute dashboard statistics from the database.

    Executes six SQL aggregation queries and assembles the result into
    a :class:`~api.schemas.stats.StatsResponse`.  No Python loops over
    raw prediction rows — all grouping and counting is done in the DB.

    Args:
        db: Async database session.
        serving_version: The model version currently loaded at startup,
            sourced from app state (not derived from prediction rows).
        prometheus_available: Whether the /metrics route is registered
            on the running application instance.

    Returns:
        Populated :class:`~api.schemas.stats.StatsResponse`.
    """
    # ── 1. Overall totals ─────────────────────────────────────────────────
    totals_row = (
        await db.execute(
            select(
                func.count(Prediction.id).label("total"),
                func.avg(Prediction.mask_confidence).label("avg_confidence"),
            )
        )
    ).one()
    total_predictions: int = int(totals_row.total or 0)
    avg_confidence: float | None = (
        float(totals_row.avg_confidence)
        if totals_row.avg_confidence is not None
        else None
    )

    # ── 2. Confidence histogram (10 equal-width buckets 0.0–1.0) ─────────
    # FLOOR(confidence * 10) maps [0.0, 0.1) → 0, …, [0.9, 1.0] → 9/10.
    # Values exactly == 1.0 produce floor=10; clamped to 9 in Python.
    bucket_col = func.floor(Prediction.mask_confidence * 10).label("bucket_idx")
    hist_rows = (
        await db.execute(
            select(bucket_col, func.count(Prediction.id).label("count"))
            .group_by(bucket_col)
            .order_by(bucket_col)
        )
    ).all()
    bucket_counts: dict[int, int] = {}
    for r in hist_rows:
        idx = min(int(r.bucket_idx or 0), 9)
        bucket_counts[idx] = bucket_counts.get(idx, 0) + r.count
    confidence_histogram = [
        ConfidenceBucket(
            bucket_label=f"{i / 10:.1f}–{(i + 1) / 10:.1f}",
            count=bucket_counts.get(i, 0),
        )
        for i in range(10)
    ]

    # ── 3. Predictions over time (last 30 calendar days, UTC) ────────────
    since = datetime.now(UTC) - timedelta(days=30)
    day_col = func.date_trunc("day", Prediction.created_at).label("day")
    timeseries_rows = (
        await db.execute(
            select(day_col, func.count(Prediction.id).label("count"))
            .where(Prediction.created_at >= since)
            .group_by(day_col)
            .order_by(day_col)
        )
    ).all()
    time_points = [
        TimePoint(date=r.day.strftime("%Y-%m-%d"), count=r.count)
        for r in timeseries_rows
    ]

    # ── 4. Feedback breakdown ─────────────────────────────────────────────
    fb_rows = (
        await db.execute(
            select(Feedback.flag, func.count(Feedback.id).label("count")).group_by(
                Feedback.flag
            )
        )
    ).all()
    fb_map: dict[str, int] = {r.flag: r.count for r in fb_rows}
    fb_good = fb_map.get("good", 0)
    fb_bad = fb_map.get("bad", 0)
    fb_uncertain = fb_map.get("uncertain", 0)
    fb_total = fb_good + fb_bad + fb_uncertain
    feedback = FeedbackBreakdown(
        good=fb_good,
        bad=fb_bad,
        uncertain=fb_uncertain,
        total=fb_total,
        bad_rate=fb_bad / fb_total if fb_total > 0 else 0.0,
    )

    # ── 5. Per-model-version stats ────────────────────────────────────────
    # Sorted numerically in Python (regex on trailing digits) so that
    # double-digit versions (e.g. unet-v12) sort after single-digit ones
    # (unet-v3), which a plain text ORDER BY would mis-order.
    mv_rows = (
        await db.execute(
            select(
                Prediction.model_version,
                func.count(Prediction.id).label("count"),
                func.avg(Prediction.mask_confidence).label("avg_conf"),
            ).group_by(Prediction.model_version)
        )
    ).all()
    per_model_version = [
        ModelVersionStat(
            model_version=r.model_version,
            prediction_count=r.count,
            avg_confidence=float(r.avg_conf) if r.avg_conf is not None else None,
        )
        for r in mv_rows
    ]
    per_model_version.sort(key=_version_numeric_key)

    # ── 6. Landmark count distribution ───────────────────────────────────
    lm_rows = (
        await db.execute(
            select(
                Prediction.landmark_count,
                func.count(Prediction.id).label("count"),
            )
            .group_by(Prediction.landmark_count)
            .order_by(Prediction.landmark_count)
        )
    ).all()
    landmark_distribution = [
        LandmarkBucket(landmark_count=r.landmark_count, count=r.count) for r in lm_rows
    ]

    # ── 7. HTTP request metrics from Prometheus registry ──────────────────
    request_metrics = _collect_request_metrics()

    business = BusinessStats(
        total_predictions=total_predictions,
        avg_confidence=avg_confidence,
        confidence_histogram=confidence_histogram,
        predictions_over_time=time_points,
        landmark_distribution=landmark_distribution,
        feedback=feedback,
        per_model_version=per_model_version,
    )
    operational = OperationalStats(
        serving_model_version=serving_version,
        prometheus_metrics_available=prometheus_available,
        request_metrics=request_metrics,
    )
    return StatsResponse(business=business, operational=operational)
