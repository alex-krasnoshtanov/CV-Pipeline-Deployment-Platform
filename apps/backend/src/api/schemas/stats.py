"""Pydantic schemas for the GET /stats response."""

from __future__ import annotations

from pydantic import BaseModel


class ConfidenceBucket(BaseModel):
    """One bar in the confidence-score histogram."""

    bucket_label: str  # e.g. "0.0–0.1"
    count: int


class TimePoint(BaseModel):
    """Prediction count for one calendar day."""

    date: str  # ISO-8601 date, e.g. "2026-05-01"
    count: int


class FeedbackBreakdown(BaseModel):
    """Aggregate feedback flag counts and derived bad-rate."""

    good: int
    bad: int
    uncertain: int
    total: int
    bad_rate: float  # bad / total; 0.0 when total == 0


class LandmarkBucket(BaseModel):
    """Prediction count grouped by landmark_count value."""

    landmark_count: int
    count: int


class ModelVersionStat(BaseModel):
    """Per-model-version prediction volume and average confidence."""

    model_version: str
    prediction_count: int
    avg_confidence: float | None


class RequestMetrics(BaseModel):
    """HTTP request counters and latency from the Prometheus instrumentator.

    Sourced in-process from the shared Prometheus registry — no network
    call required.  In a multi-worker deployment these reflect one
    worker's counters.  ``avg_latency_ms`` is None when no requests
    have been processed yet.
    """

    total_requests: int
    error_rate: float  # fraction 0.0–1.0; 0.0 when no requests
    avg_latency_ms: float | None  # None when no traffic yet


class BusinessStats(BaseModel):
    """Business-insight metrics derived from the predictions and feedback tables."""

    total_predictions: int
    avg_confidence: float | None
    confidence_histogram: list[ConfidenceBucket]
    predictions_over_time: list[TimePoint]
    landmark_distribution: list[LandmarkBucket]
    feedback: FeedbackBreakdown
    per_model_version: list[ModelVersionStat]


class OperationalStats(BaseModel):
    """Operational monitoring metrics.

    ``serving_model_version`` is the version actually loaded at startup
    (from app state / config), distinct from the versions observed in
    the predictions table (``business.per_model_version``).

    ``request_metrics`` is sourced in-process from the Prometheus registry
    populated by prometheus-fastapi-instrumentator (merged in PR #83).
    """

    serving_model_version: str | None
    prometheus_metrics_available: bool
    request_metrics: RequestMetrics


class StatsResponse(BaseModel):
    """Top-level response schema for GET /stats."""

    business: BusinessStats
    operational: OperationalStats
