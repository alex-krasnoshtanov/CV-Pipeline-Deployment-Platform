"""Prometheus custom metrics for the CV Pipeline backend.

Importing this module registers the metrics with the default registry.
The prometheus-fastapi-instrumentator's /metrics scrape will include them.
"""

from __future__ import annotations

from prometheus_client import Gauge, Histogram

low_confidence_fraction = Gauge(
    "cv_low_confidence_fraction",
    "Fraction of recent predictions with confidence below alert threshold",
)

low_confidence_alert = Gauge(
    "cv_low_confidence_alert",
    "1 if low-confidence fraction exceeds alert_low_conf_fraction, 0 otherwise",
)

inference_confidence = Histogram(
    "cv_inference_confidence",
    "Distribution of mask confidence scores from POST /infer (0-1)",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
