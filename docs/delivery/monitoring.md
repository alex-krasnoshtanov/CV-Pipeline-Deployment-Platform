# Metrics, and the one that matters

The backend exposes `/metrics` in Prometheus format through
`prometheus-fastapi-instrumentator`, wired up in
[`main.py`](../../apps/backend/src/api/main.py):

```python
Instrumentator(
    excluded_handlers=["/health", "/metrics", "/stats"],
).instrument(app).expose(app)
```

Health, metrics and stats are excluded from instrumentation on purpose. They
are polled constantly by the smoke test, the container healthcheck and
Prometheus itself, and leaving them in buries the real request traffic under
its own monitoring.

That gives the standard set for free: request counts by method, status and
handler, latency histograms, error rates.

## The model-quality signal

Standard HTTP metrics tell you the service is up. They say nothing about
whether the model is still any good, which for an ML service is the failure mode
that actually happens. So [`metrics.py`](../../apps/backend/src/api/metrics.py)
adds a histogram:

```python
inference_confidence = Histogram(
    "cv_inference_confidence",
    "Distribution of mask confidence scores from POST /infer (0-1)",
    buckets=[0.1, 0.2, ..., 1.0],
)
```

One line in the `/infer` success path records every prediction's confidence.
The distribution is the point: a model degrading on production data does not
start throwing errors, it starts producing lower-confidence masks. Watching the
shape of that histogram drift downward is the cheapest early warning available,
and it costs one `observe()` call.

## Turning the signal into an alert

A histogram on a dashboard only helps if someone is looking. The
[drift detector](../../apps/backend/src/api/services/drift_detector.py) closes
that loop: it reads the last N prediction confidences from the database and
checks what fraction fall below a threshold.

| Setting | Default |
|---|---|
| `drift_window_size` | 100 recent predictions |
| `alert_confidence_min` | 0.60 per prediction |
| `alert_low_conf_fraction` | alert above 0.20 |

When the fraction is exceeded it logs a warning, sets the
`cv_low_confidence_fraction` gauge and the binary `cv_low_confidence_alert`
gauge so both appear on the next scrape, and posts an Adaptive Card to a
Microsoft Teams channel if `TEAMS_WEBHOOK_URL` is configured. The webhook post
is best-effort: a failure is logged and never raised, because a monitoring
outage must not take out the request path it is monitoring.

That same threshold breach is what feeds the retraining flywheel described in
the main README.

## Prometheus itself

`infra/monitoring/prometheus/` builds a templated Prometheus image, published
to GHCR alongside the backend and frontend by
[`cd.yml`](../../.github/workflows/cd.yml). One image serves both the
on-premise and cloud tiers, with the scrape target injected through environment
variables at deploy time by the entrypoint, so there is no separate image per
environment to keep in sync.

It is built last in the CD job, after backend and frontend, so a problem in the
monitoring image cannot block the two pushes that matter.

## Chosen over App Insights

The original cloud design named Azure Application Insights for technical
monitoring. Prometheus replaced it: no billable Azure service, directly
scrapeable from the on-premise stack as well as the cloud one, and the same
metric names in both places. The
[deployment design notes](../architecture/deployment-cloud.md) still show App
Insights, which is one of several reasons those documents are marked as the
superseded design.

## Verified how far

`tests/test_metrics.py` asserts that `/metrics` returns 200 with a plain-text
body and that the standard request counter appears in the scrape. Neither test
imports torch, so both run in CI on every push.

A scrape from a live deployment was never captured. The on-premise stack was
running and the endpoint was reachable, but nobody saved the output before
access ended, so there is no `curl /metrics` transcript in
[`../evidence/`](../evidence/) to point at. The code, the tests and the
configuration are here; a production scrape is not.

The drift detector and this metrics work were led by Danil Sysenko.
