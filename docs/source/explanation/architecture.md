# Architecture

:::{note}
**Explanation** pages discuss concepts and design decisions. If you
want to do something specific, see [How-to](../how-to/index). If you
want the formal spec, see the [CV pipeline specification](../reference/specification.md).
:::

## Architecture Diagrams

For the latest visual architecture diagrams (rendered in Mermaid on GitHub), please refer to:
- [System Architecture](https://github.com/Gfgf96/CV-Pipeline-Deployment-Platform/blob/main/docs/architecture/system.md)
- [Deployment Topology](https://github.com/Gfgf96/CV-Pipeline-Deployment-Platform/blob/main/docs/architecture/deployment.md)
- [MLOps Loop](https://github.com/Gfgf96/CV-Pipeline-Deployment-Platform/blob/main/docs/architecture/mlops.md)

## Three components, one pipeline

The system has three delivery forms but only one inference code
path:

```
+--------------+   +--------------+   +--------------+
|   CLI        |   |   FastAPI    |   |  Azure ML    |
| cv-pipeline  |   |  backend     |   |  scoring     |
|   infer      |   |  /infer      |   |  (deployed)  |
+------+-------+   +------+-------+   +------+-------+
       |                  |                  |
       +------------------+------------------+
                          |
                          v
                +-------------------+
                |   cv_pipeline     |
                |    .infer()       |
                +-------------------+
```

Everything downstream of the three delivery forms is identical.
The CLI and the API call the same `cv_pipeline.infer()` function
with the same arguments; the Azure ML scoring script does too. **There is no duplicated inference code.**

This matters because:
- One test suite validates all three paths.
- Bug fixes propagate automatically.
- The pipeline version string in responses comes from one source
  (`cv_pipeline.__version__`), so a client can cross-reference.

## Why FastAPI

We chose FastAPI over Flask or Django-REST for three reasons:

1. **Native async support.** Long-running inference runs in a
   threadpool while the event loop serves `/health` and other
   concurrent requests. See {doc}`/autoapi/api/index` for how
   `run_in_threadpool` wraps the torch forward pass.
2. **Automatic OpenAPI.** `/docs` is auto-generated from Pydantic
   response models - zero hand-written schema.
3. **Pydantic everywhere.** Request validation, response
   serialisation, and error envelopes all flow through the same
   type system. This gives us a single source of truth for the API
   contract (specification section 4).

## Why U-Net for segmentation

The NPEC brief emphasises root tissue segmentation on lab-taken
plates - high-resolution, controlled lighting, binary foreground
vs. background. U-Net's symmetric encoder-decoder with skip
connections is the textbook match:

- Handles pixel-level precision (landmark detection requires
  sub-millimetre accuracy in the mask).
- Small enough to run on a consumer GPU (batch size 1, ~180 MB
  weights) - the target servers have limited GPU memory.
- Well-studied for biomedical imagery - any later improvement work has
  plenty of priors to draw on.

Alternatives considered and rejected:
- **Mask R-CNN** - adds instance segmentation we don't need (one
  plate = one connected root system).
- **SAM (Segment Anything)** - foundation model, too heavy for
  on-prem serving, licence friction.

## Why patch-based inference

Whole-image inference on a HADES plate (up to 8192x8192) would OOM
on any GPU we have access to. We tile the image into 1024x1024
patches with 50% overlap, run the model on each, then stitch the
probability maps back together.

Overlap is required because root tissue near patch edges would
otherwise show a discontinuity where adjacent patches don't quite
agree. At 50% overlap each output pixel is seen by 2-4 patches; we
average their probabilities before thresholding.

Trade-off: 2-4x the FLOPs vs. naive tiling, but gives visually
seamless masks. A later iteration may explore overlap-blending via
Gaussian weights for further smoothing.

## Containerisation

`docker-compose up` brings up three services on a shared network:

- **backend** (FastAPI, port 8000) - inference service
- **frontend** (Next.js, port 3000) - researcher UI
- **db** (Postgres 16, port 5432) - predictions, feedback, users.
  Tables are created via Alembic migrations and populated by the
  seed script on first startup. The `predictions` table is written
  to on every successful `/infer` call.

All three are health-checked via `/health` (backend) and
`pg_isready` (db). The frontend polls `/health` on load and shows a
red/green indicator so users know when the stack is up.

The on-prem deployment uses the same compose file through Portainer
with images pulled from GHCR.

## Roadmap items from this snapshot

These were planned when this page was first written. Most have since
shipped:

- **Data & training pipelines** - Airflow DAGs in `infra/airflow/dags/`
  run preprocessing and Azure ML training jobs (HP sweeps, conditional
  model registration on a test/val-F1 threshold).
  **Shipped.**
- **Blue-green deployment** - a blue-green rollout with a smoke-test
  rollback gate is the active deployment safeguard.
  **Shipped** on-prem and on Azure Container Apps (backend + frontend
  Running); the blue-green revision traffic-split is wired in `cd.yml`.
- **Feedback loop** - corrections from the Next.js UI write to the
  `feedback` table; a threshold crossing triggers a retrain via the
  `feedback_retrain_trigger` DAG. **Shipped.**
- **Monitoring** - the backend exposes Prometheus `/metrics` and runs a
  rolling-confidence drift detector (`apps/backend/src/api/services/drift_detector.py`)
  **Shipped.** Azure Monitor / App Insights
  dashboards remain **planned**.
