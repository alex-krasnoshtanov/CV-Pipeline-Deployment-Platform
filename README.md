# CV Pipeline — Plant Phenotyping Platform

[![CI](https://github.com/alex-krasnoshtanov/CV-Pipeline-Deployment-Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/alex-krasnoshtanov/CV-Pipeline-Deployment-Platform/actions/workflows/ci.yml)
[![Docs](https://github.com/alex-krasnoshtanov/CV-Pipeline-Deployment-Platform/actions/workflows/docs.yml/badge.svg)](https://github.com/alex-krasnoshtanov/CV-Pipeline-Deployment-Platform/actions/workflows/docs.yml)
[![CD](https://github.com/alex-krasnoshtanov/CV-Pipeline-Deployment-Platform/actions/workflows/cd.yml/badge.svg)](https://github.com/alex-krasnoshtanov/CV-Pipeline-Deployment-Platform/actions/workflows/cd.yml)
[![Images](https://github.com/alex-krasnoshtanov/CV-Pipeline-Deployment-Platform/actions/workflows/images.yml/badge.svg)](https://github.com/alex-krasnoshtanov/CV-Pipeline-Deployment-Platform/actions/workflows/images.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)

A computer-vision platform for plant organ segmentation and root-tip detection
on *Arabidopsis thaliana* seedling images — packaged as a library, served
behind an authenticated API, orchestrated by Airflow, and deployable to three
different targets (local, on-premise, Azure).

The interesting part of this repository is not the model. It is everything
around the model: the packaging boundary, the serving modes, the retraining
loop that closes on user feedback, and the fact that the same codebase deploys
to a laptop, a shared GPU server, and a managed cloud endpoint without a fork.

> **Attribution.** This began as a four-person university group project
> (Breda University of Applied Sciences, Feb–Jun 2026). It is not solely my
> work, and the [Attribution](#attribution) section below breaks down authorship
> per component with line counts. Read that section before reading this as a
> portfolio piece.

---

## Contents

- [What it does](#what-it-does)
- [System architecture](#system-architecture)
- [Model lifecycle](#model-lifecycle)
- [Quick start](#quick-start)
- [Data and model weights](#data-and-model-weights)
- [Repository layout](#repository-layout)
- [The `cv-pipeline` package](#the-cv-pipeline-package)
- [API](#api)
- [Authentication](#authentication)
- [Deployment targets](#deployment-targets)
- [Monitoring, drift and retraining](#monitoring-drift-and-retraining)
- [CI/CD](#cicd)
- [Testing](#testing)
- [Documentation](#documentation)
- [Attribution](#attribution)
- [License and reuse](#license-and-reuse)

---

## What it does

- **Segmentation + landmarks** — U-Net (ResNet-34 encoder) predicts root masks;
  a landmark stage derives root tips from the predicted mask.
- **Two serving modes from one codebase** — the backend either loads a local
  checkpoint or delegates to a remote scoring endpoint, chosen by environment
  variable. No code path is duplicated between them.
- **Explainability** — Seg-Grad-CAM heatmaps served from the same loaded model,
  so an explanation never needs a second weight download.
- **Feedback flywheel** — researchers flag bad predictions through the UI;
  flagged samples are exported, re-preprocessed against a frozen test set, and
  fed back into training.
- **Drift detection and alerting** — confidence distribution, latency and error
  rate are tracked against configurable thresholds and exposed to Prometheus.
- **Three deployment targets** — Docker Compose locally, Portainer on-premise,
  Azure Container Apps + Azure ML in the cloud.

---

## System architecture

The backend serves predictions either from a local model checkpoint or by
calling a remote managed endpoint, depending on configuration.

```mermaid
flowchart LR
    researcher["Researcher / robotic platform"]

    subgraph app["Application stack (Compose / Portainer / Container Apps)"]
        frontend["Frontend<br/>Next.js : 3000"]
        backend["Backend API<br/>FastAPI : 8000"]
        db[("Postgres 16<br/>: 5432")]
        prom["Prometheus<br/>: 9090"]
    end

    subgraph pkg["cv-pipeline package"]
        infer["infer(): segmentation + landmarks"]
    end

    subgraph cloud["Azure ML"]
        assets["Versioned data assets<br/>train / val / test"]
        registry["Model registry<br/>versioned checkpoints"]
        endpoint["Managed online endpoint"]
    end

    subgraph orch["Orchestration"]
        airflow["Airflow DAGs"]
    end

    researcher --> frontend --> backend
    researcher -->|"X-API-Key / token"| backend
    backend -->|"local mode"| infer
    backend -->|"cloud mode"| endpoint
    backend --> db
    prom -->|"scrape /metrics"| backend
    airflow -->|"submit jobs"| assets
    airflow --> registry
    registry --> endpoint
```

Per-environment diagrams (data pipeline, training pipeline, local, on-premise,
cloud) live in [`docs/architecture/`](docs/architecture/).

---

## Model lifecycle

Every arrow below is implemented as code — an Airflow DAG, an Azure ML job, or
a backend service.

```mermaid
flowchart TD
    raw["Raw plate images + masks"] --> pre["Preprocessing DAG<br/>image-level split, then patch"]
    pre --> assets["Versioned data assets<br/>train / val / test"]
    assets --> train["Training DAG<br/>U-Net (resnet34) + AdamW + cosine"]
    train --> eval["Dataset-level evaluation<br/>val_f1 / test_f1"]
    eval -->|"test_f1 >= min_f1"| reg["Register model version"]
    eval -->|"below threshold"| stop["Not registered"]
    reg --> deploy["Deploy to managed endpoint"]
    deploy --> serve["Backend serves /infer"]
    serve --> store["Store prediction in Postgres"]
    store --> flag["Researcher flags prediction<br/>POST /feedback"]
    flag --> export["Feedback export DAG"]
    export --> incr["Incremental preprocessing<br/>(frozen test set)"]
    incr --> train
```

**Why dataset-level F1.** `val_f1` and `test_f1` accumulate true and false
positives across the whole set and score once, rather than averaging per-batch
F1. Root patches are overwhelmingly background, and a per-batch average
collapses toward zero even for a strong model. See
[`train.py`](packages/cv-pipeline/src/cv_pipeline/train.py).

**Image-level splitting.** The train/val/test split happens *before* patching,
not after. Patches from one plate cannot leak across the split boundary, which
would otherwise inflate validation scores badly.

---

## Quick start

### Prerequisites

- Python 3.11
- [`uv`](https://docs.astral.sh/uv/) 0.5+
- Docker + Docker Compose

A GPU is not required. The workspace resolves CPU PyTorch wheels deliberately —
see [`pyproject.toml`](pyproject.toml) for why.

### Install

```bash
git clone https://github.com/alex-krasnoshtanov/CV-Pipeline-Deployment-Platform.git
cd CV-Pipeline-Deployment-Platform
uv sync
```

### Configure

```bash
cp configs/env/env.example configs/env/.env
```

Fill in every value marked `REQUIRED`. The compose stack refuses to start with
an unset required secret rather than falling back to a default — generate each
with `openssl rand -hex 32`.

### Run the full stack

```bash
cd infra/local
docker compose up --build
```

| Service | URL | Notes |
|---|---|---|
| Frontend | <http://localhost:3000> | Researcher UI |
| Backend API | <http://localhost:8000/docs> | OpenAPI / Swagger |
| Postgres | `localhost:5433` | Host 5433 → container 5432 |
| Prometheus | <http://localhost:9090> | Scrapes backend `/metrics` |

`docker compose down` stops everything; add `-v` to wipe volumes.

### Backend only, with reload

```bash
uv run uvicorn api.main:app \
    --reload --host 0.0.0.0 --port 8000 \
    --app-dir apps/backend/src
```

---

## Data and model weights

**Neither the dataset nor a trained checkpoint ships with this repository.**

The NPEC *Arabidopsis* plate dataset is not mine to redistribute, so it is
absent by design — `data/` contains only directory placeholders and a single
sample plate used in the documentation. The API, the CLI and the full test
suite all run without it.

Model weights resolve through a small registry in
[`weights.py`](packages/cv-pipeline/src/cv_pipeline/weights.py), which maps a
version string to a download URL and caches the result under
`~/.cache/cv-pipeline/models/`. Weights are published as GitHub Release assets:
the release endpoint streams raw bytes and needs no credentials on a public
repository.

> **Current status:** no public weights release exists yet. Until one is
> published, point `MODEL_PATH` at your own checkpoint, or run the backend in
> remote mode with `MODEL_ENDPOINT_URL`. See
> [add a new model version](docs/source/how-to/add-a-new-model-version.md) for
> the publishing flow.

To train on your own data, arrange it as:

```
my_data/
├── train/
│   ├── images/
│   └── masks/
└── val/
    ├── images/
    └── masks/
```

then run `uv run cv-pipeline train --data-dir my_data/train --val-dir my_data/val`.

---

## Repository layout

```
.
├── apps/
│   ├── backend/                  FastAPI service
│   │   └── src/api/
│   │       ├── auth/             API keys, JWT, OAuth/OIDC, sessions, bcrypt
│   │       ├── db/               SQLAlchemy models, session, seed
│   │       ├── middleware/       Request ID + exception handlers
│   │       ├── routers/          auth, explain, feedback, health, infer,
│   │       │                     monitoring, stats, users
│   │       ├── schemas/          Pydantic request/response models
│   │       ├── services/         drift, endpoint client, explain, feedback,
│   │       │                     inference, storage, stats
│   │       └── metrics.py        Prometheus instrumentation
│   └── frontend/                 Next.js researcher UI
├── packages/
│   └── cv-pipeline/              Installable package: model, infer, train, CLI
├── infra/
│   ├── local/                    docker-compose.yml (4 services)
│   ├── server/                   Portainer on-premise stack
│   ├── monitoring/prometheus/    Prometheus image + scrape config
│   ├── cloud/
│   │   ├── train_environment/    Azure ML training image + registration
│   │   ├── training_jobs/        Manual training submission
│   │   └── endpoint/             Azure ML inference endpoint image
│   └── airflow/
│       ├── dags/                 8 orchestration DAGs
│       └── training_code/        Cloud preprocess + train entry points
├── scripts/azure/                Stage data, register env, deploy endpoint
├── configs/
│   ├── env/env.example           Environment template
│   └── model/                    Training defaults
├── docs/
│   ├── architecture/             Mermaid diagrams per environment
│   └── source/                   Sphinx documentation source
├── .github/workflows/            ci, docs, images, pr-title
├── pyproject.toml                uv workspace root
└── uv.lock
```

---

## The `cv-pipeline` package

`cv-pipeline` is a standalone installable library — it runs without a server,
a database, or a cloud account.

The design rule that shapes it: **the package never loads weights itself.** The
caller passes in a loaded model. That keeps the library free of any opinion
about where checkpoints live, which is what lets the same code back a CLI, a
FastAPI service, and an Azure ML scoring script without conditionals.

```bash
uv run cv-pipeline infer --image plate.png --output results/
uv run cv-pipeline train --data-dir data/train --val-dir data/val --epochs 50
```

| Module | Responsibility |
|---|---|
| `segmentation.py` | U-Net construction and mask prediction |
| `landmarks.py` | Root-tip extraction from a predicted mask |
| `infer.py` | Orchestrates segmentation → landmarks → result schema |
| `explain.py` | Seg-Grad-CAM heatmap generation |
| `train.py` | Training loop, dataset-level metrics, checkpointing |
| `preprocessing.py` | Patching, normalisation, split handling |
| `weights.py` | Version registry, download and cache |
| `schema.py` | Typed result contract shared with the API |
| `validation.py` | Input and mask validation |

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness plus model-readiness state |
| `POST` | `/infer` | Segment an uploaded image, persist the prediction |
| `POST` | `/explain` | Seg-Grad-CAM heatmap for an image |
| `POST` | `/feedback` | Flag a prediction as incorrect |
| `GET` | `/stats` | Aggregate prediction statistics |
| `GET` | `/monitoring` | Drift and alert state |
| `*` | `/auth/*` | Login, OAuth callback, session lifecycle |
| `*` | `/users/*` | User administration |
| `GET` | `/metrics` | Prometheus exposition |

Full generated reference:
[`docs/source/reference/backend-api.md`](docs/source/reference/backend-api.md).

Every error response carries a stable machine-readable code — see
[error codes](docs/source/explanation/error-codes.md). `MODEL_NOT_READY` (503)
is returned while the lifespan handler is still loading the model, rather than
failing an inference request with a generic 500.

---

## Authentication

Four mechanisms, layered:

- **API keys** — bcrypt-hashed, for programmatic and robotic-platform clients.
- **Email + password** — bcrypt, for human researchers.
- **GitHub OAuth** — via Authlib, with an independently signed state cookie.
- **JWT sessions** — short-TTL signed tokens for browser sessions.

Credentialed CORS is enforced with an explicit origin allow-list; `*` is
rejected rather than silently permitted, because browsers refuse wildcard
origins on credentialed requests anyway.

See [security model](docs/source/explanation/security-model.md).

---

## Deployment targets

| Target | Orchestration | Notes |
|---|---|---|
| **Local** | Docker Compose | Four services, self-contained, no cloud dependency |
| **On-premise** | Portainer on a shared GPU server | Images pulled from GHCR, Traefik ingress |
| **Cloud** | Azure Container Apps + Azure ML | Managed online endpoint serves the model |

Architecture notes per target:
[local](docs/architecture/deployment-local.md) ·
[on-premise](docs/architecture/deployment-on-premise.md) ·
[cloud](docs/architecture/deployment-cloud.md)

The local stack is deliberately *not* wired to read the shared
`configs/env/.env` wholesale. Its compose file declares an explicit allow-list
of variables, so a cloud-only setting such as `MODEL_ENDPOINT_URL` cannot leak
in and silently flip local inference into remote mode.

---

## Monitoring, drift and retraining

Prometheus scrapes the backend's `/metrics`. On top of the raw counters the
backend evaluates four configurable alert conditions:

| Threshold | Default | Meaning |
|---|---|---|
| `ALERT_CONFIDENCE_MIN` | 0.60 | Floor for an individual prediction |
| `ALERT_LOW_CONF_FRACTION` | 0.20 | Share of low-confidence predictions tolerated |
| `ALERT_LATENCY_P95_MS` | 5000 | p95 inference latency ceiling |
| `ALERT_ERROR_RATE` | 0.05 | Error-rate ceiling |

`RETRAIN_FEEDBACK_THRESHOLD` (default 50) governs when accumulated feedback
triggers the retraining DAG.

---

## CI/CD

| Workflow | Trigger | Does |
|---|---|---|
| `ci.yml` | push / PR to main | Ruff lint + format check, pytest with 85% coverage gate, frontend lint/test/build |
| `cd.yml` | push to main | Builds and pushes three images, Trivy scan, on-premise deploy, smoke test, OIDC auth check, Container Apps rollout |
| `images.yml` | PR / version tag | Proves all three Dockerfiles still build; publishes semver-tagged images from a tag |
| `deploy-endpoint.yml` | manual | Redeploys the Azure ML scoring endpoint |
| `docs.yml` | docs or source changes | Generates OpenAPI markdown, builds Sphinx, uploads the site |
| `pr-title.yml` | PR | Enforces conventional-commit PR titles |

Image publishing authenticates with the built-in `GITHUB_TOKEN`. GHCR was
chosen over Docker Hub specifically so that half of the pipeline needs no
repository secrets, and the cloud half needs none either because it
authenticates through OIDC federated credentials rather than a stored client
secret.

**The two deploy chains are gated and currently skipped.** They targeted a
Portainer host on the university campus network and the university's Azure
subscription, and neither exists now, so both sit behind an `ENABLE_DEPLOY`
repository variable. Everything before them runs on every push.

They did run. [`docs/delivery/`](docs/delivery/) documents each gate and
[`docs/evidence/`](docs/evidence/) holds captured output, including a
4,086-line transcript of one full run and a health capture showing the cloud
apps on their nineteenth revision, serving, with the model loaded.

---

## Testing

```bash
uv run pytest -m unit                    # fast unit suite
uv run ruff check . && uv run ruff format --check .
cd apps/frontend && npm run test:ci      # vitest
```

Currently **426 unit tests, 92.6% line coverage** across `cv_pipeline` and
`api`, with CI failing below 85%.

---

## Documentation

[`docs/delivery/`](docs/delivery/) is the pipeline: federated cloud auth,
approval gates, the post-deploy smoke test, the Container Apps rollout, and
metrics. [`docs/evidence/`](docs/evidence/) holds the captured runs behind it.
[`docs/cost-analysis.md`](docs/cost-analysis.md) prices the cloud tier.

Sphinx sources in [`docs/source/`](docs/source/), organised on the
[Diátaxis](https://diataxis.fr/) split — tutorials, how-to guides, reference,
explanation. Build locally:

```bash
uv sync --group docs
uv run scripts/generate_openapi_docs.py
cd docs && make html
```

---

## Attribution

This is group work. The table below is derived from the original repository's
git history — files touched and lines attributed per component.

| Component | Files | Lines | My share | Dominant other author |
|---|---:|---:|---:|---|
| `infra/monitoring` | 3 | 75 | **100%** | — |
| `infra/server` | 1 | 88 | **97%** | — |
| `.github/workflows` | 5 | 900 | **61%** | Danil 221 |
| `scripts/azure` | 12 | 1,789 | **57%** | Filipp 641 |
| `apps/frontend` | 40 | 6,201 | **47%** | Maksym 2,318; Danil 981 |
| `apps/backend` | 97 | 12,456 | **43%** | Filipp 4,831; Danil 1,809 |
| `infra/airflow` | 19 | 4,413 | **31%** | Filipp 3,014 (68%) |
| `infra/cloud` | 8 | 458 | **26%** | Filipp 341 (74%) |
| `packages/cv-pipeline` | 24 | 6,838 | **21%** | Filipp 5,146 (75%) |

**In plain terms:** I led the deployment and operations surface — CI/CD,
container images, the on-premise and monitoring stacks, and the Azure
provisioning scripts — and was the largest single contributor to the backend
API and the frontend. The `cv-pipeline` package itself, the Airflow DAGs and
the Azure ML training layer were predominantly Filipp's work; I contributed to
all three but did not lead them.

Original team: Oleksii Krasnoshtanov, Filipp Lotsmanov, Danil Sysenko,
Maksym Steshkin, Marin Chiosa.

Two notes on how to read the table. Line counts are a poor proxy for
contribution — they reward whoever typed the boilerplate — so treat the
percentages as a rough shape, not a scoreboard. And commit counts are worse:
Maksym's frontend work landed in four large commits, which is why a
commit-ranked list would have understated it badly. A small number of the
original repository's commits were also authored by a GitHub Copilot agent.

---

## License and reuse

**No open-source license is offered, deliberately.**

This repository is jointly authored. I hold copyright only in my own
contributions, and I have not obtained my collaborators' consent to relicense
theirs — so attaching an MIT or Apache header would misrepresent rights I do
not hold. The code is published for review and discussion, not for reuse.

If you want to build on something here, open an issue and I will sort out
permission properly.

---

## Author

**Oleksii (Alex) Krasnoshtanov** — [@alex-krasnoshtanov](https://github.com/alex-krasnoshtanov)
