# CV Pipeline — Plant Phenotyping Platform

Computer-vision pipeline for plant organ segmentation and root-tip detection on
*Arabidopsis thaliana* seedling images, deployed across local, on-premise and
cloud targets.

Built as a year-2 group project at Breda University of Applied Sciences,
February–June 2026.

> **Status: in preparation.** This repository is being assembled from the
> original group repository. Content, structure and documentation are
> incomplete until this notice is removed.

## Components

- **`cv-pipeline`** — U-Net segmentation and landmark detection, packaged as an
  installable Python library
- **Backend** — FastAPI service exposing inference and health endpoints
- **Frontend** — Next.js interface for researchers
- **Database** — PostgreSQL for predictions, feedback and users
- **Orchestration** — Airflow DAGs for data versioning and training
- **Monitoring** — Prometheus metrics, drift detection, feedback-driven
  retraining

## Attribution

Group project. Contributors to the original repository, by commit count:

| Contributor | Commits |
|---|---:|
| Oleksii Krasnoshtanov | 249 |
| Filipp Lotsmanov | 85 |
| Danil Sysenko | 63 |
| Marin Chiosa | 33 |

A breakdown of which components each of us worked on will be added before this
repository is made public. Nothing here should be read as solely my work until
that section exists.

## Notes

Infrastructure endpoints, credentials and internal hostnames from the original
deployment are deliberately excluded. Configuration is supplied via environment
variables; see `.env.example` once added.
