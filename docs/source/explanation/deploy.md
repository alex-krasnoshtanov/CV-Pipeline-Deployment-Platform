# Deployment

The application is containerized and designed to run across multiple environments: Local, On-Premise, and Cloud.

## Deployment Targets

1. **Local (Dev)**
   - Managed via `docker compose` in `infra/local/docker-compose.yml`.
   - Runs the backend, frontend, and a local PostgreSQL 16 database.
   - Ideal for rapid iteration and testing.

2. **On-Premise (shared university server)**
   - Deployed via Portainer.
   - Images are built by GitHub Actions and pushed to GHCR (GitHub Container Registry).
   - The Portainer agent pulls these images to run on the shared GPU-enabled servers.

3. **Cloud (Azure)** — *deployed and Running.*
   - The backend and frontend run on Azure Container Apps (ACA), deployed by the `cd.yml` `az containerapp update` step on merge to `main`. Both apps report `Running` and serve a live `/health`.
   - In this target, inference would run via an Azure Machine Learning (AML) endpoint; on `main` the AML `endpoint_client` path is dormant and inference runs in-process through the local `cv-pipeline`.
   - Planned: Azure Monitor for metrics and logs. Current monitoring is the app's Prometheus `/metrics` endpoint plus a rolling-confidence drift detector.

## CI/CD Workflow
All pushes to `main` and Pull Requests trigger GitHub Actions to run the test suite, linting, and docs builds. For `main` branch merges, the CI pipeline builds Docker images and pushes them to GHCR, ensuring the Portainer instance can pull the latest versions.
