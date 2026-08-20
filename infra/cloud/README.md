# Cloud Deployment (Azure Container Apps)

This folder stores cloud deployment assets for Azure.

Target platform:

- Azure Container Apps for backend and frontend containers.
- Azure Database for PostgreSQL for managed persistence.
- Azure ML for training, model registry, and managed inference endpoints.

Suggested structure:

- `bicep/` or `terraform/`: infrastructure definitions for networking, ACA, and database resources.
- `deploy/`: environment rollout scripts for dev/stage/prod.
- `aca/`: Container Apps revisions, ingress, and secrets configuration references.

AKS is out of scope for this project architecture and should not be used as the default path.


## Deploy ordering: endpoint before backend (required for explain)

The backend Container App and the Azure ML scoring endpoint deploy on
independent schedules. The backend's cloud explain path POSTs `mode="explain"`
and parses the reply as an `ExplanationResult`.

If a new backend revision rolls out **before** the new `score.py` is live on the
endpoint, the endpoint still answers with an `InferenceResult` shape (no
`heatmap_b64`), so every cloud `/explain` returns a hard 500 until the endpoint
catches up. The frontend's 502/503/504 handler does not mask this (it is a clean
500), and `_call_endpoint_explain` raises an explicit "redeploy the endpoint
first" error to make the cause obvious in logs.

**Therefore, when shipping a change to the explain contract (`score.py` or the
`cv_pipeline` explain code):**

1. Run the **Deploy Endpoint** workflow first and confirm it is live
   (`mode="explain"` returns a heatmap). Use `rebuild_env=true` whenever the
   `cv_pipeline` wheel changed (e.g. the first time explain support is added).
2. Only then let the backend Container App revision roll out (merge to `main`).
3. Commit the updated `.deployed-contract.json` so CD's drift check goes green.

For a backwards-compatible change (the endpoint already supports `explain`),
ordering does not matter and the backend can roll out first.
