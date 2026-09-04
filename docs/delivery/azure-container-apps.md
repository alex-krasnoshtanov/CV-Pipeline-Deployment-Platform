# Container Apps rollout

The cloud tier updates two Azure Container Apps, backend and frontend, to the
sha-tagged images that `build-and-push` just pushed to GHCR. It runs on a
public runner after `azure-auth-test`, using the same federated credential and
no stored secret.

Backend and frontend go through a matrix with `fail-fast: false`, so a
transient failure on one does not abort a working deploy of the other.

This ran. [`../evidence/cd-run-27143277518.log`](../evidence/cd-run-27143277518.log)
is the full 4,086-line transcript of one execution on 8 June 2026, all eight
jobs, both Container Apps promoted. A week of deploys later the apps were on
revisions 19 and 18 and serving; see
[`../evidence/deployment-health-2026-06-08.txt`](../evidence/deployment-health-2026-06-08.txt)
for the live `/health` response, which reports `model_loaded: true` and
`serving_mode: azure_ml`.

## Update, never create

The job calls `az containerapp update --image`. It refuses to create an app
that does not exist, and says so with an error annotation naming the app and
the resource group. That is deliberate, for three reasons.

**Create takes parameters that should not live in a workflow.** Ingress
configuration, target port, and five secrets: the database password, the JWT
signing key, the session secret and two API keys. Those belong in one-time
`az containerapp secret set` and `env set` calls, not in something that runs on
every push.

**One of them is circular.** The backend's `CORS_ORIGINS`, `FRONTEND_URL` and
`OAUTH_REDIRECT_URI` all need the frontend's fully-qualified domain name, which
does not exist until the frontend app is created. A create-or-update workflow
would have to sequence that itself and would be brittle at exactly the moment
you least want brittleness. `scripts/azure/create_container_apps.py` handles it
instead: create backend, create frontend, then patch the backend's origins with
the frontend FQDN.

**Update is cheap and idempotent.** `create` with the full configuration takes
ninety seconds or more and re-applies settings that are already correct.
`update --image` is the only operation that genuinely differs between two
consecutive deploys.

A pre-flight `az containerapp show` runs before the update so a missing app
fails immediately with a clear message rather than deep inside an `az` error.

## Cut, verify, roll back

`az containerapp revision set-mode --mode multiple` runs before every deploy.
It is idempotent, and without it a Container App serves only its latest
revision, so no traffic control is possible at all.

Then the rollout, per app:

1. Find the newest active revision (the one just deployed) and the
   second-newest (the current stable one).
2. Cut **100%** of traffic to the new revision.
3. Poll it until ready. The probe differs per app: the backend has `/health`,
   the Next.js frontend has no health route, so it gets `/`. Thirty attempts,
   ten seconds apart.
4. For the backend, two hard assertions: `/health` must contain
   `"model_loaded":true`, and a real `POST /infer` with the sample plate must
   return 200 with a `landmark_count` field.
5. On any failure, restore 100% to the previous revision and exit 1.
6. On success, deactivate the old revision and emit a notice naming both.

The window in which unverified code serves traffic is only as long as the
checks take. A label-based test-before-cut would remove even that window, but
it adds plumbing that is sensitive to the `az` CLI version; the rollback gate
is the simpler thing that works.

Two details in that sequence took a while to get right, and both are commented
in the workflow:

**The readiness probe is per-app.** Polling `/health` against the frontend
returns 404 forever, which rolled the frontend back on every single deploy
until the probe was split.

**The in-loop probe is quiet, the final one is verbose.** A normal slow start
would otherwise print thirty connection errors, and a genuinely stuck deploy
would print nothing useful. So `curl` is silenced inside the loop, and one
verbose attempt runs before the rollback.

## The escape hatch, and why it stays off

`INFER_SMOKE_FATAL` controls whether a failing `/infer` rolls the deploy back.
It is `"true"`.

The backend reaches the model through a public tunnel
(`MODEL_ENDPOINT_URL` with a bearer key), so the inference path is genuinely
verifiable from a GitHub runner. Setting the flag to `false` is for the case
where the tunnel is known to be down and other changes still need to ship. Left
at `false` it produces false confidence, because the inference step would then
pass whether or not the model was ever reachable. The non-fatal branch emits a
warning that says exactly that.

**What the gate does not assert:** a non-zero `landmark_count`. The only plate
image this repository may redistribute is a web-downscaled copy, and `unet-v1`
is scale-sensitive enough to correctly return an empty mask at that
resolution. So the gate checks the request path, auth, model load and response
shape, and stops short of segmentation quality. Tightening it needs a
native-resolution fixture that is licensed for redistribution.

## Variables it needs

| Variable | Purpose |
|---|---|
| `ENABLE_DEPLOY` | must be `true` or both deploy jobs skip |
| `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` | the federated identity |
| `AZURE_RESOURCE_GROUP` | resource group holding the apps |
| `AZURE_BACKEND_APP`, `AZURE_FRONTEND_APP` | app names to update |

Plus `secrets.FEEDBACK_BLOB_CONNECTION_STRING` and
`vars.FEEDBACK_BLOB_CONTAINER` for the retraining feedback store.

The subscription and the Container App environment belonged to the university
and no longer exist, so `ENABLE_DEPLOY` is unset here and these jobs stay
skipped. Point them at your own resource group and they work as documented.

## The scoring endpoint is separate

The Azure ML managed endpoint has its own workflow,
[`deploy-endpoint.yml`](../../.github/workflows/deploy-endpoint.yml), and only
runs on manual dispatch. Re-registering the inference environment triggers an
ACR image build and re-provisions the deployment, which takes minutes, so
putting it on every push would be wasteful and risks re-rolling a live endpoint
at the wrong moment.

Instead, CD's `endpoint-contract-check` job tells you *when* a redeploy is due.
It hashes the scoring contract — `score.py`, the conda environment, the
Dockerfile and the `cv_pipeline` source — and compares it against a sentinel
written by the last successful deploy. It never deploys and never fails the
run: it writes a warning and a copy-pasteable command into the job summary, and
reports whether the environment changed, which is what decides between the fast
path and a full image rebuild.

In this repository no sentinel exists, because no endpoint has been deployed
from it. The job therefore reports "never deployed" on every run and stays
green, which is the correct answer.
