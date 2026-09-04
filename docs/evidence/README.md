# Evidence

Captured output from runs that actually happened, kept because a pipeline
described is not a pipeline demonstrated.

**These files are unedited.** They still contain the university's resource group
name, the Container App names and their FQDNs, and one reference to a course
learning outcome in a header comment. The hostnames stopped resolving when the
subscription was decommissioned. Editing a log to tidy it up would defeat the
only reason to keep one, so the prose in [`../delivery/`](../delivery/) carries
the placeholders and these captures stay verbatim.

## `cd-run-27143277518.log`

One complete CD run, 8 June 2026, 4,086 lines. Every job in
[`cd.yml`](../../.github/workflows/cd.yml) appears in it:

| Job | What the log shows |
|---|---|
| `endpoint-contract-check` | scoring contract compared to the last deploy |
| `build-and-push` | backend, frontend and Prometheus images built and pushed to GHCR |
| `scan-images` | Trivy scanning the backend image, SARIF uploaded |
| `deploy-portainer` | stack updated on the on-premise host with images pinned to the commit SHA |
| `smoke-test` | `/health` and `/` probed on the deployed services |
| `azure-auth-test` | OIDC federated credential exchanged for an Azure token |
| `deploy-azure` (×2) | both Container Apps updated, promoted and verified |

The log is a raw Actions download, so it carries ANSI escape sequences and a
job-name column. `grep` for a job name to read one chain:

```bash
grep -F 'deploy-azure' docs/evidence/cd-run-27143277518.log | less
```

## `deployment-health-2026-06-08.txt`

Three commands and their real output, taken from the live cloud deployment on
the same day:

- `az containerapp show` for both apps: `"state": "Running"`, revisions
  `--0000019` and `--0000018`
- `curl /health` on the backend's public FQDN, returning
  `{"status":"ok","model_loaded":true,...,"serving_mode":"azure_ml"}`

Nineteen backend revisions is the number worth noticing. The rollout in
[`azure-container-apps.md`](../delivery/azure-container-apps.md) ran many times,
not once.

## `training-run/`

A 50-epoch training run from 23 May 2026, logged by `cv_pipeline.train`.

| | |
|---|---|
| Run name | `local-comparison` |
| Data | 200 training pairs, 200 validation pairs |
| Best validation F1 | **0.6693**, epoch 39 |
| Final epoch | loss 0.1584, F1 0.6537, IoU 0.5124 |
| Wall clock | 4 min 13 s on an RTX 3060, ~4.8 s/epoch |

`training.log` is the run as it happened, `run_metrics.json` has every epoch's
loss, F1 and IoU, and `training-curve.png` plots it.

**This is not the run behind the published weights.** It is a small local
comparison run on a 200-pair subset, kept because it is a complete and honest
record of the training path executing end to end: dataset load, encoder weights
fetched from the Hub, 50 epochs, best-checkpoint selection, metrics written.
The released `unet-v1` checkpoint reports a validation F1 of 0.848 and was
trained on the full dataset; its own log was left on the university GPU server
and is gone.

The curve is worth a look. Validation F1 sits at roughly zero for five epochs
and then jumps to 0.48, which is the model learning to predict anything at all
instead of an all-background mask. On a segmentation task with sparse positive
pixels that is the local minimum a run either escapes early or never leaves.

It then falls back to zero twice more, at epochs 12-13 and again at 32, and
climbs out both times. The plot annotates both as learning-rate instability,
which is what `lr=2.4e-3` on a 200-pair dataset will do even under the cosine
schedule the trainer uses. The run recovers
because best-checkpoint selection keeps epoch 39 regardless of what epoch 50
looks like. A run whose final epoch happened to land in one of those holes
would have shipped a useless checkpoint if selection had been "last epoch
wins", which is the argument for selecting on the validation metric.
