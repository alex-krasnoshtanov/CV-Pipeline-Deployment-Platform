# Delivery

How a commit on `main` becomes a running service, and what stops it when it
should not.

The pipeline lives in [`.github/workflows/cd.yml`](../../.github/workflows/cd.yml).
Seven jobs, in three chains off one build:

```
build-and-push ──┬─> scan-images                    Trivy, HIGH+CRITICAL, SARIF
                 │
                 ├─> deploy-portainer ─> smoke-test  on-premise, self-hosted runner
                 │
                 └─> azure-auth-test ─> deploy-azure  Container Apps, cut + rollback

endpoint-contract-check                              independent, advisory only
```

| Gate | Document |
|---|---|
| Cloud auth with no stored credentials | [`oidc-federated-auth.md`](oidc-federated-auth.md) |
| Human approval before production | [`approval-gates.md`](approval-gates.md) |
| Post-deploy verification | [`smoke-test-gate.md`](smoke-test-gate.md) |
| Container Apps rollout and rollback | [`azure-container-apps.md`](azure-container-apps.md) |
| Metrics and drift signal | [`monitoring.md`](monitoring.md) |

Captured output from real runs is in [`../evidence/`](../evidence/), including a
4,000-line log of the whole chain executing against live infrastructure.

## What ran, and what runs here

Every job ran against real infrastructure. The run shows eight entries because
`deploy-azure` is a two-app matrix. The transcript in
[`../evidence/cd-run-27143277518.log`](../evidence/cd-run-27143277518.log)
covers one full execution on 8 June 2026: three images built and pushed,
scanned, deployed on-premise, smoke-tested, and both Container Apps promoted
behind an inference gate. A health capture from the same day shows the cloud
apps on revisions 19 and 18, serving, with the model loaded.

None of that infrastructure exists now. The Portainer host was a workstation on
the university campus network and the Azure subscription was the university's,
so both deploy chains are gated behind an `ENABLE_DEPLOY` repository variable
and stay skipped. Set it and the matching `vars.*` and they target your own
infrastructure; [`azure-container-apps.md`](azure-container-apps.md) lists what
each one needs.

What still runs on every push to `main` here: three images built and pushed to
GHCR, then scanned with Trivy. That part is verifiable from the Actions tab.

## Numbers from the original run

Measured on the group repository between 13 April and 11 June 2026.

| | |
|---|---|
| Workflow runs | 1,387 |
| Workflows | 9 |
| Merged pull requests | 102 of 131 opened |
| Coverage on `main`, 1 June 2026 | **90.82%** (1,776 statements, 163 uncovered) against an 85% CI floor |

The coverage figure comes from CI run `26759428815` on the group repository,
which is private, so the run itself is not linkable. The floor that produced it
is still enforced in [`ci.yml`](../../.github/workflows/ci.yml), and the suite
in this repository now reports 92.6% against the same 85% gate.

## Redactions

These documents describe infrastructure owned by Breda University of Applied
Sciences. Tenant, subscription and client identifiers, the campus deploy host,
the resource group and the Container App names have been replaced with
placeholders. Nothing about the mechanisms has been changed.

Attribution: the blue-green traffic split and the Prometheus work described in
[`monitoring.md`](monitoring.md) were led by Danil Sysenko. The rest of the
pipeline was mine.
