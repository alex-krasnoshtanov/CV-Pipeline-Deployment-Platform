# The post-deploy gate

A deploy that succeeds and leaves the service broken is worse than a deploy
that fails, because the pipeline reports green and nobody looks again. The
`smoke-test` job exists to make that outcome impossible.

It runs after `deploy-portainer`, on the same self-hosted runner, because the
deploy host sat on the campus network and public GitHub runners cannot route to
it.

## Wait, then probe

The backend downloads model weights on startup, so its container healthcheck
carries a 120-second `start_period`. The job polls `/health` every ten seconds
for up to five minutes before giving up. That covers both cases without special
handling: a cold first deploy where Portainer has just pulled a new image, and
a warm restart that comes back in seconds.

Then two probes:

**Backend `/health`** must return 200, and the response body must contain
`"model_loaded":true`. The second half is the important half. FastAPI answers
200 as soon as the process is up, which tells you the container started and
nothing about whether the model behind it loaded. A missing `MODEL_PATH`, an
unreachable weights URL or a corrupted cache all produce a healthy-looking
service that returns 500 on the first real request.

**Frontend `/`** must return 2xx or 3xx. Next.js redirects an unauthenticated
visit to `/login`, so anything in either range counts as serving.

Either probe failing exits 1 and the run goes red.

## What this catches

Every one of these produced a green deploy before the gate existed:

- an image that pulls and starts, then crashes on its first request
- a missing environment variable that stops the model loading
- a port mapping regression in the compose file
- a deploy that updated one service and silently left the other on the old
  image

## Configuration

The two URLs come from `vars.DEPLOY_BACKEND_URL` and
`vars.DEPLOY_FRONTEND_URL`. In the original deployment they pointed at two
ports on one campus host. They are repository variables so the pipeline can be
pointed at different infrastructure without editing the workflow, which is also
why no host address appears in this repository.

A cloud equivalent of this job was designed and never ran, because the Azure
apps were never bootstrapped with real secrets. See
[`azure-container-apps.md`](azure-container-apps.md).

## Evidence

[`../evidence/cd-run-27143277518.log`](../evidence/cd-run-27143277518.log)
contains a full run of the chain with this job passing, and
[`../evidence/deployment-health-2026-06-08.txt`](../evidence/deployment-health-2026-06-08.txt)
is a health capture taken from the deployed stack.
