# Monitoring - Prometheus

Operational metrics for the CV pipeline. The FastAPI backend already exposes
Prometheus metrics at `/metrics` (via `prometheus-fastapi-instrumentator`). This
folder builds a small Prometheus image that scrapes that endpoint; the Next.js
dashboard renders the time-series (request rate, latency p50/p95/p99, error
rate) under **Operational Monitoring**.

## One image, three environments

Neither Portainer (stack content is pushed via the API, with no repo files on
the host) nor Azure Container Apps (no bind mounts) can mount a config file from
the repo. So instead of shipping a static `prometheus.yml`, the image renders it
at start-up from environment variables. The same image runs everywhere; only the
variables change.

| Env var | Meaning | Local | On-prem (Portainer) | Cloud (ACA) |
|---|---|---|---|---|
| `BACKEND_TARGET` | host:port of the backend `/metrics` | `backend:8000` | `backend:8000` | `<backend-fqdn>:443` |
| `SCRAPE_SCHEME` | `http` or `https` | `http` | `http` | `https` |
| `PROMETHEUS_RETENTION` | TSDB retention (optional) | `15d` | `15d` | `15d` |

The frontend reads the Prometheus address from its own env var, server-side only
(never exposed to the browser):

| Env var | Set on | Value |
|---|---|---|
| `PROMETHEUS_URL` | frontend | `http://prometheus:9090` (local / on-prem), `https://<prometheus-internal-fqdn>` (cloud) |

If `PROMETHEUS_URL` is unset, the dashboard shows a "not configured" note and
everything else keeps working.

## Where it is wired

- **Local**: `infra/local/docker-compose.yml` builds this image and runs it on
  `localhost:9090`.
- **On-prem**: `infra/server/docker-compose.portainer.yml` pulls the image from
  GHCR (built by `.github/workflows/cd.yml`) and exposes it on port `2029`.
- **Cloud**: `scripts/azure/create_container_apps.py` `ensure_prometheus()` runs
  it as an internal-ingress Container App.

## Files

- `prometheus.tmpl.yml` - scrape config template with `__BACKEND_TARGET__` and
  `__SCRAPE_SCHEME__` placeholders.
- `docker-entrypoint.sh` - substitutes the placeholders and execs Prometheus.
- `Dockerfile` - `FROM prom/prometheus`, copies the template and entrypoint.

## Security note

`/metrics` is unauthenticated. Locally and on-prem it is only reachable on the
internal Docker network. In cloud the backend has external ingress, so its
`/metrics` is reachable on the public backend FQDN - acceptable for this
project, but a real production deployment would put it behind internal ingress
or a network policy.
