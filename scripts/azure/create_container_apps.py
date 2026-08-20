#!/usr/bin/env python3
"""One-time, idempotent bootstrap for the CV7 cloud deployment.

Creates -- or, on re-run, updates -- the two Azure Container Apps that the
``deploy-azure`` job in ``.github/workflows/cd.yml`` updates on every push to
main. After this script has run once, CD owns the steady state: it only swaps
the image tag (``az containerapp update --image``). Everything that must NOT
live in a per-push workflow -- secrets, env vars, ingress, the GHCR pull
credential, the ML-endpoint auth -- is configured here, exactly once.

Why a script instead of a list of ``az`` commands in a doc
----------------------------------------------------------
- Idempotent. Safe to re-run. It checks whether each app exists and
  creates or only-updates accordingly, so a half-finished run is recoverable
  and re-running never clobbers the image tag CD last deployed.
- One source of truth for the cloud env contract, committed next to the code.
- No hand-typed command can silently drift from what is in the repo.

Auth model (chosen): the backend calls the Azure ML endpoint with a service
principal whose credentials live as Container App SECRETS (not plain env
vars) and are referenced via ``secretref``. ``endpoint_client.py`` already
selects ``ClientSecretCredential`` when ``AZURE_CLIENT_SECRET`` is present and
falls back to ``DefaultAzureCredential`` otherwise -- so no code change is
needed for this path. The alternative (a managed identity granted the
"AzureML Data Scientist" role) is cleaner but needs an instructor role
assignment; see docs/cloud-setup.md "Upgrade path: managed identity".

Run
---
Populate the environment variables listed under REQUIRED_VARS / SECRET_VARS
(see docs/cloud-setup.md for exactly which value goes where), then:

    python scripts/azure/create_container_apps.py

After it prints the two FQDNs, push to main to trigger the first CD deploy.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time


def _required(name: str) -> str:
    """Read a required setting, or fail with a message naming what is missing."""
    value = os.getenv(name)
    if not value:
        raise SystemExit(
            f"{name} is not set. This script provisions Azure Container Apps "
            f"and deliberately has no defaults - set {name} explicitly so it "
            f"cannot target the wrong subscription or resource group."
        )
    return value


# --- Resource identifiers -------------------------------------------------
# Every value below is required and has no default. The original version
# defaulted to the university's own resource group, Container Apps environment
# and app names, which meant running this script without a fully populated
# environment silently targeted infrastructure belonging to someone else.
#
# The app names must match whatever your deployment workflow targets, or the
# workflow will not find the apps it is meant to update.
RESOURCE_GROUP = _required("AZURE_RESOURCE_GROUP")
ENVIRONMENT = _required("ACA_ENVIRONMENT")
REGISTRY = os.getenv(
    "IMAGE_REGISTRY",
    "ghcr.io/gfgf96/cv-pipeline-deployment-platform",
)
BACKEND_APP = _required("BACKEND_APP")
FRONTEND_APP = _required("FRONTEND_APP")
# Prometheus runs as a third app with INTERNAL ingress (reachable only from
# inside the Container Apps environment, e.g. by the frontend's server-side
# proxy). CD's deploy-azure matrix does not update this app, so it stays on the
# image tag set here; re-run this bootstrap to roll a new Prometheus image.
PROMETHEUS_APP = _required("PROMETHEUS_APP")

# --- Required configuration ----------------------------------------------
# SECRET_VARS are written as Container App secrets and referenced via
# secretref. REQUIRED_VARS are non-sensitive identifiers written as plain
# env vars. Splitting them this way keeps passwords and keys out of the
# Container App's plaintext env listing (visible to anyone with Reader).
SECRET_VARS = [
    "DB_URL",  # full managed-Postgres URL, contains the password
    "API_KEY",
    "ADMIN_API_KEY",
    "JWT_SIGNING_KEY",
    "SESSION_SECRET",
    "AZURE_CLIENT_SECRET",  # the service principal secret (Path A auth)
]
REQUIRED_VARS = [
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_ID",  # the SP app id -- MUST match AZURE_CLIENT_SECRET's SP
    "MODEL_ENDPOINT_NAME",  # e.g. hades-unet-endpoint, used by the SDK invoke
]

# Map each secret to the lowercase Container App secret name it is stored
# under. Secret names must be lowercase alphanumeric with dashes.
SECRET_NAMES = {var: var.lower().replace("_", "-") for var in SECRET_VARS}


# Resolve the Azure CLI once. On Windows `az` is `az.cmd` (a batch file) that
# CreateProcess cannot launch from a plain argument list, so the call must go
# through the shell; on POSIX `az` is a normal executable and we skip the shell
# entirely.
_AZ = shutil.which("az") or "az"
_IS_WINDOWS = os.name == "nt"


def _az(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run an ``az`` command across platforms.

    Callers may include a leading ``"az"`` in ``args`` (legacy style) or omit
    it; either works. On Windows the command is run through the shell with
    every argument double-quoted, so shell metacharacters (``& | < > ( )``)
    inside secret values are treated literally. On POSIX the argument list is
    passed directly with no shell, which needs no quoting.

    Args:
        args: Arguments following ``az`` (a leading ``"az"`` is stripped).
        check: When True, exit with the command's return code on failure.

    Returns:
        The completed process, stdout/stderr captured as text.
    """
    if args and args[0] == "az":
        args = args[1:]
    if _IS_WINDOWS:
        quoted = " ".join('"' + str(a).replace('"', '""') + '"' for a in args)
        command = f'"{_AZ}" {quoted}'
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
    else:
        result = subprocess.run([_AZ, *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"az {' '.join(args)} failed:", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)
    return result


def _run(args: list[str]) -> str:
    """Run an ``az`` command and return its stdout, raising on failure."""
    return _az(args).stdout.strip()


def _check_config() -> dict[str, str]:
    """Verify every required env var is present; return them as a dict."""
    # The cloud database is the managed Postgres. That URL differs from the
    # local-compose DB_URL in .env, which points at the `db` service and will
    # NOT resolve inside Container Apps. CLOUD_DB_URL, if set, wins -- so a
    # loaded .env cannot accidentally push the local URL to the cloud app.
    if os.getenv("CLOUD_DB_URL"):
        os.environ["DB_URL"] = os.environ["CLOUD_DB_URL"]

    needed = SECRET_VARS + REQUIRED_VARS
    missing = [name for name in needed if not os.getenv(name)]
    if missing:
        print(
            f"ERROR: missing required env vars: {', '.join(missing)}", file=sys.stderr
        )
        print("See docs/cloud-setup.md for what each value should be.", file=sys.stderr)
        raise SystemExit(1)
    return {name: os.environ[name] for name in needed}


def _app_exists(name: str) -> bool:
    """Return True if the Container App already exists in the resource group."""
    result = _az(
        [
            "containerapp",
            "show",
            "--name",
            name,
            "--resource-group",
            RESOURCE_GROUP,
            "--output",
            "none",
        ],
        check=False,
    )
    return result.returncode == 0


def _app_fqdn(name: str) -> str:
    """Return the ingress FQDN of an existing Container App."""
    return _run(
        [
            "az",
            "containerapp",
            "show",
            "--name",
            name,
            "--resource-group",
            RESOURCE_GROUP,
            "--query",
            "properties.configuration.ingress.fqdn",
            "--output",
            "tsv",
        ]
    )


def _secret_args(cfg: dict[str, str]) -> list[str]:
    """Build the ``--secrets name=value ...`` argument list."""
    pairs = [f"{SECRET_NAMES[var]}={cfg[var]}" for var in SECRET_VARS]
    return ["--secrets", *pairs]


def _backend_env_vars(cfg: dict[str, str]) -> list[str]:
    """Backend env vars. Sensitive ones use secretref; the rest are plain.

    Notes:
    - MODEL_ENDPOINT_URL is the flag that flips the backend into azure_ml
      serving mode (see api/config.py serving_mode). Its value is not used by
      the SDK invoke path -- MODEL_ENDPOINT_NAME drives that -- so any
      non-empty value works. We pass the real scoring URI when available for
      clarity. (Phase 3 will make this trigger explicit so the dummy value is
      no longer load-bearing.)
    - We deliberately do NOT set MODEL_VERSION here. In azure_ml mode the
      endpoint serves the model, but the container entrypoint still
      pre-caches weights from the registry whenever MODEL_VERSION is set,
      which would pull ~50 MB from SharePoint on every cold start for no
      reason. Omitting it skips that download cleanly.
    - COOKIE_SECURE=true because cloud ingress is HTTPS.
    - CORS_ORIGINS/FRONTEND_URL/OAUTH_REDIRECT_URI are set later, once the
      frontend FQDN is known (see update_backend_urls).
    """
    model_endpoint_url = os.getenv("MODEL_ENDPOINT_URL", "azure-ml-endpoint")
    plain = [
        f"AZURE_SUBSCRIPTION_ID={cfg['AZURE_SUBSCRIPTION_ID']}",
        f"AZURE_RESOURCE_GROUP={RESOURCE_GROUP}",
        f"AZURE_WORKSPACE_NAME={_required('AZURE_WORKSPACE_NAME')}",
        f"AZURE_TENANT_ID={cfg['AZURE_TENANT_ID']}",
        f"AZURE_CLIENT_ID={cfg['AZURE_CLIENT_ID']}",
        f"MODEL_ENDPOINT_NAME={cfg['MODEL_ENDPOINT_NAME']}",
        f"MODEL_ENDPOINT_URL={model_endpoint_url}",
        "COOKIE_SECURE=true",
        "LOG_LEVEL=INFO",
    ]
    # Optional GitHub OAuth client id (public identifier).
    if os.getenv("GITHUB_OAUTH_CLIENT_ID"):
        plain.append(f"GITHUB_OAUTH_CLIENT_ID={os.environ['GITHUB_OAUTH_CLIENT_ID']}")
    # Sensitive values resolve from the secrets set on the app.
    secretref = [f"{var}=secretref:{SECRET_NAMES[var]}" for var in SECRET_VARS]
    # Bare KEY=VALUE pairs (no flag): create uses --env-vars, update uses
    # --set-env-vars, so the caller supplies the right flag.
    return [*plain, *secretref]


def _set_registry(app: str) -> None:
    """Attach a GHCR pull credential, unless the packages are public.

    Container Apps cannot pull a private GHCR image without a credential.
    If the org's packages are public this is a no-op; set GHCR_PAT (a token
    with read:packages) and GHCR_USER to enable it.
    """
    pat = os.getenv("GHCR_PAT")
    if not pat:
        print(f"  (skipping GHCR credential for {app}; assuming public packages)")
        return
    user = os.getenv("GHCR_USER", "")
    print(f"  Setting GHCR pull credential on {app}...")
    _run(
        [
            "az",
            "containerapp",
            "registry",
            "set",
            "--name",
            app,
            "--resource-group",
            RESOURCE_GROUP,
            "--server",
            "ghcr.io",
            "--username",
            user,
            "--password",
            pat,
        ]
    )


def _registry_create_args() -> list[str]:
    """Inline ``--registry-*`` args for ``az containerapp create``, if a PAT is set.

    The credential must be present DURING create, because create provisions
    the first revision immediately and pulls the image at that moment. Returns
    an empty list when GHCR_PAT is unset (public packages need no credential).
    """
    pat = os.getenv("GHCR_PAT")
    if not pat:
        return []
    user = os.getenv("GHCR_USER", "")
    return [
        "--registry-server",
        "ghcr.io",
        "--registry-username",
        user,
        "--registry-password",
        pat,
    ]


def _set_single_mode(app: str) -> None:
    """Put the app in single-revision mode so an update rolls one 100% revision.

    The bootstrap's job is to land one healthy revision serving all traffic.
    Multiple-revision mode starts new revisions at 0% traffic, which is what
    CD wants for canary -- cd.yml re-enables it at deploy time -- but is wrong
    here.
    """
    _run(
        [
            "az",
            "containerapp",
            "revision",
            "set-mode",
            "--mode",
            "single",
            "--name",
            app,
            "--resource-group",
            RESOURCE_GROUP,
            "--output",
            "none",
        ]
    )


def _revision_suffix() -> str:
    """A unique revision suffix so each update rolls a fresh revision.

    Updating a secret's value does not restart or roll a revision on its own,
    and an env update that resolves to an unchanged template produces no new
    revision -- so a corrected secret would never be picked up. Forcing a
    unique suffix guarantees a new revision that reloads current secret values.
    """
    return f"r{int(time.time())}"


def ensure_backend(cfg: dict[str, str]) -> str:
    """Create or update the backend Container App; return its FQDN."""
    image = f"{REGISTRY}/backend:latest"
    if _app_exists(BACKEND_APP):
        print(f"{BACKEND_APP} exists -- updating secrets + env (image left to CD)...")
        _set_registry(BACKEND_APP)
        _set_single_mode(BACKEND_APP)
        _run(
            [
                "az",
                "containerapp",
                "secret",
                "set",
                "--name",
                BACKEND_APP,
                "--resource-group",
                RESOURCE_GROUP,
                *_secret_args(cfg),
            ]
        )
        # --revision-suffix forces a fresh revision that reloads the corrected
        # secret values; without it a same-template update is a no-op.
        _run(
            [
                "az",
                "containerapp",
                "update",
                "--name",
                BACKEND_APP,
                "--resource-group",
                RESOURCE_GROUP,
                "--set-env-vars",
                *_backend_env_vars(cfg),
                "--revision-suffix",
                _revision_suffix(),
                "--output",
                "none",
            ]
        )
    else:
        print(f"Creating {BACKEND_APP}...")
        _run(
            [
                "az",
                "containerapp",
                "create",
                "--name",
                BACKEND_APP,
                "--resource-group",
                RESOURCE_GROUP,
                "--environment",
                ENVIRONMENT,
                "--image",
                image,
                "--target-port",
                "8000",
                "--ingress",
                "external",
                "--min-replicas",
                "1",
                "--max-replicas",
                "3",
                # Inline so the first revision can pull during create.
                *_registry_create_args(),
                *_secret_args(cfg),
                "--env-vars",
                *_backend_env_vars(cfg),
                "--output",
                "none",
            ]
        )
    fqdn = _run(
        [
            "az",
            "containerapp",
            "show",
            "--name",
            BACKEND_APP,
            "--resource-group",
            RESOURCE_GROUP,
            "--query",
            "properties.configuration.ingress.fqdn",
            "--output",
            "tsv",
        ]
    )
    print(f"  Backend FQDN: {fqdn}")
    return fqdn


def ensure_prometheus(backend_fqdn: str) -> str:
    """Create or update the Prometheus Container App; return its internal FQDN.

    Internal ingress, so it is reachable only from inside the environment (the
    frontend's server-side proxy calls it; the browser never does). It scrapes
    the backend's PUBLIC FQDN over HTTPS on 443 -- the backend has external
    ingress, so that is the address that resolves. A single replica only: each
    replica keeps its own TSDB, so scaling out would split the metric history.
    """
    image = f"{REGISTRY}/prometheus:latest"
    env = [
        f"BACKEND_TARGET={backend_fqdn}:443",
        "SCRAPE_SCHEME=https",
    ]
    if _app_exists(PROMETHEUS_APP):
        print(f"{PROMETHEUS_APP} exists -- updating env...")
        _set_registry(PROMETHEUS_APP)
        _set_single_mode(PROMETHEUS_APP)
        _run(
            [
                "az",
                "containerapp",
                "update",
                "--name",
                PROMETHEUS_APP,
                "--resource-group",
                RESOURCE_GROUP,
                "--set-env-vars",
                *env,
                "--revision-suffix",
                _revision_suffix(),
                "--output",
                "none",
            ]
        )
    else:
        print(f"Creating {PROMETHEUS_APP}...")
        _run(
            [
                "az",
                "containerapp",
                "create",
                "--name",
                PROMETHEUS_APP,
                "--resource-group",
                RESOURCE_GROUP,
                "--environment",
                ENVIRONMENT,
                "--image",
                image,
                "--target-port",
                "9090",
                "--ingress",
                "internal",
                "--min-replicas",
                "1",
                "--max-replicas",
                "1",
                *_registry_create_args(),
                "--env-vars",
                *env,
                "--output",
                "none",
            ]
        )
    fqdn = _run(
        [
            "az",
            "containerapp",
            "show",
            "--name",
            PROMETHEUS_APP,
            "--resource-group",
            RESOURCE_GROUP,
            "--query",
            "properties.configuration.ingress.fqdn",
            "--output",
            "tsv",
        ]
    )
    print(f"  Prometheus internal FQDN: {fqdn}")
    return fqdn


def ensure_frontend(backend_fqdn: str, prometheus_fqdn: str | None) -> str:
    """Create or update the frontend Container App; return its FQDN.

    The frontend reads BACKEND_URL and PROMETHEUS_URL server-side at runtime
    (Next.js proxy routes), so it just needs the backend's HTTPS FQDN and, when
    Prometheus is deployed, its internal HTTPS FQDN. If ``prometheus_fqdn`` is
    None the operational charts render their "not configured" note and nothing
    else breaks.
    """
    image = f"{REGISTRY}/frontend:latest"
    env = [f"BACKEND_URL=https://{backend_fqdn}"]
    if prometheus_fqdn:
        env.append(f"PROMETHEUS_URL=https://{prometheus_fqdn}")
    if _app_exists(FRONTEND_APP):
        print(f"{FRONTEND_APP} exists -- updating env (image left to CD)...")
        _set_registry(FRONTEND_APP)
        _set_single_mode(FRONTEND_APP)
        _run(
            [
                "az",
                "containerapp",
                "update",
                "--name",
                FRONTEND_APP,
                "--resource-group",
                RESOURCE_GROUP,
                "--set-env-vars",
                *env,
                "--revision-suffix",
                _revision_suffix(),
                "--output",
                "none",
            ]
        )
    else:
        print(f"Creating {FRONTEND_APP}...")
        _run(
            [
                "az",
                "containerapp",
                "create",
                "--name",
                FRONTEND_APP,
                "--resource-group",
                RESOURCE_GROUP,
                "--environment",
                ENVIRONMENT,
                "--image",
                image,
                "--target-port",
                "3000",
                "--ingress",
                "external",
                # min 1 for demo-day reliability (no cold start mid-demo). Switch
                # to 0 after the block if off-hours cost matters -- see cost doc.
                "--min-replicas",
                "1",
                "--max-replicas",
                "3",
                *_registry_create_args(),
                "--env-vars",
                *env,
                "--output",
                "none",
            ]
        )
    fqdn = _run(
        [
            "az",
            "containerapp",
            "show",
            "--name",
            FRONTEND_APP,
            "--resource-group",
            RESOURCE_GROUP,
            "--query",
            "properties.configuration.ingress.fqdn",
            "--output",
            "tsv",
        ]
    )
    print(f"  Frontend FQDN: {fqdn}")
    return fqdn


def update_backend_urls(backend_fqdn: str, frontend_fqdn: str) -> None:
    """Set the backend URLs that depend on the frontend FQDN being known.

    With the Next.js proxy the browser only ever talks to the frontend
    origin, so CORS is not strictly required for the app to work -- but we
    set CORS_ORIGINS to the frontend FQDN anyway so any direct browser call
    to the backend (e.g. a curl from the FastAPI docs) is not blocked.
    """
    print("Setting backend CORS_ORIGINS / FRONTEND_URL / OAUTH_REDIRECT_URI...")
    _run(
        [
            "az",
            "containerapp",
            "update",
            "--name",
            BACKEND_APP,
            "--resource-group",
            RESOURCE_GROUP,
            "--set-env-vars",
            f"CORS_ORIGINS=https://{frontend_fqdn}",
            f"FRONTEND_URL=https://{frontend_fqdn}",
            f"OAUTH_REDIRECT_URI=https://{backend_fqdn}/auth/github/callback",
            "--output",
            "none",
        ]
    )


def wire_prometheus_only() -> None:
    """Add live metrics to an already-bootstrapped deployment.

    Creates (or refreshes) ONLY the Prometheus Container App and points the
    frontend's PROMETHEUS_URL at it. Unlike the full bootstrap this needs no
    secrets and never rewrites the backend secrets/env or the frontend's other
    env vars -- it only upserts PROMETHEUS_URL (--set-env-vars merges) -- so it
    is safe to run against the live apps CD already owns.

    Use when backend + frontend exist but the operational charts still show
    "Live metrics are not configured" (i.e. the deployment was bootstrapped
    before Prometheus was wired up). Needs only `az login`; set GHCR_PAT /
    GHCR_USER as well if the org's GHCR packages are private.
    """
    if not _app_exists(BACKEND_APP):
        print(
            f"ERROR: {BACKEND_APP} does not exist. Run the full bootstrap "
            "(python scripts/azure/create_container_apps.py) first.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not _app_exists(FRONTEND_APP):
        print(
            f"ERROR: {FRONTEND_APP} does not exist. Run the full bootstrap first.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    backend_fqdn = _app_fqdn(BACKEND_APP)
    print(f"Backend FQDN: {backend_fqdn}")

    prometheus_fqdn = ensure_prometheus(backend_fqdn)

    # Upsert ONLY PROMETHEUS_URL on the frontend. --set-env-vars merges, so
    # BACKEND_URL and every other env var already on the app are preserved.
    print(f"Pointing {FRONTEND_APP} PROMETHEUS_URL at the Prometheus FQDN...")
    _set_single_mode(FRONTEND_APP)
    _run(
        [
            "az",
            "containerapp",
            "update",
            "--name",
            FRONTEND_APP,
            "--resource-group",
            RESOURCE_GROUP,
            "--set-env-vars",
            f"PROMETHEUS_URL=https://{prometheus_fqdn}",
            "--revision-suffix",
            _revision_suffix(),
            "--output",
            "none",
        ]
    )

    print("\n=== Prometheus wired ===")
    print(f"Prometheus: https://{prometheus_fqdn} (internal to the environment)")
    print("Frontend now has PROMETHEUS_URL; reload the dashboard to see charts.")


def main() -> None:
    cfg = _check_config()
    backend_fqdn = ensure_backend(cfg)
    # Prometheus before the frontend so its internal FQDN can be injected as
    # PROMETHEUS_URL when the frontend is created/updated.
    prometheus_fqdn = ensure_prometheus(backend_fqdn)
    frontend_fqdn = ensure_frontend(backend_fqdn, prometheus_fqdn)
    update_backend_urls(backend_fqdn, frontend_fqdn)

    print("\n=== Bootstrap complete ===")
    print(f"Backend:    https://{backend_fqdn}")
    print(f"Frontend:   https://{frontend_fqdn}")
    print(f"Prometheus: https://{prometheus_fqdn} (internal to the environment)")
    print("\nVerify, then push to main to let CD take over:")
    print(f"  curl https://{backend_fqdn}/health")
    print(f"  open  https://{frontend_fqdn}")


if __name__ == "__main__":
    # `prometheus-only` adds live metrics to an existing deployment without
    # touching secrets; no argument runs the full first-time bootstrap.
    if len(sys.argv) > 1 and sys.argv[1] == "prometheus-only":
        wire_prometheus_only()
    else:
        main()
