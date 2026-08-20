"""Hash the Azure ML endpoint "scoring contract" to detect when a redeploy is needed.

The cloud endpoint is expensive to redeploy (re-registering the inference
environment triggers an ACR image build, then the Kubernetes deployment is
re-provisioned -- minutes, not seconds). So CD must NOT redeploy on every push.
This script gives both halves of the semi-automatic flow a single, identical
definition of "did the scoring contract change":

- CD (.github/workflows/cd.yml) runs ``check`` on every push to main. It is
  stdlib-only on purpose, so the drift job needs no pip install and no Azure
  auth. On a mismatch it warns (non-blocking) that a manual redeploy is due.
- The manual deploy workflow (deploy-endpoint.yml) runs ``write`` after a
  successful deploy to record what is now live.

Two independent digests are tracked because they have different deploy costs:

- ``score``: just ``infra/cloud/endpoint/score.py``. This is the deployment's
  ``code_configuration`` -- changing it needs only a redeploy, NOT an env
  rebuild (fast path: ``rebuild_env=false``).
- ``env``:  ``conda.yml`` + ``Dockerfile`` + the whole ``cv_pipeline`` source
  (the library baked into the inference image's wheel). Changing any of these
  means the image must be rebuilt (``rebuild_env=true``).

Splitting them lets the deploy workflow skip the slow ACR rebuild for a
score.py-only change, which is the common case once the explain mode exists.

Exit codes for ``check``:
    0  current contract matches the sentinel (nothing to deploy)
    2  sentinel file missing or unreadable (never deployed / first run)
    3  contract changed since last deploy (redeploy due)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Repo root = three levels up from scripts/azure/endpoint_contract.py.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Files whose change forces only a redeploy (score) vs a full env rebuild (env).
SCORE_FILES = ["infra/cloud/endpoint/score.py"]
ENV_FILES = ["infra/cloud/endpoint/conda.yml", "infra/cloud/endpoint/Dockerfile"]
ENV_GLOB_ROOT = "packages/cv-pipeline/src/cv_pipeline"

DEFAULT_SENTINEL = "infra/cloud/endpoint/.deployed-contract.json"


def _hash_files(rel_paths: list[str]) -> str:
    """Return a deterministic SHA-256 over the given files (path + bytes).

    The relative path is folded into the hash so that renaming a file -- not
    just editing it -- counts as a change. Missing files contribute a fixed
    marker rather than raising, so a deleted file is still a detectable change.

    Args:
        rel_paths: Repo-relative file paths, already in the desired order.

    Returns:
        Hex SHA-256 digest as a string.
    """
    h = hashlib.sha256()
    for rel in rel_paths:
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        fpath = REPO_ROOT / rel
        if fpath.is_file():
            # Normalise CRLF/CR -> LF so the digest is identical whether the
            # files were checked out on Windows (local) or Linux (CI). Without
            # this, a local Windows working tree would hash differently from the
            # CI checkout and the drift check would fire on every push.
            data = fpath.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            h.update(data)
        else:
            h.update(b"<MISSING>")
        h.update(b"\0")
    return h.hexdigest()


def _cv_pipeline_sources() -> list[str]:
    """Return sorted repo-relative paths of every cv_pipeline source file.

    Sorting makes the digest independent of filesystem walk order.

    Returns:
        Sorted list of repo-relative ``.py`` paths under the package source.
    """
    root = REPO_ROOT / ENV_GLOB_ROOT
    files = sorted(p for p in root.rglob("*.py"))
    return [str(p.relative_to(REPO_ROOT).as_posix()) for p in files]


def current_contract() -> dict[str, str]:
    """Compute the current score and env digests from the working tree.

    Returns:
        Mapping with ``score`` and ``env`` hex digests.
    """
    score = _hash_files(SCORE_FILES)
    env = _hash_files(ENV_FILES + _cv_pipeline_sources())
    return {"score": score, "env": env}


def _load_sentinel(path: Path) -> dict | None:
    """Load the recorded contract, or None if it is missing/unreadable.

    Args:
        path: Path to the sentinel JSON file.

    Returns:
        Parsed sentinel dict, or None.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def cmd_print(_args: argparse.Namespace) -> int:
    """Print the current contract digests as JSON."""
    print(json.dumps(current_contract(), indent=2))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Compare the current contract to the sentinel and report drift.

    Writes a short, CI-friendly summary to stdout describing which part (if
    any) changed and what the deploy workflow should do about it.
    """
    sentinel_path = REPO_ROOT / args.sentinel
    current = current_contract()
    recorded = _load_sentinel(sentinel_path)

    if recorded is None:
        print(f"No deployed-contract sentinel at {args.sentinel} (never deployed).")
        print("score_changed=true")
        print("env_changed=true")
        return 2

    score_changed = recorded.get("score") != current["score"]
    env_changed = recorded.get("env") != current["env"]

    print(f"score_changed={'true' if score_changed else 'false'}")
    print(f"env_changed={'true' if env_changed else 'false'}")

    if not score_changed and not env_changed:
        print("Endpoint scoring contract matches the last deploy. Nothing to do.")
        return 0

    if env_changed:
        print("Environment inputs changed (conda/Dockerfile/cv_pipeline).")
        print("-> redeploy with rebuild_env=true (rebuilds the inference image).")
    else:
        print("Only score.py changed.")
        print("-> redeploy with rebuild_env=false (fast: no image rebuild).")
    return 3


def cmd_write(args: argparse.Namespace) -> int:
    """Record the current contract as the deployed sentinel.

    Run by the deploy workflow after a successful deploy so the next push's
    drift check goes green.
    """
    sentinel_path = REPO_ROOT / args.sentinel
    payload = current_contract()
    payload["model_version"] = args.model_version or ""
    payload["git_sha"] = args.git_sha or ""
    payload["deployed_at"] = datetime.now(timezone.utc).isoformat()
    payload["note"] = "Written by deploy-endpoint.yml after a successful deploy."
    sentinel_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.sentinel}: {json.dumps(payload, indent=2)}")
    return 0


def main() -> int:
    """Entry point: dispatch print/check/write subcommands."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("print", help="Print current contract digests as JSON.")

    p_check = sub.add_parser("check", help="Compare current contract to sentinel.")
    p_check.add_argument("--sentinel", default=DEFAULT_SENTINEL)

    p_write = sub.add_parser("write", help="Record current contract as deployed.")
    p_write.add_argument("--sentinel", default=DEFAULT_SENTINEL)
    p_write.add_argument("--model-version", default="")
    p_write.add_argument("--git-sha", default="")

    args = parser.parse_args()
    handlers = {"print": cmd_print, "check": cmd_check, "write": cmd_write}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
