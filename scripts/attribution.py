"""Recompute the per-component attribution table in the README.

Two independent measures, because either one alone misleads:

* **Surviving lines** -- ``git blame`` over the final tree, so it answers
  "whose lines are in the code that shipped" rather than "who typed the most at
  some point". Rewrites, reverts and churn do not inflate it.
* **Commits touching the component**, a rough proxy for involvement that is not
  distorted by one person happening to write the verbose parts.

Generated and captured files are excluded. Lockfiles alone are 18,744 lines,
and counting them would credit whoever last ran ``npm install`` with 8,533.

Usage::

    python scripts/attribution.py path/to/group-repo

The group repository is private and lives on a university account, so this
cannot run in CI; it is here so the README's numbers can be reproduced and
checked rather than taken on trust.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_REPO = Path("../archive/2025-26d-fai2-adsai-group-suicidesquad7")

# Every alias each person committed under, collapsed to one name. Four of the
# five used more than one; Danil used four.
IDENTITY = {
    "OleksiiKrasnoshtanov240247": "Oleksii",
    "Filipp Lotsmanov": "Filipp",
    "FilippLotsmanov240843": "Filipp",
    "Danil Sysenko": "Danil",
    "DanilSysenko244760": "Danil",
    "Danildogsuppy": "Danil",
    "willbebettertoday": "Danil",
    "MarinChiosa246602": "Marin",
    "swif": "Maksym",
    "MaksymSteshkin242689": "Maksym",
    "copilot-swe-agent[bot]": "Copilot agent",
    "github-classroom[bot]": "GitHub Classroom",
}

COMPONENTS = [
    "packages/cv-pipeline",
    "apps/backend",
    "apps/frontend",
    "infra/airflow",
    "infra/cloud",
    "infra/monitoring",
    "infra/server",
    "scripts/azure",
    ".github/workflows",
    "docs",
]

EXCLUDE_NAMES = {"uv.lock", "package-lock.json", "poetry.lock", "Pipfile.lock"}
EXCLUDE_SUFFIX = {
    ".lock",
    ".whl",
    ".png",
    ".jpg",
    ".jpeg",
    ".ico",
    ".svg",
    ".pdf",
    ".log",
    ".txt",
}
EXCLUDE_PREFIX = ("docs/evidence/",)


def run(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    ).stdout


def component_of(path: str) -> str | None:
    for c in COMPONENTS:
        if path == c or path.startswith(c + "/"):
            return c
    return None


def included(path: str) -> bool:
    p = Path(path)
    if p.name in EXCLUDE_NAMES or p.suffix.lower() in EXCLUDE_SUFFIX:
        return False
    return not path.startswith(EXCLUDE_PREFIX)


def main() -> int:
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REPO
    if not (repo / ".git").exists():
        print(
            f"not a git repository: {repo}\n"
            "Pass a clone of the group repository as the first argument.",
            file=sys.stderr,
        )
        return 1

    files = [f for f in run(repo, "ls-files").splitlines() if f.strip()]
    tracked = [f for f in files if component_of(f) and included(f)]
    skipped = [f for f in files if component_of(f) and not included(f)]
    print(f"blaming {len(tracked)} files, skipping {len(skipped)} generated/captured\n")

    lines: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for i, f in enumerate(tracked, 1):
        if i % 40 == 0:
            print(f"  ... {i}/{len(tracked)}")
        comp = component_of(f)
        for line in run(
            repo, "blame", "--line-porcelain", "-w", "HEAD", "--", f
        ).splitlines():
            if line.startswith("author "):
                who = line[7:].strip()
                lines[comp][IDENTITY.get(who, who)] += 1

    commits: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for comp in COMPONENTS:
        for name in run(
            repo, "log", "--all", "--no-merges", "--format=%an", "--", comp
        ).splitlines():
            if name.strip():
                commits[comp][IDENTITY.get(name.strip(), name.strip())] += 1

    payload = {
        "lines": {k: dict(v) for k, v in lines.items()},
        "commits": {k: dict(v) for k, v in commits.items()},
        "skipped": skipped,
        "n_files": len(tracked),
    }
    out = Path("attribution.json")
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
