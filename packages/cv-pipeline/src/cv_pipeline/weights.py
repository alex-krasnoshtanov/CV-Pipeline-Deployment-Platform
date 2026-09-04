"""Weight checkpoint download and caching.

Local analogue of the cloud model registry: a directory on disk plus
the code that knows how to populate it. Every version the package
knows about has an entry in REGISTRY mapping its version string to a
download URL. First call to get_weights() downloads; subsequent calls
return the cached path.

Design decisions:
- Cache directory defaults to ~/.cache/cv-pipeline/models (follows
  the XDG Base Directory convention). Overridable via
  CV_PIPELINE_CACHE_DIR. Docker containers mount this as a volume so
  weights persist across container restarts.
- REGISTRY is a Python dict rather than a JSON file shipped with the
  package. A dict is simpler, import-time typo-checked, and can be
  replaced with a JSON loader later without breaking callers.
  Alternative considered: per-version URL in an env var - rejected
  because it defers the "what models does this package support" answer
  to deployment config, which is the wrong layer.
- Downloads stream in 1 MB chunks so a 50 MB weight file does not blow
  up memory. We write to a temp file and rename on success so an
  interrupted download cannot leave a corrupt file in the cache.
- A registry entry may pin the file's SHA-256. When it does, a
  downloaded file whose digest differs is deleted and the call fails,
  which turns a corrupted or substituted download into an error rather
  than a model that silently predicts nonsense. Verification runs after
  download and not on a cache hit: re-hashing 98 MB on every process
  start costs more than it protects, and the temp-file rename already
  rules out a truncated cache entry. Delete the cached file to force a
  fresh download and a fresh check.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WeightSpec:
    """Where a checkpoint lives and what it should hash to."""

    #: Direct download URL returning raw bytes.
    url: str

    #: Expected SHA-256 hex digest. None disables the check, which should
    #: only be the case for a version whose digest has not been recorded.
    sha256: str | None = None


# Version string -> where to get it. Add a new entry per new model.
# URLs must return the raw binary, so weights are published as GitHub
# Release assets: the download endpoint streams octet-stream directly
# and needs no credentials for a public repository. If a host returns
# HTML instead, the download fails loudly - see _download below.
#
# A bare string is still accepted in place of a WeightSpec, for a local
# or throwaway URL that has no digest to pin.
REGISTRY: dict[str, WeightSpec | str] = {
    "unet-v1": WeightSpec(
        url=(
            "https://github.com/alex-krasnoshtanov/CV-Pipeline-Deployment-Platform"
            "/releases/download/weights-v1/unet-v1.pth"
        ),
        sha256="36908fab40c9388f6834cbd0a2278f2ad173cd1597d889cf477b6d31fc747f71",
    ),
}

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "cv-pipeline" / "models"


def get_cache_dir() -> Path:
    """Return the directory where cached weights are stored.

    Reads CV_PIPELINE_CACHE_DIR from the environment if set, otherwise
    uses the XDG default. Creates the directory if missing.

    Returns:
        Absolute path to the cache directory.
    """
    raw = os.getenv("CV_PIPELINE_CACHE_DIR")
    cache = Path(raw).expanduser() if raw else _DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def list_versions() -> list[str]:
    """Return the sorted list of known model versions."""
    return sorted(REGISTRY)


def get_weights(version: str) -> Path:
    """Return the local path to the weights file for the given version.

    Downloads from REGISTRY[version] if the file is not already cached.

    Args:
        version: A key from REGISTRY, e.g. "unet-v1".

    Returns:
        Absolute path to the local .pth file.

    Raises:
        KeyError: If version is not in REGISTRY.
        RuntimeError: If the download fails or returns non-binary
            content (indicates a wrong or preview-page URL).
    """
    if version not in REGISTRY:
        raise KeyError(
            f"Unknown version '{version}'. "
            f"Known versions: {list_versions()}. "
            f"Add a new entry to REGISTRY in cv_pipeline/weights.py."
        )

    target = get_cache_dir() / f"{version}.pth"
    if target.exists():
        logger.info("Using cached weights at '%s'.", target)
        return target

    spec = _spec(version)
    logger.info("Downloading weights for '%s' from %s.", version, spec.url)
    _download(spec.url, target)
    if spec.sha256:
        _verify(target, spec.sha256)
    logger.info("Weights saved to '%s'.", target)
    return target


def _spec(version: str) -> WeightSpec:
    """Return REGISTRY[version] as a WeightSpec, wrapping a bare URL."""
    entry = REGISTRY[version]
    return entry if isinstance(entry, WeightSpec) else WeightSpec(url=entry)


def _verify(path: Path, expected: str) -> None:
    """Check a downloaded file's SHA-256 and delete it if it does not match.

    Deleting matters: leaving the bad file in place would make the next
    call a cache hit, so one corrupted download would poison every run
    that followed.

    Raises:
        RuntimeError: If the digest differs from expected.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for '{path.name}': expected {expected}, "
            f"got {actual}. The download was corrupted or the release asset "
            f"was replaced. The bad file has been deleted; retry, and if it "
            f"persists check the digest pinned in cv_pipeline/weights.py."
        )
    logger.debug("Checksum verified for '%s'.", path.name)


def _download(url: str, target: Path) -> None:
    """Stream a file from url to target.

    Writes to <target>.tmp first and renames on success so a failed
    download cannot leave a truncated file in the cache.
    """
    # Lazy import: requests is a dep of many ML libraries but we keep
    # the import out of module load to keep cv_pipeline import cheap.
    import requests

    with requests.get(
        url,
        stream=True,
        allow_redirects=True,
        timeout=60,
    ) as response:
        response.raise_for_status()

        # Guard against hosts that answer a download URL with a landing
        # or preview page. A direct download returns octet-stream or
        # similar; HTML means the URL is wrong, the asset is missing, or
        # the release is private.
        content_type = response.headers.get("Content-Type", "")
        if "html" in content_type.lower():
            raise RuntimeError(
                f"Expected binary response from {url} but got "
                f"content-type={content_type!r}. The release asset is "
                f"probably missing or not public. Check the tag and asset "
                f"name, or point REGISTRY at any host that returns raw bytes."
            )

        tmp = target.with_suffix(target.suffix + ".tmp")
        with tmp.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
        tmp.replace(target)
