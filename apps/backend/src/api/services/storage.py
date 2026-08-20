"""Storage backend for feedback images.

The feedback flywheel needs the input image to outlive the request so
a 'good' or relabeled prediction can later be paired with its mask.
This module owns *where* those bytes go and *how* to name them, behind
one interface so the inference path does not care which deployment it
runs in.

Two backends:
- ``local``: a mounted volume. Available for any deployment; requires no
  Azure credentials. The whole volume is the feedback area, so keys are
  stored relative to it. Used by the local dev stack.
- ``blob``: Azure Blob Storage, writing into the container backing the
  Azure ML ``workspaceblobstore`` datastore. Used by the on-premise
  (Portainer) and cloud deployments so the bytes sit next to the existing
  ``hades-*`` data assets and feed the Airflow retraining loop. Because
  that store is shared, the blob backend namespaces every object under a
  ``feedback/`` prefix.

Canonical object key (per-user folder, keyed by prediction id):
    raw/{user_id}/{prediction_id}{suffix}

The stored ``image_uri`` is the locator the export task (later chunk)
resolves: a filesystem path for ``local`` (``/<root>/raw/...``), an
``azureml://`` datastore URI for ``blob`` (``.../paths/feedback/raw/...``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Protocol

from api.config import get_settings

# Prefix applied only by the blob backend, which shares its store with
# the hades-* data assets and therefore needs a namespace. The local
# volume is dedicated to feedback, so it does not repeat this.
_BLOB_NAMESPACE = "feedback"


class StorageBackend(Protocol):
    """Minimal contract every storage backend implements."""

    def uri_for(self, key: str) -> str:
        """Return the durable locator stored in ``predictions.image_uri``.

        Args:
            key: Object key, e.g. ``raw/<uid>/<pid>.png``.

        Returns:
            A backend-specific locator string.
        """
        ...

    def write(self, key: str, data: bytes) -> None:
        """Write *data* at *key*. Blocking; call from a threadpool.

        Args:
            key: Object key under which to store the bytes.
            data: Raw image bytes.

        Raises:
            OSError: On a local filesystem write failure.
            Exception: On an Azure Blob upload failure.
        """
        ...


class LocalStorageBackend:
    """Writes images to a directory on a mounted volume."""

    def __init__(self, root: str) -> None:
        """Initialise the backend.

        Args:
            root: Absolute directory that holds the feedback tree.
        """
        self._root = Path(root)

    def uri_for(self, key: str) -> str:
        return str(self._root / key)

    def write(self, key: str, data: bytes) -> None:
        dest = self._root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)


class BlobStorageBackend:
    """Writes images to the workspaceblobstore-backed container.

    The configured container must be the one Azure ML's
    ``workspaceblobstore`` datastore points at, so the returned
    ``azureml://`` URI and the written blob refer to the same bytes.
    Objects are namespaced under ``feedback/`` because the store is
    shared with the existing data assets.
    """

    def __init__(self, connection_string: str, container: str) -> None:
        """Initialise the backend.

        Args:
            connection_string: Storage account connection string.
            container: Container name backing ``workspaceblobstore``.
        """
        self._connection_string = connection_string
        self._container = container
        self._container_client = None

    def _get_container_client(self):
        """Return a cached ContainerClient, building it on first use.

        The backend is a process singleton (``get_storage_backend`` is
        ``lru_cache``-d), so the client and its HTTP session are created
        once and reused across writes rather than per call. The SDK import
        stays lazy so local deployments without azure-storage-blob still
        work.
        """
        if self._container_client is None:
            from azure.storage.blob import BlobServiceClient

            service = BlobServiceClient.from_connection_string(self._connection_string)
            self._container_client = service.get_container_client(self._container)
        return self._container_client

    def _blob_name(self, key: str) -> str:
        """Return the namespaced blob name for a canonical key."""
        return f"{_BLOB_NAMESPACE}/{key}"

    def uri_for(self, key: str) -> str:
        return f"azureml://datastores/workspaceblobstore/paths/{self._blob_name(key)}"

    def write(self, key: str, data: bytes) -> None:
        container = self._get_container_client()
        container.upload_blob(name=self._blob_name(key), data=data, overwrite=True)


def feedback_image_key(user_id: str, prediction_id: str, suffix: str) -> str:
    """Build the canonical object key for a feedback image.

    The key is relative to each backend's feedback root. The blob
    backend adds its own ``feedback/`` namespace; the local backend
    writes it directly under the mounted volume.

    Args:
        user_id: Owning user's UUID as a string.
        prediction_id: Prediction UUID as a string.
        suffix: File suffix including the dot, e.g. ``.png``.

    Returns:
        The object key, e.g. ``raw/<uid>/<pid>.png``.
    """
    clean_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return f"raw/{user_id}/{prediction_id}{clean_suffix}"


@lru_cache(maxsize=1)
def get_storage_backend() -> StorageBackend:
    """Return the configured storage backend (built once).

    Returns:
        A ``LocalStorageBackend`` or ``BlobStorageBackend``.

    Raises:
        ValueError: If ``blob`` is selected without connection settings.
    """
    settings = get_settings()
    if settings.feedback_storage_backend == "blob":
        if not (
            settings.feedback_storage_blob_connection_string
            and settings.feedback_storage_blob_container
        ):
            raise ValueError(
                "feedback_storage_backend='blob' requires "
                "FEEDBACK_STORAGE_BLOB_CONNECTION_STRING and "
                "FEEDBACK_STORAGE_BLOB_CONTAINER to be set."
            )
        return BlobStorageBackend(
            connection_string=(
                settings.feedback_storage_blob_connection_string.get_secret_value()
            ),
            container=settings.feedback_storage_blob_container,
        )
    return LocalStorageBackend(settings.feedback_storage_local_dir)
