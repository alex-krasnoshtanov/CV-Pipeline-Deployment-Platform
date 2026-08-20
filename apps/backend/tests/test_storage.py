"""Unit tests for api.services.storage.

Covers the canonical key builder, the local-filesystem backend, the
blob backend's naming (without touching the Azure SDK), and the
settings-driven backend selection.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from api.services import storage
from api.services.storage import (
    BlobStorageBackend,
    LocalStorageBackend,
    feedback_image_key,
    get_storage_backend,
)
from pydantic import SecretStr


@pytest.mark.unit
class TestFeedbackImageKey:
    """Tests for the canonical object-key builder."""

    def test_adds_missing_dot_to_suffix(self) -> None:
        """A suffix without a leading dot is normalised."""
        assert feedback_image_key("u1", "p1", "png") == "raw/u1/p1.png"

    def test_keeps_existing_dot(self) -> None:
        """A suffix that already has a dot is left unchanged."""
        assert feedback_image_key("u1", "p1", ".tif") == "raw/u1/p1.tif"


@pytest.mark.unit
class TestLocalStorageBackend:
    """Tests for the local-filesystem backend."""

    def test_write_creates_parents_and_file(self, tmp_path: Path) -> None:
        """write() creates intermediate directories and stores the bytes."""
        backend = LocalStorageBackend(str(tmp_path))

        backend.write("raw/u1/p1.png", b"data")

        written = tmp_path / "raw" / "u1" / "p1.png"
        assert written.read_bytes() == b"data"

    def test_uri_for_is_path_under_root(self, tmp_path: Path) -> None:
        """uri_for() returns the on-disk path under the configured root."""
        backend = LocalStorageBackend(str(tmp_path))

        assert backend.uri_for("raw/u1/p1.png") == str(tmp_path / "raw/u1/p1.png")


@pytest.mark.unit
class TestBlobStorageBackend:
    """Tests for the blob backend's naming (no SDK calls)."""

    def test_uri_is_namespaced_azureml_path(self) -> None:
        """uri_for() namespaces the key under feedback/ as an azureml URI."""
        backend = BlobStorageBackend("conn", "container")

        assert backend.uri_for("raw/u1/p1.png") == (
            "azureml://datastores/workspaceblobstore/paths/feedback/raw/u1/p1.png"
        )


@pytest.mark.unit
class TestGetStorageBackend:
    """Tests for settings-driven backend selection."""

    def setup_method(self) -> None:
        """Clear the cached backend before each test."""
        get_storage_backend.cache_clear()

    def teardown_method(self) -> None:
        """Clear the cached backend after each test."""
        get_storage_backend.cache_clear()

    def test_defaults_to_local(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 'local' setting yields a LocalStorageBackend."""
        settings = MagicMock()
        settings.feedback_storage_backend = "local"
        settings.feedback_storage_local_dir = "/data/feedback"
        monkeypatch.setattr(storage, "get_settings", lambda: settings)

        assert isinstance(get_storage_backend(), LocalStorageBackend)

    def test_blob_without_credentials_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Selecting 'blob' without connection settings raises ValueError."""
        settings = MagicMock()
        settings.feedback_storage_backend = "blob"
        settings.feedback_storage_blob_connection_string = None
        settings.feedback_storage_blob_container = None
        monkeypatch.setattr(storage, "get_settings", lambda: settings)

        with pytest.raises(ValueError, match="requires"):
            get_storage_backend()

    def test_blob_with_credentials_builds_blob_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A fully-configured 'blob' setting yields a BlobStorageBackend."""
        settings = MagicMock()
        settings.feedback_storage_backend = "blob"
        settings.feedback_storage_blob_connection_string = SecretStr("conn")
        settings.feedback_storage_blob_container = "cont"
        monkeypatch.setattr(storage, "get_settings", lambda: settings)

        assert isinstance(get_storage_backend(), BlobStorageBackend)
