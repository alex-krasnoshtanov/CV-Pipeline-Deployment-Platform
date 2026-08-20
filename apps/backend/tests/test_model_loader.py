"""Unit tests for api.services.model_loader."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from api.services.model_loader import load_model


@pytest.mark.unit
class TestLoadModelFromPath:
    """Tests for loading via MODEL_PATH env var."""

    @patch("api.services.model_loader.SegmentationModel")
    def test_loads_from_explicit_path(
        self,
        mock_seg: MagicMock,
        tmp_path: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When MODEL_PATH is set and the file exists, load from that path."""
        weights_file = tmp_path / "model.pth"
        weights_file.write_bytes(b"fake")
        monkeypatch.setenv("MODEL_PATH", str(weights_file))
        monkeypatch.delenv("MODEL_VERSION", raising=False)

        load_model()

        mock_seg.assert_called_once_with(weights_file)

    def test_missing_path_raises_file_not_found(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When MODEL_PATH points to a nonexistent file, raise FileNotFoundError."""
        monkeypatch.setenv("MODEL_PATH", "/nonexistent/model.pth")
        monkeypatch.delenv("MODEL_VERSION", raising=False)

        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_model()


@pytest.mark.unit
class TestLoadModelFromVersion:
    """Tests for loading via MODEL_VERSION or registry fallback."""

    @patch("api.services.model_loader.SegmentationModel")
    def test_loads_by_version_env(
        self,
        mock_seg: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When MODEL_VERSION is set, load that specific version."""
        monkeypatch.delenv("MODEL_PATH", raising=False)
        monkeypatch.setenv("MODEL_VERSION", "unet-v2")

        load_model()

        mock_seg.assert_called_once_with("unet-v2")

    @patch("api.services.model_loader.SegmentationModel")
    @patch("api.services.model_loader.REGISTRY", {"unet-v1": "https://example.com"})
    def test_falls_back_to_first_registry_entry(
        self,
        mock_seg: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When neither env var is set, use the first REGISTRY key."""
        monkeypatch.delenv("MODEL_PATH", raising=False)
        monkeypatch.delenv("MODEL_VERSION", raising=False)

        load_model()

        mock_seg.assert_called_once_with("unet-v1")
