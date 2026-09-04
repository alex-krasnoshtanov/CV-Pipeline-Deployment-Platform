"""Unit tests for cv_pipeline.weights."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
from cv_pipeline import weights


@pytest.mark.unit
class TestWeightsCacheDir:
    """Tests for cache directory resolution."""

    def test_get_cache_dir_honours_env_var(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """CV_PIPELINE_CACHE_DIR should override the default cache path."""
        custom_cache = tmp_path / "custom-cache"
        monkeypatch.setenv("CV_PIPELINE_CACHE_DIR", str(custom_cache))

        result = weights.get_cache_dir()

        assert result == custom_cache
        assert result.exists()
        assert result.is_dir()

    def test_get_cache_dir_creates_directory_if_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The cache directory should be created when it does not exist."""
        default_cache = tmp_path / "xdg-default" / "cv-pipeline" / "models"
        monkeypatch.delenv("CV_PIPELINE_CACHE_DIR", raising=False)
        monkeypatch.setattr(weights, "_DEFAULT_CACHE_DIR", default_cache)

        assert not default_cache.exists()

        result = weights.get_cache_dir()

        assert result == default_cache
        assert result.exists()
        assert result.is_dir()


@pytest.mark.unit
class TestGetWeights:
    """Tests for high-level weight path resolution and caching."""

    def test_get_weights_raises_for_unknown_version(self) -> None:
        """Unknown version keys should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown version"):
            weights.get_weights("missing-version")

    def test_get_weights_uses_cache_without_redownload(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Second call should return cached file without any HTTP request."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("CV_PIPELINE_CACHE_DIR", str(cache_dir))
        monkeypatch.setattr(
            weights,
            "REGISTRY",
            {"test-v1": "https://example.test/test-v1.pth"},
        )

        def fake_download(url: str, target: Path) -> None:
            """Simulate a successful download by writing a small file."""
            target.write_bytes(b"fake-weights")

        monkeypatch.setattr(weights, "_download", fake_download)

        first = weights.get_weights("test-v1")
        assert first.exists()

        with patch("requests.get", side_effect=AssertionError("requests.get called")):
            second = weights.get_weights("test-v1")

        assert second == first
        assert second.read_bytes() == b"fake-weights"


@pytest.mark.unit
class TestDownload:
    """Tests for low-level streaming download behavior."""

    def test_download_raises_for_html_content_type(
        self,
        tmp_path: Path,
    ) -> None:
        """HTML responses should be rejected with a RuntimeError."""

        class FakeResponse:
            """Minimal requests-like response object for testing."""

            status_code = 200
            headers = {"Content-Type": "text/html; charset=utf-8"}

            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
                return False

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                yield b"<html></html>"

        target = tmp_path / "weights.pth"
        with patch("requests.get", return_value=FakeResponse()):
            with pytest.raises(RuntimeError, match="Expected binary response"):
                weights._download("https://example.test/file", target)

        assert not target.exists()


@pytest.mark.unit
class TestChecksumVerification:
    """Tests for the SHA-256 pin on a registry entry."""

    def test_matching_digest_passes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A download whose digest matches is kept and returned."""
        payload = b"fake-weights"
        digest = hashlib.sha256(payload).hexdigest()
        monkeypatch.setenv("CV_PIPELINE_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setattr(
            weights,
            "REGISTRY",
            {
                "test-v1": weights.WeightSpec(
                    url="https://example.test/w.pth", sha256=digest
                )
            },
        )
        monkeypatch.setattr(
            weights, "_download", lambda url, target: target.write_bytes(payload)
        )

        result = weights.get_weights("test-v1")

        assert result.read_bytes() == payload

    def test_mismatched_digest_raises_and_deletes_the_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A corrupted download must not survive as a cache hit.

        Leaving it in place would make every later call return the bad
        file without re-downloading, so one bad byte would poison the
        cache permanently.
        """
        monkeypatch.setenv("CV_PIPELINE_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setattr(
            weights,
            "REGISTRY",
            {
                "test-v1": weights.WeightSpec(
                    url="https://example.test/w.pth", sha256="00" * 32
                )
            },
        )
        monkeypatch.setattr(
            weights, "_download", lambda url, target: target.write_bytes(b"corrupted")
        )

        with pytest.raises(RuntimeError, match="Checksum mismatch"):
            weights.get_weights("test-v1")

        assert not (tmp_path / "cache" / "test-v1.pth").exists()

    def test_a_bare_url_entry_still_works(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """REGISTRY accepts a plain string for an unpinned version."""
        monkeypatch.setenv("CV_PIPELINE_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setattr(
            weights, "REGISTRY", {"test-v1": "https://example.test/w.pth"}
        )
        monkeypatch.setattr(
            weights, "_download", lambda url, target: target.write_bytes(b"x")
        )

        assert weights.get_weights("test-v1").read_bytes() == b"x"

    def test_the_shipped_unet_v1_entry_pins_a_real_digest(self) -> None:
        """Guards against a placeholder digest reaching a release.

        A ``<digest>`` string or a truncated hash would disable the check
        for everyone while still looking configured.
        """
        spec = weights._spec("unet-v1")

        assert spec.sha256 is not None
        assert len(spec.sha256) == 64
        assert set(spec.sha256) <= set("0123456789abcdef")
        assert spec.url.startswith("https://github.com/")
        assert "/releases/download/" in spec.url
