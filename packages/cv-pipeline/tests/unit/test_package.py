"""Unit tests for cv_pipeline package-level exports."""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestPackageExports:
    """Tests for package-level imports and version."""

    def test_version_is_string(self) -> None:
        """__version__ should be a string."""
        from cv_pipeline import __version__

        assert isinstance(__version__, str)

    def test_version_matches_expected(self) -> None:
        """__version__ should match the current release."""
        from cv_pipeline import __version__

        assert __version__ == "0.1.0"

    def test_infer_is_importable(self) -> None:
        """infer should be importable from the package root."""
        from cv_pipeline import infer

        assert callable(infer)

    def test_version_module_is_importable(self) -> None:
        """_version module should be importable directly."""
        from cv_pipeline._version import __version__

        assert __version__ == "0.1.0"
