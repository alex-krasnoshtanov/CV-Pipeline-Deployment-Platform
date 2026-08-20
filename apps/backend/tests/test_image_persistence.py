"""Unit tests for api.services.image_persistence.

The persistence task is intentionally fail-safe: every failure path is
logged and swallowed so a storage or database problem never turns a
successful prediction into a user-facing error. These tests pin that
behaviour and confirm the tempfile is always cleaned up.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from api.services import image_persistence
from api.services.image_persistence import (
    _safe_unlink,
    persist_inference_image,
)
from sqlalchemy.exc import SQLAlchemyError


class _FakeSession:
    """Async-context-manager stand-in for an AsyncSession."""

    def __init__(self) -> None:
        """Initialise with awaitable execute/commit mocks."""
        self.execute = AsyncMock()
        self.commit = AsyncMock()

    async def __aenter__(self) -> _FakeSession:
        """Enter the context, returning self as the session."""
        return self

    async def __aexit__(self, *exc: object) -> bool:
        """Exit the context without suppressing exceptions."""
        return False


@pytest.fixture
def tmp_image(tmp_path: Path) -> Path:
    """Write a small tempfile to stand in for the uploaded image."""
    src = tmp_path / "upload.png"
    src.write_bytes(b"image-bytes")
    return src


def _install_backend(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a fake storage backend and return it for assertions."""
    backend = MagicMock()
    backend.write = MagicMock()
    backend.uri_for = MagicMock(return_value="/data/feedback/raw/u/p.png")
    monkeypatch.setattr(
        image_persistence,
        "get_storage_backend",
        lambda: backend,
    )
    return backend


def _install_session(monkeypatch: pytest.MonkeyPatch) -> _FakeSession:
    """Point SessionLocal at a fresh fake session and return it."""
    session = _FakeSession()
    monkeypatch.setattr(
        image_persistence.db_session,
        "SessionLocal",
        lambda: session,
    )
    return session


@pytest.mark.unit
class TestPersistInferenceImage:
    """Tests for persist_inference_image."""

    @pytest.mark.anyio
    async def test_happy_path_writes_and_records(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_image: Path,
    ) -> None:
        """Bytes are stored, image_uri recorded, and tempfile removed."""
        backend = _install_backend(monkeypatch)
        session = _install_session(monkeypatch)

        await persist_inference_image(
            prediction_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            src_path=tmp_image,
            suffix=".png",
        )

        backend.write.assert_called_once()
        session.execute.assert_awaited_once()
        session.commit.assert_awaited_once()
        assert not tmp_image.exists()

    @pytest.mark.anyio
    async def test_unreadable_tempfile_returns_quietly(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A missing tempfile is logged and skipped, never raised."""
        backend = _install_backend(monkeypatch)
        missing = tmp_path / "gone.png"

        await persist_inference_image(
            prediction_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            src_path=missing,
            suffix=".png",
        )

        backend.write.assert_not_called()

    @pytest.mark.anyio
    async def test_storage_write_failure_skips_db(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_image: Path,
    ) -> None:
        """A storage write failure is swallowed and the DB is untouched."""
        backend = _install_backend(monkeypatch)
        backend.write.side_effect = RuntimeError("storage down")
        session = _install_session(monkeypatch)

        await persist_inference_image(
            prediction_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            src_path=tmp_image,
            suffix=".png",
        )

        session.execute.assert_not_awaited()
        assert not tmp_image.exists()

    @pytest.mark.anyio
    async def test_unset_session_factory_skips_db(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_image: Path,
    ) -> None:
        """A null SessionLocal is logged and skipped without raising."""
        _install_backend(monkeypatch)
        monkeypatch.setattr(
            image_persistence.db_session,
            "SessionLocal",
            None,
        )

        await persist_inference_image(
            prediction_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            src_path=tmp_image,
            suffix=".png",
        )

        assert not tmp_image.exists()

    @pytest.mark.anyio
    async def test_db_error_is_swallowed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_image: Path,
    ) -> None:
        """A DB failure is logged, not raised, and the tempfile removed."""
        _install_backend(monkeypatch)
        session = _install_session(monkeypatch)
        session.execute.side_effect = SQLAlchemyError("boom")

        await persist_inference_image(
            prediction_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            src_path=tmp_image,
            suffix=".png",
        )

        assert not tmp_image.exists()


@pytest.mark.unit
class TestSafeUnlink:
    """Tests for the _safe_unlink helper."""

    def test_removes_existing_file(self, tmp_path: Path) -> None:
        """An existing file is deleted."""
        target = tmp_path / "x.bin"
        target.write_bytes(b"x")

        _safe_unlink(target)

        assert not target.exists()

    def test_missing_file_is_noop(self, tmp_path: Path) -> None:
        """A missing file does not raise."""
        _safe_unlink(tmp_path / "nope.bin")

    def test_unlink_error_is_swallowed(self) -> None:
        """An OSError during unlink is logged, not raised."""
        bad = MagicMock()
        bad.unlink.side_effect = OSError("permission denied")

        _safe_unlink(bad)

        bad.unlink.assert_called_once()
