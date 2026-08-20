"""Unit tests for api.db.session."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from api.db import session as session_module


@pytest.mark.unit
class TestInitDb:
    """Tests for the init_db database initialisation function."""

    def test_empty_url_raises_value_error(self) -> None:
        """An empty URL string should raise ValueError."""
        with pytest.raises(ValueError, match="DB_URL is empty or unset"):
            session_module.init_db("")

    def test_none_url_raises_value_error(self) -> None:
        """A None URL should raise ValueError."""
        with pytest.raises(ValueError, match="DB_URL is empty or unset"):
            session_module.init_db("")

    @patch("api.db.session.async_sessionmaker")
    @patch("api.db.session.create_async_engine")
    def test_valid_url_sets_engine_and_session(
        self,
        mock_engine: MagicMock,
        mock_sessionmaker: MagicMock,
    ) -> None:
        """A valid URL should create the engine and session factory."""
        original_engine = session_module.engine
        original_session = session_module.SessionLocal

        try:
            session_module.init_db("postgresql+asyncpg://u:p@localhost/db")

            mock_engine.assert_called_once()
            mock_sessionmaker.assert_called_once()
            assert session_module.engine is not None
            assert session_module.SessionLocal is not None
        finally:
            session_module.engine = original_engine
            session_module.SessionLocal = original_session


@pytest.mark.unit
class TestGetDb:
    """Tests for the get_db async generator."""

    @pytest.mark.anyio
    async def test_raises_when_not_initialised(self) -> None:
        """get_db should raise RuntimeError if init_db was never called."""
        original = session_module.SessionLocal
        try:
            session_module.SessionLocal = None

            with pytest.raises(RuntimeError, match="Database not initialised"):
                async for _ in session_module.get_db():
                    pass
        finally:
            session_module.SessionLocal = original

    @pytest.mark.anyio
    async def test_yields_session_when_initialised(self) -> None:
        """get_db should yield an AsyncSession when SessionLocal is set."""
        mock_session = AsyncMock()

        mock_factory = MagicMock(return_value=mock_session)
        original = session_module.SessionLocal

        try:
            session_module.SessionLocal = mock_factory

            sessions = []
            async for db in session_module.get_db():
                sessions.append(db)

            assert len(sessions) == 1
        finally:
            session_module.SessionLocal = original
