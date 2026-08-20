"""Async database session factory for the FastAPI backend"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

engine: AsyncEngine | None = None
SessionLocal: async_sessionmaker[AsyncSession] | None = None


def init_db(url: str) -> None:
    """Create the async engine and session factory.

    Must be called once during application startup (inside the
    FastAPI ``lifespan``) before any route handler calls ``get_db``.

    Args:
        url: Full SQLAlchemy async connection string,
             e.g. ``postgresql+asyncpg://user:pass@host/db``.

    Raises:
        ValueError: If *url* is empty or None.
    """
    global engine, SessionLocal  # noqa: PLW0603

    if not url:
        raise ValueError(
            "DB_URL is empty or unset. Cannot initialise the database engine."
        )

    engine = create_async_engine(url, future=True)
    SessionLocal = async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session, closing it when the request finishes.

    Yields:
        An ``AsyncSession`` scoped to the current request.

    Raises:
        RuntimeError: If called before ``init_db()`` has run.
    """
    if SessionLocal is None:
        raise RuntimeError(
            "Database not initialised. Call init_db() during app startup."
        )
    async with SessionLocal() as db:
        yield db
