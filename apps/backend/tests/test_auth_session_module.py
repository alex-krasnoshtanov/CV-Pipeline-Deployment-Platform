"""Unit tests for api.auth.session.

Tests the three public functions: create_session, get_user_from_session,
delete_session. All are async; we use pytest.mark.anyio (same pattern as
test_session.py for the DB session factory).

The DB is mocked with AsyncMock throughout — these are unit tests, no
Postgres required.
"""

from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from api.auth.session import (  # noqa: E402
    DEFAULT_SESSION_TTL,
    create_session,
    delete_session,
    get_user_from_session,
)
from api.db.models import Session, User  # noqa: E402


def _make_db() -> AsyncMock:
    """Minimal AsyncMock that satisfies the session function signatures."""
    db = AsyncMock()
    db.add = MagicMock()  # synchronous in SQLAlchemy
    return db


@pytest.mark.unit
class TestCreateSession:
    @pytest.mark.anyio
    async def test_adds_session_row_with_correct_user_id(self):
        """create_session must db.add() a Session bound to the given user_id."""
        db = _make_db()
        user_id = uuid.uuid4()

        async def _refresh(obj):
            obj.id = uuid.uuid4()

        db.refresh.side_effect = _refresh

        await create_session(db, user_id)

        db.add.assert_called_once()
        added = db.add.call_args[0][0]
        assert isinstance(added, Session)
        assert added.user_id == user_id

    @pytest.mark.anyio
    async def test_expires_at_uses_provided_ttl(self):
        """expires_at on the new row must be approximately now + ttl."""
        db = _make_db()
        ttl = timedelta(hours=3)

        async def _refresh(obj):
            obj.id = uuid.uuid4()

        db.refresh.side_effect = _refresh

        before = datetime.now(UTC)
        await create_session(db, uuid.uuid4(), ttl=ttl)
        after = datetime.now(UTC)

        added = db.add.call_args[0][0]
        assert before + ttl <= added.expires_at <= after + ttl

    @pytest.mark.anyio
    async def test_expires_at_uses_default_ttl_when_not_provided(self):
        """Default TTL is 7 days."""
        db = _make_db()

        async def _refresh(obj):
            obj.id = uuid.uuid4()

        db.refresh.side_effect = _refresh

        before = datetime.now(UTC)
        await create_session(db, uuid.uuid4())
        after = datetime.now(UTC)

        added = db.add.call_args[0][0]
        assert before + DEFAULT_SESSION_TTL <= added.expires_at
        assert added.expires_at <= after + DEFAULT_SESSION_TTL

    @pytest.mark.anyio
    async def test_commits_and_refreshes_once(self):
        """create_session must commit and refresh exactly once each."""
        db = _make_db()

        async def _refresh(obj):
            obj.id = uuid.uuid4()

        db.refresh.side_effect = _refresh

        await create_session(db, uuid.uuid4())

        db.commit.assert_awaited_once()
        db.refresh.assert_awaited_once()

    @pytest.mark.anyio
    async def test_runs_opportunistic_cleanup_delete(self):
        """create_session must execute a DELETE to clean expired rows."""
        db = _make_db()

        async def _refresh(obj):
            obj.id = uuid.uuid4()

        db.refresh.side_effect = _refresh

        await create_session(db, uuid.uuid4())

        # The only db.execute call is the DELETE for expired session cleanup.
        db.execute.assert_awaited_once()


@pytest.mark.unit
class TestGetUserFromSession:
    @pytest.mark.anyio
    async def test_malformed_uuid_returns_none_without_db_call(self):
        """A cookie that isn't a valid UUID should short-circuit before the DB."""
        db = _make_db()
        result = await get_user_from_session(db, "definitely-not-a-uuid")
        assert result is None
        db.execute.assert_not_awaited()

    @pytest.mark.anyio
    async def test_none_session_id_returns_none(self):
        """None instead of a string should be handled cleanly."""
        db = _make_db()
        result = await get_user_from_session(db, None)  # type: ignore[arg-type]
        assert result is None

    @pytest.mark.anyio
    async def test_missing_row_returns_none(self):
        """No DB row for the session UUID -> anonymous caller."""
        db = _make_db()
        db_result = MagicMock()
        db_result.first.return_value = None
        db.execute.return_value = db_result

        result = await get_user_from_session(db, str(uuid.uuid4()))
        assert result is None

    @pytest.mark.anyio
    async def test_valid_non_expired_session_returns_user(self):
        """A session row whose expires_at is in the future should return its user."""
        db = _make_db()

        session_row = MagicMock(spec=Session)
        session_row.expires_at = datetime.now(UTC) + timedelta(days=1)

        user = MagicMock(spec=User)
        user.id = uuid.uuid4()

        db_result = MagicMock()
        db_result.first.return_value = (session_row, user)
        db.execute.return_value = db_result

        result = await get_user_from_session(db, str(uuid.uuid4()))
        assert result is user

    @pytest.mark.anyio
    async def test_expired_session_returns_none(self):
        """A session row whose expires_at is in the past should be rejected."""
        db = _make_db()

        session_row = MagicMock(spec=Session)
        # One second in the past — expired
        session_row.expires_at = datetime.now(UTC) - timedelta(seconds=1)

        user = MagicMock(spec=User)
        db_result = MagicMock()
        db_result.first.return_value = (session_row, user)
        db.execute.return_value = db_result

        result = await get_user_from_session(db, str(uuid.uuid4()))
        assert result is None

    @pytest.mark.anyio
    async def test_db_exception_returns_none_not_raises(self):
        """A DB failure must degrade to anonymous, not 500 the request.

        This tests the except Exception block that protects callers from
        transient DB hiccups during session lookup.
        """
        db = _make_db()
        db.execute.side_effect = RuntimeError("connection lost")

        result = await get_user_from_session(db, str(uuid.uuid4()))
        assert result is None


@pytest.mark.unit
class TestDeleteSession:
    @pytest.mark.anyio
    async def test_valid_uuid_executes_delete_and_commits(self):
        db = _make_db()
        await delete_session(db, str(uuid.uuid4()))
        db.execute.assert_awaited_once()
        db.commit.assert_awaited_once()

    @pytest.mark.anyio
    async def test_malformed_uuid_is_noop(self):
        """A bad cookie value must not touch the DB at all (idempotent)."""
        db = _make_db()
        await delete_session(db, "not-a-uuid-at-all")
        db.execute.assert_not_awaited()
        db.commit.assert_not_awaited()

    @pytest.mark.anyio
    async def test_none_session_id_is_noop(self):
        """None session_id must not touch the DB."""
        db = _make_db()
        await delete_session(db, None)  # type: ignore[arg-type]
        db.execute.assert_not_awaited()

    @pytest.mark.anyio
    async def test_nonexistent_session_does_not_raise(self):
        """Deleting a session that doesn't exist must be silent (idempotent).

        The logout endpoint calls this without first checking if the row
        exists, so it must never raise.
        """
        db = _make_db()
        # execute returns a result that represents 0 rows deleted — still fine
        await delete_session(db, str(uuid.uuid4()))
        # No assertion needed: the test passes if no exception is raised
