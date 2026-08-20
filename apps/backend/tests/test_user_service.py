"""Unit tests for api.services.user_service."""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock

import bcrypt
import pytest
from api.services.user_service import create_user, generate_api_key


@pytest.mark.unit
class TestGenerateApiKey:
    """Tests for API key generation."""

    def test_returns_32_char_hex_string(self) -> None:
        """The key should be a 32-character hexadecimal string."""
        key = generate_api_key()

        assert isinstance(key, str)
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)

    def test_keys_are_unique(self) -> None:
        """Consecutive calls should produce different keys."""
        keys = {generate_api_key() for _ in range(10)}

        assert len(keys) == 10


@pytest.mark.unit
class TestCreateUser:
    """Tests for user creation."""

    @pytest.mark.anyio
    async def test_returns_user_and_plaintext_key(self) -> None:
        """create_user should return a (User, plaintext_key) tuple."""
        db = MagicMock()
        db.add = MagicMock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock()

        user, plaintext_key = await create_user(
            db=db,
            name="Alice",
            role="researcher",
        )

        assert user.name == "Alice"
        assert user.role == "researcher"
        assert isinstance(plaintext_key, str)
        assert len(plaintext_key) == 32

    @pytest.mark.anyio
    async def test_stored_hash_validates_against_key(self) -> None:
        """The bcrypt hash on the user should match the returned key."""
        db = MagicMock()
        db.add = MagicMock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock()

        user, plaintext_key = await create_user(
            db=db,
            name="Bob",
            role="admin",
        )

        assert bcrypt.checkpw(
            plaintext_key.encode("utf-8"),
            user.api_key_hash.encode("utf-8"),
        )

    @pytest.mark.anyio
    async def test_key_sha256_matches_plaintext(self) -> None:
        """The stored SHA-256 digest should match the plaintext key."""
        db = MagicMock()
        db.add = MagicMock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock()

        user, plaintext_key = await create_user(
            db=db,
            name="Carol",
            role="researcher",
        )

        expected = hashlib.sha256(
            plaintext_key.encode("utf-8"),
        ).hexdigest()
        assert user.key_sha256 == expected

    @pytest.mark.anyio
    async def test_persists_to_database(self) -> None:
        """create_user should call db.add, db.commit, and db.refresh."""
        db = MagicMock()
        db.add = MagicMock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock()

        await create_user(db=db, name="Dave", role="researcher")

        db.add.assert_called_once()
        db.commit.assert_called_once()
        db.refresh.assert_called_once()
