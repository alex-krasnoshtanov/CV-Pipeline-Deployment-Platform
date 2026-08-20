"""Unit tests for api.auth.password.

hash_password and verify_password are pure functions with no external
dependencies beyond bcrypt. We test the happy path, the mismatch path,
and the defensive error-handling branch (malformed/None hash returns
False instead of raising).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from api.auth.password import hash_password, verify_password  # noqa: E402


@pytest.mark.unit
class TestHashPassword:
    def test_returns_bcrypt_prefixed_string(self):
        """bcrypt hashes always start with the $2b$ identifier."""
        result = hash_password("secret")
        assert result.startswith("$2b$")

    def test_returns_string_type(self):
        """Return type must be str so it can be stored in a Text column."""
        assert isinstance(hash_password("anything"), str)

    def test_different_calls_produce_different_hashes(self):
        """Two hashes of the same plaintext must differ (random salt)."""
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2

    def test_hash_length_is_60(self):
        """bcrypt hashes are always exactly 60 characters."""
        assert len(hash_password("password")) == 60


@pytest.mark.unit
class TestVerifyPassword:
    def test_correct_password_returns_true(self):
        hashed = hash_password("correct")
        assert verify_password("correct", hashed) is True

    def test_wrong_password_returns_false(self):
        hashed = hash_password("correct")
        assert verify_password("wrong", hashed) is False

    def test_empty_string_matches_hash_of_empty_string(self):
        hashed = hash_password("")
        assert verify_password("", hashed) is True

    def test_case_sensitive(self):
        """Passwords are case-sensitive — 'Secret' != 'secret'."""
        hashed = hash_password("Secret")
        assert verify_password("secret", hashed) is False

    def test_malformed_hash_returns_false_not_raises(self):
        """A corrupt hash row in the DB should not 500 the login endpoint.

        This exercises the except (ValueError, TypeError): return False path
        inside verify_password.
        """
        assert verify_password("anything", "not-a-valid-bcrypt-hash") is False

    def test_none_hash_returns_false_not_raises(self):
        """Passing None as hash (type error) is caught and returns False.

        This can happen if the column is NULL in the DB (OAuth-only user)
        and the caller forgets to guard. The function must never raise.
        """
        result = verify_password("anything", None)  # type: ignore[arg-type]
        assert result is False
