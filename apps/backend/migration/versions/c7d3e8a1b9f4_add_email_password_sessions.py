"""add email + password + sessions for human web login

Revision ID: c7d3e8a1b9f4
Revises: 44d5bbcd4d4a
Create Date: 2026-05-11

Adds email + password_hash to users so humans can log in with
credentials instead of out-of-band API keys, and a sessions table
so we can issue httpOnly session cookies (revocable, server-side).

The existing api_key_hash / key_sha256 columns remain — legacy
seeded users still authenticate via X-API-Key. New web users will
have password_hash set and api_key_hash NULL. The CHECK constraint
enforces at least one auth method per row.

We DO NOT use JWT here on purpose. Server-side sessions cost one
extra indexed DB lookup per request (already true for X-API-Key,
so no regression) and give us free revocation: deleting the row
logs the user out everywhere immediately.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c7d3e8a1b9f4"
down_revision: Union[str, None] = "44d5bbcd4d4a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Relax NOT NULL on api_key_hash / key_sha256 so a row can
    # represent an email+password user with no API key.
    op.alter_column("users", "api_key_hash", nullable=True)
    op.alter_column("users", "key_sha256", nullable=True)

    # New columns for human web login. Both nullable because
    # existing seeded users (API-key only) will not have them.
    op.add_column(
        "users",
        sa.Column("email", sa.Text(), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column("password_hash", sa.Text(), nullable=True),
    )

    # Email must be unique across non-NULL values. Postgres allows
    # multiple NULLs in a UNIQUE column by default, which is exactly
    # what we want — legacy users keep working.
    op.create_unique_constraint("uq_users_email", "users", ["email"])

    # Every user row must have at least one way to authenticate.
    # Without this, a row could exist with no auth method at all
    # and silently break the dependency chain at runtime.
    op.create_check_constraint(
        "ck_users_has_auth_method",
        "users",
        "api_key_hash IS NOT NULL OR password_hash IS NOT NULL",
    )

    # Sessions table. The primary key IS the cookie value — a UUID
    # is 122 bits of entropy, which is more than enough for a
    # session identifier. We do NOT hash it: unlike API keys, the
    # session ID is short-lived (default 7 days) and revocable in
    # one DELETE, so storing it in the clear is acceptable.
    op.create_table(
        "sessions",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column(
            "user_id",
            sa.UUID(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "expires_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
    )
    # Index for fast cleanup queries (DELETE WHERE expires_at < now()).
    op.create_index(
        "ix_sessions_expires_at",
        "sessions",
        ["expires_at"],
    )


def downgrade() -> None:
    """Downgrade schema.

    Note: this assumes every existing user row has an api_key_hash
    and key_sha256 populated. If you created email-only users via
    the new register flow, the downgrade will fail when alembic
    tries to re-apply NOT NULL. That is intentional — you should
    not silently lose users on a rollback.
    """
    op.drop_index("ix_sessions_expires_at", table_name="sessions")
    op.drop_table("sessions")

    op.drop_constraint("ck_users_has_auth_method", "users", type_="check")
    op.drop_constraint("uq_users_email", "users", type_="unique")
    op.drop_column("users", "password_hash")
    op.drop_column("users", "email")

    op.alter_column("users", "key_sha256", nullable=False)
    op.alter_column("users", "api_key_hash", nullable=False)
