"""add OAuth columns to users

- Adds: oauth_provider, oauth_subject, last_login_at
- Replaces ck_users_has_auth_method with ck_users_has_any_credential
  that covers all three credential types
- Adds composite unique constraint on (oauth_provider, oauth_subject)

Notes:
- email and password_hash columns already exist (c7d3e8a1b9f4).
- api_key_hash and key_sha256 are already nullable (c7d3e8a1b9f4).
- We rename the CHECK constraint because its definition is changing.
  Replacing in-place is safer than ALTERing because Postgres has no
  ALTER CHECK syntax for the constraint expression.

Revision ID: a8f3d09c1b72
Revises: c7d3e8a1b9f4
Create Date: 2026-05-18
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a8f3d09c1b72"
down_revision: Union[str, None] = "c7d3e8a1b9f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # New OAuth columns. All nullable so existing rows are unaffected:
    # API-key-only and email+password users have no OAuth identity.
    op.add_column("users", sa.Column("oauth_provider", sa.Text(), nullable=True))
    op.add_column("users", sa.Column("oauth_subject", sa.Text(), nullable=True))
    op.add_column(
        "users",
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Composite uniqueness. The Postgres default treats NULLs as
    # distinct, so multiple (NULL, NULL) rows do not violate this —
    # which is what we want for non-OAuth users.
    op.create_unique_constraint(
        "uq_users_oauth_identity",
        "users",
        ["oauth_provider", "oauth_subject"],
    )

    # Drop the old two-way CHECK and replace with a three-way one
    # that also accepts OAuth users.
    op.drop_constraint("ck_users_has_auth_method", "users", type_="check")
    op.create_check_constraint(
        "ck_users_has_any_credential",
        "users",
        "api_key_hash IS NOT NULL "
        "OR password_hash IS NOT NULL "
        "OR oauth_subject IS NOT NULL",
    )


def downgrade() -> None:
    # Restore the two-way constraint first so we never end up with
    # a row that has only OAuth credentials and no way to authenticate
    # after the downgrade.
    op.drop_constraint("ck_users_has_any_credential", "users", type_="check")
    op.create_check_constraint(
        "ck_users_has_auth_method",
        "users",
        "api_key_hash IS NOT NULL OR password_hash IS NOT NULL",
    )
    op.drop_constraint("uq_users_oauth_identity", "users", type_="unique")
    op.drop_column("users", "last_login_at")
    op.drop_column("users", "oauth_subject")
    op.drop_column("users", "oauth_provider")
