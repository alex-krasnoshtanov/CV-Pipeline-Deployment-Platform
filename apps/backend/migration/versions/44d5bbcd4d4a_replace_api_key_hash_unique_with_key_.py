"""replace api_key_hash unique with key_sha256 column

Revision ID: 44d5bbcd4d4a
Revises: 66bff61fad56
Create Date: 2026-05-05
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "44d5bbcd4d4a"
down_revision: Union[str, None] = "66bff61fad56"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Step 1: add as nullable so existing rows don't fail.
    op.add_column(
        "users",
        sa.Column("key_sha256", sa.String(length=64), nullable=True),
    )

    # Step 2: backfill existing rows with empty string so NOT NULL
    # can be applied. These users will need to be re-seeded since
    # we cannot recover the SHA-256 digest from a bcrypt hash.
    op.execute("UPDATE users SET key_sha256 = '' WHERE key_sha256 IS NULL")

    # Step 3: set NOT NULL now that all rows have a value.
    op.alter_column("users", "key_sha256", nullable=False)

    # Step 4: add named unique constraint.
    op.create_unique_constraint(
        "uq_users_key_sha256",
        "users",
        ["key_sha256"],
    )

    # Step 5: drop old unique constraint on api_key_hash.
    op.drop_constraint("users_api_key_hash_key", "users", type_="unique")


def downgrade() -> None:
    # Reverse: restore api_key_hash unique, drop key_sha256 column.
    op.create_unique_constraint(
        "users_api_key_hash_key",
        "users",
        ["api_key_hash"],
    )
    op.drop_constraint("uq_users_key_sha256", "users", type_="unique")
    op.drop_column("users", "key_sha256")
