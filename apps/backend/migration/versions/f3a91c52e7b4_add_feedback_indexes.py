"""add feedback indexes on flag and exported_at

Adds indexes that back the hot WHERE clauses used by the retrain trigger
(ready-set counting) and the review queue:

* ``ix_feedback_flag`` — plain index on ``flag``.
* ``ix_feedback_exported_at_pending`` — partial index on rows where
  ``exported_at IS NULL``, matching the export query's predicate exactly
  while staying small (it only indexes pending rows).

Revision ID: f3a91c52e7b4
Revises: e7a2f1c4d8b9
Create Date: 2026-06-05
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f3a91c52e7b4"
down_revision: Union[str, None] = "e7a2f1c4d8b9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_feedback_flag",
        "feedback",
        ["flag"],
        unique=False,
    )
    op.create_index(
        "ix_feedback_exported_at_pending",
        "feedback",
        ["exported_at"],
        unique=False,
        postgresql_where=sa.text("exported_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_feedback_exported_at_pending", table_name="feedback")
    op.drop_index("ix_feedback_flag", table_name="feedback")
