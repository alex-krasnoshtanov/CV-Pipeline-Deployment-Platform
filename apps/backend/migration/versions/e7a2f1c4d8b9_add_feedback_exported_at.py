"""add exported_at to feedback

- Adds: feedback.exported_at (nullable TIMESTAMPTZ)

Nullable so existing feedback rows are unaffected. NULL means the row's
corrected mask has not yet been staged into a training data asset; the
retraining export task stamps it once consumed, so the same correction
is never ingested into more than one training run.

Revision ID: e7a2f1c4d8b9
Revises: b1c4e7f2a9d3
Create Date: 2026-06-03
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e7a2f1c4d8b9"
down_revision: Union[str, None] = "b1c4e7f2a9d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "feedback",
        sa.Column("exported_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("feedback", "exported_at")
