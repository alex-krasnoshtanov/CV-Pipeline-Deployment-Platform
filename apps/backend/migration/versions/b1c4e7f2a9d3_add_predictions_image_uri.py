"""add image_uri to predictions

- Adds: predictions.image_uri (nullable Text)

Nullable so existing prediction rows are unaffected and anonymous /infer
calls (which are never persisted) stay NULL. Populated asynchronously by
the feedback image-persistence background task.

Revision ID: b1c4e7f2a9d3
Revises: a8f3d09c1b72
Create Date: 2026-06-03
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b1c4e7f2a9d3"
down_revision: Union[str, None] = "a8f3d09c1b72"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "predictions",
        sa.Column("image_uri", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("predictions", "image_uri")
