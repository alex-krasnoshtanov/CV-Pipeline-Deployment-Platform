"""add FK and query indexes for predictions, feedback, sessions

Revision ID: d4f1a7c2e9b8
Revises: a8f3d09c1b72
Create Date: 2026-06-06

Postgres does not auto-create indexes on foreign-key columns, and the
only index in the schema so far is ix_sessions_expires_at. The /stats
dashboard (stats_service.py) and the session layer now run queries
that scan these columns. This adds:

- ix_predictions_created_at: backs the /stats time-series query
  (stats_service.py: WHERE created_at >= since, grouped by day).
- ix_sessions_user_id: backs session cleanup (WHERE user_id = ...)
  and the session->user join, plus ON DELETE CASCADE on user removal.
  FK column.
- ix_predictions_user_id: FK column (-> users.id); supports per-user
  prediction-history joins and lookups.
- ix_feedback_prediction_id: FK column (-> predictions.id); supports
  feedback->prediction joins and referential-integrity checks.
- ix_feedback_user_id: FK column (-> users.id); supports feedback->user
  joins and ON DELETE behaviour.

GROUP-BY-only columns (feedback.flag, predictions.model_version,
predictions.landmark_count) are intentionally NOT indexed: a btree
index does not accelerate a full-table aggregate.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4f1a7c2e9b8"
down_revision: Union[str, None] = "a8f3d09c1b72"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: add FK and query indexes."""
    op.create_index(
        "ix_predictions_created_at",
        "predictions",
        ["created_at"],
    )
    op.create_index(
        "ix_predictions_user_id",
        "predictions",
        ["user_id"],
    )
    op.create_index(
        "ix_feedback_prediction_id",
        "feedback",
        ["prediction_id"],
    )
    op.create_index(
        "ix_feedback_user_id",
        "feedback",
        ["user_id"],
    )
    op.create_index(
        "ix_sessions_user_id",
        "sessions",
        ["user_id"],
    )


def downgrade() -> None:
    """Downgrade schema: drop the indexes added in this revision."""
    op.drop_index("ix_sessions_user_id", table_name="sessions")
    op.drop_index("ix_feedback_user_id", table_name="feedback")
    op.drop_index("ix_feedback_prediction_id", table_name="feedback")
    op.drop_index("ix_predictions_user_id", table_name="predictions")
    op.drop_index("ix_predictions_created_at", table_name="predictions")
