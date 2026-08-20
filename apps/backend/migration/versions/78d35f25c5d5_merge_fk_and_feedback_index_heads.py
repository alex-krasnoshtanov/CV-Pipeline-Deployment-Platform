"""merge fk/query-index and feedback-index migration heads

Revision ID: 78d35f25c5d5
Revises: d4f1a7c2e9b8, f3a91c52e7b4
Create Date: 2026-06-08 14:07:19.085599

Two migration branches were created independently off a8f3d09c1b72 and
both were merged into the mainline, leaving the revision tree with two
heads:

    a8f3d09c1b72
      |-> b1c4e7f2a9d3 -> e7a2f1c4d8b9 -> f3a91c52e7b4  (feedback indexes)
      |-> d4f1a7c2e9b8                                  (FK + query indexes)

``alembic upgrade head`` is ambiguous with more than one head and exits
non-zero. The container entrypoint runs it under ``set -e`` before
starting uvicorn, so the whole backend container aborts before it ever
listens on its port. This revision carries no schema changes; it only
re-joins the two lineages into a single head so ``upgrade head`` is
unambiguous again.

Alternative considered: rewrite the ``down_revision`` of one leaf
migration to chain it after the other. That also collapses to one head
but rewrites already-merged history and reorders migrations that may have
applied on some environments. A merge revision is non-destructive and is
exactly what ``alembic merge heads`` generates, so it is preferred here.
"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "78d35f25c5d5"
down_revision: Union[str, Sequence[str], None] = ("d4f1a7c2e9b8", "f3a91c52e7b4")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Merge revision only: no schema changes. Both parent branches are
    # already applied by the time alembic reaches this node.
    pass


def downgrade() -> None:
    """Downgrade schema."""
    # No-op: alembic manages the branch/unbranch bookkeeping from the
    # revision graph; there is nothing to undo here.
    pass
