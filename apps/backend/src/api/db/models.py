"""SQLAlchemy models for the backend database.

Defines the five tables: users, predictions, feedback, model_versions,
and sessions. Users must be defined before Prediction, Feedback, and
Session because all three reference users.id via FK.

User has two parallel auth paths that intentionally coexist:
- api_key_hash + key_sha256: legacy / service-account flow (CLI,
  Streamlit, robotic platform). Bound to seed.py + POST /users.
- email + password_hash: new human-web flow. Bound to POST
  /auth/register and POST /auth/login.

A CHECK constraint guarantees every row has at least one path. The
auth dependency picks which path to use based on what the client
sent (cookie vs X-API-Key header).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)

    # API-key auth path. Nullable for email-only users.
    api_key_hash: Mapped[str | None] = mapped_column(Text)
    key_sha256: Mapped[str | None] = mapped_column(
        String(64),
        unique=True,
    )

    # Email + password auth path. Nullable for API-key-only users
    # (the legacy seeded "Default Researcher" / "Admin" accounts).
    email: Mapped[str | None] = mapped_column(Text, unique=True)
    password_hash: Mapped[str | None] = mapped_column(Text)

    # OAuth credential — Path 3 / P3-03.
    # (oauth_provider, oauth_subject) is the OAuth identity key.
    # Email travels with these but is NOT the identity (emails change,
    # provider subjects do not).
    oauth_provider: Mapped[str | None] = mapped_column(Text)
    oauth_subject: Mapped[str | None] = mapped_column(Text)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    role: Mapped[str] = mapped_column(
        Text,
        default="researcher",
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "role IN ('researcher', 'admin')",
            name="ck_users_role",
        ),
        CheckConstraint(
            "api_key_hash IS NOT NULL "
            "OR password_hash IS NOT NULL "
            "OR oauth_subject IS NOT NULL",
            name="ck_users_has_any_credential",
        ),
        UniqueConstraint(
            "oauth_provider",
            "oauth_subject",
            name="uq_users_oauth_identity",
        ),
    )


class Session(Base):
    """Server-side session row.

    The session id IS the cookie value sent to the browser. We
    chose server-side sessions over JWT for two reasons:
    1. Revocation is one DELETE. JWT revocation needs a blacklist.
    2. We already do a DB hit per request for X-API-Key auth, so
       the "extra" round-trip is not actually extra.

    Sessions are short-lived (default 7 days) and expired rows are
    cleaned up opportunistically on login. A periodic cron is
    optional but not required for correctness — get_user_from_session
    refuses to return a User for an expired row regardless of
    whether the row has been physically deleted yet.
    """

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )
    image_filename: Mapped[str] = mapped_column(Text, nullable=False)

    image_width_px: Mapped[int] = mapped_column(Integer, nullable=False)

    image_height_px: Mapped[int] = mapped_column(Integer, nullable=False)

    # Durable locator for the input image, written asynchronously after
    # the response by the feedback-persistence background task. NULL
    # until that task succeeds (or for anonymous calls, which are never
    # persisted). The export step pairs this image with its mask.
    image_uri: Mapped[str | None] = mapped_column(Text)

    plate_id: Mapped[str | None] = mapped_column(Text)

    experiment_id: Mapped[str | None] = mapped_column(Text)

    pipeline_version: Mapped[str] = mapped_column(Text, nullable=False)

    model_version: Mapped[str] = mapped_column(Text, nullable=False)

    mask_b64: Mapped[str] = mapped_column(Text, nullable=False)

    mask_confidence: Mapped[float] = mapped_column(nullable=False)

    landmark_count: Mapped[int] = mapped_column(Integer, nullable=False)

    landmarks: Mapped[list] = mapped_column(JSON, nullable=False)

    # Nullable: anonymous /infer calls leave this NULL.
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id")
    )

    __table_args__ = (
        CheckConstraint(
            "mask_confidence >= 0 AND mask_confidence <= 1",
            name="ck_predictions_mask_confidence_range",
        ),
    )


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("predictions.id"), nullable=False
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id")
    )
    flag: Mapped[str] = mapped_column(Text, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    corrected_mask_b64: Mapped[str | None] = mapped_column(Text)
    # Set by the retraining export task once this row's corrected mask has
    # been staged into a hades-raw-upload data asset version. NULL means the
    # row is still pending export; stamping it prevents re-ingesting the same
    # correction into a later training run.
    exported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(
            "flag IN ('good', 'bad', 'uncertain')",
            name="ck_feedback_flag",
        ),
        Index("ix_feedback_flag", "flag"),
        Index(
            "ix_feedback_exported_at_pending",
            "exported_at",
            postgresql_where=text("exported_at IS NULL"),
        ),
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    registered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )
    model_version: Mapped[str] = mapped_column(
        Text,
        unique=True,
        nullable=False,
    )

    val_f1: Mapped[float | None] = mapped_column(Float)

    test_f1: Mapped[float | None] = mapped_column(Float)

    azure_ml_version: Mapped[int | None] = mapped_column(Integer)

    notes: Mapped[str | None] = mapped_column(Text)
