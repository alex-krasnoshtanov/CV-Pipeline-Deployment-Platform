"""Backend configuration loaded once at startup.

Why a class instead of os.getenv scattered everywhere?
1. Type-checked at startup. alert_latency_p95_ms is int, not "5000"-string.
2. Discoverable. One file lists every env var the backend reads.
3. Validated once. Bad config crashes the container at boot, not on
   the first request three hours into demo day.
4. Easy to override in tests via Settings(_env_file=None, db_url=...).

Why not move every os.getenv at once? Big-bang refactors break tests
in non-obvious ways. The plan was to migrate the high-impact spots
first (model loader, main.py, anywhere with a default that should be
typed) and leave the one-shot scripts (seed.py) alone.

NOTE (2026-06): that migration has NOT happened yet. Nothing imports
``get_settings()`` — ``model_loader.py`` and ``main.py`` still read
``os.getenv`` directly, so this ``Settings`` class is the intended
target of that migration, not a description of the live state. Wire
``get_settings()`` into those modules (or delete this file) before
relying on any field defined here.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All backend env vars in one place."""

    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    # Database
    db_url: str = ""  # empty -> init_db raises ValueError, which is fine

    # Auth
    api_key: str = ""  # plaintext key for the seeded researcher user
    admin_api_key: str = ""  # plaintext key for the seeded admin user

    # Model resolution: MODEL_PATH wins, else MODEL_VERSION, else first
    # entry in REGISTRY. Mutual exclusivity is checked in load_model.
    model_path: str | None = None
    model_version: str | None = None
    model_endpoint_url: str | None = None
    model_endpoint_key: str | None = None

    # CORS
    cors_origins: str = ""

    # Logging
    log_level: str = "INFO"

    # Monitoring thresholds (specification §13)
    alert_confidence_min: float = 0.60
    alert_low_conf_fraction: float = 0.20
    alert_latency_p95_ms: int = 5000
    alert_error_rate: float = 0.05
    retrain_feedback_threshold: int = 50

    # Drift detection rolling window
    drift_window_size: int = 100  # recent predictions to sample
    # Optional Teams Incoming Webhook; when unset the webhook step is skipped.
    teams_webhook_url: str | None = None
    # Feedback image storage (the retraining flywheel's raw landing zone).
    # backend: 'local' (mounted volume, default) or 'blob' (Azure).
    # local_dir must be a persisted volume in the container, or images
    # vanish on restart. blob_* are only read when backend == 'blob' and
    # the container must back the workspaceblobstore datastore.
    feedback_storage_backend: str = "local"
    feedback_storage_local_dir: str = "/data/feedback"
    feedback_storage_blob_connection_string: SecretStr | None = None
    feedback_storage_blob_container: str | None = None

    @property
    def serving_mode(self) -> str:
        """'azure_ml' if MODEL_ENDPOINT_URL is set, else 'local'."""
        return "azure_ml" if self.model_endpoint_url else "local"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings, building it lazily once.

    @lru_cache rather than a module-level constant so tests can call
    get_settings.cache_clear() before injecting overrides.
    """
    return Settings()
