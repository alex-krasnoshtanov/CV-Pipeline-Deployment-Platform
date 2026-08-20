"""Authentication dependencies for the backend API."""

from api.auth.dependencies import (
    SESSION_COOKIE_NAME,
    optional_user,
    require_admin,
    require_user,
)

__all__ = [
    "SESSION_COOKIE_NAME",
    "optional_user",
    "require_admin",
    "require_user",
]
