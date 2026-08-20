"""Backend API package."""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["middleware", "routers"]


def __getattr__(name: str) -> ModuleType:
    """Lazily expose backend subpackages without eager imports."""
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
