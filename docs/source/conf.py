"""Sphinx configuration for the CV Pipeline documentation site."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------

project = "CV Pipeline"
author = "Oleksii Krasnoshtanov, Filipp Lotsmanov, Danil Sysenko, Marin Chiosa"
copyright = "2026, the CV Pipeline contributors"

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "packages" / "cv-pipeline" / "src"))
sys.path.insert(0, str(_root / "apps" / "backend" / "src"))

# Set up minimal environment for autodoc imports
if not os.getenv("DB_URL"):
    os.environ["DB_URL"] = "postgresql+asyncpg://dummy:dummy@localhost:5432/dummy"
if not os.getenv("API_KEY"):
    os.environ["API_KEY"] = "dummy-api-key"
if not os.getenv("ADMIN_API_KEY"):
    os.environ["ADMIN_API_KEY"] = "dummy-admin-key"
if not os.getenv("MODEL_VERSION"):
    os.environ["MODEL_VERSION"] = "unet-v1"

try:
    import importlib.metadata as _meta

    _pkg_version = _meta.version("cv-pipeline")
except Exception:
    _pkg_version = "unknown"

version = _pkg_version
release = _pkg_version

# ---------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "autoapi.extension",
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/.ipynb_checkpoints",
]

# ---------------------------------------------------------------------
# AutoAPI
# ---------------------------------------------------------------------

autoapi_type = "python"
autoapi_dirs = [
    str(_root / "packages" / "cv-pipeline" / "src" / "cv_pipeline"),
    str(_root / "apps" / "backend" / "src" / "api"),
]
autoapi_root = "autoapi"
autoapi_keep_files = False
autoapi_add_toctree_entry = True

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]

autoapi_member_order = "bysource"
autoapi_python_class_content = "both"

autoapi_ignore = [
    "*/tests/*",
    "*/__pycache__/*",
    "*/_version.py",
    "*/api/db/*",
    "*/api/db/models.py",
    "*/api/db/__init__.py",
]

# ---------------------------------------------------------------------
# Napoleon
# ---------------------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# ---------------------------------------------------------------------
# autodoc
# ---------------------------------------------------------------------

autodoc_mock_imports = [
    "torch",
    "torchvision",
    "segmentation_models_pytorch",
    "cv2",
    "fastapi",
    "pydantic",
    "pydantic_settings",
    "starlette",
    "sqlalchemy",
    "PIL",
    "numpy",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

# ---------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# ---------------------------------------------------------------------
# MyST
# ---------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# ---------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------

html_theme = "furo"
html_title = f"CV Pipeline {release}"
html_static_path = ["_static"]
html_show_sourcelink = True
html_copy_source = False

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": (
        "https://github.com/Gfgf96/"
        "CV-Pipeline-Deployment-Platform/"
    ),
    "source_branch": "main",
    "source_directory": "docs/source/",
}

# ---------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------

suppress_warnings = [
    # autoapi machinery
    "autoapi",
    "autoapi.toc_reference",
    # autodoc renders modules that mock imports - emits noise
    "autodoc",
    "autodoc.import_object",
    # Cross-references that intersphinx cannot resolve
    "ref",
    "ref.python",
    "ref.class",
    "ref.doc",
    "ref.ref",
    # Documents in multiple toctrees (we use both nested and root toctrees)
    "toc",
    "toc.not_included",
    "toc.not_readable",
    "toc.duplicate_entry",
    "toc.excluded",
    # Docutils RST parser warnings/errors (3rd-party docstring quirks)
    "docutils",
    # MyST highlighter relaxed-mode fallbacks (cosmetic)
    "misc.highlighting_failure",
    "misc",
    "app.add_directive",
]

# Disable nitpicky mode: don't fail on every cross-reference miss.
nitpicky = False

# Set keep_warnings=True to log without failing - paired with above
# suppress list, gives us a clean -W exit while still showing issues
# in HTML if relevant.
keep_warnings = True
