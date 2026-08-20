"""Test configuration for the Airflow DAG helpers.

Airflow places the ``dags/`` directory on ``sys.path`` at runtime, which is
why DAG modules import their siblings flatly (e.g. ``import feedback_export``
or ``from azure_helpers import get_ml_client``). pytest does not do this, so
running these tests from the repo root would fail to import the helper under
test. Replicate Airflow's behaviour by prepending the sibling ``dags/`` folder
to ``sys.path`` before collection.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DAGS_DIR = Path(__file__).resolve().parent.parent / "dags"
if str(_DAGS_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGS_DIR))
