"""Model loading abstraction for the backend service."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from cv_pipeline.segmentation import SegmentationModel
from cv_pipeline.weights import REGISTRY

logger = logging.getLogger(__name__)


def load_model() -> SegmentationModel:
    """Resolve and load the model based on environment variables.

    Priority:
      1. MODEL_PATH (explicit filesystem path) - used for dev or when
         pointing at a custom-trained checkpoint.
      2. MODEL_VERSION (registry version name) - used in Docker and
         the default flow.
      3. Fallback to the first entry in REGISTRY.

    Returns:
        A loaded SegmentationModel ready for inference calls.

    Raises:
        FileNotFoundError: If MODEL_PATH is set but the file is absent.
        RuntimeError: If weight download fails.
    """
    path_env = os.getenv("MODEL_PATH")
    version_env = os.getenv("MODEL_VERSION")

    if path_env:
        path = Path(path_env)
        if not path.exists():
            raise FileNotFoundError(
                f"MODEL_PATH='{path_env}' is set but the file does not exist."
            )
        logger.info("Loading model from explicit path '%s'.", path)
        return SegmentationModel(path)

    version = version_env or next(iter(REGISTRY))
    logger.info("Loading model by version '%s'.", version)
    return SegmentationModel(version)
