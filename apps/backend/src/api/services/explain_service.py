"""Explainability service: where to run Grad-CAM, per serving mode.

Grad-CAM needs direct, gradient-enabled access to the model weights, which the
two serving modes provide differently:

- ``local`` / on-prem: the backend already loaded a ``SegmentationModel`` at
  startup and holds it on ``app.state.model``. We reuse that exact object --
  no second copy, no extra memory -- and run Grad-CAM in a worker thread with a
  timeout.

- ``azure_ml`` (cloud): the backend holds no local model (inference is
  delegated to the remote endpoint). Explanation is delegated the same way: we
  POST to the same endpoint with ``mode="explain"`` and it runs Seg-Grad-CAM on
  the model it already has loaded. The Container App therefore needs no local
  model and no weight download -- this is what removed the old request-time
  weight fetch that made cloud explanation unreliable.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
from cv_pipeline import explain as _pipeline_explain
from cv_pipeline.schema import ExplanationResult, Metadata
from fastapi import FastAPI

if TYPE_CHECKING:
    # Imported for typing only: the in-memory model exists in local/on-prem
    # mode; the runtime module must not depend on torch in cloud mode.
    from cv_pipeline.segmentation import SegmentationModel

logger = logging.getLogger(__name__)

# Upper bound for a single LOCAL Grad-CAM call (cloud is bounded by the endpoint
# client's own request timeout). On CPU a large plate at the 1024-px cap can
# take 30-90 s; without a bound, one slow call would tie up a worker thread and
# the awaiting request. Configurable via env.
#
# Caveat: Python cannot force-kill a running thread, so on timeout the worker is
# *abandoned* (anyio detaches it) -- it runs to completion and is then
# discarded. The benefit is that the event loop and the caller get a prompt,
# clean timeout instead of hanging.
_DEFAULT_EXPLAIN_TIMEOUT_S = 90.0


def _explain_timeout_seconds() -> float:
    """Return the per-call LOCAL explanation timeout in seconds (env-configurable)."""
    raw = os.getenv("EXPLAIN_TIMEOUT_SECONDS")
    if not raw:
        return _DEFAULT_EXPLAIN_TIMEOUT_S
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "Invalid EXPLAIN_TIMEOUT_SECONDS=%r; using default %.0fs.",
            raw,
            _DEFAULT_EXPLAIN_TIMEOUT_S,
        )
        return _DEFAULT_EXPLAIN_TIMEOUT_S


def get_explain_model(app: FastAPI) -> SegmentationModel:
    """Return the in-memory model to run Grad-CAM on (local/on-prem only).

    In ``azure_ml`` mode the explanation is delegated to the remote endpoint
    (see :func:`run_explanation`), so this is never called there.

    Args:
        app: The FastAPI application (provides ``app.state``).

    Returns:
        The ``SegmentationModel`` loaded at startup.

    Raises:
        RuntimeError: If no model was loaded at startup (a failed startup the
            health check already reports as not ready).
    """
    model = getattr(app.state, "model", None)
    if model is None:
        raise RuntimeError(
            "No in-memory model is available to explain. The backend started "
            "in local mode but model loading did not complete."
        )
    return model


def _resolve_and_explain(
    app: FastAPI, image_path: Path, metadata: Metadata
) -> ExplanationResult:
    """Resolve the in-memory model and run the (blocking) Grad-CAM pipeline.

    Runs in a worker thread (see :func:`run_explanation`), local/on-prem only.

    Args:
        app: The FastAPI application.
        image_path: Path to the uploaded image on disk.
        metadata: Pass-through metadata for the response.

    Returns:
        The computed ``ExplanationResult``.
    """
    model = get_explain_model(app)
    return _pipeline_explain(image_path=image_path, model=model, metadata=metadata)


async def run_explanation(
    app: FastAPI, image_path: Path, metadata: Metadata
) -> ExplanationResult:
    """Compute a Seg-Grad-CAM explanation, dispatching on serving mode.

    - ``azure_ml``: delegate to the model endpoint (``mode="explain"``). The
      endpoint client bounds its own HTTP call, so no local thread or timeout
      is needed here.
    - ``local`` / on-prem: run the blocking Grad-CAM on the in-memory model in a
      worker thread, bounded by ``EXPLAIN_TIMEOUT_SECONDS``. On expiry the
      thread is abandoned and ``TimeoutError`` is raised (router maps it to 504).

    Args:
        app: The FastAPI application.
        image_path: Path to the uploaded image on disk.
        metadata: Pass-through metadata for the response.

    Returns:
        The computed ``ExplanationResult``.

    Raises:
        TimeoutError: If a local explanation exceeds the configured timeout.
        RuntimeError: If the cloud endpoint call fails.
    """
    state = getattr(app.state, "service_state", None)
    serving_mode = getattr(state, "serving_mode", "local")

    if serving_mode == "azure_ml":
        # Imported lazily so local/on-prem deployments never import the client.
        from api.services.endpoint_client import run_endpoint_explanation

        logger.info("Explain: delegating to Azure ML endpoint (mode=explain).")
        return await run_endpoint_explanation(image_path, metadata)

    timeout = _explain_timeout_seconds()
    try:
        # abandon_on_cancel lets the await return promptly on timeout rather
        # than blocking until the (uncancellable) CPU work finishes.
        work = anyio.to_thread.run_sync(
            _resolve_and_explain,
            app,
            image_path,
            metadata,
            abandon_on_cancel=True,
        )
    except TypeError:
        # anyio < 4 spells the same flag `cancellable`.
        work = anyio.to_thread.run_sync(
            _resolve_and_explain,
            app,
            image_path,
            metadata,
            cancellable=True,
        )
    return await asyncio.wait_for(work, timeout=timeout)
