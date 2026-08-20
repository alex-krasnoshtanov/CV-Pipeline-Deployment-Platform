"""Client for calling the HADES segmentation model endpoint.

The model is served by an Azure ML (Kubernetes/Arc) online endpoint on the
BUas cluster. That endpoint's scoring URI is a private campus IP that is NOT
reachable from Azure Container Apps, so we do **not** use the Azure ML SDK
``invoke()`` here: management-plane ``invoke()`` still POSTs to the endpoint's
private ``scoring_uri``, which Azure cannot route to (it times out).

Instead we POST directly to the public NGROK tunnel BUas runs in front of the
endpoint (per the NGROK guide). Two env vars drive it:

- ``MODEL_ENDPOINT_URL``: the public scoring URL, e.g.
  ``https://endpoint.example.invalid/port-REDACTED/api/v1/endpoint/hades-unet-endpoint/score``
- ``MODEL_ENDPOINT_KEY``: the endpoint key, sent as ``Authorization: Bearer <key>``

The same endpoint serves both request modes via the JSON ``mode`` field:
``infer`` returns a segmentation result, ``explain`` returns a Seg-Grad-CAM
heatmap computed on the same loaded model. This is why cloud explanation needs
no local model in the Container App and no request-time weight download.

This path needs no Azure AD credential at all -- the endpoint key is the only
auth -- which is why the service-principal env vars are irrelevant here.

Design note -- why stdlib ``urllib`` rather than ``requests``/``httpx``:
the call is a single JSON POST already offloaded to a worker thread, so the
extra dependency buys nothing. ``urllib`` keeps this module dependency-free and
mirrors the mentor's reference example exactly. If richer retry/backoff is ever
needed, swapping to ``httpx`` is a localized change inside ``_post``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal

from cv_pipeline.schema import ExplanationResult, InferenceResult, Metadata
from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

# Bound the scoring call: the model runs on a shared GPU box and the request
# crosses an NGROK tunnel, so allow headroom -- but cap it so a hung tunnel
# fails fast instead of holding the worker thread indefinitely.
_REQUEST_TIMEOUT_S = 60


def _build_payload(
    image_path: Path,
    metadata: Metadata,
    mode: Literal["infer", "explain"] = "infer",
) -> bytes:
    """Build the JSON request body the scoring script expects.

    Args:
        image_path: Path to the image file to score.
        metadata: Caller-supplied metadata (``plate_id``/``experiment_id`` may
            be ``None``).
        mode: ``"infer"`` for a segmentation result, ``"explain"`` for a
            Grad-CAM heatmap. The scoring script dispatches on this field.

    Returns:
        UTF-8 encoded JSON bytes ready to use as the HTTP request body.
    """
    image_b64 = base64.b64encode(image_path.read_bytes()).decode()
    payload = {
        "image_b64": image_b64,
        "filename": image_path.name,
        "plate_id": metadata.plate_id,
        "experiment_id": metadata.experiment_id,
        "mode": mode,
    }
    return json.dumps(payload).encode()


def _post(payload: bytes) -> dict[str, Any]:
    """POST a prepared body to the endpoint and return the parsed JSON dict.

    Shared by the inference and explanation paths -- only the payload ``mode``
    and how the caller parses the result differ.

    Args:
        payload: JSON request body from :func:`_build_payload`.

    Returns:
        The decoded response as a dict.

    Raises:
        RuntimeError: If the endpoint env vars are missing, the endpoint
            returns a non-2xx status, the request times out or cannot connect,
            or the response body is not valid JSON.
    """
    url = os.environ.get("MODEL_ENDPOINT_URL")
    key = os.environ.get("MODEL_ENDPOINT_KEY")
    if not url or not key:
        raise RuntimeError(
            "MODEL_ENDPOINT_URL and MODEL_ENDPOINT_KEY must both be set for "
            "cloud (azure_ml) serving mode."
        )

    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_S) as response:
            raw = response.read().decode()
    except urllib.error.HTTPError as exc:
        # 401/403 -> bad/expired endpoint key; 5xx -> scoring script error.
        detail = exc.read().decode(errors="replace")[:500]
        logger.error("Endpoint returned HTTP %s: %s", exc.code, detail)
        raise RuntimeError(f"Model endpoint error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        # Tunnel down / DNS / timeout -- the request never got an HTTP reply.
        logger.error("Could not reach model endpoint %s: %s", url, exc.reason)
        raise RuntimeError(f"Model endpoint unreachable: {exc.reason}") from exc

    # Azure ML scoring can return a JSON string that itself contains JSON;
    # unwrap one level if needed, matching the previous SDK behaviour.
    try:
        result = json.loads(raw)
        if isinstance(result, str):
            result = json.loads(result)
    except json.JSONDecodeError as exc:
        logger.error("Endpoint response was not valid JSON: %s", raw[:500])
        raise RuntimeError("Model endpoint returned a non-JSON response.") from exc

    return result


def _call_endpoint(image_path: Path, metadata: Metadata) -> InferenceResult:
    """POST an inference request and parse the segmentation result."""
    result = _post(_build_payload(image_path, metadata, mode="infer"))
    return InferenceResult.from_dict(result)


def _call_endpoint_explain(image_path: Path, metadata: Metadata) -> ExplanationResult:
    """POST an explanation request and parse the Grad-CAM result.

    Guards against deploy skew: if the live endpoint still runs an older
    ``score.py`` without explain support, it returns an ``InferenceResult``
    shape (no ``heatmap_b64``). We turn that into an explicit, diagnosable error
    instead of an opaque ``KeyError`` from ``from_dict``. The real fix is deploy
    ordering -- see the runbook in infra/cloud/README.md.
    """
    result = _post(_build_payload(image_path, metadata, mode="explain"))
    if "heatmap_b64" not in result:
        raise RuntimeError(
            "Endpoint returned no explanation (no heatmap_b64). The deployed "
            "score.py likely predates explain support -- redeploy the endpoint "
            "before the backend (see infra/cloud/README.md)."
        )
    return ExplanationResult.from_dict(result)


async def run_endpoint_inference(
    image_path: Path, metadata: Metadata
) -> InferenceResult:
    """Async wrapper -- offloads the blocking inference HTTP call to a thread.

    Args:
        image_path: Path to the image file to score.
        metadata: Caller-supplied metadata passed through to the response.

    Returns:
        The parsed inference result returned by the model endpoint.
    """
    return await run_in_threadpool(_call_endpoint, image_path, metadata)


async def run_endpoint_explanation(
    image_path: Path, metadata: Metadata
) -> ExplanationResult:
    """Async wrapper -- offloads the blocking explanation HTTP call to a thread.

    Cloud explanation reuses the same model endpoint as inference, with the
    request ``mode`` set to ``explain``. The endpoint runs Seg-Grad-CAM on the
    model it already has loaded, so the Container App needs no local model and
    no weight download.

    Args:
        image_path: Path to the image file to explain.
        metadata: Caller-supplied metadata passed through to the response.

    Returns:
        The parsed explanation result returned by the model endpoint.
    """
    return await run_in_threadpool(_call_endpoint_explain, image_path, metadata)
