"""Seg-Grad-CAM explainability for the segmentation model.

This module produces a visual explanation of *why* the U-Net classified
pixels as root tissue. It implements Seg-Grad-CAM (Vinogradova et al.,
2020) -- the segmentation-aware adaptation of Grad-CAM.

Why Grad-CAM and not SHAP/LIME/Integrated-Gradients
----------------------------------------------------
The deployed model is a dense (per-pixel) segmenter, not a classifier, so
classifier-oriented attribution methods need adapting either way. Grad-CAM
is the cheapest faithful option here: a single forward + one ``autograd.grad``
call per patch, and the last decoder block already sits at full patch
resolution, so the heatmap is sharp without upsampling tricks. SHAP/LIME are
many forward passes (too slow for an interactive tab); Integrated Gradients
is noisier and needs more tuning for a worse-looking result.

How it stays faithful to the real inference path
-------------------------------------------------
Inference is patch-based (``SegmentationModel.predict``): pad -> tile into
256x256 patches with overlap -> predict -> stitch with overlap averaging.
The CAM is computed *the same way* -- we reuse the model's own padding,
patching and reconstruction helpers so the heatmap lines up pixel-for-pixel
with the mask the user already sees.

Cost control
------------
Grad-CAM needs gradients, so it cannot use the ``torch.no_grad`` fast path.
A full 4096x4096 plate would be ~1000 patch backward passes (minutes on CPU).
We therefore cap the processed image to ``max_side`` (default 1024) and
upsample the resulting heatmap. The CAM is for human eyeballing, not pixel
scoring, so the downscale is acceptable and keeps the tab interactive.

Thread-safety
-------------
In local/on-prem mode the same ``SegmentationModel`` object is shared with the
``/infer`` path. We capture the target-layer activation through a forward hook
that writes into a ``threading.local`` store, so a concurrent ``/infer``
forward running in another worker thread writes to *its* thread's slot and
never corrupts ours. ``autograd.grad`` (not ``.backward()``) is used, so no
gradients accumulate on the shared model's parameters. In ``eval`` mode the
parameters are read-only, so concurrent forwards are safe.
"""

from __future__ import annotations

import base64
import io
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from cv_pipeline._version import __version__
from cv_pipeline.preprocessing import detect_petri_dish
from cv_pipeline.schema import ExplanationResult, Metadata
from cv_pipeline.segmentation import SegmentationModel
from cv_pipeline.validation import validate_image

logger = logging.getLogger(__name__)

METHOD = "seg-grad-cam"

# Per-thread activation store. Written by the forward hook, read right after
# the forward in the SAME thread (see module docstring on thread-safety).
_THREAD_STORE = threading.local()

# Guards the one-time, idempotent hook attachment per model object.
_HOOK_LOCK = threading.Lock()


def explain(
    image_path: str | Path,
    model: SegmentationModel,
    metadata: Metadata | None = None,
    crop: bool = True,
    max_side: int = 1024,
    gradcam_batch_size: int = 8,
) -> ExplanationResult:
    """Compute a Seg-Grad-CAM heatmap for a single image.

    Mirrors the preprocessing of :func:`cv_pipeline.infer.infer` (validate,
    optional dish crop) so the heatmap aligns with the segmentation mask, then
    runs Grad-CAM over the same patch grid and maps the result back into the
    original image frame.

    Args:
        image_path: Path to the input image file.
        model: A loaded ``SegmentationModel``. The caller creates it once and
            reuses it (the backend passes either the in-memory inference model
            or a lazily loaded CPU model -- see the backend explain service).
        metadata: Optional metadata passed through to the response.
        crop: Whether to crop to the Petri dish first (matches ``infer`` so the
            heatmap and the mask cover the same region). Defaults to ``True``.
        max_side: Longest-side cap for the processed image. Larger images are
            downscaled before Grad-CAM and the heatmap is upsampled back. Keeps
            the call interactive on large HADES plates.
        gradcam_batch_size: Patches per forward/backward. Smaller than the
            inference batch size because the autograd graph is held in memory.

    Returns:
        An :class:`ExplanationResult` whose ``heatmap_b64`` is a grayscale PNG
        (0 = no attribution, 255 = peak) at the original image dimensions.

    Raises:
        ValidationError: If the input image fails any validation check.
    """
    image_path = Path(image_path)
    if metadata is None:
        metadata = Metadata()

    logger.info("Starting Seg-Grad-CAM explanation for '%s'.", image_path.name)

    # --- 1. Validate (same gate as infer) ----------------------------
    image = validate_image(image_path)
    original_h, original_w = image.shape[:2]

    # --- 2. Optional dish crop (same as infer) ------------------------
    crop_x1, crop_y1 = 0, 0
    if crop:
        crop_x1, crop_y1, crop_x2, crop_y2 = detect_petri_dish(image)
        image = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    crop_h, crop_w = image.shape[:2]

    # --- 3. Downscale for tractable backward passes -------------------
    downscaled = max(crop_h, crop_w) > max_side
    if downscaled:
        scale = max_side / max(crop_h, crop_w)
        proc = cv2.resize(
            image,
            (max(1, int(crop_w * scale)), max(1, int(crop_h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        proc = image

    # --- 4. Resolve target layer + attach the capture hook (once) -----
    target_layer, target_name = _resolve_target_layer(model.model)
    _ensure_hook(model.model, target_layer)

    # --- 5. Patch the processed image exactly like predict() ----------
    # Reuse the model's own helpers so the CAM grid matches the mask grid.
    prepared = model._prepare_channels(proc)
    proc_h, proc_w = prepared.shape[:2]
    padding = model._calculate_padding(proc_h, proc_w)
    padded = model._pad_image(prepared, padding)
    patches = model._create_patches(padded)  # (N, C, ps, ps) uint8
    padded_h, padded_w = padded.shape[:2]

    # --- 6. Grad-CAM per patch ----------------------------------------
    cams = _gradcam_over_patches(
        model.model,
        patches,
        model.device,
        model.patch_size,
        gradcam_batch_size,
    )

    # --- 7. Stitch with the same overlap averaging as the mask --------
    reconstructed = model._reconstruct(cams, padded_h, padded_w)
    cam_proc = model._remove_padding(reconstructed, padding)  # (proc_h, proc_w)

    # --- 8. Normalise to [0, 1] ---------------------------------------
    peak = float(cam_proc.max())
    if peak > 0:
        cam_norm = cam_proc / peak
    else:
        # ReLU killed everything (e.g. nothing looked like root). A flat zero
        # heatmap is a valid, honest result -- the model saw no salient region.
        cam_norm = cam_proc

    # --- 9. Upsample back to the crop size, then into the full frame --
    if (cam_norm.shape[0], cam_norm.shape[1]) != (crop_h, crop_w):
        cam_crop = cv2.resize(
            cam_norm, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR
        )
    else:
        cam_crop = cam_norm

    cam_full = np.zeros((original_h, original_w), dtype=np.float32)
    cam_full[crop_y1 : crop_y1 + crop_h, crop_x1 : crop_x1 + crop_w] = cam_crop

    # --- 10. Encode + build result ------------------------------------
    heatmap_b64 = _encode_heatmap(cam_full)

    result = ExplanationResult(
        pipeline_version=__version__,
        model_version=model.model_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        image_filename=image_path.name,
        image_width_px=original_w,
        image_height_px=original_h,
        metadata=metadata,
        method=METHOD,
        target_layer=target_name,
        downscaled=downscaled,
        heatmap_peak=round(peak, 6),
        heatmap_b64=heatmap_b64,
    )
    logger.info(
        "Explanation complete -- layer '%s', peak %.4f, downscaled=%s.",
        target_name,
        peak,
        downscaled,
    )
    return result


# ---- Grad-CAM internals ----------------------------------------------


def _resolve_target_layer(torch_model: torch.nn.Module) -> tuple[torch.nn.Module, str]:
    """Pick the convolutional layer to attribute against.

    Prefers the last decoder block of an ``smp`` U-Net, which sits at full
    patch resolution and gives the sharpest heatmap. Falls back to the last
    ``Conv2d`` in the model (excluding the segmentation head) so the function
    still works if the architecture or library internals change.

    Args:
        torch_model: The underlying ``torch.nn.Module`` (``model.model``).

    Returns:
        A tuple of (layer module, human-readable layer name).

    Raises:
        RuntimeError: If no usable convolutional layer can be found.
    """
    decoder = getattr(torch_model, "decoder", None)
    blocks = getattr(decoder, "blocks", None)
    if blocks is not None and len(blocks) > 0:
        return blocks[-1], "decoder.blocks[-1]"

    # Fallback: last Conv2d that is not part of the segmentation head.
    last_conv: torch.nn.Module | None = None
    last_name = ""
    for name, module in torch_model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and "segmentation_head" not in name:
            last_conv, last_name = module, name
    if last_conv is None:
        raise RuntimeError("No convolutional layer found to attribute against.")
    logger.warning(
        "Decoder blocks not found; falling back to Conv2d layer '%s'.", last_name
    )
    return last_conv, last_name


def _ensure_hook(torch_model: torch.nn.Module, target: torch.nn.Module) -> None:
    """Attach the activation-capture forward hook once per model object.

    Idempotent: a flag attribute on the model records that the hook is already
    attached, so repeated explain calls do not stack hooks. The hook writes the
    target layer's output into a thread-local slot (see module docstring).

    Args:
        torch_model: The underlying module (used only to hold the flag).
        target: The layer whose output activation we capture.
    """
    if getattr(torch_model, "_gradcam_hook_attached", False):
        return
    with _HOOK_LOCK:
        if getattr(torch_model, "_gradcam_hook_attached", False):
            return

        def _hook(_module: torch.nn.Module, _inp: object, output: torch.Tensor) -> None:
            _THREAD_STORE.activation = output

        target.register_forward_hook(_hook)
        torch_model._gradcam_hook_attached = True


def _gradcam_over_patches(
    torch_model: torch.nn.Module,
    patches: np.ndarray,
    device: str,
    patch_size: int,
    batch_size: int,
) -> np.ndarray:
    """Run Seg-Grad-CAM on every patch and return per-patch heatmaps.

    Args:
        torch_model: The underlying segmentation module (in ``eval`` mode).
        patches: ``(N, C, ps, ps)`` uint8 array from ``_create_patches``.
        device: Torch device string.
        patch_size: Expected output side length; CAMs are resized to this so
            the reconstruction helper can place them.
        batch_size: Patches per forward/backward.

    Returns:
        ``(N, patch_size, patch_size)`` float32 array of unnormalised CAMs.
    """
    tensor = torch.from_numpy(patches).float() / 255.0
    out: list[np.ndarray] = []

    for i in range(0, len(tensor), batch_size):
        batch = tensor[i : i + batch_size].to(device)
        cam = _gradcam_for_batch(torch_model, batch, patch_size)
        out.append(cam.cpu().numpy())

    return np.concatenate(out, axis=0)


def _gradcam_for_batch(
    torch_model: torch.nn.Module,
    batch: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """Compute the CAM for one batch of patches.

    Seg-Grad-CAM: backprop the summed logits over the predicted-root pixels,
    average the gradients per channel to get weights, take a ReLU'd weighted
    sum of the activation maps.

    Args:
        torch_model: The segmentation module.
        batch: ``(B, C, ps, ps)`` float tensor on the target device.
        patch_size: Output side length to resize each CAM to.

    Returns:
        ``(B, patch_size, patch_size)`` float tensor of unnormalised CAMs.
    """
    _THREAD_STORE.activation = None

    # Gradients required: do NOT use torch.no_grad here.
    logits = torch_model(batch)  # (B, 1, ps, ps)
    activation = getattr(_THREAD_STORE, "activation", None)
    if activation is None:
        raise RuntimeError("Grad-CAM hook did not capture an activation.")

    # Pixels of interest = predicted root (logit > 0 <=> sigmoid > 0.5).
    roi = logits > 0
    score = logits[roi].sum() if bool(roi.any()) else logits.sum()

    # autograd.grad does not touch param.grad, so the shared model stays clean.
    grads = torch.autograd.grad(score, activation)[0]  # (B, K, h, w)
    weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, K, 1, 1)
    cam = torch.relu((weights * activation).sum(dim=1))  # (B, h, w)

    if cam.shape[-2:] != (patch_size, patch_size):
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(patch_size, patch_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    return cam.detach()


def _encode_heatmap(cam: np.ndarray) -> str:
    """Encode a [0, 1] heatmap as a base64 grayscale PNG string.

    Args:
        cam: ``(H, W)`` float array in [0, 1].

    Returns:
        Base64-encoded PNG string (single channel, 0-255).
    """
    arr = np.clip(cam * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
