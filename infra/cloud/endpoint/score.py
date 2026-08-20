"""Scoring script for Azure ML Kubernetes online endpoint.

Azure ML calls init() once at startup, run() on every request. The same loaded
model serves two request modes, selected by the "mode" field of the JSON body:

- "infer"   (default): returns an InferenceResult (mask + landmarks).
- "explain": returns an ExplanationResult (Seg-Grad-CAM heatmap).

Explain reuses the exact model that served the prediction, so the explanation
is faithful to the deployed version. Grad-CAM needs gradients, so the explain
branch must NOT run under torch.no_grad -- run() does not wrap anything, and
cv_pipeline.explain manages its own autograd, so the two modes coexist safely.
"""

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

model = None


def init():
    """Load the model from the path Azure ML provides."""
    global model
    from cv_pipeline.segmentation import SegmentationModel

    model_dir = Path(os.environ["AZUREML_MODEL_DIR"])
    logger.info("AZUREML_MODEL_DIR contents: %s", list(model_dir.rglob("*")))

    # Find the .pth file -- handles both MLflow artifacts and direct uploads
    pth_files = sorted(model_dir.rglob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth file in {model_dir}")

    checkpoint = pth_files[0]
    logger.info("Loading model from %s", checkpoint)
    model = SegmentationModel(str(checkpoint))
    logger.info("Model loaded.")


def run(raw_data):
    """Run inference or explanation on a base64-encoded image.

    Input JSON:
        {"image_b64": "<base64>", "filename": "plate.png",
         "mode": "infer" | "explain", "plate_id": ..., "experiment_id": ...}

    Output JSON:
        mode "infer"   -> InferenceResult dict (mask + landmarks)
        mode "explain" -> ExplanationResult dict (heatmap_b64 + metadata)
    """
    import base64

    from cv_pipeline.schema import Metadata

    data = json.loads(raw_data)
    mode = data.get("mode", "infer")
    image_bytes = base64.b64decode(data["image_b64"])
    filename = data.get("filename", "input.png")
    suffix = Path(filename).suffix or ".png"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = Path(tmp.name)

    metadata = Metadata(
        plate_id=data.get("plate_id"),
        experiment_id=data.get("experiment_id"),
    )

    try:
        if mode == "explain":
            from cv_pipeline.explain import explain

            logger.info("Scoring request: mode=explain, file=%s", filename)
            result = explain(image_path=tmp_path, model=model, metadata=metadata)
        elif mode == "infer":
            from cv_pipeline import infer

            logger.info("Scoring request: mode=infer, file=%s", filename)
            result = infer(image_path=tmp_path, model=model, metadata=metadata)
        else:
            # Fail loudly: a misrouted or typo'd mode must not silently return
            # inference output dressed up as the caller's requested mode.
            raise ValueError(
                f"Unknown scoring mode: {mode!r} (expected 'infer' or 'explain')."
            )
        return json.dumps(result.to_dict())
    finally:
        tmp_path.unlink(missing_ok=True)
