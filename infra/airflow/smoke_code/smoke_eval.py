"""Smoke evaluation job: compare a candidate model against the champion.

Runs on the Azure ML compute (lambda-0) inside the cv-pipeline-training
environment, which has torch and the cv_pipeline package installed.

Both models are scored with the SAME procedure the endpoint uses: each plate
is run through cv_pipeline.infer (the exact function score.py calls), which
validates the image, crops to the petri dish, segments the crop, and maps the
mask back into full-plate coordinates. Because infer returns a full-size mask
(the crop is pasted back onto a zero canvas at its offset), the prediction
aligns pixel-for-pixel with the full-plate ground-truth masks in hades-smoke.

This is deliberately NOT predict_mask on the raw photo (the model would see
input it never sees in production) and NOT the SegmentationDataset/_evaluate
path (which resizes whole images to 256x256 -- only valid on pre-patched
training tiles, meaningless on raw plates).

F1 is accumulated at the pixel level across all plates (dataset-level
tp/fp/fn, scored once) rather than averaging per-image F1, matching
cv_pipeline._evaluate so one large plate cannot dominate the score.

The verdict (promote yes/no plus both F1 scores) is written to the job's
'verdict' output as verdict.json for Airflow to read back.

The pure helpers (pair_images_and_masks, f1_from_counts, decide_promote) are
kept import-light: torch/PIL/cv_pipeline are imported lazily inside
_f1_for_model, so these helpers can be unit-tested without the ML stack.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Same set the training dataset accepts, so a pair lifted from the test set
# pairs here exactly as it does in training.
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    """Parse the arguments Azure ML passes (mounted folders + versions)."""
    p = argparse.ArgumentParser(description="Champion-vs-candidate smoke eval.")
    p.add_argument(
        "--champion-dir",
        type=Path,
        required=True,
        help="Mounted folder of the currently-deployed (champion) model version.",
    )
    p.add_argument(
        "--candidate-dir",
        type=Path,
        required=True,
        help="Mounted folder of the newly-registered (candidate) model version.",
    )
    p.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="hades-smoke folder containing images/ and masks/ subdirectories.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write verdict.json (the job's 'verdict' output).",
    )
    p.add_argument("--champion-version", type=str, required=True)
    p.add_argument("--candidate-version", type=str, required=True)
    p.add_argument(
        "--margin",
        type=float,
        default=0.005,
        help="Candidate must beat champion F1 by at least this much to promote.",
    )
    return p.parse_args()


def _find_mask(image_path: Path, masks_dir: Path) -> Path | None:
    """Find the mask for an image, mirroring cloud_preprocess.find_root_mask.

    Tries, in order: an identically-named mask (``<stem>.<ext>``), the
    project's ``<stem>_root_mask.{tif,tiff,png}`` convention, then a
    ``<stem>*root_mask*`` glob. Matching the training preprocessing means the
    smoke set pairs exactly the way the pipeline already pairs the test data it
    was lifted from, instead of assuming a single naming scheme.

    Args:
        image_path: The image whose mask we want.
        masks_dir: Directory of mask files.

    Returns:
        The matching mask path, or None if nothing matches.
    """
    stem = image_path.stem
    # 1. Identical filename stem (e.g. plate_001.png / plate_001.png).
    for ext in sorted(_IMAGE_EXTENSIONS):
        candidate = masks_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    # 2. Project convention: plate_001 -> plate_001_root_mask.{tif,tiff,png}.
    for ext in (".tif", ".tiff", ".png"):
        candidate = masks_dir / f"{stem}_root_mask{ext}"
        if candidate.exists():
            return candidate
    # 3. Looser fallback for any <stem>*root_mask* variant.
    matches = sorted(masks_dir.glob(f"{stem}*root_mask*"))
    return matches[0] if matches else None


def pair_images_and_masks(eval_dir: Path) -> list[tuple[Path, Path]]:
    """Pair images to masks using the project's naming convention.

    Args:
        eval_dir: Folder with images/ and masks/ subdirectories.

    Returns:
        List of (image_path, mask_path) pairs.

    Raises:
        FileNotFoundError: If a subdirectory is missing or no pair matches.
    """
    images_dir = eval_dir / "images"
    masks_dir = eval_dir / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(
            f"Expected images/ and masks/ subdirectories under '{eval_dir}'."
        )

    pairs: list[tuple[Path, Path]] = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        mask = _find_mask(img, masks_dir)
        if mask is not None:
            pairs.append((img, mask))

    if not pairs:
        raise FileNotFoundError(
            f"No image/mask pairs found under '{eval_dir}' (looked for "
            f"<stem>.<ext> and <stem>_root_mask.<ext> in masks/)."
        )
    logger.info("Matched %d image/mask pair(s).", len(pairs))
    return pairs


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """Return the F1 score from accumulated pixel counts.

    Args:
        tp: True positives.
        fp: False positives.
        fn: False negatives.

    Returns:
        F1 in [0, 1]; 0.0 when there are no positive pixels at all.
    """
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def decide_promote(champion_f1: float, candidate_f1: float, margin: float) -> bool:
    """Return True if the candidate beats the champion by at least the margin.

    Args:
        champion_f1: F1 of the currently-deployed model.
        candidate_f1: F1 of the newly-registered model.
        margin: Minimum improvement required to promote (guards against noise).

    Returns:
        Whether the candidate should be promoted.
    """
    return candidate_f1 >= champion_f1 + margin


def _find_checkpoint(model_dir: Path) -> Path:
    """Return the first .pth under model_dir, deterministically.

    sorted() makes selection deterministic and identical to score.py's loader,
    so the gate scores the exact weights the endpoint would serve even if a
    model folder ever contains more than one .pth. Works for both mlflow_model
    and custom_model registry layouts (both contain the checkpoint .pth).

    Args:
        model_dir: Mounted model folder Azure ML provides.

    Returns:
        Path to the checkpoint file.

    Raises:
        FileNotFoundError: If no .pth exists under model_dir.
    """
    pth_files = sorted(model_dir.rglob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth file found under '{model_dir}'.")
    logger.info("Using checkpoint %s", pth_files[0])
    return pth_files[0]


def _decode_mask_b64(mask_b64: str):
    """Decode infer()'s base64 PNG mask into a uint8 numpy array.

    Args:
        mask_b64: Base64-encoded PNG (mode 'L', values 0/255), full-plate size.

    Returns:
        (H, W) uint8 numpy array.
    """
    import base64
    import io

    import numpy as np
    from PIL import Image

    raw = base64.b64decode(mask_b64)
    return np.array(Image.open(io.BytesIO(raw)).convert("L"))


def _f1_for_model(model_dir: Path, pairs: list[tuple[Path, Path]]) -> float:
    """Dataset-level F1 for one model over all pairs, via the endpoint path.

    Loads the model the way the endpoint does, then runs cv_pipeline.infer on
    each raw plate -- the exact function score.py calls, crop included. The
    returned mask is full-plate sized, so it is compared directly against the
    full-plate ground truth. Pixel tp/fp/fn are accumulated across all plates
    and scored once.

    The ML imports are local so the module's pure helpers stay importable
    (and unit-testable) without torch/PIL/cv_pipeline installed.

    Args:
        model_dir: Mounted model folder.
        pairs: Image/mask pairs to score against.

    Returns:
        Dataset-level F1 in [0, 1].

    Raises:
        RuntimeError: If a ground-truth mask cannot be read.
    """
    import numpy as np
    from cv_pipeline.infer import infer
    from cv_pipeline.segmentation import SegmentationModel
    from PIL import Image

    model = SegmentationModel(str(_find_checkpoint(model_dir)))
    tp = fp = fn = 0

    for img_path, mask_path in pairs:
        # Exact endpoint path: validate -> crop to dish -> predict -> map the
        # mask back to full-plate coordinates. infer reads the file itself.
        result = infer(image_path=img_path, model=model)
        pred = _decode_mask_b64(result.mask_b64)

        gt_img = Image.open(mask_path)
        if gt_img is None:
            raise RuntimeError(f"Failed to read mask '{mask_path}'.")
        gt = np.array(gt_img.convert("L"))

        # Predictions are full-plate sized; GT should match. If a GT mask was
        # ever saved at a different size, align it with nearest-neighbour (no
        # label interpolation) so the pixel comparison stays valid.
        if gt.shape != pred.shape:
            gt = np.array(
                Image.fromarray(gt).resize(
                    (pred.shape[1], pred.shape[0]), Image.NEAREST
                )
            )

        pred_bin = pred > 0
        gt_bin = gt > 0
        tp += int(np.logical_and(pred_bin, gt_bin).sum())
        fp += int(np.logical_and(pred_bin, ~gt_bin).sum())
        fn += int(np.logical_and(~pred_bin, gt_bin).sum())

    f1 = f1_from_counts(tp, fp, fn)
    logger.info("tp=%d fp=%d fn=%d -> F1=%.4f", tp, fp, fn, f1)
    return f1


def main() -> None:
    """Score both models, decide promote/skip, and write the verdict."""
    args = parse_args()
    pairs = pair_images_and_masks(args.eval_dir)

    logger.info("Scoring champion (v%s)...", args.champion_version)
    champion_f1 = _f1_for_model(args.champion_dir, pairs)
    logger.info("Scoring candidate (v%s)...", args.candidate_version)
    candidate_f1 = _f1_for_model(args.candidate_dir, pairs)

    promote = decide_promote(champion_f1, candidate_f1, args.margin)

    verdict = {
        "champion_version": args.champion_version,
        "candidate_version": args.candidate_version,
        "champion_f1": round(champion_f1, 4),
        "candidate_f1": round(candidate_f1, 4),
        "margin": args.margin,
        "promote": promote,
        "n_images": len(pairs),
    }
    logger.info("VERDICT: %s", verdict)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output_dir / "verdict.json"
    out_file.write_text(json.dumps(verdict, indent=2))
    logger.info("Wrote verdict to %s", out_file)


if __name__ == "__main__":
    main()
