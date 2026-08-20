"""Azure ML incremental preprocessing script for HADES image data.

Unlike cloud_preprocess.py (clean-slate fresh split), this script
preserves a frozen test set so models can be compared on a stable
held-out set across versions.

This script runs on the Azure ML cluster. It:
  1. Copies the existing TRAIN, VAL, and TEST patches through unchanged.
  2. Splits the NEW raw images at the SOURCE-IMAGE level into train/val
     (preserving the 70:20 -> 0.7778/0.2222 proportion), then patches each
     image into its assigned split and appends those patches.

Why image-level for the new data: patches are generated with overlap, so
two patches from one image share pixels. Routing a whole image into a
single split keeps its overlapping patches together and prevents them
leaking across the train/val boundary. Existing patches are NOT
reshuffled across splits -- train stays train, val stays val -- so no new
cross-split overlap is introduced. (Historical overlap baked into older
asset versions, before this change, cannot be undone here because the
stored patches carry no source-image provenance; this script stops adding
to it.)

New data never enters the test set. Test contents are identical to the
existing test asset, only renamed into the output sequence.

Azure ML mounts data assets and blob paths as local folders. Because the
inputs are FUSE-mounted (every file touch is a network round-trip), file
enumeration and copying dominate runtime. Two optimisations target that:
  * collect_existing pairs files via two directory listings + a set
    intersection instead of one network stat() per file.
  * copy phases run over a thread pool: each copy is an I/O wait that
    releases the GIL, so concurrent copies overlap their network waits.
Progress is logged periodically so a slow copy is distinguishable from
a stalled one.
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PATCH_SIZE = 256
OVERLAP = 0.5
SHRINK = 20

# Train/val ratio for the NEW images, preserving the original 70:20
# proportion. Test is frozen and excluded from this split.
TRAIN_RATIO_IN_POOL = 0.7 / (0.7 + 0.2)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Incremental HADES preprocessing with frozen test set.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Staging folder with images/ and masks/ subdirs.",
    )
    parser.add_argument(
        "--existing-train-dir",
        type=Path,
        required=True,
        help="Current hades-train data asset.",
    )
    parser.add_argument(
        "--existing-val-dir",
        type=Path,
        required=True,
        help="Current hades-val data asset.",
    )
    parser.add_argument(
        "--existing-test-dir",
        type=Path,
        required=True,
        help="Current hades-test data asset (frozen, passed through).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the new split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible new-image train/val assignment.",
    )
    parser.add_argument(
        "--copy-workers",
        type=int,
        default=32,
        help="Parallel worker threads for file-copy operations.",
    )
    return parser.parse_args()


# ---- mask dtype handling -------------------------------------------------


def _to_uint8_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a mask to uint8 without destroying its nonzero labels.

    Masks may arrive 8- or 16-bit, encoded as either {0, 1} or {0, 255}.
    Dividing a {0, 1} 16-bit mask by 256 rounds every root pixel to 0 and
    silently deletes the label. Patches are binarised at training time
    (mask > 0), so we only need the nonzero pixels to survive the cast:
    scale by 256 only when values genuinely exceed the 8-bit range.
    """
    if mask.dtype == np.uint8:
        return mask
    if mask.max() > 255:
        return (mask / 256).astype(np.uint8)
    return mask.astype(np.uint8)


# ---- Petri dish detection ------------------------------------------------


def detect_petri_dish(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Detect petri dish and return square crop coordinates (y1, y2, x1, x2)."""
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3
        else image.copy()
    )
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((20, 20), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    x += SHRINK
    y += SHRINK
    w -= 2 * SHRINK
    h -= 2 * SHRINK

    size = max(w, h)
    cx, cy = x + w // 2, y + h // 2

    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(image.shape[1], x1 + size)
    y2 = min(image.shape[0], y1 + size)

    return y1, y2, x1, x2


# ---- Patching ------------------------------------------------------------


def calculate_padding(h: int, w: int) -> tuple[int, int, int, int]:
    """Calculate mirror padding so (size - patch_size) % step == 0."""
    step = int(PATCH_SIZE * (1 - OVERLAP))

    def pad_needed(size: int) -> int:
        if size < PATCH_SIZE:
            return PATCH_SIZE - size
        remainder = (size - PATCH_SIZE) % step
        return (step - remainder) % step if remainder != 0 else 0

    h_pad = pad_needed(h)
    w_pad = pad_needed(w)
    return h_pad // 2, h_pad - h_pad // 2, w_pad // 2, w_pad - w_pad // 2


def pad_array(
    arr: np.ndarray, top: int, bottom: int, left: int, right: int
) -> np.ndarray:
    """Pad with BORDER_REFLECT_101."""
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return arr
    return cv2.copyMakeBorder(arr, top, bottom, left, right, cv2.BORDER_REFLECT_101)


def create_patches(arr: np.ndarray) -> list[np.ndarray]:
    """Split array into overlapping patches."""
    step = int(PATCH_SIZE * (1 - OVERLAP))
    h, w = arr.shape[:2]
    patches = []
    for y in range(0, h - PATCH_SIZE + 1, step):
        for x in range(0, w - PATCH_SIZE + 1, step):
            patches.append(arr[y : y + PATCH_SIZE, x : x + PATCH_SIZE])
    return patches


# ---- Mask finding --------------------------------------------------------


def find_root_mask(image_path: Path, mask_dir: Path) -> Path | None:
    """Find root mask for an image using multiple naming patterns."""
    stem = image_path.stem
    patterns = [
        f"{stem}_root_mask.tif",
        f"{stem}_root_mask.tiff",
        f"{stem}_root_mask.png",
    ]
    for pattern in patterns:
        candidate = mask_dir / pattern
        if candidate.exists():
            return candidate
    matches = list(mask_dir.glob(f"{stem}*root_mask*"))
    return matches[0] if matches else None


# ---- Process one pair ----------------------------------------------------


def process_pair(
    image_path: Path, mask_path: Path
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Process one image+mask pair into patches."""
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        logger.warning("Cannot read image: %s", image_path.name)
        return None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.warning("Cannot read mask: %s", mask_path.name)
        return None

    if not np.any(mask > 0):
        return None

    if image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)
    # Preserve nonzero label pixels when casting the mask (see helper).
    mask = _to_uint8_mask(mask)

    coords = detect_petri_dish(image)
    if coords is not None:
        y1, y2, x1, x2 = coords
        image = image[y1:y2, x1:x2]
        mask = mask[y1:y2, x1:x2]

    h = min(image.shape[0], mask.shape[0])
    w = min(image.shape[1], mask.shape[1])
    image = image[:h, :w]
    mask = mask[:h, :w]

    top, bottom, left, right = calculate_padding(h, w)
    image = pad_array(image, top, bottom, left, right)
    mask = pad_array(mask, top, bottom, left, right)

    image_patches = create_patches(image)
    mask_patches = create_patches(mask)

    if len(image_patches) != len(mask_patches):
        logger.warning("Patch count mismatch for %s", image_path.name)
        return None

    return list(zip(image_patches, mask_patches))


# ---- Source pair discovery ----------------------------------------------


def collect_source_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    """Return (image_path, mask_path) pairs for every raw image with a mask.

    Args:
        raw_dir: Staging folder containing images/ and masks/ subdirs.

    Returns:
        Source pairs, sorted by image name for determinism. Images with no
        matching mask are logged and skipped.
    """
    img_dir = raw_dir / "images"
    mask_dir = raw_dir / "masks"

    if not img_dir.exists():
        logger.warning("No images/ directory in raw-dir.")
        return []

    images = sorted(
        f
        for f in img_dir.iterdir()
        if f.suffix.lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
    )
    logger.info("Found %d raw images in staging.", len(images))

    pairs: list[tuple[Path, Path]] = []
    for img_path in images:
        mask_path = find_root_mask(img_path, mask_dir)
        if mask_path is None:
            logger.warning("No mask found for %s - skipping.", img_path.name)
            continue
        pairs.append((img_path, mask_path))
    return pairs


# ---- Existing patch collection ------------------------------------------


def collect_existing(split_dir: Path, label: str) -> list[tuple[str, str]]:
    """Collect (image_path, mask_path) pairs from an existing split dir.

    Pairs by filename. Over a FUSE mount a per-file ``exists()`` is a
    network stat; for ~100k+ patches that dominates runtime. Instead,
    list ``images/`` and ``masks/`` once each and intersect their names
    in memory, so only two directory listings hit the network.

    Args:
        split_dir: Path to an existing split asset (has images/ and masks/).
        label: Split name for logging.

    Returns:
        List of (image_path, mask_path) string tuples, sorted by name.
    """
    pairs: list[tuple[str, str]] = []

    if split_dir is None or not split_dir.exists():
        logger.info("No existing %s directory - skipping.", label)
        return pairs

    img_dir = split_dir / "images"
    mask_dir = split_dir / "masks"

    if not img_dir.exists() or not mask_dir.exists():
        logger.info("No images/masks in existing %s - skipping.", label)
        return pairs

    img_names = {f.name for f in img_dir.iterdir()}
    mask_names = {f.name for f in mask_dir.iterdir()}
    common = sorted(img_names & mask_names)

    pairs = [(str(img_dir / name), str(mask_dir / name)) for name in common]

    logger.info("Collected %d existing %s pairs.", len(pairs), label)
    return pairs


# ---- Parallel copy -------------------------------------------------------


def copy_many(jobs: list[tuple[str, str]], workers: int, label: str) -> int:
    """Copy a list of (src, dst) file pairs concurrently.

    File copies are I/O-bound and release the GIL while waiting on the
    network, so a thread pool overlaps their waits for a large speedup
    over sequential copying on a FUSE-mounted datastore. Progress is
    logged periodically so a slow run is distinguishable from a stall.

    Args:
        jobs: List of (src_path, dst_path) string tuples.
        workers: Number of worker threads.
        label: Label for progress logging.

    Returns:
        The number of files copied.

    Raises:
        Exception: Propagates the first copy error encountered.
    """
    total = len(jobs)
    if total == 0:
        return 0

    done = 0
    log_every = max(1, total // 20)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(shutil.copy2, src, dst) for src, dst in jobs]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % log_every == 0 or done == total:
                logger.info("  %s: copied %d/%d files", label, done, total)
    return done


# ---- Split materialisation ----------------------------------------------


def copy_through(
    pairs: list[tuple[str, str]],
    out_img: Path,
    out_mask: Path,
    start_idx: int,
    workers: int,
    label: str,
) -> int:
    """Copy existing patch pairs into a split, renumbering from start_idx.

    Args:
        pairs: Existing (image, mask) path pairs to copy unchanged.
        out_img: Destination images/ directory.
        out_mask: Destination masks/ directory.
        start_idx: First patch index to use (existing patches occupy a
            contiguous range; new patches continue after it).
        workers: Copy worker threads.
        label: Split name for logging.

    Returns:
        The next free patch index after the copied pairs.
    """
    jobs: list[tuple[str, str]] = []
    idx = start_idx
    for img_src, mask_src in pairs:
        fname = f"patch_{idx:06d}.png"
        jobs.append((img_src, str(out_img / fname)))
        jobs.append((mask_src, str(out_mask / fname)))
        idx += 1
    copy_many(jobs, workers, label)
    return idx


def patch_new_images(
    image_pairs: list[tuple[Path, Path]],
    out_img: Path,
    out_mask: Path,
    start_idx: int,
    label: str,
) -> int:
    """Patch each new source image and write its patches into a split.

    All patches from one image are written into the same split (the image
    was already assigned to this split), so overlapping patches never
    straddle the train/val boundary. Patches are written directly rather
    than via the copy pool because the new-image count per run is small
    (feedback deltas) and the arrays are already in memory.

    Args:
        image_pairs: New (image, mask) source pairs assigned to this split.
        out_img: Destination images/ directory.
        out_mask: Destination masks/ directory.
        start_idx: First patch index to use (continues after existing).
        label: Split name for logging.

    Returns:
        The next free patch index after the written patches.
    """
    idx = start_idx
    skipped = 0
    for img_path, mask_path in image_pairs:
        result = process_pair(img_path, mask_path)
        if result is None:
            skipped += 1
            continue
        for img_patch, mask_patch in result:
            fname = f"patch_{idx:06d}.png"
            cv2.imwrite(str(out_img / fname), img_patch)
            cv2.imwrite(str(out_mask / fname), mask_patch)
            idx += 1
    logger.info(
        "  %s: added patches from %d new image(s) (%d skipped).",
        label,
        len(image_pairs) - skipped,
        skipped,
    )
    return idx


# ---- Main ----------------------------------------------------------------


def main() -> None:
    """Pass existing splits through; append new images at the image level."""
    args = parse_args()

    # ---- 1. Discover NEW source pairs and assign them at the image level ----
    new_pairs = collect_source_pairs(args.raw_dir)
    rng = random.Random(args.seed)
    rng.shuffle(new_pairs)
    n_train = int(len(new_pairs) * TRAIN_RATIO_IN_POOL)
    new_train_imgs = new_pairs[:n_train]
    new_val_imgs = new_pairs[n_train:]
    logger.info(
        "New images: %d (train: %d, val: %d). Test receives no new data.",
        len(new_pairs),
        len(new_train_imgs),
        len(new_val_imgs),
    )

    # ---- 2. Collect existing train and val patches (passed through as-is) ----
    existing_train = collect_existing(args.existing_train_dir, "train")
    existing_val = collect_existing(args.existing_val_dir, "val")

    # ---- 3. Prepare output directories ----
    for split in ("train", "val", "test"):
        (args.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (args.output_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    # ---- 4. Copy frozen test set through unchanged (parallel) ----
    test_img_dir = args.existing_test_dir / "images"
    test_mask_dir = args.existing_test_dir / "masks"

    out_test_img = args.output_dir / "test" / "images"
    out_test_mask = args.output_dir / "test" / "masks"

    if not (test_img_dir.exists() and test_mask_dir.exists()):
        raise RuntimeError(
            "Existing test directory missing images/ or masks/. "
            "Frozen test set requires a valid existing hades-test asset."
        )

    test_img_names = {f.name for f in test_img_dir.iterdir()}
    test_mask_names = {f.name for f in test_mask_dir.iterdir()}
    test_common = sorted(test_img_names & test_mask_names)

    test_jobs: list[tuple[str, str]] = []
    for idx, name in enumerate(test_common):
        fname = f"patch_{idx:06d}.png"
        test_jobs.append((str(test_img_dir / name), str(out_test_img / fname)))
        test_jobs.append((str(test_mask_dir / name), str(out_test_mask / fname)))

    test_count = len(test_common)
    copy_many(test_jobs, args.copy_workers, "test")
    logger.info("Copied %d frozen test patches through unchanged.", test_count)

    # ---- 5. TRAIN: copy existing through, then append new train images ----
    out_train_img = args.output_dir / "train" / "images"
    out_train_mask = args.output_dir / "train" / "masks"
    next_idx = copy_through(
        existing_train, out_train_img, out_train_mask, 0, args.copy_workers, "train"
    )
    patch_new_images(new_train_imgs, out_train_img, out_train_mask, next_idx, "train")

    # ---- 6. VAL: copy existing through, then append new val images ----
    out_val_img = args.output_dir / "val" / "images"
    out_val_mask = args.output_dir / "val" / "masks"
    next_idx = copy_through(
        existing_val, out_val_img, out_val_mask, 0, args.copy_workers, "val"
    )
    patch_new_images(new_val_imgs, out_val_img, out_val_mask, next_idx, "val")

    # ---- 7. Final counts ----
    logger.info("Incremental preprocessing complete.")
    for split_name in ("train", "val", "test"):
        count = len(list((args.output_dir / split_name / "images").iterdir()))
        logger.info("  %s: %d patches", split_name, count)


if __name__ == "__main__":
    main()
