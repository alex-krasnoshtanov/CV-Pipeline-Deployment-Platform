"""Azure ML preprocessing script for HADES image data.

This script runs on the Azure ML cluster. It:
  1. Lists the raw image/mask source pairs from staging.
  2. Splits at the SOURCE-IMAGE level (70/20/10) before any patching, so
     every patch produced from one image stays entirely within one split.
  3. Patches each image (crop petri dish -> overlapping patches) into its
     assigned split's output directory.

Why split before patching: patches are generated with overlap, so two
patches from the same source image share pixels. Pooling all patches and
splitting at the patch level lets those near-duplicate tiles land on
opposite sides of the train/val/test boundary, leaking information into
the evaluation sets and inflating metrics. Assigning whole images to a
split first makes that impossible -- this matches the original research
methodology. The split key is the source image; all of an image's patches
move together.

Azure ML mounts data assets and blob paths as local folders.
"""

from __future__ import annotations

import argparse
import logging
import random
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
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess HADES data.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Staging folder with images/ and masks/ subdirs.",
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
        help="Random seed for reproducible image-level splits.",
    )
    return parser.parse_args()


# ---- mask dtype handling -------------------------------------------------


def _to_uint8_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a mask to uint8 without destroying its nonzero labels.

    Masks may arrive 8- or 16-bit, encoded as either {0, 1} or {0, 255}.
    The previous code divided every 16-bit mask by 256, which rounds a
    {0, 1} 16-bit mask down to all zeros and silently deletes the label.
    Because patches are binarised at training time (mask > 0), we only
    need the nonzero pixels to survive the cast: scale by 256 only when
    the values genuinely exceed the 8-bit range.

    Args:
        mask: Mask array of any integer dtype.

    Returns:
        A uint8 mask with its nonzero structure preserved.
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


def split_images(
    pairs: list[tuple[Path, Path]], seed: int
) -> dict[str, list[tuple[Path, Path]]]:
    """Split source-image pairs into train/val/test at the image level.

    The split is performed on whole images BEFORE patching, so every patch
    derived from one image lands in exactly one split. This is the change
    that prevents overlapping patches from leaking across the split
    boundary.

    Args:
        pairs: Source (image, mask) pairs from collect_source_pairs.
        seed: Random seed controlling the shuffle.

    Returns:
        Mapping of split name to its assigned source pairs.
    """
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


# ---- Main ----------------------------------------------------------------


def main() -> None:
    """Split source images, then patch each split independently."""
    args = parse_args()

    # ---- 1. Discover source pairs and split them at the image level ----
    pairs = collect_source_pairs(args.raw_dir)
    if not pairs:
        raise RuntimeError("No image/mask source pairs found in raw-dir.")

    splits = split_images(pairs, args.seed)
    logger.info(
        "Image-level split of %d source images - train: %d, val: %d, test: %d.",
        len(pairs),
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )

    # ---- 2. Prepare output directories ----
    for split in ("train", "val", "test"):
        (args.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (args.output_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    # ---- 3. Patch each split's images into that split only ----
    for split_name, split_pairs in splits.items():
        out_img = args.output_dir / split_name / "images"
        out_mask = args.output_dir / split_name / "masks"

        patch_idx = 0
        skipped = 0
        for img_path, mask_path in split_pairs:
            result = process_pair(img_path, mask_path)
            if result is None:
                skipped += 1
                continue
            for img_patch, mask_patch in result:
                fname = f"patch_{patch_idx:06d}.png"
                cv2.imwrite(str(out_img / fname), img_patch)
                cv2.imwrite(str(out_mask / fname), mask_patch)
                patch_idx += 1

        logger.info(
            "  %s: %d patches from %d image(s) (%d skipped).",
            split_name,
            patch_idx,
            len(split_pairs),
            skipped,
        )

    logger.info("Preprocessing complete.")
    for split_name in ("train", "val", "test"):
        count = len(list((args.output_dir / split_name / "images").iterdir()))
        logger.info("  %s: %d patches", split_name, count)


if __name__ == "__main__":
    main()
