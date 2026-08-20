"""Stage HADES data: crop petri dishes, create 256x256 patches, split train/val.

Split strategy:
  - train: Y2B_23 (train/) + Y2B_24 (all) + Y2B_25 (train_)
  - val:   Y2B_25 (val_)
  - test:  Y2B_23 (test/)

Usage:
    python stage_patches.py --src D:\\data --out D:\\data\\hades-patches --dry-run
    python stage_patches.py --src D:\\data --out D:\\data\\hades-patches
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PATCH_SIZE = 256
OVERLAP = 0.5
SHRINK = 20


# ── petri dish detection ─────────────────────────────────────────


def detect_petri_dish(
    image: np.ndarray, shrink: int = SHRINK
) -> tuple[int, int, int, int] | None:
    """Detect petri dish and return square crop coordinates.

    Returns (y1, y2, x1, x2) or None if detection fails.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((20, 20), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    x += shrink
    y += shrink
    w -= 2 * shrink
    h -= 2 * shrink

    size = max(w, h)
    cx, cy = x + w // 2, y + h // 2

    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(image.shape[1], x1 + size)
    y2 = min(image.shape[0], y1 + size)

    return y1, y2, x1, x2


# ── padding ──────────────────────────────────────────────────────


def calculate_padding(
    h: int, w: int, patch_size: int = PATCH_SIZE, overlap: float = OVERLAP
) -> tuple[int, int, int, int]:
    """Calculate mirror padding so (size - patch_size) % step == 0."""
    step = int(patch_size * (1 - overlap))

    def pad_needed(size: int) -> int:
        if size < patch_size:
            return patch_size - size
        remainder = (size - patch_size) % step
        return (step - remainder) % step if remainder != 0 else 0

    h_pad = pad_needed(h)
    w_pad = pad_needed(w)
    return h_pad // 2, h_pad - h_pad // 2, w_pad // 2, w_pad - w_pad // 2


def pad_array(
    arr: np.ndarray, top: int, bottom: int, left: int, right: int
) -> np.ndarray:
    """Pad with BORDER_REFLECT_101 (mirror without edge repeat)."""
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return arr
    return cv2.copyMakeBorder(arr, top, bottom, left, right, cv2.BORDER_REFLECT_101)


# ── manual patching ──────────────────────────────────────────────


def create_patches(
    arr: np.ndarray, patch_size: int = PATCH_SIZE, overlap: float = OVERLAP
) -> list[np.ndarray]:
    """Split array into overlapping patches. Returns list of patches."""
    step = int(patch_size * (1 - overlap))
    h, w = arr.shape[:2]
    patches = []
    for y in range(0, h - patch_size + 1, step):
        for x in range(0, w - patch_size + 1, step):
            patches.append(arr[y : y + patch_size, x : x + patch_size])
    return patches


# ── mask finding ─────────────────────────────────────────────────


def find_root_mask(image_path: Path, mask_dir: Path) -> Path | None:
    """Find root mask for an image using multiple naming patterns."""
    stem = image_path.stem
    patterns = [
        f"{stem}_root_mask.tif",
        f"{stem}_root_mask.tiff",
        f"{stem}-Fish Eye Corrected_root_mask.tif",
        f"{stem}-Fish Eye Corrected_root_mask.tiff",
    ]
    for pattern in patterns:
        candidate = mask_dir / pattern
        if candidate.exists():
            return candidate
    # Also try glob for any match
    matches = list(mask_dir.glob(f"{stem}*root_mask*"))
    return matches[0] if matches else None


# ── pair collection ──────────────────────────────────────────────


def collect_pairs(src: Path) -> dict[str, list[tuple[Path, Path, str]]]:
    """Collect all image+root_mask pairs, assigned to train or val.

    Split strategy:
      - train: Y2B_23 train/ + Y2B_24 (all) + Y2B_25 (train_)
      - val:   Y2B_25 (val_)
      - test:  Y2B_23 test/

    Returns: {'train': [...], 'val': [...], 'test': [...]} where each item is
             (image_path, mask_path, dataset_name).
    """
    splits: dict[str, list[tuple[Path, Path, str]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    # Y2B_23: train/ → train, test/ → test
    y23 = src / "Y2B_23"
    if y23.exists():
        mask_dir = y23 / "masks"
        for subdir, split in (("train", "train"), ("test", "test")):
            img_dir = y23 / "images" / subdir
            if not img_dir.exists():
                continue
            for img in sorted(img_dir.glob("*.png")):
                mask = find_root_mask(img, mask_dir)
                if mask:
                    splits[split].append((img, mask, "Y2B_23"))

    # Y2B_24: all → train (masks in annotator subdirs)
    y24 = src / "Y2B_24"
    if y24.exists():
        for img in sorted((y24 / "images").glob("*.png")):
            parts = img.stem.split("_")
            if len(parts) < 2:
                continue
            person_mask_dir = y24 / "masks" / parts[1]
            mask = (
                find_root_mask(img, person_mask_dir)
                if person_mask_dir.exists()
                else None
            )
            if mask:
                splits["train"].append((img, mask, "Y2B_24"))

    # Y2B_25: train_ → train, val_ → val (masks in annotator subdirs)
    y25 = src / "Y2B_25"
    if y25.exists():
        for img in sorted((y25 / "images").glob("*.png")):
            parts = img.stem.split("_")
            if len(parts) < 2:
                continue
            person_mask_dir = y25 / "masks" / parts[1]
            mask = (
                find_root_mask(img, person_mask_dir)
                if person_mask_dir.exists()
                else None
            )
            if mask:
                split = "val" if img.name.startswith("val_") else "train"
                splits[split].append((img, mask, "Y2B_25"))

    return splits


# ── single pair processing ───────────────────────────────────────


def process_pair(
    image_path: Path, mask_path: Path
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Process one image+mask pair through the full pipeline.

    Returns list of (image_patch, mask_patch) or None if skipped.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        logger.warning("Cannot read image: %s", image_path.name)
        return None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.warning("Cannot read mask: %s", mask_path.name)
        return None

    # Skip blank masks
    if not np.any(mask > 0):
        return None

    # Handle dtype
    if image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)
    if mask.dtype == np.uint16:
        mask = (mask / 256).astype(np.uint8)

    # Crop petri dish
    coords = detect_petri_dish(image)
    if coords is not None:
        y1, y2, x1, x2 = coords
        image = image[y1:y2, x1:x2]
        mask = mask[y1:y2, x1:x2]

    # Ensure same dimensions
    h = min(image.shape[0], mask.shape[0])
    w = min(image.shape[1], mask.shape[1])
    image = image[:h, :w]
    mask = mask[:h, :w]

    # Pad
    top, bottom, left, right = calculate_padding(h, w)
    image = pad_array(image, top, bottom, left, right)
    mask = pad_array(mask, top, bottom, left, right)

    # Patch
    image_patches = create_patches(image)
    mask_patches = create_patches(mask)

    if len(image_patches) != len(mask_patches):
        logger.warning("Patch count mismatch for %s", image_path.name)
        return None

    return list(zip(image_patches, mask_patches))


# ── main ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage HADES data as 256x256 patches for training.",
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Parent directory containing Y2B_23, Y2B_24, Y2B_25.",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output directory for patched dataset."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Count patches without writing files."
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN — no files will be written.")

    # Collect pairs
    splits = collect_pairs(args.src)
    logger.info(
        "Found %d train pairs, %d val pairs, %d test pairs.",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )

    # Process and save
    stats: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        pairs = splits[split_name]
        if not pairs:
            stats[split_name] = {"images": 0, "patches": 0, "skipped": 0}
            continue

        out_img = args.out / split_name / "images"
        out_mask = args.out / split_name / "masks"
        if not args.dry_run:
            out_img.mkdir(parents=True, exist_ok=True)
            out_mask.mkdir(parents=True, exist_ok=True)

        total_patches = 0
        processed = 0
        skipped = 0

        for img_path, mask_path, dataset in pairs:
            patch_pairs = process_pair(img_path, mask_path)

            if patch_pairs is None:
                skipped += 1
                continue

            processed += 1
            for idx, (img_patch, mask_patch) in enumerate(patch_pairs):
                fname = f"{img_path.stem}_patch_{idx:04d}.png"
                if not args.dry_run:
                    cv2.imwrite(str(out_img / fname), img_patch)
                    cv2.imwrite(str(out_mask / fname), mask_patch)
                total_patches += 1

            if processed % 50 == 0:
                logger.info(
                    "  %s: %d/%d images processed...", split_name, processed, len(pairs)
                )

        stats[split_name] = {
            "images": processed,
            "patches": total_patches,
            "skipped": skipped,
        }
        logger.info(
            "%s: %d images → %d patches (%d skipped).",
            split_name,
            processed,
            total_patches,
            skipped,
        )

    # Summary
    print("\n=== SUMMARY ===")
    for split_name, s in stats.items():
        print(
            f"  {split_name}: {s['images']} images "
            f"→ {s['patches']} patches ({s['skipped']} skipped)"
        )
    total = sum(s["patches"] for s in stats.values())
    print(f"  TOTAL: {total} patches")
    if args.dry_run:
        print("\n  (dry run — re-run without --dry-run to write files)")


if __name__ == "__main__":
    main()
