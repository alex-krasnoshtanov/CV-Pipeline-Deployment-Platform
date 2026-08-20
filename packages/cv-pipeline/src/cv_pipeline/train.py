"""Model training for the cv-pipeline package.

Trains a U-Net segmentation model on image-mask pairs and saves the best
checkpoint based on validation F1 score. The checkpoint format is
compatible with ``SegmentationModel._load_checkpoint`` so trained models
can be used for inference immediately.

Expected data directory structure::

    data_dir/
        images/
            plate_001.png
            plate_002.png
        masks/
            plate_001.png
            plate_002.png

Images and masks are matched by filename stem. Masks must be
single-channel with 0 for background and 255 (or any non-zero value) for
root pixels.

Design notes (see also the block-B reference recipe this restores):

* Metrics are computed at the DATASET level. We accumulate true/false
  positives and false negatives over the whole val/test set and compute a
  single F1/IoU at the end. We do NOT average a per-batch F1, because
  HADES patches are mostly pure background: a background-only patch has
  tp=fp=fn=0 (F1 is 0/0) and scoring it 0.0 collapses the mean even for a
  perfect model. Accumulation makes background patches contribute only
  true negatives, which never enter F1.
* Empty (background-only) patches are subsampled for TRAINING only, so the
  loss is not dominated by easy negatives. Validation/test keep every
  patch so the reported score reflects the real data distribution.
* Light augmentation (flip / small rotation / brightness) is applied to
  TRAINING patches only.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cv_pipeline._version import __version__

logger = logging.getLogger(__name__)

# Supported image extensions for dataset loading.
_IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ---- result dataclasses ---------------------------------------------


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""

    epoch: int
    train_loss: float
    val_f1: float
    val_iou: float

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for JSON serialisation."""
        return {
            "epoch": self.epoch,
            "train_loss": round(self.train_loss, 4),
            "val_f1": round(self.val_f1, 4),
            "val_iou": round(self.val_iou, 4),
        }


@dataclass
class TrainingResult:
    """Complete result of a training run."""

    run_name: str
    pipeline_version: str
    epochs: list[EpochMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_val_f1: float = 0.0
    training_completed: str = ""

    def to_dict(self) -> dict:
        """Convert to a plain dictionary matching run_metrics.json schema."""
        return {
            "run_name": self.run_name,
            "pipeline_version": self.pipeline_version,
            "epochs": [e.to_dict() for e in self.epochs],
            "best_epoch": self.best_epoch,
            "best_val_f1": round(self.best_val_f1, 4),
            "training_completed": self.training_completed,
        }


# ---- augmentation ----------------------------------------------------


def _augment_pair(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply light, label-preserving augmentation to one train patch.

    Operates on a float image in [0, 1] and a binary mask in {0, 1}.

    Why these three and not more: roots imaged top-down have a roughly
    fixed orientation (tips point down under gravitropism), so we use a
    horizontal flip and a small rotation rather than arbitrary flips that
    would create unrealistic upside-down roots. Brightness jitter covers
    lighting/condensation variation. The mask is rotated with nearest-
    neighbour so it stays binary; the image uses linear interpolation.

    Randomness uses the ``random`` module (not numpy): PyTorch seeds
    ``random`` per DataLoader worker automatically but does NOT seed numpy,
    so numpy-based augmentation would repeat identically across workers.

    Args:
        image: Grayscale patch as float32 in [0, 1], shape (H, W).
        mask: Binary patch as float32 in {0, 1}, shape (H, W).

    Returns:
        The augmented (image, mask) pair, same shapes and dtypes.
    """
    # Horizontal flip.
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])

    # Small rotation.
    angle = random.uniform(-15.0, 15.0)
    if abs(angle) > 0.1:
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(
            image,
            rot,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = cv2.warpAffine(
            mask,
            rot,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    # Brightness jitter (image only).
    if random.random() < 0.5:
        factor = random.uniform(0.85, 1.15)
        image = np.clip(image * factor, 0.0, 1.0).astype(np.float32)

    return image, mask


# ---- empty-patch balancing ------------------------------------------


def _balance_empty_patches(
    pairs: list[tuple[Path, Path]],
    empty_patch_ratio: float,
    seed: int,
) -> list[tuple[Path, Path]]:
    """Keep all root patches and subsample background-only patches.

    ``empty_patch_ratio`` is how many empty patches to keep per root patch:
    0.75 keeps 0.75x as many empties as patches that contain root. We keep
    some empties (not zero) so the model still learns what "not a root"
    looks like and does not over-predict.

    This reads each mask once. On very large mounted datasets that scan can
    be slow; if it ever becomes a bottleneck, emit a manifest of non-empty
    patch names during preprocessing and read that instead. We deliberately
    do NOT cache here: a stale cache silently trains on the wrong split, a
    failure mode that is hard to notice.

    Args:
        pairs: All matched (image, mask) pairs in a split.
        empty_patch_ratio: Empties to keep per root patch (>= 0).
        seed: Seed for the deterministic empty-patch sample.

    Returns:
        The kept pairs (all root pairs followed by the sampled empties).
    """
    root_pairs: list[tuple[Path, Path]] = []
    empty_pairs: list[tuple[Path, Path]] = []
    for img_path, mask_path in pairs:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None and np.count_nonzero(mask) > 0:
            root_pairs.append((img_path, mask_path))
        else:
            empty_pairs.append((img_path, mask_path))

    n_keep = min(len(empty_pairs), int(len(root_pairs) * empty_patch_ratio))
    random.Random(seed).shuffle(empty_pairs)
    kept = root_pairs + empty_pairs[:n_keep]

    logger.info(
        "Empty-patch balancing: %d root + %d empty kept (of %d empty available), "
        "ratio=%.2f.",
        len(root_pairs),
        n_keep,
        len(empty_pairs),
        empty_patch_ratio,
    )
    return kept


# ---- dataset ---------------------------------------------------------


class SegmentationDataset(Dataset):
    """Dataset for loading image-mask pairs from a directory.

    Expects the directory to contain ``images/`` and ``masks/``
    subdirectories with files matched by filename stem.

    Args:
        data_dir: Root directory containing ``images/`` and ``masks/``.
        patch_size: Images and masks are resized to this square size.
        in_channels: 1 for grayscale, 3 for RGB.
        empty_patch_ratio: If set, keep all root patches and only this many
            empty patches per root patch (e.g. 0.75). Use for TRAIN. Leave
            ``None`` for val/test so the real distribution is preserved.
        augment: If ``True``, apply light augmentation in ``__getitem__``.
            Use for TRAIN only.
        seed: Seed for the deterministic empty-patch sample.

    Raises:
        FileNotFoundError: If ``images/`` or ``masks/`` subdirectory is
            missing, or if no matched pairs are found.
    """

    def __init__(
        self,
        data_dir: Path,
        patch_size: int = 256,
        in_channels: int = 1,
        empty_patch_ratio: float | None = None,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.augment = augment

        images_dir = data_dir / "images"
        masks_dir = data_dir / "masks"

        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"Expected 'images/' subdirectory in '{data_dir}', not found."
            )
        if not masks_dir.is_dir():
            raise FileNotFoundError(
                f"Expected 'masks/' subdirectory in '{data_dir}', not found."
            )

        # Collect image paths and match to masks by stem.
        image_paths = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
        )

        # List mask filenames once upfront to avoid per-file .exists()
        # calls, which fail on network-mounted filesystems (Azure ML
        # FUSE mounts drop connections under rapid stat() calls).
        existing_masks = set(f.name for f in masks_dir.iterdir())

        self.pairs: list[tuple[Path, Path]] = []
        for img_path in image_paths:
            # Try to find a mask with the same stem, any supported extension.
            mask_candidates = [
                masks_dir / f"{img_path.stem}{ext}" for ext in _IMAGE_EXTENSIONS
            ]
            mask_path = next(
                (m for m in mask_candidates if m.name in existing_masks),
                None,
            )
            if mask_path is not None:
                self.pairs.append((img_path, mask_path))

        if not self.pairs:
            raise FileNotFoundError(
                f"No matched image-mask pairs found in '{data_dir}'. "
                f"Ensure 'images/' and 'masks/' contain files with matching stems."
            )

        # Train-only: drop most background patches so the loss is not
        # dominated by easy negatives. Runs after pairing, before training.
        if empty_patch_ratio is not None:
            self.pairs = _balance_empty_patches(self.pairs, empty_patch_ratio, seed)

        logger.info(
            "Dataset loaded: %d pairs from '%s' (augment=%s).",
            len(self.pairs),
            data_dir,
            self.augment,
        )

    def __len__(self) -> int:
        """Return the number of image-mask pairs."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and preprocess an image-mask pair.

        Args:
            idx: Index of the pair to load.

        Returns:
            A tuple of (image_tensor, mask_tensor). Image is float32 in
            [0, 1] with shape (C, H, W). Mask is float32 in {0, 1} with
            shape (1, H, W).
        """
        img_path, mask_path = self.pairs[idx]

        # Load image.
        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Failed to read image: '{img_path}'.")

        # Load mask as grayscale.
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: '{mask_path}'.")

        # Convert image channels.
        if self.in_channels == 1:
            if len(image.shape) == 3:
                if image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif self.in_channels == 3:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # Resize to patch_size.
        image = cv2.resize(
            image, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR
        )
        mask = cv2.resize(
            mask, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST
        )

        # Normalise image to [0, 1].
        image = image.astype(np.float32) / 255.0
        # Binarise mask to {0, 1}.
        mask = (mask > 0).astype(np.float32)

        # Augment (train only). Done on the single-channel float image and
        # the binary mask, before channel dims are added.
        if self.augment and self.in_channels == 1:
            image, mask = _augment_pair(image, mask)

        # Add channel dims.
        if self.in_channels == 1:
            image = np.expand_dims(image, axis=0)  # (1, H, W)
        else:
            image = np.transpose(image, (2, 0, 1))  # (3, H, W)
        mask = np.expand_dims(mask, axis=0)  # (1, H, W)

        return torch.from_numpy(image), torch.from_numpy(mask)


# ---- loss function ---------------------------------------------------


class DiceLoss(nn.Module):
    """Differentiable Dice loss for binary segmentation.

    Operates on raw logits — applies sigmoid internally. A smoothing term
    prevents division by zero when both prediction and target are empty,
    and the sigmoid output is clamped away from 0/1 for numerical safety.

    Args:
        smooth: Smoothing constant added to numerator and denominator.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the Dice loss.

        Args:
            logits: Raw model output (N, 1, H, W).
            targets: Binary ground truth (N, 1, H, W) in {0, 1}.

        Returns:
            Scalar Dice loss value.
        """
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """BCE + Dice loss for binary segmentation.

    Combines BCEWithLogitsLoss (pixel-level cross-entropy) with DiceLoss
    (region-level overlap). Both operate on raw logits.

    Args:
        bce_weight: Weight for the BCE component.
        dice_weight: Weight for the Dice component.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the combined loss.

        Args:
            logits: Raw model output (N, 1, H, W).
            targets: Binary ground truth (N, 1, H, W) in {0, 1}.

        Returns:
            Weighted sum of BCE and Dice loss.
        """
        return self.bce_weight * self.bce(
            logits, targets
        ) + self.dice_weight * self.dice(logits, targets)


# ---- metrics ---------------------------------------------------------


def _metric_counts(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Return (tp, fp, fn) pixel counts for one batch.

    We return raw counts, not a per-batch F1, so the caller can accumulate
    over the whole val/test set and compute a single dataset-level F1. This
    is the fix for the "strange metrics" plateau: a background-only patch
    has tp=fp=fn=0, and scoring that as F1=0.0 (and averaging per batch)
    collapses the mean even when the model is perfect — badly so, because
    with shuffle=False the empty patches cluster into whole batches.
    Accumulating counts makes empty patches contribute only true negatives,
    which never appear in F1.

    Args:
        predictions: Raw logits (N, 1, H, W).
        targets: Binary ground truth (N, 1, H, W) in {0, 1}.
        threshold: Probability threshold for binarisation.

    Returns:
        A tuple of (tp, fp, fn) as floats.
    """
    with torch.no_grad():
        probs = torch.sigmoid(predictions)
        binary = (probs >= threshold).float()

        tp = (binary * targets).sum().item()
        fp = (binary * (1.0 - targets)).sum().item()
        fn = ((1.0 - binary) * targets).sum().item()

    return tp, fp, fn


def _f1_iou_from_counts(tp: float, fp: float, fn: float) -> tuple[float, float]:
    """Compute F1 and IoU from accumulated pixel counts.

    Args:
        tp: True-positive pixel count.
        fp: False-positive pixel count.
        fn: False-negative pixel count.

    Returns:
        A tuple of (f1, iou). Both in [0, 1]. Returns 0.0 only when there
        are no positives and no predicted positives anywhere in the set.
    """
    f1_denom = 2.0 * tp + fp + fn
    iou_denom = tp + fp + fn
    f1 = (2.0 * tp) / f1_denom if f1_denom > 0 else 0.0
    iou = tp / iou_denom if iou_denom > 0 else 0.0
    return f1, iou


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Run the model over a loader and return dataset-level (F1, IoU).

    Accumulates tp/fp/fn across all batches, then scores once. Shared by
    the per-epoch validation loop here and the held-out test evaluation in
    the Azure ML training script, so both report the same kind of number.

    Args:
        model: The model to evaluate (set to eval mode internally).
        loader: DataLoader yielding (images, masks).
        device: Torch device string.
        threshold: Probability threshold for binarisation.

    Returns:
        A tuple of (f1, iou) at the dataset level.
    """
    model.eval()
    tp_sum = 0.0
    fp_sum = 0.0
    fn_sum = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            tp, fp, fn = _metric_counts(logits, masks, threshold)
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
    return _f1_iou_from_counts(tp_sum, fp_sum, fn_sum)


# ---- main training function ------------------------------------------


def train(
    data_dir: Path,
    val_dir: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = "cuda",
    run_name: str | None = None,
    in_channels: int = 1,
    patch_size: int = 256,
    num_workers: int = 4,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 15,
    train_empty_patch_ratio: float | None = 0.75,
    augment: bool = True,
    on_epoch_end: Callable[[int, float, float, float], None] | None = None,
) -> TrainingResult:
    """Train a U-Net segmentation model on image-mask pairs.

    Builds the same architecture used by ``SegmentationModel`` and saves a
    checkpoint compatible with ``SegmentationModel._load_checkpoint``.

    Args:
        data_dir: Directory containing ``images/`` and ``masks/`` for
            training.
        val_dir: Directory containing ``images/`` and ``masks/`` for
            validation.
        output_dir: Where to write ``best_model.pth``, ``run_metrics.json``.
        epochs: Maximum number of training epochs (early stopping may end
            sooner).
        batch_size: Training and validation batch size.
        lr: Initial learning rate for the AdamW optimiser.
        device: Torch device string (``'cuda'``, ``'cpu'``, ``'cuda:0'``).
        run_name: Identifier for this run. Defaults to a timestamp.
        in_channels: 1 for grayscale, 3 for RGB.
        patch_size: Images are resized to this square size.
        num_workers: Number of parallel data-loading workers.
        weight_decay: AdamW weight decay (L2 regularisation).
        early_stopping_patience: Stop if val F1 does not improve for this
            many epochs. Set very high to effectively disable.
        train_empty_patch_ratio: Empty patches kept per root patch in the
            TRAINING set. ``None`` keeps all patches. Validation always
            keeps all patches.
        augment: Apply light augmentation to TRAINING patches.
        on_epoch_end: Optional callback invoked after each epoch with
            ``(epoch, train_loss, val_f1, val_iou)`` for live logging.

    Returns:
        A ``TrainingResult`` with per-epoch metrics and best checkpoint
        info.

    Raises:
        FileNotFoundError: If data directories are missing or empty.
        RuntimeError: On CUDA OOM or other fatal training errors.
    """
    import segmentation_models_pytorch as smp

    if run_name is None:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # Resolve device.
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU.")
        device = "cpu"

    logger.info(
        "Starting training run '%s' — epochs=%d, batch_size=%d, lr=%s, "
        "weight_decay=%s, device=%s.",
        run_name,
        epochs,
        batch_size,
        lr,
        weight_decay,
        device,
    )

    # ---- data --------------------------------------------------------
    data_dir = Path(data_dir)
    val_dir = Path(val_dir)
    output_dir = Path(output_dir)

    # Training set: balanced + augmented. Validation: full distribution,
    # no augmentation, so the reported F1 is honest.
    train_dataset = SegmentationDataset(
        data_dir,
        patch_size=patch_size,
        in_channels=in_channels,
        empty_patch_ratio=train_empty_patch_ratio,
        augment=augment,
    )
    val_dataset = SegmentationDataset(
        val_dir,
        patch_size=patch_size,
        in_channels=in_channels,
        empty_patch_ratio=None,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    logger.info(
        "Data loaded — %d training pairs, %d validation pairs.",
        len(train_dataset),
        len(val_dataset),
    )

    # ---- model -------------------------------------------------------
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=1,
        activation=None,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model created — %s parameters.", f"{total_params:,}")

    # ---- optimiser, loss, schedule ------------------------------------
    # AdamW (decoupled weight decay) generalises better than plain Adam for
    # this task. Cosine annealing decays the LR smoothly over the planned
    # horizon; with early stopping it simply will not reach the floor.
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = CombinedLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(epochs, 1), eta_min=1e-6
    )

    # ---- training loop -----------------------------------------------
    result = TrainingResult(
        run_name=run_name,
        pipeline_version=__version__,
    )

    best_val_f1 = 0.0
    best_state_dict = None
    epochs_without_improvement = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # -- train --
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimiser.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            # Gradient clipping guards against rare exploding-gradient steps.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # -- validate (dataset-level F1/IoU) --
        avg_val_f1, avg_val_iou = _evaluate(model, val_loader, device)

        # Step the LR schedule once per epoch.
        scheduler.step()

        elapsed = time.time() - epoch_start

        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_f1=avg_val_f1,
            val_iou=avg_val_iou,
        )
        result.epochs.append(epoch_metrics)

        if on_epoch_end is not None:
            on_epoch_end(epoch, avg_train_loss, avg_val_f1, avg_val_iou)

        logger.info(
            "Epoch %d/%d — loss=%.4f, val_f1=%.4f, val_iou=%.4f, lr=%.2e (%.1fs)",
            epoch,
            epochs,
            avg_train_loss,
            avg_val_f1,
            avg_val_iou,
            optimiser.param_groups[0]["lr"],
            elapsed,
        )

        # -- track best + early stopping --
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
            logger.info("New best model at epoch %d — val_f1=%.4f.", epoch, avg_val_f1)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d — no val_f1 improvement for "
                    "%d epochs (best=%.4f).",
                    epoch,
                    early_stopping_patience,
                    best_val_f1,
                )
                break

    # ---- save outputs ------------------------------------------------
    result.best_epoch = (
        max((e for e in result.epochs), key=lambda e: e.val_f1).epoch
        if result.epochs
        else 0
    )
    result.best_val_f1 = best_val_f1
    result.training_completed = datetime.now(timezone.utc).isoformat()

    # Save checkpoint in the format SegmentationModel expects.
    checkpoint_path = output_dir / "best_model.pth"
    if best_state_dict is not None:
        checkpoint = {
            "model_state_dict": best_state_dict,
            "model_version": f"unet-local-{run_name}",
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info("Best checkpoint saved to '%s'.", checkpoint_path)
    else:
        logger.warning("No checkpoint saved — training produced no improvement.")

    # Save run metrics.
    metrics_path = output_dir / "run_metrics.json"
    metrics_path.write_text(
        json.dumps(result.to_dict(), indent=2),
        encoding="utf-8",
    )
    logger.info("Run metrics saved to '%s'.", metrics_path)

    logger.info(
        "Training complete — best val_f1=%.4f at epoch %d.",
        result.best_val_f1,
        result.best_epoch,
    )
    return result
