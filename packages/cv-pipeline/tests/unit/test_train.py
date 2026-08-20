"""Unit tests for cv_pipeline.train."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
from cv_pipeline.train import (
    CombinedLoss,
    DiceLoss,
    EpochMetrics,
    SegmentationDataset,
    TrainingResult,
    _f1_iou_from_counts,
    _metric_counts,
    train,
)

# ---- helpers ---------------------------------------------------------


def _create_tiny_dataset(
    base: Path,
    name: str = "train",
    n: int = 2,
    size: int = 32,
) -> Path:
    """Create a directory with tiny image-mask pairs for testing.

    Args:
        base: Parent directory.
        name: Subdirectory name (e.g. 'train', 'val').
        n: Number of image-mask pairs.
        size: Square image size in pixels.

    Returns:
        Path to the created data directory.
    """
    data_dir = base / name
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    for i in range(n):
        img = np.random.randint(0, 256, (size, size), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 255

        cv2.imwrite(str(images_dir / f"img_{i}.png"), img)
        cv2.imwrite(str(masks_dir / f"img_{i}.png"), mask)

    return data_dir


class _TinyUnet(nn.Module):
    """Minimal model that mimics U-Net input/output shapes for testing.

    Accepts (B, C, H, W) and returns (B, 1, H, W) logits.
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning single-channel logits."""
        return self.conv(x)


# ---- dataclass tests -------------------------------------------------


@pytest.mark.unit
class TestEpochMetrics:
    """Tests for EpochMetrics dataclass."""

    def test_to_dict_rounds_values(self) -> None:
        """to_dict should round floats to 4 decimal places."""
        m = EpochMetrics(
            epoch=1,
            train_loss=0.123456,
            val_f1=0.654321,
            val_iou=0.432198,
        )
        d = m.to_dict()

        assert d["epoch"] == 1
        assert d["train_loss"] == 0.1235
        assert d["val_f1"] == 0.6543
        assert d["val_iou"] == 0.4322


@pytest.mark.unit
class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_to_dict_includes_all_fields(self) -> None:
        """to_dict should contain all expected top-level keys."""
        r = TrainingResult(
            run_name="test",
            pipeline_version="0.1.0",
            epochs=[EpochMetrics(1, 0.5, 0.6, 0.4)],
            best_epoch=1,
            best_val_f1=0.6,
            training_completed="2026-05-01T12:00:00Z",
        )
        d = r.to_dict()

        assert d["run_name"] == "test"
        assert d["pipeline_version"] == "0.1.0"
        assert len(d["epochs"]) == 1
        assert d["best_epoch"] == 1
        assert d["best_val_f1"] == 0.6

    def test_empty_epochs_is_valid(self) -> None:
        """A result with no epochs should serialise cleanly."""
        r = TrainingResult(run_name="empty", pipeline_version="0.1.0")
        d = r.to_dict()

        assert d["epochs"] == []
        assert d["best_epoch"] == 0


# ---- loss functions --------------------------------------------------


@pytest.mark.unit
class TestDiceLoss:
    """Tests for the DiceLoss module."""

    def test_perfect_prediction_returns_near_zero(self) -> None:
        """Identical logits and targets should produce near-zero loss."""
        targets = torch.ones(1, 1, 4, 4)
        logits = torch.full((1, 1, 4, 4), 5.0)

        loss = DiceLoss()(logits, targets)

        assert loss.item() < 0.1

    def test_all_wrong_returns_high_loss(self) -> None:
        """Completely wrong predictions should produce high loss."""
        targets = torch.ones(1, 1, 4, 4)
        logits = torch.full((1, 1, 4, 4), -5.0)

        loss = DiceLoss()(logits, targets)

        assert loss.item() > 0.8

    def test_output_is_differentiable(self) -> None:
        """DiceLoss output should support backward pass."""
        logits = torch.randn(1, 1, 4, 4, requires_grad=True)
        targets = torch.ones(1, 1, 4, 4)

        loss = DiceLoss()(logits, targets)
        loss.backward()

        assert logits.grad is not None


@pytest.mark.unit
class TestCombinedLoss:
    """Tests for the CombinedLoss module."""

    def test_returns_scalar(self) -> None:
        """CombinedLoss should return a scalar tensor."""
        logits = torch.randn(1, 1, 4, 4)
        targets = torch.ones(1, 1, 4, 4)

        loss = CombinedLoss()(logits, targets)

        assert loss.dim() == 0

    def test_output_is_differentiable(self) -> None:
        """CombinedLoss should support backward pass."""
        logits = torch.randn(1, 1, 4, 4, requires_grad=True)
        targets = torch.ones(1, 1, 4, 4)

        loss = CombinedLoss()(logits, targets)
        loss.backward()

        assert logits.grad is not None


# ---- metric computation ----------------------------------------------


@pytest.mark.unit
class TestMetrics:
    """Tests for the dataset-level metric helpers.

    These pin the fix for the per-batch-averaging bug: a correctly
    predicted background-only patch must NOT drag the score to zero once
    counts are accumulated across the set.
    """

    def test_counts_perfect_prediction(self) -> None:
        """A perfect root prediction yields all tp, no fp/fn."""
        logits = torch.full((1, 1, 4, 4), 5.0)  # sigmoid -> ~1
        targets = torch.ones(1, 1, 4, 4)

        tp, fp, fn = _metric_counts(logits, targets)

        assert tp == 16.0
        assert fp == 0.0
        assert fn == 0.0

    def test_counts_all_wrong(self) -> None:
        """Predicting empty against a full target yields all fn."""
        logits = torch.full((1, 1, 4, 4), -5.0)  # sigmoid -> ~0
        targets = torch.ones(1, 1, 4, 4)

        tp, fp, fn = _metric_counts(logits, targets)

        assert tp == 0.0
        assert fp == 0.0
        assert fn == 16.0

    def test_f1_iou_from_counts_perfect(self) -> None:
        """Perfect counts give F1 and IoU of 1.0."""
        f1, iou = _f1_iou_from_counts(tp=100.0, fp=0.0, fn=0.0)

        assert f1 == pytest.approx(1.0)
        assert iou == pytest.approx(1.0)

    def test_f1_iou_from_counts_empty_set(self) -> None:
        """No positives anywhere returns 0.0 by convention."""
        f1, iou = _f1_iou_from_counts(tp=0.0, fp=0.0, fn=0.0)

        assert f1 == 0.0
        assert iou == 0.0

    def test_empty_patch_does_not_sink_score(self) -> None:
        """A correct root patch + a correct empty patch must give F1 = 1.0.

        The old per-batch metric scored the empty patch 0.0 and averaged,
        giving ~0.5 for a perfect model. Accumulating counts must give 1.0.
        """
        tp = fp = fn = 0.0
        batches = [
            (torch.full((1, 1, 4, 4), 5.0), torch.ones(1, 1, 4, 4)),  # root
            (torch.full((1, 1, 4, 4), -5.0), torch.zeros(1, 1, 4, 4)),  # empty
        ]
        for logits, targets in batches:
            a, b, c = _metric_counts(logits, targets)
            tp += a
            fp += b
            fn += c

        f1, iou = _f1_iou_from_counts(tp, fp, fn)

        assert f1 == pytest.approx(1.0)
        assert iou == pytest.approx(1.0)


# ---- dataset ---------------------------------------------------------


@pytest.mark.unit
class TestSegmentationDataset:
    """Tests for the SegmentationDataset."""

    def test_loads_matching_pairs(self, tmp_path: Path) -> None:
        """Dataset should discover image-mask pairs matched by stem."""
        data_dir = _create_tiny_dataset(tmp_path, "data", n=3, size=32)
        ds = SegmentationDataset(data_dir, patch_size=32, in_channels=1)

        assert len(ds) == 3

    def test_getitem_returns_tensors(self, tmp_path: Path) -> None:
        """Each item should be a tuple of (image_tensor, mask_tensor)."""
        data_dir = _create_tiny_dataset(tmp_path, "data", n=1, size=32)
        ds = SegmentationDataset(data_dir, patch_size=32, in_channels=1)

        image, mask = ds[0]

        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert image.shape == (1, 32, 32)
        assert mask.shape == (1, 32, 32)

    def test_augment_preserves_shapes(self, tmp_path: Path) -> None:
        """Augmentation must not change tensor shapes or mask binarity."""
        data_dir = _create_tiny_dataset(tmp_path, "data", n=1, size=32)
        ds = SegmentationDataset(data_dir, patch_size=32, in_channels=1, augment=True)

        image, mask = ds[0]

        assert image.shape == (1, 32, 32)
        assert mask.shape == (1, 32, 32)
        # Mask stays binary after nearest-neighbour rotation.
        unique = set(torch.unique(mask).tolist())
        assert unique.issubset({0.0, 1.0})

    def test_empty_patch_ratio_filters_background(self, tmp_path: Path) -> None:
        """With ratio=0, only root patches survive; empty patches dropped."""
        data_dir = tmp_path / "mixed"
        images_dir = data_dir / "images"
        masks_dir = data_dir / "masks"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        # Two root patches, three empty patches.
        for i in range(2):
            cv2.imwrite(
                str(images_dir / f"root_{i}.png"),
                np.random.randint(0, 256, (32, 32), dtype=np.uint8),
            )
            m = np.zeros((32, 32), dtype=np.uint8)
            m[8:24, 8:24] = 255
            cv2.imwrite(str(masks_dir / f"root_{i}.png"), m)
        for i in range(3):
            cv2.imwrite(
                str(images_dir / f"empty_{i}.png"),
                np.random.randint(0, 256, (32, 32), dtype=np.uint8),
            )
            cv2.imwrite(
                str(masks_dir / f"empty_{i}.png"),
                np.zeros((32, 32), dtype=np.uint8),
            )

        ds = SegmentationDataset(
            data_dir, patch_size=32, in_channels=1, empty_patch_ratio=0.0
        )

        assert len(ds) == 2

    def test_missing_images_dir_raises(self, tmp_path: Path) -> None:
        """A data directory without images/ should raise FileNotFoundError."""
        data_dir = tmp_path / "empty"
        data_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="images"):
            SegmentationDataset(data_dir)

    def test_no_matching_pairs_raises(self, tmp_path: Path) -> None:
        """No overlapping stems between images/ and masks/ should raise."""
        data_dir = tmp_path / "mismatch"
        images_dir = data_dir / "images"
        masks_dir = data_dir / "masks"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        cv2.imwrite(str(images_dir / "a.png"), np.zeros((32, 32), dtype=np.uint8))
        cv2.imwrite(str(masks_dir / "b.png"), np.zeros((32, 32), dtype=np.uint8))

        with pytest.raises(FileNotFoundError, match="No matched"):
            SegmentationDataset(data_dir)


# ---- train() happy path with mocked model ----------------------------


@pytest.mark.unit
class TestTrainHappyPath:
    """Tests for successful training runs."""

    @patch("torch.cuda.is_available", return_value=False)
    @patch("segmentation_models_pytorch.Unet")
    @patch("torch.save")
    def test_one_epoch_with_real_data(
        self,
        mock_save: MagicMock,
        mock_unet: MagicMock,
        mock_cuda: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Running 1 epoch with tiny images should exercise the full loop."""
        train_dir = _create_tiny_dataset(tmp_path, "train", n=2, size=32)
        val_dir = _create_tiny_dataset(tmp_path, "val", n=2, size=32)
        out_dir = tmp_path / "out"

        mock_unet.return_value = _TinyUnet(in_channels=1)

        result = train(
            data_dir=train_dir,
            val_dir=val_dir,
            output_dir=out_dir,
            epochs=1,
            batch_size=2,
            lr=1e-3,
            device="cpu",
            run_name="test-loop",
            patch_size=32,
        )

        assert len(result.epochs) == 1
        assert result.run_name == "test-loop"
        assert result.best_val_f1 >= 0.0
        assert (out_dir / "run_metrics.json").exists()

    @patch("torch.cuda.is_available", return_value=False)
    @patch("cv_pipeline.train.SegmentationDataset")
    @patch("cv_pipeline.train.DataLoader")
    @patch("segmentation_models_pytorch.Unet")
    @patch("torch.save")
    def test_returns_result_with_run_name(
        self,
        mock_save: MagicMock,
        mock_unet: MagicMock,
        mock_loader: MagicMock,
        mock_dataset: MagicMock,
        mock_cuda: MagicMock,
        tmp_path: Path,
    ) -> None:
        """train() should return a result object with a non-null run_name."""
        data_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        out_dir = tmp_path / "out"
        data_dir.mkdir()
        val_dir.mkdir()

        mock_model = MagicMock()
        param = torch.nn.Parameter(torch.randn(1, 1))
        mock_model.parameters.return_value = [param]
        mock_unet.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_loader.return_value = []

        result = train(
            data_dir=data_dir,
            val_dir=val_dir,
            output_dir=out_dir,
            epochs=1,
            batch_size=1,
            device="cpu",
        )

        assert result.run_name is not None

    @patch("torch.cuda.is_available", return_value=False)
    @patch("cv_pipeline.train.SegmentationDataset")
    @patch("cv_pipeline.train.DataLoader")
    @patch("segmentation_models_pytorch.Unet")
    @patch("torch.save")
    def test_writes_metrics_file(
        self,
        mock_save: MagicMock,
        mock_unet: MagicMock,
        mock_loader: MagicMock,
        mock_dataset: MagicMock,
        mock_cuda: MagicMock,
        tmp_path: Path,
    ) -> None:
        """train() should write run_metrics.json to the output directory."""
        data_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        out_dir = tmp_path / "out"
        data_dir.mkdir()
        val_dir.mkdir()

        mock_model = MagicMock()
        param = torch.nn.Parameter(torch.randn(1, 1))
        mock_model.parameters.return_value = [param]
        mock_unet.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_loader.return_value = []

        train(
            data_dir=data_dir,
            val_dir=val_dir,
            output_dir=out_dir,
            epochs=1,
            batch_size=1,
            device="cpu",
        )

        assert (out_dir / "run_metrics.json").exists()

    @patch("torch.cuda.is_available", return_value=False)
    @patch("cv_pipeline.train.SegmentationDataset")
    @patch("cv_pipeline.train.DataLoader")
    @patch("segmentation_models_pytorch.Unet")
    @patch("torch.save")
    def test_custom_run_name_passes_through(
        self,
        mock_save: MagicMock,
        mock_unet: MagicMock,
        mock_loader: MagicMock,
        mock_dataset: MagicMock,
        mock_cuda: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A user-supplied run_name should appear in the result."""
        data_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        out_dir = tmp_path / "out"
        data_dir.mkdir()
        val_dir.mkdir()

        mock_model = MagicMock()
        param = torch.nn.Parameter(torch.randn(1, 1))
        mock_model.parameters.return_value = [param]
        mock_unet.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_loader.return_value = []

        result = train(
            data_dir=data_dir,
            val_dir=val_dir,
            output_dir=out_dir,
            epochs=1,
            batch_size=1,
            device="cpu",
            run_name="my-experiment",
        )

        assert result.run_name == "my-experiment"


# ---- train() error paths --------------------------------------------


@pytest.mark.unit
class TestTrainErrors:
    """Tests for training error handling."""

    def test_missing_data_dir_raises(self, tmp_path: Path) -> None:
        """train() should raise when the training data directory does not exist."""
        val_dir = tmp_path / "val"
        val_dir.mkdir()
        missing_dir = tmp_path / "nonexistent"

        with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
            train(
                data_dir=missing_dir,
                val_dir=val_dir,
                output_dir=tmp_path / "out",
                epochs=1,
                batch_size=1,
                device="cpu",
            )

    def test_missing_val_dir_raises(self, tmp_path: Path) -> None:
        """train() should raise when the validation data directory does not exist."""
        data_dir = tmp_path / "train"
        data_dir.mkdir()
        missing_dir = tmp_path / "nonexistent"

        with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
            train(
                data_dir=data_dir,
                val_dir=missing_dir,
                output_dir=tmp_path / "out",
                epochs=1,
                batch_size=1,
                device="cpu",
            )
