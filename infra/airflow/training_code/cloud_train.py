"""Azure ML training script for the HADES CV pipeline.

This script runs on the Azure ML cluster. It wraps the existing
cv_pipeline.train.train() function, optionally evaluates the best
checkpoint on a held-out test set, and logs all metrics to MLflow
so they appear in the Azure ML experiment dashboard.

Azure ML mounts data assets as local folders, so the script
receives --train-dir, --val-dir, and optionally --test-dir as
regular paths.

Test evaluation uses the same dataset-level F1/IoU as training
validation (accumulate tp/fp/fn over the whole set, score once), so
the test number is directly comparable to val_f1 and is not deflated
by background-only patches.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow
import torch
from cv_pipeline.train import SegmentationDataset, _evaluate, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments passed by the Azure ML job."""
    parser = argparse.ArgumentParser(description="Train U-Net on HADES data.")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--min-f1",
        type=float,
        default=0.75,
        help="Minimum test F1 to mark model as eligible for registration.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="hades-unet",
        help="Name for the registered model in Azure ML.",
    )
    return parser.parse_args()


def mlflow_epoch_callback(
    epoch: int, train_loss: float, val_f1: float, val_iou: float
) -> None:
    """Log per-epoch metrics to MLflow in real time."""
    mlflow.log_metrics(
        {
            "train_loss": float(train_loss),
            "val_f1": float(val_f1),
            "val_iou": float(val_iou),
        },
        step=epoch,
    )


def evaluate_on_test(
    checkpoint_path: Path,
    test_dir: Path,
    device: str,
    batch_size: int,
    num_workers: int,
) -> tuple[float, float]:
    """Evaluate the best checkpoint on the held-out test set.

    Builds the same architecture used for inference, loads the trained
    weights, and computes a single dataset-level (F1, IoU) via the shared
    ``_evaluate`` helper. The test set is loaded with no empty-patch
    filtering and no augmentation, so it reflects the real distribution.

    Args:
        checkpoint_path: Path to best_model.pth saved by train().
        test_dir: Directory with images/ and masks/ subdirectories.
        device: Torch device string.
        batch_size: Batch size for evaluation.
        num_workers: DataLoader workers.

    Returns:
        Tuple of (test_f1, test_iou).
    """
    import segmentation_models_pytorch as smp

    logger.info("Evaluating best checkpoint on test set: %s", test_dir)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation=None,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    test_dataset = SegmentationDataset(test_dir, patch_size=256, in_channels=1)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    test_f1, test_iou = _evaluate(model, test_loader, device)

    logger.info("Test evaluation — test_f1=%.4f, test_iou=%.4f.", test_f1, test_iou)
    return test_f1, test_iou


def main() -> None:
    """Run training, optionally evaluate on test set, and log metrics to MLflow."""
    args = parse_args()

    mlflow.autolog(disable=True)

    mlflow.log_params(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "device": args.device,
            "num_workers": args.num_workers,
            "min_f1_threshold": args.min_f1,
        }
    )

    logger.info("Starting training on Azure ML cluster.")
    logger.info("  train-dir:  %s", args.train_dir)
    logger.info("  val-dir:    %s", args.val_dir)
    logger.info("  test-dir:   %s", args.test_dir or "not provided")
    logger.info("  output-dir: %s", args.output_dir)

    # ---- Train ----
    result = train(
        data_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        run_name=args.run_name,
        num_workers=args.num_workers,
        on_epoch_end=mlflow_epoch_callback,
    )

    best_f1 = float(result.best_val_f1)
    best_epoch = int(result.best_epoch)

    mlflow.log_metrics(
        {
            "best_val_f1": best_f1,
            "best_epoch": best_epoch,
        }
    )

    logger.info(
        "Training complete — best_val_f1=%.4f at epoch %d.",
        best_f1,
        best_epoch,
    )

    # ---- Evaluate on test set (if provided) ----
    checkpoint_path = args.output_dir / "best_model.pth"

    if not checkpoint_path.exists():
        logger.warning("No checkpoint saved — skipping test evaluation.")
        mlflow.log_param("model_registered", False)
        return

    if args.test_dir is None:
        logger.info("No --test-dir provided — skipping test evaluation.")
        mlflow.log_param("model_registered", False)
        return

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    test_f1, test_iou = evaluate_on_test(
        checkpoint_path=checkpoint_path,
        test_dir=args.test_dir,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    mlflow.log_metrics(
        {
            "test_f1": test_f1,
            "test_iou": test_iou,
        }
    )

    # ---- Conditionally register based on test F1 ----
    if test_f1 >= args.min_f1:
        logger.info(
            "test_f1 %.4f >= threshold %.4f — registering model.",
            test_f1,
            args.min_f1,
        )
        mlflow.log_artifact(str(checkpoint_path), artifact_path="model")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, args.model_name)
        mlflow.log_param("model_registered", True)
        logger.info("Model registered as '%s'.", args.model_name)
    else:
        logger.warning(
            "test_f1 %.4f < threshold %.4f — model NOT registered.",
            test_f1,
            args.min_f1,
        )
        mlflow.log_param("model_registered", False)


if __name__ == "__main__":
    main()
