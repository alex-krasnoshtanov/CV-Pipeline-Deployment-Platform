"""Command-line interface for the cv-pipeline package.

Provides the ``cv-pipeline infer`` and ``cv-pipeline train`` commands.

Usage::

    cv-pipeline infer --image plate_001.tif --output results/
    cv-pipeline train --data-dir ./data/train --val-dir ./data/val --output-dir ./models
    cv-pipeline version
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from cv_pipeline._version import __version__
from cv_pipeline.schema import ErrorResponse, Metadata
from cv_pipeline.validation import ValidationError


def main() -> None:
    """Entry point for the ``cv-pipeline`` console script."""
    parser = _build_parser()
    args = parser.parse_args()

    _setup_logging(args.verbose)

    if args.command == "infer":
        _run_infer(args)
    elif args.command == "train":
        _run_train(args)
    elif args.command == "version":
        print(f"cv-pipeline {__version__}")
    else:
        parser.print_help()
        sys.exit(1)


# ---- argument parsing ------------------------------------------------


def _add_verbose_flag(parser: argparse.ArgumentParser) -> None:
    """Attach the --verbose/-v flag to a parser.

    We add the flag to both the top-level parser and each subparser so
    ``cv-pipeline -v infer ...`` and ``cv-pipeline infer -v ...`` both
    work. Without this the flag is only accepted before the subcommand,
    which is surprising to users coming from git, docker, etc. (UX #4).
    """
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands.

    Returns:
        Configured ``ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        prog="cv-pipeline",
        description="NPEC plant organ segmentation and root tip detection.",
    )
    _add_verbose_flag(parser)

    subparsers = parser.add_subparsers(dest="command")

    # --- infer subcommand ---
    infer_parser = subparsers.add_parser(
        "infer",
        help="Run inference on a single image.",
        description=(
            "Run inference on a single image and write the mask + result "
            "JSON to the output directory. Specify a model via --model "
            "<path> OR --version <n>; if neither is given, the first "
            "registered version is used. Model weights are downloaded and "
            "cached automatically on first use when needed."
        ),
    )
    _add_verbose_flag(infer_parser)
    infer_parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input image file.",
    )
    infer_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to write result files to.",
    )
    # Either --model <path> (explicit file) or --version <n> (registry lookup).
    # Mutually exclusive because accepting both would be ambiguous.
    source_group = infer_parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to a .pth model checkpoint on disk.",
    )
    source_group.add_argument(
        "--version",
        type=str,
        default=None,
        help="Registered model version (downloaded and cached on first use).",
    )
    infer_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for mask binarisation (default: 0.5).",
    )
    infer_parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Skip Petri dish detection and cropping.",
    )
    infer_parser.add_argument(
        "--plate-id",
        type=str,
        default=None,
        help="Optional plate identifier to include in the result.",
    )
    infer_parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Optional experiment identifier to include in the result.",
    )
    infer_parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Optional ISO 8601 capture timestamp to include in the result.",
    )

    # --- train subcommand ---
    train_parser = subparsers.add_parser(
        "train",
        help="Train a segmentation model on image-mask pairs.",
        description=(
            "Train a U-Net segmentation model on a dataset of image-mask "
            "pairs. Expects data directories with images/ and masks/ "
            "subdirectories. Saves best_model.pth, run_metrics.json, and "
            "training.log to the output directory."
        ),
    )
    _add_verbose_flag(train_parser)
    train_parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory of training images and mask pairs.",
    )
    train_parser.add_argument(
        "--val-dir",
        type=Path,
        required=True,
        help="Directory of validation images and mask pairs.",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write best_model.pth and run_metrics.json.",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16).",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate (default: 1e-4).",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device: cuda, cpu, or cuda:N (default: cuda).",
    )
    train_parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Identifier for this run (default: timestamp).",
    )
    train_parser.add_argument(
        "--in-channels",
        type=int,
        default=1,
        choices=[1, 3],
        help="Input channels: 1 for grayscale, 3 for RGB (default: 1).",
    )

    # --- version subcommand ---
    version_parser = subparsers.add_parser(
        "version",
        help="Print the package version.",
    )
    _add_verbose_flag(version_parser)

    return parser


# ---- inference command -----------------------------------------------


def _run_infer(args: argparse.Namespace) -> None:
    """Execute the infer subcommand.

    Loads the model, runs inference, and writes the mask and JSON
    result to the output directory. On validation failure, writes
    the error JSON to stderr and exits with code 1.

    Args:
        args: Parsed CLI arguments.
    """
    from cv_pipeline.infer import infer
    from cv_pipeline.segmentation import SegmentationModel
    from cv_pipeline.weights import REGISTRY

    logger = logging.getLogger(__name__)

    # Validate input image early — exists, is a file, readable.
    # cv_pipeline.validation would catch these later, but reporting
    # here produces a clearer error message and avoids loading the
    # model when we already know the input is broken.
    if not args.image.exists():
        _exit_with_error(
            error_code="CORRUPT_FILE",
            message=f"Image file not found: '{args.image}'.",
        )
    if not args.image.is_file():
        _exit_with_error(
            error_code="CORRUPT_FILE",
            message=f"Image path is not a file: '{args.image}'.",
        )
    if not os.access(args.image, os.R_OK):
        _exit_with_error(
            error_code="CORRUPT_FILE",
            message=f"Image file is not readable: '{args.image}'.",
        )

    # Resolve model source: explicit --model, registry --version, or
    # default to the first registered entry. Guard against an empty
    # REGISTRY — without this, a fresh clone with no weights downloaded
    # raises StopIteration and surfaces a Python traceback to the user,
    # which violates AC #497 "clear error message, not Python traceback".
    if args.model is not None:
        if not args.model.exists():
            _exit_with_error(
                error_code="CORRUPT_FILE",
                message=f"Model file not found: '{args.model}'.",
            )
        model_source = str(args.model)
    elif args.version is not None:
        if not REGISTRY:
            _exit_with_error(
                error_code="MODEL_NOT_READY",
                message=(
                    "No model versions are registered, so '--version' cannot "
                    f"be used with '{args.version}'. Run "
                    "'python scripts/download_weights.py' first, or pass "
                    "--model <path>."
                ),
            )
        if args.version not in REGISTRY:
            known_versions = ", ".join(sorted(REGISTRY))
            _exit_with_error(
                error_code="INVALID_ARGUMENT",
                message=(
                    f"Unknown model version '{args.version}'. "
                    f"Known versions: {known_versions}."
                ),
            )
        model_source = args.version
    else:
        if not REGISTRY:
            _exit_with_error(
                error_code="MODEL_NOT_READY",
                message=(
                    "No model registered and neither --model nor --version "
                    "was provided. Complete the documented model-weight setup "
                    "for this environment and then use --version <model-version>, "
                    "or pass --model <path>."
                ),
            )
        model_source = next(iter(REGISTRY))
        logger.info("No --model or --version given; defaulting to '%s'.", model_source)

    # Load model before creating the output directory — otherwise a
    # failing model load leaves an empty directory behind.
    try:
        model = SegmentationModel(model_source)
    except Exception as exc:
        _exit_with_error(
            error_code="CORRUPT_FILE",
            message=f"Failed to load model: {exc}",
        )

    # Now create the output directory — we know we have a valid input
    # and a working model, so we will actually write something.
    try:
        args.output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _exit_with_error(
            error_code="OUTPUT_WRITE_FAILED",
            message=f"Failed to create output directory '{args.output}': {exc}",
        )

    # Build metadata from optional flags.
    metadata = Metadata(
        plate_id=args.plate_id,
        experiment_id=args.experiment_id,
        timestamp=args.timestamp,
    )

    # Run inference.
    try:
        result = infer(
            image_path=args.image,
            model=model,
            metadata=metadata,
            threshold=args.threshold,
            crop=not args.no_crop,
        )
    except ValidationError as exc:
        _exit_with_error(error_code=exc.error_code, message=exc.message)

    # Write outputs.
    stem = args.image.stem
    result_dict = result.to_dict()

    json_path = args.output / f"{stem}_result.json"
    json_path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
    logger.info("Result JSON written to '%s'.", json_path)

    mask_path = args.output / f"{stem}_mask.png"
    _write_mask(result.mask_b64, mask_path)
    logger.info("Mask written to '%s'.", mask_path)

    print(f"Inference complete: {result.landmark_count} landmark(s) detected.")
    print(f"  Result: {json_path}")
    print(f"  Mask:   {mask_path}")


# ---- training command ------------------------------------------------


def _run_train(args: argparse.Namespace) -> None:
    """Execute the train subcommand.

    Validates inputs, runs training, writes outputs, and exits with
    the appropriate code per task_328 §9.3.

    Exit codes:
        0 — Training completed successfully, checkpoint saved.
        1 — Training completed but best validation F1 < 0.5.
        2 — Fatal error (data not found, CUDA OOM, config error).

    Args:
        args: Parsed CLI arguments.
    """
    from cv_pipeline.train import train

    train_logger = logging.getLogger(__name__)

    # Set up file logging to output_dir/training.log.
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _exit_with_error(
            error_code="OUTPUT_WRITE_FAILED",
            message=f"Failed to create output directory '{args.output_dir}': {exc}",
        )

    log_path = args.output_dir / "training.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    # Validate directories exist before starting.
    if not args.data_dir.is_dir():
        train_logger.error("Training data directory not found: '%s'.", args.data_dir)
        sys.exit(2)
    if not args.val_dir.is_dir():
        train_logger.error("Validation data directory not found: '%s'.", args.val_dir)
        sys.exit(2)

    try:
        result = train(
            data_dir=args.data_dir,
            val_dir=args.val_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            run_name=args.run_name,
            in_channels=args.in_channels,
        )
    except FileNotFoundError as exc:
        train_logger.error("Data error: %s", exc)
        sys.exit(2)
    except RuntimeError as exc:
        train_logger.error("Training failed: %s", exc)
        sys.exit(2)

    print(
        f"Training complete — best val F1: "
        f"{result.best_val_f1:.4f} at epoch "
        f"{result.best_epoch}."
    )
    print(f"  Checkpoint: {args.output_dir / 'best_model.pth'}")
    print(f"  Metrics:    {args.output_dir / 'run_metrics.json'}")
    print(f"  Log:        {log_path}")

    if result.best_val_f1 < 0.5:
        train_logger.warning(
            "Best validation F1 (%.4f) is below 0.5 threshold.",
            result.best_val_f1,
        )
        sys.exit(1)

    sys.exit(0)


# ---- helpers ---------------------------------------------------------


def _write_mask(mask_b64: str, path: Path) -> None:
    """Decode a base64 PNG string and write it to disk.

    Args:
        mask_b64: Base64-encoded PNG data.
        path: Destination file path.
    """
    import base64

    raw = base64.b64decode(mask_b64)
    path.write_bytes(raw)


def _exit_with_error(error_code: str, message: str) -> None:
    """Write an error JSON to stderr and exit with code 1.

    Args:
        error_code: Machine-readable error code.
        message: Human-readable error message.
    """
    error = ErrorResponse(
        error_code=error_code,
        message=message,
        pipeline_version=__version__,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    sys.stderr.write(json.dumps(error.to_dict(), indent=2) + "\n")
    sys.exit(1)


def _setup_logging(verbose: bool) -> None:
    """Configure logging for the CLI.

    Args:
        verbose: If ``True``, set level to DEBUG. Otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
