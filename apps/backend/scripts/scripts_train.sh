#!/usr/bin/env bash
# scripts/train.sh — Train a segmentation model using the backend Docker image.
#
# Usage:
#   ./scripts/train.sh <data_dir> [output_dir] [epochs]
#
# The data directory must contain train/ and val/ subdirectories,
# each with images/ and masks/ inside:
#
#   my_data/
#       train/
#           images/
#           masks/
#       val/
#           images/
#           masks/
#
# The trained checkpoint (best_model.pth) and metrics (run_metrics.json)
# are written to the output directory on your host machine.

set -euo pipefail

DATA_DIR="${1:?Usage: ./scripts/train.sh <data_dir> [output_dir] [epochs]}"
OUTPUT_DIR="${2:-./models}"
EPOCHS="${3:-50}"

# Resolve to absolute paths for Docker volume mounts.
DATA_DIR="$(cd "$DATA_DIR" && pwd)"
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"

echo "Training configuration:"
echo "  Data:    $DATA_DIR"
echo "  Output:  $OUTPUT_DIR"
echo "  Epochs:  $EPOCHS"
echo ""

docker run --rm \
    --entrypoint cv-pipeline \
    -v "$DATA_DIR":/data:ro \
    -v "$OUTPUT_DIR":/output \
    cv-platform/backend:local \
    train \
        --data-dir /data/train \
        --val-dir /data/val \
        --output-dir /output \
        --epochs "$EPOCHS"
