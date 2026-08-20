# scripts/train.ps1 — Train a segmentation model using the backend Docker image.
#
# Usage:
#   .\scripts\train.ps1 <data_dir> [output_dir] [epochs]
#
# The data directory must contain train\ and val\ subdirectories,
# each with images\ and masks\ inside.

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$DataDir,

    [Parameter(Position=1)]
    [string]$OutputDir = ".\models",

    [Parameter(Position=2)]
    [int]$Epochs = 50
)

$ErrorActionPreference = "Stop"

$DataDir = (Resolve-Path $DataDir).Path
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}
$OutputDir = (Resolve-Path $OutputDir).Path

Write-Host "Training configuration:"
Write-Host "  Data:    $DataDir"
Write-Host "  Output:  $OutputDir"
Write-Host "  Epochs:  $Epochs"
Write-Host ""

docker run --rm `
    --entrypoint cv-pipeline `
    -v "${DataDir}:/data:ro" `
    -v "${OutputDir}:/output" `
    cv-platform/backend:local `
    train `
        --data-dir /data/train `
        --val-dir /data/val `
        --output-dir /output `
        --epochs $Epochs
