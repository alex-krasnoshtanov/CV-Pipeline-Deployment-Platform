@echo off
REM scripts\train.bat — Train a segmentation model using the backend Docker image.
REM
REM Usage:
REM   scripts\train.bat <data_dir> [output_dir] [epochs]
REM
REM The data directory must contain train\ and val\ subdirectories,
REM each with images\ and masks\ inside.

setlocal

if "%~1"=="" (
    echo Usage: scripts\train.bat ^<data_dir^> [output_dir] [epochs]
    exit /b 1
)

set "DATA_DIR=%~f1"
if "%~2"=="" (set "OUTPUT_DIR=%cd%\models") else (set "OUTPUT_DIR=%~f2")
if "%~3"=="" (set "EPOCHS=50") else (set "EPOCHS=%~3")

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Training configuration:
echo   Data:    %DATA_DIR%
echo   Output:  %OUTPUT_DIR%
echo   Epochs:  %EPOCHS%
echo.

docker run --rm ^
    --entrypoint cv-pipeline ^
    -v "%DATA_DIR%":/data:ro ^
    -v "%OUTPUT_DIR%":/output ^
    cv-platform/backend:local ^
    train ^
        --data-dir /data/train ^
        --val-dir /data/val ^
        --output-dir /output ^
        --epochs %EPOCHS%
