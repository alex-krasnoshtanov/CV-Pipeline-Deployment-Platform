# CV Pipeline Specification

**Version**: 0.2.0  
**Last updated**: 2026-04-18  
**Status**: Stable for v0.x. Section numbering is referenced from other
documents, so sections are never renumbered — only appended or amended in
place.

This document is the contract between the `cv-pipeline` package and all consumers: the FastAPI backend, the CLI, Azure ML training and inference jobs, and any robotic platform integrating the pipeline. Anyone reading this document should know exactly what to pass in and exactly what to expect back, without reading the source code.

---

## Table of Contents

- [CV Pipeline Specification](#cv-pipeline-specification)
  - [Table of Contents](#table-of-contents)
  - [1. Scope and Purpose](#1-scope-and-purpose)
    - [Requirement coverage](#requirement-coverage)
  - [2. Input Specification](#2-input-specification)
    - [2.1 Image File](#21-image-file)
    - [2.2 Plant Species](#22-plant-species)
    - [2.3 Image Source](#23-image-source)
    - [2.4 Optional Metadata Fields](#24-optional-metadata-fields)
  - [3. Output Specification](#3-output-specification)
    - [3.1 Segmentation Mask](#31-segmentation-mask)
    - [3.2 Landmark Coordinates](#32-landmark-coordinates)
    - [3.3 Confidence Scores](#33-confidence-scores)
  - [4. JSON Response Schema](#4-json-response-schema)
    - [Field reference](#field-reference)
  - [5. Edge Cases and Error Handling](#5-edge-cases-and-error-handling)
  - [6. Error Response Structure](#6-error-response-structure)
  - [7. Versioning and Future Work](#7-versioning-and-future-work)
    - [Out of Scope for v0.x: Batch Inference](#out-of-scope-for-v0x-batch-inference)
  - [8. Change Log](#8-change-log)
  - [9. Training CLI](#9-training-cli)
  - [10. Database schema](#10-database-schema)
  - [11. Feedback API](#11-feedback-api)
  - [12. Health check](#12-health-check)
  - [13. Monitoring thresholds](#13-monitoring-thresholds)

---

## 1. Scope and Purpose

The `cv-pipeline` package provides the model architecture, preprocessing, and postprocessing logic for Petri-dish seedling images. Weight loading and endpoint exposure are the responsibility of the serving layer (the backend container locally/on-prem, or the Azure ML managed endpoint in cloud).

The package accepts images of Petri dishes containing *Arabidopsis thaliana* seedlings and returns:

- a binary segmentation mask identifying root pixels
- pixel-space coordinates of each detected root tip
- confidence scores for the mask and for each landmark

The pipeline is intentionally agnostic about the caller. The same `infer()` function is called by the CLI, by the FastAPI `/infer` endpoint, and by Azure ML inference jobs. No robot-platform-specific logic lives inside the package.

The primary image source is the HADES phenotyping installation at NPEC. Other image sources are accepted at runtime provided the images meet the constraints below.

### 1.1 Serving layer responsibility

The `infer()` function has the signature:

```python
infer(model: torch.nn.Module, image: Union[Path, bytes], metadata: dict | None) -> InferenceResult
```

The caller (backend startup event, CLI entrypoint, Azure ML scoring script) is responsible for:

- Loading weights from the appropriate source (local path or Azure ML registry)
- Passing the loaded model to `infer()`
- Handling model version metadata

The package never touches the filesystem for weight loading.

### 1.2 Environment contract

| Context | How weights are provided |
|---|---|
| CLI (`cv-pipeline infer`) | `--weights <path>` argument, required |
| CLI (`cv-pipeline train`) | Writes weights to `--output-dir`, does not read them |
| Backend container (local/on-prem) | `MODEL_PATH` environment variable, loaded at startup |
| Backend container (cloud) | `MODEL_ENDPOINT_URL` + `MODEL_ENDPOINT_KEY` environment variables, calls Azure ML |
| Azure ML inference job | Scoring script receives model path from Azure ML framework |
| Azure ML training job | Training script writes checkpoint, submits to registry |

No code path should make assumptions about which context it is running in. The calling layer (CLI entrypoint, FastAPI startup event, Azure ML scoring script) is responsible for weight resolution. The `infer()` function receives a loaded model object.

### Requirement coverage

The four domain requirements the pipeline was built against, and where this
specification addresses each:

| Requirement | Where this spec addresses it |
|---|---|
| Accept a plant image as input | Section 2.1 (accepted formats, resolution, file size) |
| Output segmentation masks, landmark locations, and confidence scores | Sections 3.1, 3.2, 3.3, and the full JSON schema in section 4 |
| Modular and robotic-platform agnostic | Section 1 (caller-agnostic `infer()` interface); section 4 (self-contained JSON response that any platform can parse) |
| Handle multiple users with secure access | Out of scope for this document — authentication is implemented at the FastAPI layer, not inside `cv-pipeline` |

---

## 2. Input Specification

### 2.1 Image File

| Property | Accepted values | Notes |
|---|---|---|
| File formats | `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg` | Multi-page TIFF: only the first page is read |
| Colour modes | Grayscale (single channel), RGB (three channels) | RGBA images: alpha channel is dropped and a `WARNING` is logged; CMYK is rejected |
| Resolution (minimum) | 256 × 256 pixels | Images below this threshold are rejected |
| Resolution (maximum) | 8192 × 8192 pixels | Images above this threshold are rejected |
| File size (maximum) | 50 MB | Applies to the raw file on disk, before decoding |
| Bit depth | 8-bit or 16-bit per channel | 16-bit images are normalised to 8-bit before inference |

The pipeline does not resize images internally before inference. The segmentation mask returned is the same pixel dimensions as the input image (see section 3.1).

### 2.2 Plant Species

The model is trained exclusively on *Arabidopsis thaliana* seedlings in Petri dishes. Other species are not rejected at the API level — the pipeline will run and return a result — but prediction quality is undefined and no accuracy guarantees apply. Callers using other species must note this in their own documentation.

### 2.3 Image Source

| Source | Support status |
|---|---|
| HADES phenotyping installation (NPEC) | Fully supported |
| Flatbed scanner images of Petri dishes | Accepted (no guarantee on accuracy) |
| Smartphone or consumer camera | Accepted (no guarantee on accuracy) |
| Synthetic or rendered images | Accepted (no guarantee on accuracy) |

### 2.4 Optional Metadata Fields

Metadata is passed as a JSON object alongside the image. All fields are optional. If omitted, the fields appear as `null` in the response.

| Field | Type | Example | Description |
|---|---|---|---|
| `plate_id` | `string` | `"PL-2024-001"` | Identifier for the Petri dish or plate |
| `experiment_id` | `string` | `"EXP-NPEC-42"` | Identifier for the experiment this image belongs to |
| `timestamp` | `string` (ISO 8601) | `"2026-04-14T09:30:00Z"` | Time the image was captured |

#### Example metadata payload (API)

```json
{
  "plate_id": "PL-2024-001",
  "experiment_id": "EXP-NPEC-42",
  "timestamp": "2026-04-14T09:30:00Z"
}
```

#### Example CLI usage

```bash
cv-pipeline infer \
  --image plate_001.tif \
  --weights ./models/best_model.pth \
  --output results/ \
  --plate-id PL-2024-001 \
  --experiment-id EXP-NPEC-42 \
  --timestamp 2026-04-14T09:30:00Z
```

The CLI loads the checkpoint fresh on each invocation. For repeated inference, use the API (which keeps the model warm in memory). For the API, weight loading is handled at container startup via the `MODEL_PATH` environment variable.

---

## 3. Output Specification

### 3.1 Segmentation Mask

| Property | Value |
|---|---|
| Format | PNG (lossless, single-channel 8-bit) |
| Resolution | Same as the input image (no cropping or resizing is applied) |
| Pixel value: background | `0` |
| Pixel value: root | `255` |
| Colour mode | Grayscale |
| Filename (CLI) | `<input_stem>_mask.png` saved to the output directory |
| Encoding (API) | Base64-encoded PNG string in the JSON response field `mask_b64` |

The mask is a binary map. Every pixel that the model classifies as root tissue is set to 255. All other pixels are 0. There are no intermediate values and no multi-class labels in this version of the pipeline.

The CLI writes two files per input image into the output directory:

| File | Convention | Example |
|---|---|---|
| Segmentation mask | `<input_stem>_mask.png` | `plate_001_mask.png` |
| Full JSON result | `<input_stem>_result.json` | `plate_001_result.json` |

The JSON result file contains the complete response schema from section 4, with `mask_b64` included. If the output directory does not exist, the CLI creates it.

### 3.2 Landmark Coordinates

Root tip coordinates are returned as a list of objects. Each object contains a zero-based `id`, integer pixel coordinates `x` and `y`, and a per-landmark `confidence` score (see section 3.3).

| Property | Value |
|---|---|
| Reference frame | Pixel space of the input image |
| Origin | Top-left corner of the image (0, 0) |
| x-axis | Increases to the right |
| y-axis | Increases downward |
| Coordinate type | Integer pixel coordinates |
| Ordering | No guaranteed order between tips |

Each entry in the `landmarks` list corresponds to one detected root tip. If no root tips are detected, the list is empty (`[]`) and `landmark_count` is `0`. This is a valid non-error response.

### 3.3 Confidence Scores

Two kinds of confidence score are returned.

**Mask confidence** (`mask_confidence`): a single float in `[0.0, 1.0]` representing the model's mean pixel-level confidence across all pixels classified as root. This is the mean of the sigmoid output values for root-classified pixels. A value of `1.0` means the model was maximally confident on every root pixel. A value below `0.5` indicates the model was uncertain about most of what it labelled as root. If no pixels are classified as root, the mean is undefined; by convention `mask_confidence` is set to `0.0` in this case.

**Per-landmark confidence** (`confidence` inside each landmark object): a float in `[0.0, 1.0]` representing the model's confidence that the detected point is a true root tip. Derived from the heatmap peak intensity at that coordinate.

There is no hard threshold enforced by the pipeline itself. Callers are responsible for deciding what confidence level is acceptable for their use case. A threshold of `0.5` is a reasonable starting point.

---

## 4. JSON Response Schema

This is the complete structure of a successful inference response. The same schema applies to the API response body, the CLI output JSON file, and the Azure ML inference job output.

```json
{
  "pipeline_version": "0.2.0",
  "model_version": "unet-v2",
  "timestamp": "2026-04-14T09:31:05Z",
  "image_filename": "plate_001.tif",
  "image_width_px": 2048,
  "image_height_px": 2048,
  "metadata": {
    "plate_id": "PL-2024-001",
    "experiment_id": "EXP-NPEC-42",
    "timestamp": "2026-04-14T09:30:00Z"
  },
  "mask_b64": "<base64-encoded PNG string>",
  "mask_confidence": 0.87,
  "landmark_count": 3,
  "landmarks": [
    {
      "id": 0,
      "x": 412,
      "y": 893,
      "confidence": 0.94
    },
    {
      "id": 1,
      "x": 1105,
      "y": 654,
      "confidence": 0.81
    },
    {
      "id": 2,
      "x": 788,
      "y": 1420,
      "confidence": 0.76
    }
  ]
}
```

### Field reference

| Field | Type | Nullable | Description |
|---|---|---|---|
| `pipeline_version` | string | No | Semantic version of the `cv-pipeline` package |
| `model_version` | string | No | Name and version of the registered model used for inference |
| `timestamp` | string (ISO 8601) | No | UTC time the inference completed |
| `image_filename` | string | No | Filename of the input image (not the full path) |
| `image_width_px` | integer | No | Width of the input image in pixels |
| `image_height_px` | integer | No | Height of the input image in pixels |
| `metadata.plate_id` | string | Yes | Passed through from the input metadata |
| `metadata.experiment_id` | string | Yes | Passed through from the input metadata |
| `metadata.timestamp` | string (ISO 8601) | Yes | Passed through from the input metadata |
| `mask_b64` | string | No | Base64-encoded PNG of the binary segmentation mask |
| `mask_confidence` | float [0, 1] | No | Mean confidence across root-classified pixels |
| `landmark_count` | integer | No | Number of root tips detected; 0 is a valid value |
| `landmarks` | array | No | List of landmark objects; empty array if none detected |
| `landmarks[].id` | integer | No | Zero-based index of this landmark |
| `landmarks[].x` | integer | No | x coordinate in input pixel space |
| `landmarks[].y` | integer | No | y coordinate in input pixel space |
| `landmarks[].confidence` | float [0, 1] | No | Confidence that this point is a true root tip |

---

## 5. Edge Cases and Error Handling

| Case | Trigger condition | Pipeline behaviour | HTTP status | Error code |
|---|---|---|---|---|
| Unsupported file type | Extension not in `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg` | Rejected before decoding | 422 | `UNSUPPORTED_FILE_TYPE` |
| CMYK colour mode | Image decoded as CMYK | Rejected after format check | 422 | `UNSUPPORTED_COLOR_MODE` |
| File too large | File size > 50 MB | Rejected before decoding | 413 | `FILE_TOO_LARGE` |
| Image too small | Either dimension < 256 px | Rejected after decoding | 422 | `IMAGE_TOO_SMALL` |
| Image too large | Either dimension > 8192 px | Rejected after decoding | 422 | `IMAGE_TOO_LARGE` |
| Corrupt or unreadable file | File cannot be decoded as an image | Rejected after decode attempt | 422 | `CORRUPT_FILE` |
| No landmarks detected | Model finds no root tips | Valid response returned; `landmark_count` is `0`, `landmarks` is `[]` | 200 | — |
| Low mask confidence | `mask_confidence` below caller's threshold | Valid response returned; the confidence value is present in the response and the caller decides how to handle it | 200 | — |
| No roots segmented | Mask is entirely zero | Valid response returned; `mask_confidence` is `0.0`, `landmark_count` is `0` | 200 | — |

Note: the pipeline does not raise an error for low confidence or empty predictions. These are valid model outputs. It is the caller's responsibility — the API route handler, the CLI wrapper, or the robotic platform — to decide whether a low-confidence result should be flagged, stored, or retried.

---

## 6. Error Response Structure

All error responses from the API follow this structure:

```json
{
  "error_code": "FILE_TOO_LARGE",
  "message": "The uploaded file is 62.4 MB, which exceeds the 50 MB limit.",
  "pipeline_version": "0.2.0",
  "timestamp": "2026-04-14T09:31:05Z"
}
```

| Field | Type | Description |
|---|---|---|
| `error_code` | string | Machine-readable code from the table in section 5 |
| `message` | string | Human-readable explanation with specific values where available |
| `pipeline_version` | string | Package version at the time of the error |
| `timestamp` | string (ISO 8601) | UTC time the error occurred |

The CLI outputs this same structure to stderr as JSON, in addition to a non-zero exit code.

---

## 7. Versioning and Future Work

`pipeline_version` follows semantic versioning (`MAJOR.MINOR.PATCH`).

- A `PATCH` increment means internal changes only; the response schema is unchanged.
- A `MINOR` increment means new optional fields may appear in the response. Existing fields are unchanged. Callers that ignore unknown fields are unaffected.
- A `MAJOR` increment means breaking changes to the schema. Callers must update.

`model_version` is the name and version tag of the model as registered in Azure ML. The format is:

```
<architecture>-v<azure_ml_version>
```

Examples: `unet-v1`, `unet-v2`, `unet-v12`. The `<architecture>` segment is fixed per registered model name in Azure ML. The `<azure_ml_version>` segment is the integer version assigned by the Azure ML model registry when the model is registered. The training pipeline is responsible for writing this string into the model's metadata so that the serving layer can expose it in the response. Callers that need reproducibility should log both `pipeline_version` and `model_version`.

### Out of Scope for v0.x: Batch Inference

This specification covers single-image inference only. The data pipeline processes multiple images in sequence by calling `infer()` in a loop. Batch inference is deferred deliberately: it is only worth the added surface if single-image inference proves too slow in the pipeline, and the measured threshold for revisiting it is >3s per image on the target server. Anyone implementing batch processing should update this section first, since it is the contract other components read.

Any future `batch_infer()` interface — accepting a list of image paths and returning a list of results — will require a minor version bump when added.

---

## 8. Change Log

| Version | Date | Author | Summary |
|---|---|---|---|
| 0.1.0 | 2026-04-15 | Krasnoshtanov, Alex | Initial draft |
| 0.1.1 | 2026-04-15 | Krasnoshtanov, Alex | Fix landmarks prose to match schema; add brief requirement mapping; RGBA now logs a warning; add CLI result JSON filename convention; pin model_version format; add batch inference future-work note |
| 0.2.0 | 2026-04-18 | Krasnoshtanov, Alex | Separate package from serving layer (section 1 rewrite); add environment contract table; add `--weights` to CLI infer; add training CLI (section 9); add database schema (section 10); add feedback API (section 11); add health check (section 12); add monitoring thresholds (section 13); update batch inference scope with concrete trigger; fix model_version references to match integer format |

---

## 9. Training CLI

The CLI exposes two top-level commands: `cv-pipeline infer` and `cv-pipeline train`.

### 9.1 `cv-pipeline train`

Trains the model on a local dataset and saves a checkpoint.

```bash
cv-pipeline train \
  --data-dir ./data/train \
  --val-dir ./data/val \
  --output-dir ./models \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-4 \
  --device cuda \
  --run-name experiment-001
```

| Argument | Type | Required | Description |
|---|---|---|---|
| `--data-dir` | path | yes | Directory of training images and mask pairs |
| `--val-dir` | path | yes | Directory of validation images and mask pairs |
| `--output-dir` | path | yes | Where to write `best_model.pth` and `run_metrics.json` |
| `--epochs` | int | no, default 50 | Number of training epochs |
| `--batch-size` | int | no, default 16 | Batch size |
| `--lr` | float | no, default 1e-4 | Initial learning rate |
| `--device` | str | no, default cuda | `cuda`, `cpu`, or `cuda:N` |
| `--run-name` | str | no, default timestamp | Identifier for this run in logs and output filenames |

### 9.2 Output

A successful training run writes to `--output-dir`:

| File | Description |
|---|---|
| `best_model.pth` | Checkpoint with highest validation F1 |
| `run_metrics.json` | Per-epoch loss, F1, IoU; final test metrics |
| `training.log` | Full Python logging output |

`run_metrics.json` structure:

```json
{
  "run_name": "experiment-001",
  "pipeline_version": "0.2.0",
  "epochs": [
    {"epoch": 1, "train_loss": 0.52, "val_f1": 0.61, "val_iou": 0.44},
    {"epoch": 2, "train_loss": 0.41, "val_f1": 0.69, "val_iou": 0.51}
  ],
  "best_epoch": 38,
  "best_val_f1": 0.79,
  "training_completed": "2026-05-01T14:22:00Z"
}
```

### 9.3 Exit codes

| Code | Meaning |
|---|---|
| 0 | Training completed successfully, checkpoint saved |
| 1 | Training completed but validation F1 below 0.5 threshold; checkpoint saved with warning |
| 2 | Fatal error (data not found, CUDA OOM, config error) |

---

## 10. Database schema

All three deployment environments (local, on-prem, cloud) use the same Postgres 16 schema. Schema migrations are managed with Alembic.

### 10.1 predictions

Stores every inference result. Written by the backend after a successful call to the model service.

```sql
CREATE TABLE predictions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    image_filename  TEXT NOT NULL,
    image_width_px  INTEGER NOT NULL,
    image_height_px INTEGER NOT NULL,
    plate_id        TEXT,
    experiment_id   TEXT,
    pipeline_version TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    mask_b64        TEXT NOT NULL,
    mask_confidence REAL NOT NULL CHECK (mask_confidence >= 0 AND mask_confidence <= 1),
    landmark_count  INTEGER NOT NULL,
    landmarks       JSONB NOT NULL,
    user_id         UUID REFERENCES users(id)
);
```

### 10.2 feedback

Stores researcher flags on predictions. Written by POST /feedback. Read by the retraining trigger logic.

```sql
CREATE TABLE feedback (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    prediction_id   UUID NOT NULL REFERENCES predictions(id),
    user_id         UUID REFERENCES users(id),
    flag            TEXT NOT NULL CHECK (flag IN ('good', 'bad', 'uncertain')),
    notes           TEXT,
    corrected_mask_b64 TEXT
);
```

The `corrected_mask_b64` field is optional. When a researcher provides a corrected mask (via the frontend's correction tool), it is stored here and used as a supervised label in the next retraining run. Rows without a corrected mask are still useful — they act as a signal that the prediction needs human review.

### 10.3 users

Stores API key hashes for authentication. Passwords are never stored in plaintext.

```sql
CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    name            TEXT NOT NULL,
    api_key_hash    TEXT NOT NULL UNIQUE,
    role            TEXT NOT NULL DEFAULT 'researcher'
                        CHECK (role IN ('researcher', 'admin'))
);
```

### 10.4 model_versions

Tracks which model version was used for each deployment. Read by the monitoring dashboard to annotate metric timelines with version changes.

```sql
CREATE TABLE model_versions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    registered_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version   TEXT NOT NULL UNIQUE,
    val_f1          REAL,
    test_f1         REAL,
    azure_ml_version INTEGER,
    notes           TEXT
);
```

---

## 11. Feedback API

### 11.1 POST /feedback

Records a researcher's assessment of a prediction.

**Request body**:

```json
{
  "prediction_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "flag": "bad",
  "notes": "Root tip 2 is wrong, the model missed the lateral root.",
  "corrected_mask_b64": null
}
```

| Field | Type | Required | Values |
|---|---|---|---|
| `prediction_id` | UUID string | yes | Must reference an existing prediction |
| `flag` | string | yes | `good`, `bad`, `uncertain` |
| `notes` | string | no | Free-text, max 2000 chars |
| `corrected_mask_b64` | string | no | Base64-encoded PNG mask, same spec as prediction output |

**Response (200)**:

```json
{
  "feedback_id": "a1b2c3d4-...",
  "prediction_id": "3fa85f64-...",
  "flag": "bad",
  "created_at": "2026-05-15T10:30:00Z"
}
```

**Retraining trigger logic**:

The retraining pipeline checks the feedback table on a schedule and on-demand. It triggers a new training run when either of these conditions is true:
- Accumulated `bad` feedback records since the last training run >= 50
- Weekly schedule fires (regardless of feedback count)

This threshold (50) is configurable via the `RETRAIN_FEEDBACK_THRESHOLD` environment variable. Its default is 50.

---

## 12. Health check

### 12.1 GET /health

Returns the current status of the backend service.

**Response (200 — healthy)**:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "unet-v2",
  "pipeline_version": "0.2.0",
  "serving_mode": "local",
  "timestamp": "2026-05-01T09:00:00Z"
}
```

**Response (503 — model not yet loaded)**:

```json
{
  "status": "loading",
  "model_loaded": false,
  "pipeline_version": "0.2.0",
  "timestamp": "2026-05-01T08:59:45Z"
}
```

The endpoint returns HTTP 200 only when `model_loaded` is `true`. The compose healthcheck uses this:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 10s
  start_period: 40s
  retries: 5
```

`start_period: 40s` is required because the first CUDA kernel compilation after container start takes 20-35 seconds on a consumer GPU.

`serving_mode` is one of `local` (weights from `MODEL_PATH`) or `azure_ml` (calls `MODEL_ENDPOINT_URL`).

---

## 13. Monitoring thresholds

These are the concrete values used in Azure Monitor alert rules and in the retraining trigger logic. All are configurable via environment variables with these defaults.

| Metric | Alert condition | Env var | Default |
|---|---|---|---|
| Mean prediction confidence | < 0.60 over a 1-hour window | `ALERT_CONFIDENCE_MIN` | 0.60 |
| Fraction of predictions with confidence < 0.50 | > 20% over a 1-hour window | `ALERT_LOW_CONF_FRACTION` | 0.20 |
| Inference latency p95 | > 5000 ms over a 5-minute window | `ALERT_LATENCY_P95_MS` | 5000 |
| Error rate | > 5% of requests return 4xx or 5xx over 5 minutes | `ALERT_ERROR_RATE` | 0.05 |
| Test F1 regression | New model test F1 < current production model test F1 - 0.01 | — | hardcoded in promotion gate |
| Feedback-triggered retrain | Accumulated `bad` feedback since last run >= 50 | `RETRAIN_FEEDBACK_THRESHOLD` | 50 |

The confidence threshold of 0.60 is set above 0.50 (the "uncertain" point) to give an early warning before predictions degrade to random. The best validation F1 from earlier work was 0.7847. The absolute floor for model registration is F1 >= 0.75 (see the training-pipeline design note in `docs/architecture/`). These two numbers are consistent: a model that passes registration should produce mean confidence above 0.60 on typical HADES images.

**Auto-rollback rule**: if the confidence mean drops below 0.60 AND the new model version is less than 24 hours old, automatically roll back the endpoint traffic split to 100% on the previous version and page the team.

---
