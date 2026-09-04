# Package Plan — cv-pipeline and Application Services

**Version**: 0.1.0  
**Status**: Updated to match implemented code  
**ADO**: #265  
**Last updated**: 2026-04-20

---

## 1. Repository Structure

The project follows a uv workspace monorepo pattern. The cv-pipeline library
is a standalone installable package with no knowledge of the serving layer.
Backend and frontend are separate applications that consume the library.

```text
cv-platform-workspace/
├── packages/
│   └── cv-pipeline/                # Installable library (pip install dist/*.whl)
│       ├── pyproject.toml
│       ├── src/
│       │   └── cv_pipeline/
│       │       ├── __init__.py     # Re-exports infer() and __version__
│       │       ├── _version.py     # Single source of truth for version string
│       │       ├── cli.py          # argparse entrypoints: infer, version
│       │       ├── infer.py        # infer() → InferenceResult (top-level pipeline)
│       │       ├── validation.py   # Input checks (extension, size, dimensions, colour)
│       │       ├── preprocessing.py# Petri dish detection and cropping
│       │       ├── segmentation.py # SegmentationModel: U-Net loading + patch inference
│       │       ├── landmarks.py    # Root tip detection from segmentation output
│       │       ├── schema.py       # Dataclass models: InferenceResult, Landmark, etc.
│       │       └── py.typed        # PEP 561 marker for type checker support
│       └── tests/
│           ├── conftest.py         # Shared fixtures (sample_image_path, tmp_output_dir)
│           ├── fixtures/           # Sample test images
│           ├── unit/
│           │   ├── test_cli.py
│           │   ├── test_infer.py
│           │   ├── test_landmarks.py
│           │   ├── test_package.py
│           │   ├── test_preprocessing.py
│           │   ├── test_schema.py
│           │   ├── test_segmentation.py
│           │   └── test_validation.py
│           └── integration/
│               └── README.md       # Integration tests
├── apps/
│   ├── backend/                    # FastAPI service (not pip-installable, consumes cv-pipeline)
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   ├── tests/                  # API endpoint tests (TestClient)
│   │   │   └── .gitkeep
│   │   └── src/
│   │       └── api/
│   │           ├── __init__.py
│   │           ├── main.py         # FastAPI app with lifespan model loading
│   │           ├── routers/
│   │           │   ├── health.py   # GET /health — readiness probe
│   │           │   └── infer.py    # POST /infer — image → segmentation result
│   │           └── schemas/
│   │               ├── __init__.py
│   │               └── health.py   # Pydantic HealthResponse model
│   └── frontend/                   # Next.js application
│       ├── Dockerfile
│       ├── package.json
│       └── src/
├── infra/
│   ├── local/
│   │   └── docker-compose.yml      # 3-service stack: backend, frontend, postgres
│   ├── server/
│   │   └── README.md               # Portainer deployment instructions
│   └── cloud/
│       └── README.md               # Azure ML deployment assets
├── configs/
│   ├── env/
│   │   └── .env.example            # All required environment variables
│   └── model/
│       └── base.yaml               # Default inference settings
├── docs/
│   ├── architecture/               # Four deployment diagrams
│   └── package_plan.md             # This file
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                  # Lint (ruff) + test (pytest ≥80% coverage)
│   │   ├── cd.yml                  # Build images → GHCR → Portainer webhook
│   │   └── pr-title.yml            # Conventional commit title enforcement
│   └── copilot-instructions.md
├── pyproject.toml                  # uv workspace root
└── uv.lock
```

## 2. cv-pipeline Module Responsibilities

Each module has a single responsibility. The inference pipeline flows
top-to-bottom through these modules in order:

```text
infer.py (orchestrator)
  ├── 1. validation.py     → validate_image(path) → np.ndarray
  ├── 2. preprocessing.py  → crop_to_dish(image) → np.ndarray
  ├── 3. segmentation.py   → model.predict(image) → prob_map
  ├── 4. landmarks.py      → detect_landmarks(prob_map, mask) → list[Landmark]
  └── 5. schema.py         → InferenceResult (assembled from above)
```

| Module | Purpose | Key class/function |
|--------|---------|-------------------|
| `_version.py` | Single source of truth for `__version__` | `__version__ = "0.1.0"` |
| `cli.py` | Command-line interface using `argparse` | `main()` → subcommands: `infer`, `version` |
| `infer.py` | Top-level pipeline orchestrator | `infer(image_path, model, metadata, ...) → InferenceResult` |
| `validation.py` | Input image validation (extension, size, dimensions, colour mode) | `validate_image(path) → np.ndarray` raises `ValidationError` |
| `preprocessing.py` | Petri dish detection via Otsu threshold + morphology, returns cropped region | `crop_to_dish(image) → np.ndarray` |
| `segmentation.py` | U-Net model loading + patch-based inference with overlap averaging | `SegmentationModel(path).predict(image) → prob_map` |
| `landmarks.py` | Root tip detection: separates plants by expected horizontal position, finds bottommost point per plant | `detect_landmarks(prob_map, mask, ...) → list[Landmark]` |
| `schema.py` | Data structures for pipeline output using Python `dataclasses` | `InferenceResult`, `Landmark`, `Metadata`, `ErrorResponse` |

## 3. Public Interface

These are the only symbols consumers (backend, CLI, Azure ML jobs) should import.
Everything prefixed with `_` is internal.

### 3.1 `infer()`

```python
from cv_pipeline import infer
from cv_pipeline.schema import InferenceResult, Metadata
from cv_pipeline.segmentation import SegmentationModel

model = SegmentationModel("weights.pth")

result: InferenceResult = infer(
    image_path="plate_001.png",
    model=model,
    metadata=Metadata(plate_id="PL-001"),
    threshold=0.5,
    crop=True,
    num_plants=5,
    plant_start=350,
    plant_step=500,
    roi_width=250,
)
```

**Returns**: `InferenceResult` dataclass containing:
- `mask_b64`: base64-encoded PNG of binary segmentation mask
- `landmarks`: list of `Landmark(id, x, y, confidence)`
- `mask_confidence`: mean probability of root-classified pixels
- `image_filename`, `image_width_px`, `image_height_px`
- `pipeline_version`, `model_version`, `timestamp`
- `metadata`: pass-through `Metadata` object

**Raises**: `ValidationError` with machine-readable `error_code` and human-readable `message` for any input that fails validation.

### 3.2 `SegmentationModel`

```python
from cv_pipeline.segmentation import SegmentationModel

model = SegmentationModel(
    model_path="weights.pth",
    patch_size=256,
    overlap=0.5,
    device=None,  # auto-detects CUDA
)

prob_map = model.predict(image)  # → (H, W) float32 [0, 1]
binary_mask, confidence = model.predict_mask(image)  # → (uint8 0/255, float)
```

The caller creates the model once and reuses it across calls. The backend
loads it at startup via the `lifespan` context manager. The CLI loads it
once per invocation.

### 3.3 `validate_image()`

```python
from cv_pipeline.validation import validate_image, ValidationError

try:
    image_np = validate_image(Path("input.png"))
except ValidationError as exc:
    print(f"[{exc.error_code}] {exc.message}")
```

**Validation checks** (in order):
1. File extension: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`
2. File size: ≤ 50 MB
3. Decodability: Pillow can open the file
4. Colour mode: grayscale or RGB (RGBA alpha dropped, CMYK rejected)
5. Minimum dimensions: ≥ 256 × 256 px
6. Maximum dimensions: ≤ 8192 × 8192 px

### 3.4 CLI

```bash
# Run inference
cv-pipeline infer \
    --image plate_001.png \
    --output results/ \
    --model weights.pth \
    --threshold 0.5 \
    --plate-id PL-2024-001

# Print version
cv-pipeline version
```

The CLI is implemented with `argparse` (stdlib). Entry point is
`cv_pipeline.cli:main`, registered in `pyproject.toml` under
`[project.scripts]`.

### 3.5 What cv-pipeline Does NOT Export

- Any HTTP client or server code
- Any database client or ORM models
- Any Azure ML SDK imports
- Any Docker or infrastructure code
- Model weight files (`.pth`) — never in the repo or wheel

## 4. Backend Architecture

### 4.1 Model Serving Abstraction

The backend resolves the serving mode at startup from environment variables:

```python
# apps/backend/src/api/main.py


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = ServiceState(...)
    if os.getenv("MODEL_ENDPOINT_URL"):
        state.serving_mode = "azure_ml"  # Delegates to Azure ML endpoint
    else:
        model = SegmentationModel(os.getenv("MODEL_PATH"))
        state.serving_mode = "local"  # Uses local weights
    ...
```

Route handlers receive the loaded model via `request.app.state` and call
`cv_pipeline.infer()` without knowing whether inference is local or cloud.
This is the single place in the codebase that contains the local vs cloud
branching logic.

### 4.2 Endpoints

| Method | Path | Purpose | Auth | Response Model |
|--------|------|---------|------|----------------|
| GET | `/health` | Readiness probe | None | `HealthResponse` (Pydantic) |
| POST | `/infer` | Image → segmentation + landmarks | API key | `InferenceResponse` (Pydantic) |
| POST | `/feedback` | User flags a bad prediction | API key | `FeedbackResponse` (Pydantic) |

### 4.3 Authentication

API key authentication via `X-API-Key` header. Keys stored as bcrypt hashes
in the users table. The `/health` endpoint is exempt from authentication.
Implemented as a FastAPI `Depends` dependency.

## 5. Dependency Graph

```text
cv-pipeline (library, installable via wheel)
  ├── torch >= 2.0
  ├── segmentation-models-pytorch >= 0.3
  ├── opencv-python-headless >= 4.8
  ├── Pillow >= 10.0
  ├── numpy >= 1.26
  └── (no web framework, no database, no cloud SDK)

apps/backend (FastAPI service, not installable)
  ├── cv-pipeline          (workspace dependency)
  ├── fastapi >= 0.110
  ├── uvicorn >= 0.29
  ├── pydantic >= 2.0      (API response models only — cv-pipeline uses dataclasses)
  ├── sqlalchemy >= 2.0 (feedback storage)
  ├── asyncpg >= 0.29 (async Postgres driver)
  └── azure-ai-ml >= 1.15 (cloud serving mode, optional)

apps/frontend (Next.js app, not installable)
  ├── next >= 16
  ├── react >= 19
  └── recharts >= 3
```

**Design decision — dataclasses vs Pydantic**: The cv-pipeline library uses
stdlib `dataclasses` for its response schemas (`InferenceResult`, `Landmark`,
etc.) to keep the package dependency-free from web frameworks. The FastAPI
backend wraps these in Pydantic models at the API boundary for automatic
validation and OpenAPI documentation. This separation ensures the library
remains usable outside of a web context (CLI, Azure ML jobs, notebooks).

## 6. Data Flow Diagram

```text
                        ┌─────────────────────┐
                        │   Plant Image (PNG)  │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ 1. validate_image()          │
                    │    extension, size, decode,   │
                    │    colour, dimensions         │
                    └──────────────┬──────────────┘
                                   │ np.ndarray (H,W) or (H,W,3)
                    ┌──────────────▼──────────────┐
                    │ 2. crop_to_dish()            │
                    │    Otsu → morphology →        │
                    │    contour → square crop      │
                    └──────────────┬──────────────┘
                                   │ np.ndarray (cropped)
                    ┌──────────────▼──────────────┐
                    │ 3. model.predict()           │
                    │    pad → patch → U-Net →      │
                    │    reconstruct → prob_map     │
                    └──────────────┬──────────────┘
                                   │ float32 (H,W) [0,1]
                    ┌──────────────▼──────────────┐
                    │ 4. detect_landmarks()        │
                    │    separate plants → find     │
                    │    root tips → confidence     │
                    └──────────────┬──────────────┘
                                   │ list[Landmark]
                    ┌──────────────▼──────────────┐
                    │ 5. InferenceResult           │
                    │    mask_b64 + landmarks +     │
                    │    confidence + metadata      │
                    └─────────────────────────────┘
```

## 7. Design Decisions Log

| Decision | Rationale | Alternative Considered |
|----------|-----------|----------------------|
| argparse over Click/Typer for CLI | Zero external dependencies for the library. Stdlib argparse is sufficient for 2 subcommands. | Click (adds dependency), Typer (adds Click + typing-extensions) |
| dataclasses over Pydantic for cv-pipeline schemas | Keeps the library free of web framework dependencies. Pydantic is used only at the API boundary. | Pydantic everywhere (would couple library to FastAPI ecosystem) |
| Patch-based inference with overlap averaging | Handles arbitrarily large HADES images (up to 8192×8192) without OOM. Block B validated this approach. | Resize to fixed resolution (loses detail), tiling without overlap (boundary artefacts) |
| Model loaded once at startup (not per request) | Inference latency < 5s. Loading the model per request would add ~10s overhead. | Lazy loading on first request (cold start penalty) |
| Separate validation module | Validation errors return specific error codes (UNSUPPORTED_FILE_TYPE, IMAGE_TOO_SMALL, etc.) that the API can map to HTTP status codes. | Inline validation in infer() (less reusable, harder to test) |
| Connected component separation for plant instances | HADES dishes have 5 plants at known horizontal positions. Using expected positions + component overlap is more robust than pure clustering. | K-means clustering (fragile with touching roots), watershed (over-segments) |

## 8. Versioning Strategy

- **Package version**: `cv_pipeline._version.__version__` — single source of truth, updated in one file
- **Model version**: read from checkpoint metadata key `model_version` (e.g. `"unet-v1"`), falls back to `"unet-v0"`
- **Data version**: Azure ML registered data assets with version tags (`train_v1`, `train_v2`, etc.)
- **API version**: FastAPI app `version` parameter, currently `"0.1.0"`
- **Container version**: GHCR image tagged with git SHA (`sha-<hash>`) and `latest`