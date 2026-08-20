# Add a new model version

When retraining produces a better checkpoint, register it in the
model registry so the CLI and API can serve it without code changes.

## Current state (Sprint 2)

The registry is a Python dict in `packages/cv-pipeline/src/cv_pipeline/weights.py`
mapping version string to download URL:

```python
REGISTRY: dict[str, str] = {
    "unet-v1": "https://example.invalid/...?download=1",
}
```

Checksum verification is a TODO for Sprint 3 - for now only the URL
is stored. The pipeline downloads the file, checks that the server
returns binary (not an HTML preview page), writes to a temp file, and
renames on success so interrupted downloads leave no corrupt cache.

This will move to Azure ML Model Registry in Sprint 3 - but until
then, adding a model is a PR against this dict.

## Steps

### 1. Upload the .pth to Azure Blob

Ask the PO for blob container credentials, then:

```bash
az storage blob upload \
    --account-name <our-account> \
    --container-name model-weights \
    --name unet-v2.pth \
    --file ./best_model.pth
```

Note the public download URL.

### 2. (Optional) Record the SHA-256

Checksum verification is a TODO for Sprint 3. You can record it in a
comment for future use:

```bash
sha256sum best_model.pth
# e.g. a3f1... best_model.pth
```

### 3. Open a PR adding the entry

```python
REGISTRY: dict[str, str] = {
    "unet-v1": "https://example.invalid/...?download=1",
    "unet-v2": "https://example.invalid/.../unet-v2.pth?download=1",
}
```

### 4. Update the default (optional)

If `unet-v2` should be the new default when no `--version` is
specified, keep it first in the dict. Python dicts preserve insertion
order, and the CLI uses `next(iter(REGISTRY))` as fallback.

### 5. Test

```bash
uv run cv-pipeline infer --image test.png --output results/ --version unet-v2
```

On first run the weights are downloaded from the blob URL and cached in
`~/.cache/cv-pipeline/models/`. (SHA-256 verification is a Sprint 3
TODO - the download will fail loudly if the server returns HTML instead
of binary, preventing corrupt cache entries.)

## Sprint 3 migration notes

When we cut over to Azure ML Model Registry, this procedure is
replaced by:

```bash
az ml model register \
    --name unet \
    --version 2 \
    --path ./best_model.pth
```

The Python registry dict becomes a thin cache populated at startup
from the ML API - no PR needed to add new versions.
