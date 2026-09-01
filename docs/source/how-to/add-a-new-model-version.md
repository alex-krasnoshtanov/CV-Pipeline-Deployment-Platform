# Add a new model version

When retraining produces a better checkpoint, register it in the model
registry so the CLI and the API can serve it without a code change to
any call site.

## How the registry works

The registry is a Python dict in
`packages/cv-pipeline/src/cv_pipeline/weights.py` mapping a version
string to a download URL:

```python
REGISTRY: dict[str, str] = {
    "unet-v1": (
        "https://github.com/Gfgf96/CV-Pipeline-Deployment-Platform/"
        "releases/download/weights%2Funet-v1/unet-v1.pth"
    ),
}
```

Weights are published as **GitHub Release assets**. The release download
endpoint streams `application/octet-stream` directly and needs no
credentials for a public repository, which keeps first-run setup to a
single `uv run` with nothing to configure.

`get_weights()` downloads on first use and caches under
`~/.cache/cv-pipeline/models/` (override with `CV_PIPELINE_CACHE_DIR`).
The download writes to a temp file and renames on success, so an
interrupted download cannot leave a corrupt file in the cache.

## Steps

### 1. Publish the checkpoint as a release asset

Tag the release under a `weights/` prefix so model releases sort apart
from application releases:

```bash
gh release create weights/unet-v2 \
    --title "unet-v2 weights" \
    --notes "U-Net checkpoint. See docs for training configuration." \
    ./best_model.pth#unet-v2.pth
```

The `#unet-v2.pth` suffix sets the asset filename, which is what the
download URL resolves against.

### 2. Record the SHA-256

Include it in the release notes so a consumer can verify the download:

```bash
sha256sum best_model.pth
```

### 3. Open a PR adding the entry

```python
REGISTRY: dict[str, str] = {
    "unet-v1": (
        "https://github.com/Gfgf96/CV-Pipeline-Deployment-Platform/"
        "releases/download/weights%2Funet-v1/unet-v1.pth"
    ),
    "unet-v2": (
        "https://github.com/Gfgf96/CV-Pipeline-Deployment-Platform/"
        "releases/download/weights%2Funet-v2/unet-v2.pth"
    ),
}
```

Note the `%2F`: the tag contains a slash, which has to stay
percent-encoded inside the URL path.

### 4. Update the default (optional)

If `unet-v2` should be the default when no `--version` is given, keep it
first in the dict. Python dicts preserve insertion order and the CLI
uses `next(iter(REGISTRY))` as its fallback.

### 5. Test

```bash
uv run cv-pipeline infer --image test.png --output results/ --version unet-v2
```

On first run the weights download and cache; subsequent runs reuse the
cached file.

## Why a dict rather than a model registry

A dict is import-time typo-checked, reviewable in a PR diff, and has no
runtime dependency on a cloud control plane — a consumer who only wants
inference does not need an Azure subscription to resolve a version.

Where an Azure ML workspace is available, `MODEL_ENDPOINT_URL` bypasses
this path entirely: the backend delegates inference to the managed
endpoint and never downloads weights. The registry is the local and
on-premise story; the endpoint is the cloud one.
