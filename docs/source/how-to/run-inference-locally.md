# Run inference on a local image

Use the `cv-pipeline infer` CLI to process a single image without
starting a server.

## One-liner

Use `--version` to let the pipeline auto-download the checkpoint on
first run (cached under `~/.cache/cv-pipeline/models/`):

```bash
uv run cv-pipeline infer \
    --image path/to/plate.png \
    --output results/ \
    --version unet-v1
```

If you already have a local checkpoint file, pass it directly with
`--model`:

```bash
uv run cv-pipeline infer \
    --image path/to/plate.png \
    --output results/ \
    --model path/to/checkpoint.pth
```

Outputs:

- `results/plate_mask.png` - binary mask of root tissue
- `results/plate_result.json` - full response matching
  section 4 of the [CV pipeline specification](../reference/specification.md)

## Batch processing

There is no native `batch` subcommand yet. Loop
with your shell:

```bash
for img in data/plates/*.tif; do
    uv run cv-pipeline infer --image "$img" --output results/
done
```

## Attaching metadata

The Metadata dictionary is stored in the JSON output and is useful
for NPEC experiments where plates are tagged by ID:

```bash
uv run cv-pipeline infer \
    --image plate.png \
    --output results/ \
    --plate-id PL-2024-001 \
    --experiment-id EXP-NPEC-42 \
    --timestamp 2026-04-20T09:00:00Z
```

All three metadata flags are optional.

## Disabling the Petri-dish crop

By default the pipeline finds the Petri dish edge and crops to it.
For HADES plates where the dish fills the frame, skip the detection:

```bash
uv run cv-pipeline infer --image plate.png --output results/ --no-crop
```

## Tweaking the mask threshold

The default `--threshold 0.5` binarises the probability mask at 0.5.
Lower values keep more tissue (risk of false positives); higher
values are stricter:

```bash
uv run cv-pipeline infer --image plate.png --output results/ --threshold 0.65
```

## Common errors

:::{list-table}
:header-rows: 1

* - Error code
  - What happened
  - What to do
* - `IMAGE_TOO_SMALL`
  - Image is under 256x256
  - Upscale or use a different plate
* - `UNSUPPORTED_FILE_TYPE`
  - Extension not in .png/.jpg/.jpeg/.tif/.tiff
  - Convert the image first
* - `MODEL_NOT_READY`
  - No `--version`/`--model` and weights not cached
  - Pass `--version unet-v1` to trigger auto-download
* - `CORRUPT_FILE`
  - Path doesn't exist or isn't readable
  - Check the path - the error message tells you which one
:::

See {doc}`../explanation/error-codes` for the full spec.
