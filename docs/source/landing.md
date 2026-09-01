# CV Pipeline

Computer vision pipeline for plant organ segmentation and root tip
detection on *Arabidopsis thaliana* seedling images, built as the
Built as a university group project in applied data science and AI.

This documentation is organised along the
[Diátaxis framework](https://diataxis.fr/): four pages for four
different needs.

```{list-table}
:widths: 30 70
:header-rows: 1

* - If you want to...
  - Read this
* - Get started with a worked example
  - {doc}`tutorials/quickstart`
* - Do a specific task step by step
  - {doc}`how-to/index`
* - Look up a function, class, or endpoint
  - {doc}`reference/index`
* - Understand why the system is built this way
  - {doc}`explanation/index`
```

---

## What this pipeline does

- **In:** one plant image (JPEG/PNG/TIFF, 256–8192 px per side, ≤50 MB).
- **Out:** a binary segmentation mask over root tissue, a list of root-tip
  landmarks with pixel coordinates, and confidence scores for each.

The code ships in three forms, all driven by the same package:

- **Library:** `import cv_pipeline; cv_pipeline.infer(image_path=..., model=...)`
- **CLI:** `cv-pipeline infer --image plate.png --output results/`
- **HTTP API:** `POST /infer` on the FastAPI backend (X-API-Key auth).

Full contract: see the CV pipeline specification §4.

## Current scope

| Phase | Delivered |
|---|---|
| 1 | Package plan, architecture diagrams, API specification |
| 2 | cv-pipeline package (U-Net + landmarks), FastAPI service, CLI, Docker, local deploy |
| 3 | Azure ML data pipelines + cloud training (Airflow DAGs running Azure ML jobs) |
| 4 | Azure deployment + monitoring + feedback loop |
| 5 | Integration testing, demo, polish |
