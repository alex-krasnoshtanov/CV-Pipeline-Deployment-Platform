# Error codes

Every 4xx/5xx response - from both the CLI and the API - carries a
stable `error_code` alongside the human-readable `message`.
Clients should branch on the code, never on the message text (which
may be rephrased without notice).

Source of truth: section 5 of the API contract specification (coursework, not published here).

## The envelope

Every error response has this shape:

```json
{
  "error_code": "IMAGE_TOO_SMALL",
  "message": "Image 200x200 is below the 256x256 minimum.",
  "pipeline_version": "0.1.0",
  "timestamp": "2026-04-22T14:03:19+00:00",
  "request_id": "c3d4e5-..."
}
```

`request_id` echoes the `X-Request-ID` header and is useful for
cross-referencing server logs.

## Code table

```{list-table}
:widths: 20 10 40 30
:header-rows: 1

* - `error_code`
  - HTTP
  - When
  - What to do
* - `UNAUTHORIZED`
  - 401
  - Missing or invalid `X-API-Key` header
  - Check `.env`; make sure the client reads the same value
* - `FILE_TOO_LARGE`
  - 413
  - Upload exceeds 50 MB
  - Downscale before upload; HADES images at 4096x4096 are under the limit
* - `UNSUPPORTED_FILE_TYPE`
  - 422
  - Extension not in `.png/.jpg/.jpeg/.tif/.tiff`
  - Convert the file (`.bmp`, `.webp` not accepted)
* - `UNSUPPORTED_COLOR_MODE`
  - 422
  - Image is CMYK or palette-indexed
  - Re-save as RGB or grayscale
* - `IMAGE_TOO_SMALL`
  - 422
  - Either dimension under 256 px
  - The U-Net will not see enough features at that size
* - `IMAGE_TOO_LARGE`
  - 422
  - Either dimension over 8192 px
  - Even with patch-based inference this overflows practical RAM
* - `CORRUPT_FILE`
  - 422
  - Decoder cannot read the file, or path is not a file
  - Verify the file isn't truncated; recheck the path
* - `INTERNAL_SERVER_ERROR`
  - 500
  - Unexpected failure in the pipeline
  - Look up `request_id` in the backend logs; file an issue
* - `MODEL_NOT_READY`
  - 503
  - Backend is running but the model hasn't loaded yet
  - Retry after 5-10 s; startup usually takes < 30 s
```

## Why this design

We separated code from message because:

1. **Stable contract.** Translating messages, rewording errors for
   UX, or appending debug info never changes the code. Clients can
   build switch statements against codes without breaking on
   cosmetic changes.

2. **Programmatic handling.** A robotic platform should re-upload
   on `CORRUPT_FILE` (probably a transient SD-card read) but not
   on `IMAGE_TOO_SMALL` (a real user error). Switching on message
   substrings is fragile.

3. **Observability.** The same codes appear in backend logs and
   monitoring dashboards. Grep-ability across CLI stderr, API
   response bodies, and server logs is a feature.

## When a code doesn't cover the case

If the pipeline hits something we didn't anticipate (an OOM inside
torch, a segfault from opencv) - the global exception handler in
`api.middleware.exception_handlers` catches it and returns
`INTERNAL_SERVER_ERROR`. The full traceback goes to backend logs
with the same `request_id`, so you can still trace it.

We explicitly don't raise `INTERNAL_SERVER_ERROR` from our own
code. It's the signal of "something we didn't plan for" - if a new
failure mode becomes common, it gets its own code in a spec bump
(e.g. `MODEL_NOT_READY` was added in task_328 v0.2.1 after exactly
this pattern).

## Adding a new code

1. Add the row to task_328 section 5 in a minor version bump (e.g. 0.2.1 -> 0.2.2).
2. Add the code to `cv_pipeline.validation.ValidationError` or the
   appropriate raise site.
3. If HTTP-visible, add the mapping in `api.routers.infer._ERROR_STATUS`.
4. Add a test that exercises the new path.
5. Mention it in the error table above.

The whole round-trip is a 30-minute PR.
