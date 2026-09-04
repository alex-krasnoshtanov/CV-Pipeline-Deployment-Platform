# Call the inference API

The FastAPI backend exposes `POST /infer` and `GET /health`. Every
endpoint except `/health` requires a valid `X-API-Key` header.

## Prerequisites

- Backend running at `http://localhost:8000` (see
  {doc}`../tutorials/quickstart` step 5 for how to start it).
- Your `API_KEY` exported as an environment variable.

## From the command line

```bash
curl -X POST http://localhost:8000/infer \
    -H "X-API-Key: $API_KEY" \
    -F "image=@plate.png" \
    -F "plate_id=PL-2024-001" \
    | python -m json.tool
```

`-F "image=@plate.png"` is the multipart upload. The `@` is important -
it tells curl to read the file, not send the string "plate.png".

## From Python (requests)

```python
import os
import requests

API_KEY = os.environ["API_KEY"]
API_URL = "http://localhost:8000"

with open("plate.png", "rb") as f:
    response = requests.post(
        f"{API_URL}/infer",
        headers={"X-API-Key": API_KEY},
        files={"image": ("plate.png", f, "image/png")},
        data={
            "plate_id": "PL-2024-001",
            "experiment_id": "EXP-NPEC-42",
        },
        timeout=30,
    )

response.raise_for_status()
result = response.json()
print(f"Detected {result['landmark_count']} landmarks")
```

## From Python (httpx async)

```python
import asyncio
import os
import httpx


async def infer(image_path: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        with open(image_path, "rb") as f:
            response = await client.post(
                "http://localhost:8000/infer",
                headers={"X-API-Key": os.environ["API_KEY"]},
                files={"image": f},
            )
    response.raise_for_status()
    return response.json()


result = asyncio.run(infer("plate.png"))
```

## Interactive Swagger UI

Start the backend and open <http://localhost:8000/docs>.
Swagger UI lets you authorize with your API key (green "Authorize"
button top right) and run every endpoint directly from the browser.

## Handling errors

All 4xx/5xx responses have the same envelope:

```json
{
  "error_code": "UNAUTHORIZED",
  "message": "Missing X-API-Key header.",
  "pipeline_version": "0.1.0",
  "timestamp": "2026-04-22T14:00:00+00:00",
  "request_id": "c3d4e5-..."
}
```

Robust Python client pattern:

```python
response = requests.post(...)
if response.status_code >= 400:
    err = response.json()
    match err["error_code"]:
        case "UNAUTHORIZED":
            raise RuntimeError("API key rejected - check .env")
        case "IMAGE_TOO_SMALL":
            raise ValueError(f"Image too small: {err['message']}")
        case "MODEL_NOT_READY":
            # Retry after a few seconds - the server is still loading.
            time.sleep(5)
        case _:
            raise RuntimeError(f"{err['error_code']}: {err['message']}")
result = response.json()
```

Full error code list:
{doc}`../explanation/error-codes`.

## Tracing requests

Every response carries `X-Request-ID` in its headers. Log it next
to any error you hit - the backend logs are indexed by the same ID
and will give you the full server-side trace:

```python
response = requests.post(...)
log.info("inference_request", request_id=response.headers["X-Request-ID"])
```
