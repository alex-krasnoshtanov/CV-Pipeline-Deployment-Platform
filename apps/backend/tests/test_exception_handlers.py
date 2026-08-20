"""Unit tests for the global exception handlers.

Exercises all branches of http_exception_handler, validation_exception_handler,
and unhandled_exception_handler via a minimal FastAPI app.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from api.middleware.exception_handlers import install  # noqa: E402

_app = FastAPI()
install(_app)


@_app.get("/raise-http-structured")
async def _raise_http_structured():
    raise StarletteHTTPException(
        status_code=404,
        detail={"error_code": "NOT_FOUND", "message": "Resource not found"},
    )


@_app.get("/raise-http-plain")
async def _raise_http_plain():
    """Raise an HTTPException with a plain-string detail (not a dict)."""
    raise StarletteHTTPException(status_code=403, detail="Access forbidden")


@_app.get("/raise-unhandled")
async def _raise_unhandled():
    raise RuntimeError("Something went very wrong")


@_app.get("/raise-validation")
async def _raise_validation(q: int):  # int parameter forces validation
    return {"q": q}


@pytest.fixture(autouse=True)
def _patch_pipeline_version():
    """Avoid importing cv_pipeline just to get its version string."""
    with patch("api.middleware.exception_handlers._pipeline_version", "test-version"):
        yield


@pytest.fixture
def client():
    with TestClient(_app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# http_exception_handler — structured-detail branch (lines 54-64)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_http_exception_structured_detail(client: TestClient):
    response = client.get("/raise-http-structured")
    assert response.status_code == 404
    body = response.json()
    assert body["error_code"] == "NOT_FOUND"
    assert body["message"] == "Resource not found"
    assert body["pipeline_version"] == "test-version"
    assert "detail" in body


# ---------------------------------------------------------------------------
# http_exception_handler — plain-string branch (lines 65-74, previously uncovered)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_http_exception_plain_string_detail(client: TestClient):
    """Plain string detail hits the else branch and wraps in HTTP_ERROR."""
    response = client.get("/raise-http-plain")
    assert response.status_code == 403
    body = response.json()
    assert body["error_code"] == "HTTP_ERROR"
    assert body["message"] == "Access forbidden"
    assert body["pipeline_version"] == "test-version"


# ---------------------------------------------------------------------------
# unhandled_exception_handler (lines 102-103, previously uncovered)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unhandled_exception_returns_500(client: TestClient):
    """Unhandled exceptions reach the backstop and return a structured 500."""
    response = client.get("/raise-unhandled")
    assert response.status_code == 500
    body = response.json()
    assert body["error_code"] == "INTERNAL_SERVER_ERROR"
    assert "unexpected error" in body["message"].lower()
    assert body["pipeline_version"] == "test-version"


# ---------------------------------------------------------------------------
# validation_exception_handler
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validation_error_returns_422(client: TestClient):
    response = client.get("/raise-validation", params={"q": "not-an-int"})
    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "VALIDATION_ERROR"
    assert "validation" in body["message"].lower()
