"""Unit tests for the Azure ML endpoint client (cloud serving path).

These cover the three units in ``api.services.endpoint_client``:

- ``_build_payload``: image + metadata to JSON request bytes.
- ``_call_endpoint``: the synchronous scoring call, including the
  success path, the double-encoded-JSON unwrap, missing configuration,
  and the HTTP/URL/JSON failure modes.
- ``run_endpoint_inference``: the async wrapper that offloads the
  blocking call to a worker thread.

No real endpoint is contacted. ``urllib.request.urlopen`` is patched at
``urllib.request.urlopen`` (the module calls it via ``import
urllib.request``), and the schema collaborator is patched at
``api.services.endpoint_client.InferenceResult`` so these tests stay
decoupled from the schema's own validation, which is covered in the
cv-pipeline test suite.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from api.services import endpoint_client
from cv_pipeline.schema import Metadata

_URL = "https://cradle.example/score"
_KEY = "endpoint-secret"


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    """Write a small byte blob to a temp .png and return its path."""
    path = tmp_path / "plate_001.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nFAKEIMAGEBYTES")
    return path


@pytest.fixture
def endpoint_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set both endpoint env vars so _call_endpoint passes its config check."""
    monkeypatch.setenv("MODEL_ENDPOINT_URL", _URL)
    monkeypatch.setenv("MODEL_ENDPOINT_KEY", _KEY)


def _urlopen_cm(body: bytes) -> MagicMock:
    """Return a urlopen mock usable as a context manager yielding ``body``.

    The module reads the response via ``with urlopen(...) as response:``
    so the mock must support the context-manager protocol and expose a
    ``read()`` returning the raw bytes.
    """
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = body
    cm.__exit__.return_value = False
    return cm


@pytest.mark.unit
class TestBuildPayload:
    """_build_payload serialises the image and metadata to JSON bytes."""

    def test_encodes_image_and_metadata(self, image_file: Path) -> None:
        """Payload carries the base64 image, filename, and both identifiers."""
        meta = Metadata(plate_id="PL-1", experiment_id="EXP-1")

        payload = json.loads(endpoint_client._build_payload(image_file, meta))

        assert (
            payload["image_b64"] == base64.b64encode(image_file.read_bytes()).decode()
        )
        assert payload["filename"] == "plate_001.png"
        assert payload["plate_id"] == "PL-1"
        assert payload["experiment_id"] == "EXP-1"

    def test_null_metadata_serialises_as_none(self, image_file: Path) -> None:
        """Absent identifiers appear as JSON null, not as missing keys."""
        payload = json.loads(endpoint_client._build_payload(image_file, Metadata()))

        assert payload["plate_id"] is None
        assert payload["experiment_id"] is None


@pytest.mark.unit
class TestCallEndpoint:
    """_call_endpoint POSTs to the endpoint and parses the response."""

    @pytest.mark.parametrize(
        ("url", "key"),
        [(None, None), (_URL, None), (None, _KEY)],
    )
    def test_missing_config_raises(
        self,
        image_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        url: str | None,
        key: str | None,
    ) -> None:
        """Either env var absent raises before any network call is made."""
        monkeypatch.delenv("MODEL_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("MODEL_ENDPOINT_KEY", raising=False)
        if url:
            monkeypatch.setenv("MODEL_ENDPOINT_URL", url)
        if key:
            monkeypatch.setenv("MODEL_ENDPOINT_KEY", key)

        with pytest.raises(RuntimeError, match="must both be set"):
            endpoint_client._call_endpoint(image_file, Metadata())

    def test_success_returns_parsed_result(
        self, image_file: Path, endpoint_env: None
    ) -> None:
        """A 2xx JSON body is parsed and handed to InferenceResult.from_dict."""
        result_dict = {"mask_b64": "AAAA", "landmark_count": 0, "landmarks": []}
        cm = _urlopen_cm(json.dumps(result_dict).encode())

        with (
            patch("urllib.request.urlopen", return_value=cm) as mock_open,
            patch("api.services.endpoint_client.InferenceResult") as mock_ir,
        ):
            out = endpoint_client._call_endpoint(image_file, Metadata())

        mock_ir.from_dict.assert_called_once_with(result_dict)
        assert out is mock_ir.from_dict.return_value

        # The outgoing request carries the bearer key and is a POST.
        sent = mock_open.call_args.args[0]
        assert sent.get_full_url() == _URL
        assert sent.get_header("Authorization") == f"Bearer {_KEY}"
        assert sent.method == "POST"

    def test_double_encoded_json_is_unwrapped(
        self, image_file: Path, endpoint_env: None
    ) -> None:
        """A JSON string that itself contains JSON is unwrapped one level."""
        inner = {"mask_b64": "AAAA", "landmark_count": 0, "landmarks": []}
        body = json.dumps(json.dumps(inner)).encode()
        cm = _urlopen_cm(body)

        with (
            patch("urllib.request.urlopen", return_value=cm),
            patch("api.services.endpoint_client.InferenceResult") as mock_ir,
        ):
            endpoint_client._call_endpoint(image_file, Metadata())

        mock_ir.from_dict.assert_called_once_with(inner)

    def test_http_error_becomes_runtime_error(
        self, image_file: Path, endpoint_env: None
    ) -> None:
        """An HTTP error status is surfaced as a RuntimeError with the code."""
        err = urllib.error.HTTPError(_URL, 500, "Server Error", {}, io.BytesIO(b"boom"))

        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(RuntimeError, match="500"):
                endpoint_client._call_endpoint(image_file, Metadata())

    def test_url_error_becomes_runtime_error(
        self, image_file: Path, endpoint_env: None
    ) -> None:
        """A transport failure (no HTTP reply) raises an unreachable error."""
        err = urllib.error.URLError("tunnel down")

        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(RuntimeError, match="unreachable"):
                endpoint_client._call_endpoint(image_file, Metadata())

    def test_non_json_response_becomes_runtime_error(
        self, image_file: Path, endpoint_env: None
    ) -> None:
        """A non-JSON body raises rather than propagating a decode error."""
        cm = _urlopen_cm(b"<html>not json</html>")

        with patch("urllib.request.urlopen", return_value=cm):
            with pytest.raises(RuntimeError, match="non-JSON"):
                endpoint_client._call_endpoint(image_file, Metadata())


@pytest.mark.unit
class TestRunEndpointInference:
    """The async wrapper delegates to _call_endpoint on a worker thread."""

    def test_delegates_to_call_endpoint(self, image_file: Path) -> None:
        """The wrapper returns the underlying result and forwards its args."""
        sentinel = object()
        meta = Metadata(plate_id="PL-9")

        with patch.object(
            endpoint_client, "_call_endpoint", return_value=sentinel
        ) as mock_call:
            out = asyncio.run(endpoint_client.run_endpoint_inference(image_file, meta))

        assert out is sentinel
        mock_call.assert_called_once_with(image_file, meta)
