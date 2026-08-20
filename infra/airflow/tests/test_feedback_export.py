"""Unit tests for the feedback bridge helpers.

These cover the pure logic and the DB/blob wrappers in
``feedback_export`` with mocked psycopg2 connections and a mocked Azure
``ContainerClient`` — no database or storage account is contacted. The
Airflow DAG modules themselves are not imported here (they require an
Airflow runtime); the testable logic lives in ``feedback_export``.
"""

from __future__ import annotations

import base64
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import feedback_export as fx
import pytest

_PNG = b"\x89PNG\r\n\x1a\nMASKBYTES"
_MASK_B64 = base64.b64encode(_PNG).decode()
_URI = "azureml://datastores/workspaceblobstore/paths/feedback/raw/u/p.png"


def _row(prediction_id, predicted, corrected, uri=_URI, fid=None):
    """Build a fetch_good_set-style dict row."""
    return {
        "feedback_id": fid or uuid.uuid4(),
        "prediction_id": prediction_id,
        "image_uri": uri,
        "predicted_mask_b64": predicted,
        "corrected_mask_b64": corrected,
        "flag": "good" if corrected is None else "bad",
        "created_at": None,
    }


def _conn_with_cursor() -> tuple[MagicMock, MagicMock]:
    """Return (conn, cursor) where conn.cursor() is a context manager."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn, cursor


class TestSelectLabels:
    """Label selection: corrected over predicted, latest per prediction."""

    def test_prefers_corrected_over_predicted(self) -> None:
        """A corrected mask is used as the label when present."""
        pid = uuid.uuid4()
        rows = [_row(pid, "PRED", "CORR")]

        pairs, ids = fx.select_labels(rows)

        assert pairs[pid] == (_URI, "CORR")
        assert len(ids) == 1

    def test_uses_predicted_when_no_correction(self) -> None:
        """A 'good' row (no correction) labels with the predicted mask."""
        pid = uuid.uuid4()

        pairs, _ = fx.select_labels([_row(pid, "PRED", None)])

        assert pairs[pid] == (_URI, "PRED")

    def test_latest_per_prediction_wins(self) -> None:
        """A later row supersedes an earlier one for the same prediction."""
        pid = uuid.uuid4()
        rows = [_row(pid, "PRED", "OLD"), _row(pid, "PRED", "NEW")]

        pairs, ids = fx.select_labels(rows)

        assert pairs[pid] == (_URI, "NEW")
        assert len(ids) == 2  # both stamped, even the superseded one


class TestBlobKeyFromUri:
    """Parsing the in-container blob key from a datastore URI."""

    def test_extracts_key_after_paths(self) -> None:
        """The key is everything after the /paths/ marker."""
        assert fx.blob_key_from_uri(_URI) == "feedback/raw/u/p.png"

    @pytest.mark.parametrize(
        "bad", ["/data/local/p.png", "https://x/p.png", "azureml://no-paths"]
    )
    def test_rejects_non_datastore_uri(self, bad: str) -> None:
        """A non-datastore URI raises ValueError."""
        with pytest.raises(ValueError):
            fx.blob_key_from_uri(bad)


class TestStagePairs:
    """Materialising image/mask pairs into the staging layout."""

    def _container(self, image_bytes: bytes = b"IMG") -> MagicMock:
        container = MagicMock()
        container.download_blob.return_value.readall.return_value = image_bytes
        return container

    def test_writes_image_and_mask(self, tmp_path: Path) -> None:
        """A valid pair lands as images/<id>.png and masks/<id>.png."""
        pid = uuid.uuid4()
        pairs = {pid: (_URI, _MASK_B64)}

        staged = fx.stage_pairs(self._container(b"IMAGE"), pairs, tmp_path)

        assert staged == 1
        assert (tmp_path / "images" / f"{pid}.png").read_bytes() == b"IMAGE"
        assert (tmp_path / "masks" / f"{pid}_root_mask.png").read_bytes() == _PNG

    def test_skips_bad_uri(self, tmp_path: Path) -> None:
        """A non-datastore image URI is skipped, not fatal."""
        pid = uuid.uuid4()
        pairs = {pid: ("/local/path.png", _MASK_B64)}

        assert fx.stage_pairs(self._container(), pairs, tmp_path) == 0

    def test_skips_bad_base64_mask(self, tmp_path: Path) -> None:
        """An invalid base64 mask is skipped."""
        pid = uuid.uuid4()
        pairs = {pid: (_URI, "not!base64!")}

        assert fx.stage_pairs(self._container(), pairs, tmp_path) == 0

    def test_skips_unreadable_blob(self, tmp_path: Path) -> None:
        """A blob download failure is logged and skipped."""
        pid = uuid.uuid4()
        container = MagicMock()
        container.download_blob.side_effect = RuntimeError("404")

        assert fx.stage_pairs(container, {pid: (_URI, _MASK_B64)}, tmp_path) == 0


class TestDbWrappers:
    """fetch_good_set, count_pending, and mark_exported against a mock cursor."""

    def test_fetch_good_set_maps_rows_to_dicts(self) -> None:
        """Cursor description names become dict keys for each row."""
        conn, cursor = _conn_with_cursor()
        cursor.description = [
            SimpleNamespace(name="feedback_id"),
            SimpleNamespace(name="prediction_id"),
        ]
        cursor.fetchall.return_value = [(1, 2), (3, 4)]

        rows = fx.fetch_good_set(conn)

        assert rows == [
            {"feedback_id": 1, "prediction_id": 2},
            {"feedback_id": 3, "prediction_id": 4},
        ]

    def test_count_pending_returns_int(self) -> None:
        """count_pending returns the scalar from COUNT(*)."""
        conn, cursor = _conn_with_cursor()
        cursor.fetchone.return_value = (7,)

        assert fx.count_pending(conn) == 7

    def test_mark_exported_updates_and_commits(self) -> None:
        """A non-empty id list issues one UPDATE and commits."""
        conn, cursor = _conn_with_cursor()
        ids = [uuid.uuid4(), uuid.uuid4()]

        fx.mark_exported(conn, ids)

        cursor.execute.assert_called_once()
        assert cursor.execute.call_args.args[1] == (ids,)
        conn.commit.assert_called_once()

    def test_mark_exported_noop_on_empty(self) -> None:
        """An empty id list does nothing — no query, no commit."""
        conn, _ = _conn_with_cursor()

        fx.mark_exported(conn, [])

        conn.commit.assert_not_called()
