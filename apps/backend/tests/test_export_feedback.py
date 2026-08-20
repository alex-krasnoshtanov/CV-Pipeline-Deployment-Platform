"""Unit tests for the corrected-feedback staging producer.

The pure helpers (URL rewrite, dedupe, image resolution, pair writing)
are tested against a real temp filesystem and real base64. The
``export_feedback`` orchestrator is tested with a mocked session so no
database is required — matching the project's service-test style.
"""

from __future__ import annotations

import base64
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from api.tools import export_feedback as ef

_PNG = b"\x89PNG\r\n\x1a\nMASKBYTES"
_MASK_B64 = base64.b64encode(_PNG).decode()


def _row(prediction_id, mask_b64, image_uri, created):
    """Build a select row tuple matching _select_pending's column order."""
    return (uuid.uuid4(), prediction_id, mask_b64, created, image_uri)


@pytest.mark.unit
class TestPureHelpers:
    """URL rewrite, dedupe, image resolution, and pair writing."""

    def test_sync_url_strips_async_driver(self) -> None:
        """The asyncpg suffix is removed for the sync engine."""
        assert (
            ef._sync_url("postgresql+asyncpg://u:p@h:5432/db")
            == "postgresql://u:p@h:5432/db"
        )

    def test_latest_per_prediction_keeps_newest(self) -> None:
        """A second correction of the same prediction supersedes the first."""
        pid = uuid.uuid4()
        older = datetime(2026, 1, 1, tzinfo=timezone.utc)
        newer = datetime(2026, 1, 2, tzinfo=timezone.utc)
        rows = [
            _row(pid, "OLD", "/img/a.png", older),
            _row(pid, "NEW", "/img/b.png", newer),
        ]

        latest = ef._latest_per_prediction(rows)

        assert latest == {pid: ("NEW", "/img/b.png")}

    def test_resolve_image_bytes_reads_local_file(self, tmp_path: Path) -> None:
        """A local filesystem image_uri is read back verbatim."""
        img = tmp_path / "plate.png"
        img.write_bytes(b"IMAGE")

        assert ef._resolve_image_bytes(str(img)) == b"IMAGE"

    @pytest.mark.parametrize(
        "uri",
        [None, "", "azureml://datastores/workspaceblobstore/paths/x.png"],
    )
    def test_resolve_image_bytes_skips_unreadable(self, uri) -> None:
        """Missing, empty, or blob URIs resolve to None, not an error."""
        assert ef._resolve_image_bytes(uri) is None

    def test_resolve_image_bytes_missing_file_returns_none(
        self, tmp_path: Path
    ) -> None:
        """A path that does not exist resolves to None."""
        assert ef._resolve_image_bytes(str(tmp_path / "nope.png")) is None

    def test_write_pair_writes_both_files(self, tmp_path: Path) -> None:
        """A valid pair lands as images/<id>.png and masks/<id>.png."""
        (tmp_path / "images").mkdir()
        (tmp_path / "masks").mkdir()
        pid = uuid.uuid4()

        ok = ef._write_pair(tmp_path, pid, b"IMG", _MASK_B64)

        assert ok is True
        assert (tmp_path / "images" / f"{pid}.png").read_bytes() == b"IMG"
        assert (tmp_path / "masks" / f"{pid}_root_mask.png").read_bytes() == _PNG

    def test_write_pair_rejects_bad_base64(self, tmp_path: Path) -> None:
        """An invalid base64 mask is skipped and reported as failure."""
        (tmp_path / "images").mkdir()
        (tmp_path / "masks").mkdir()

        ok = ef._write_pair(tmp_path, uuid.uuid4(), b"IMG", "not!base64!")

        assert ok is False


@pytest.mark.unit
class TestExportFeedback:
    """The orchestrator: select, write, stamp, commit."""

    def _session_returning(self, rows: list[tuple]) -> MagicMock:
        """Mock a session whose first execute() returns rows, second is the update."""
        session = MagicMock()
        select_result = MagicMock()
        select_result.all.return_value = rows
        session.execute.side_effect = [select_result, MagicMock()]
        return session

    def test_writes_pairs_and_stamps(self, tmp_path: Path) -> None:
        """A valid corrected row is written and all rows are stamped + committed."""
        img = tmp_path / "src.png"
        img.write_bytes(b"IMAGE")
        pid = uuid.uuid4()
        rows = [_row(pid, _MASK_B64, str(img), datetime.now(timezone.utc))]
        session = self._session_returning(rows)

        written = ef.export_feedback(session, tmp_path / "stage")

        assert written == 1
        assert (tmp_path / "stage" / "images" / f"{pid}.png").exists()
        assert (tmp_path / "stage" / "masks" / f"{pid}_root_mask.png").exists()
        # select + update were both issued, and the transaction committed.
        assert session.execute.call_count == 2
        session.commit.assert_called_once()

    def test_empty_queue_is_noop(self, tmp_path: Path) -> None:
        """No pending rows means no write, no update, no commit."""
        session = MagicMock()
        empty = MagicMock()
        empty.all.return_value = []
        session.execute.return_value = empty

        written = ef.export_feedback(session, tmp_path / "stage")

        assert written == 0
        session.commit.assert_not_called()
        assert session.execute.call_count == 1  # only the select ran

    def test_blob_uri_row_is_stamped_but_not_written(self, tmp_path: Path) -> None:
        """A blob-backed row writes nothing locally but is still consumed."""
        pid = uuid.uuid4()
        blob = "azureml://datastores/workspaceblobstore/paths/feedback/raw/x.png"
        rows = [_row(pid, _MASK_B64, blob, datetime.now(timezone.utc))]
        session = self._session_returning(rows)

        written = ef.export_feedback(session, tmp_path / "stage")

        assert written == 0
        assert not (tmp_path / "stage" / "images" / f"{pid}.png").exists()
        # still stamped + committed so it does not re-surface every run.
        assert session.execute.call_count == 2
        session.commit.assert_called_once()
