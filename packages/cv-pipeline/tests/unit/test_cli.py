"""Unit tests for cv_pipeline.cli argument parsing and helpers."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from cv_pipeline.cli import _build_parser, _exit_with_error, _setup_logging, _write_mask
from cv_pipeline.schema import InferenceResult, Landmark, Metadata
from PIL import Image


def _make_test_image(tmp_path: Path, name: str = "test.png", size: int = 300) -> Path:
    """Create a valid grayscale test image on disk."""
    img = Image.fromarray(np.zeros((size, size), dtype=np.uint8))
    path = tmp_path / name
    img.save(path)
    return path


def _fake_result(filename: str = "test.png") -> InferenceResult:
    """Build a deterministic fake inference result."""
    return InferenceResult(
        pipeline_version="0.1.0",
        model_version="unet-v1",
        timestamp="2026-05-01T12:00:00Z",
        image_filename=filename,
        image_width_px=300,
        image_height_px=300,
        metadata=Metadata(),
        mask_b64=base64.b64encode(b"fake-mask-png").decode("ascii"),
        mask_confidence=0.85,
        landmark_count=1,
        landmarks=[Landmark(id=0, x=150, y=200, confidence=0.9)],
    )


# ---- parser tests: infer subcommand ---------------------------------


@pytest.mark.unit
class TestBuildParserInfer:
    """Tests for CLI infer subcommand argument parsing."""

    def test_infer_subcommand_exists(self) -> None:
        """Parser should recognise the infer subcommand."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "infer",
                "--image",
                "test.png",
                "--output",
                "out/",
                "--model",
                "model.pth",
            ]
        )

        assert args.command == "infer"
        assert args.image == Path("test.png")
        assert args.output == Path("out/")
        assert args.model == Path("model.pth")

    def test_version_subcommand_exists(self) -> None:
        """Parser should recognise the version subcommand."""
        parser = _build_parser()
        args = parser.parse_args(["version"])

        assert args.command == "version"

    def test_infer_default_threshold(self) -> None:
        """Default threshold should be 0.5."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "infer",
                "--image",
                "test.png",
                "--output",
                "out/",
                "--model",
                "model.pth",
            ]
        )

        assert args.threshold == 0.5

    def test_infer_custom_threshold(self) -> None:
        """Custom threshold should be parsed correctly."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "infer",
                "--image",
                "test.png",
                "--output",
                "out/",
                "--model",
                "model.pth",
                "--threshold",
                "0.7",
            ]
        )

        assert args.threshold == 0.7

    def test_infer_no_crop_flag(self) -> None:
        """--no-crop flag should set no_crop to True."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "infer",
                "--image",
                "test.png",
                "--output",
                "out/",
                "--model",
                "model.pth",
                "--no-crop",
            ]
        )

        assert args.no_crop is True

    def test_infer_optional_metadata_flags(self) -> None:
        """Optional metadata flags should be parsed."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "infer",
                "--image",
                "test.png",
                "--output",
                "out/",
                "--model",
                "model.pth",
                "--plate-id",
                "PL-001",
                "--experiment-id",
                "EXP-42",
                "--timestamp",
                "2026-04-17T12:00:00Z",
            ]
        )

        assert args.plate_id == "PL-001"
        assert args.experiment_id == "EXP-42"
        assert args.timestamp == "2026-04-17T12:00:00Z"

    def test_infer_metadata_defaults_to_none(self) -> None:
        """Optional metadata should default to None."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "infer",
                "--image",
                "test.png",
                "--output",
                "out/",
                "--model",
                "model.pth",
            ]
        )

        assert args.plate_id is None
        assert args.experiment_id is None
        assert args.timestamp is None


# ---- parser tests: train subcommand ---------------------------------


@pytest.mark.unit
class TestBuildParserTrain:
    """Tests for CLI train subcommand argument parsing."""

    def test_train_subcommand_exists(self) -> None:
        """Parser should recognise the train subcommand with required args."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "train",
                "--data-dir",
                "data/train",
                "--val-dir",
                "data/val",
                "--output-dir",
                "models/",
            ]
        )

        assert args.command == "train"
        assert args.data_dir == Path("data/train")
        assert args.val_dir == Path("data/val")
        assert args.output_dir == Path("models/")

    def test_train_default_epochs(self) -> None:
        """Default epochs should be 50."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "train",
                "--data-dir",
                "data/train",
                "--val-dir",
                "data/val",
                "--output-dir",
                "models/",
            ]
        )

        assert args.epochs == 50

    def test_train_default_batch_size(self) -> None:
        """Default batch_size should be 16."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "train",
                "--data-dir",
                "data/train",
                "--val-dir",
                "data/val",
                "--output-dir",
                "models/",
            ]
        )

        assert args.batch_size == 16

    def test_train_default_learning_rate(self) -> None:
        """Default lr should be 1e-4."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "train",
                "--data-dir",
                "data/train",
                "--val-dir",
                "data/val",
                "--output-dir",
                "models/",
            ]
        )

        assert args.lr == pytest.approx(1e-4)

    def test_train_custom_hyperparameters(self) -> None:
        """Custom training hyperparameters should be parsed correctly."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "train",
                "--data-dir",
                "data/train",
                "--val-dir",
                "data/val",
                "--output-dir",
                "models/",
                "--epochs",
                "10",
                "--batch-size",
                "8",
                "--lr",
                "0.001",
                "--device",
                "cpu",
                "--run-name",
                "experiment-001",
            ]
        )

        assert args.epochs == 10
        assert args.batch_size == 8
        assert args.lr == pytest.approx(0.001)
        assert args.device == "cpu"
        assert args.run_name == "experiment-001"

    def test_train_run_name_defaults_to_none(self) -> None:
        """run_name should default to None when not provided."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "train",
                "--data-dir",
                "data/train",
                "--val-dir",
                "data/val",
                "--output-dir",
                "models/",
            ]
        )

        assert args.run_name is None


# ---- helpers ---------------------------------------------------------


@pytest.mark.unit
class TestWriteMask:
    """Tests for the mask writing helper."""

    def test_writes_decodable_png(self, tmp_path: Path) -> None:
        """Written mask should be a valid file."""
        png_bytes = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        mask_b64 = png_bytes.decode("ascii")
        path = tmp_path / "mask.png"

        _write_mask(mask_b64, path)

        assert path.exists()
        assert path.stat().st_size > 0

    def test_writes_correct_bytes(self, tmp_path: Path) -> None:
        """Written bytes should match the decoded base64."""
        original = b"test data for mask"
        mask_b64 = base64.b64encode(original).decode("ascii")
        path = tmp_path / "mask.bin"

        _write_mask(mask_b64, path)

        assert path.read_bytes() == original


@pytest.mark.unit
class TestExitWithError:
    """Tests for the error exit helper."""

    def test_exits_with_code_1(self) -> None:
        """_exit_with_error should raise SystemExit with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            _exit_with_error(
                error_code="TEST_ERROR",
                message="test message",
            )

        assert exc_info.value.code == 1


@pytest.mark.unit
class TestSetupLogging:
    """Tests for logging configuration."""

    def test_verbose_does_not_raise(self) -> None:
        """Verbose logging setup should complete without error."""
        _setup_logging(verbose=True)

    def test_non_verbose_does_not_raise(self) -> None:
        """Non-verbose logging setup should complete without error."""
        _setup_logging(verbose=False)


# ---- main() entry point ---------------------------------------------


@pytest.mark.unit
class TestMainFunction:
    """Tests for the main() entry point."""

    def test_version_command(self, capsys: pytest.CaptureFixture) -> None:
        """cv-pipeline version should print version string."""
        from cv_pipeline.cli import main

        with patch("sys.argv", ["cv-pipeline", "version"]):
            main()

        captured = capsys.readouterr()
        assert "cv-pipeline" in captured.out
        assert "0.1.0" in captured.out

    def test_no_command_exits(self) -> None:
        """No subcommand should exit with code 1."""
        from cv_pipeline.cli import main

        with patch("sys.argv", ["cv-pipeline"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1

    def test_infer_missing_image_exits(self) -> None:
        """Infer with missing image file should exit with error."""
        from cv_pipeline.cli import main

        with patch(
            "sys.argv",
            [
                "cv-pipeline",
                "infer",
                "--image",
                "nonexistent.png",
                "--output",
                "out/",
                "--model",
                "fake.pth",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1

    def test_infer_missing_model_exits(self, tmp_path: Path) -> None:
        """Infer with missing model file should exit with error."""
        from cv_pipeline.cli import main

        img_path = _make_test_image(tmp_path)

        with patch(
            "sys.argv",
            [
                "cv-pipeline",
                "infer",
                "--image",
                str(img_path),
                "--output",
                str(tmp_path / "out"),
                "--model",
                "nonexistent.pth",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1


# ---- _run_infer execution -------------------------------------------


@pytest.mark.unit
class TestRunInfer:
    """Tests for the full infer command execution path."""

    def test_happy_path_writes_output_files(
        self,
        tmp_path: Path,
    ) -> None:
        """A successful infer run should write JSON and mask files."""
        from cv_pipeline.cli import main

        img_path = _make_test_image(tmp_path, name="plate_001.png")
        model_path = tmp_path / "fake.pth"
        model_path.write_bytes(b"fake-checkpoint")
        output_dir = tmp_path / "results"

        mock_model = MagicMock()
        result = _fake_result(filename="plate_001.png")

        with (
            patch(
                "sys.argv",
                [
                    "cv-pipeline",
                    "infer",
                    "--image",
                    str(img_path),
                    "--output",
                    str(output_dir),
                    "--model",
                    str(model_path),
                ],
            ),
            patch(
                "cv_pipeline.segmentation.SegmentationModel",
                return_value=mock_model,
            ),
            patch("cv_pipeline.infer.infer", return_value=result),
        ):
            main()

        assert (output_dir / "plate_001_result.json").exists()
        assert (output_dir / "plate_001_mask.png").exists()

        with open(output_dir / "plate_001_result.json") as f:
            data = json.load(f)
        assert data["pipeline_version"] == "0.1.0"
        assert data["landmark_count"] == 1

    def test_version_flag_loads_from_registry(
        self,
        tmp_path: Path,
    ) -> None:
        """--version should resolve through the weights registry."""
        from cv_pipeline.cli import main

        img_path = _make_test_image(tmp_path)
        output_dir = tmp_path / "results"

        mock_model = MagicMock()
        result = _fake_result()

        with (
            patch(
                "sys.argv",
                [
                    "cv-pipeline",
                    "infer",
                    "--image",
                    str(img_path),
                    "--output",
                    str(output_dir),
                    "--version",
                    "unet-v1",
                ],
            ),
            patch("cv_pipeline.weights.REGISTRY", {"unet-v1": "https://example.com"}),
            patch(
                "cv_pipeline.segmentation.SegmentationModel",
                return_value=mock_model,
            ),
            patch("cv_pipeline.infer.infer", return_value=result),
        ):
            main()

        assert (output_dir / "test_result.json").exists()

    def test_defaults_to_first_registry_entry(
        self,
        tmp_path: Path,
    ) -> None:
        """No --model or --version should default to first REGISTRY entry."""
        from cv_pipeline.cli import main

        img_path = _make_test_image(tmp_path)
        output_dir = tmp_path / "results"

        mock_model = MagicMock()
        result = _fake_result()

        with (
            patch(
                "sys.argv",
                [
                    "cv-pipeline",
                    "infer",
                    "--image",
                    str(img_path),
                    "--output",
                    str(output_dir),
                ],
            ),
            patch("cv_pipeline.weights.REGISTRY", {"unet-v1": "https://example.com"}),
            patch(
                "cv_pipeline.segmentation.SegmentationModel",
                return_value=mock_model,
            ),
            patch("cv_pipeline.infer.infer", return_value=result),
        ):
            main()

        assert (output_dir / "test_result.json").exists()

    def test_empty_registry_no_flags_exits(self, tmp_path: Path) -> None:
        """No model source and empty REGISTRY should exit with code 1."""
        from cv_pipeline.cli import main

        img_path = _make_test_image(tmp_path)

        with (
            patch(
                "sys.argv",
                [
                    "cv-pipeline",
                    "infer",
                    "--image",
                    str(img_path),
                    "--output",
                    str(tmp_path / "out"),
                ],
            ),
            patch("cv_pipeline.weights.REGISTRY", {}),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1

    def test_validation_error_exits_with_code_1(
        self,
        tmp_path: Path,
    ) -> None:
        """ValidationError from pipeline should exit with code 1."""
        from cv_pipeline.cli import main
        from cv_pipeline.validation import ValidationError

        img_path = _make_test_image(tmp_path)
        model_path = tmp_path / "fake.pth"
        model_path.write_bytes(b"fake")

        mock_model = MagicMock()

        with (
            patch(
                "sys.argv",
                [
                    "cv-pipeline",
                    "infer",
                    "--image",
                    str(img_path),
                    "--output",
                    str(tmp_path / "out"),
                    "--model",
                    str(model_path),
                ],
            ),
            patch(
                "cv_pipeline.segmentation.SegmentationModel",
                return_value=mock_model,
            ),
            patch(
                "cv_pipeline.infer.infer",
                side_effect=ValidationError("IMAGE_TOO_SMALL", "too small"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1

    def test_model_load_failure_exits(self, tmp_path: Path) -> None:
        """SegmentationModel construction failure should exit with code 1."""
        from cv_pipeline.cli import main

        img_path = _make_test_image(tmp_path)
        model_path = tmp_path / "corrupt.pth"
        model_path.write_bytes(b"not-a-checkpoint")

        with (
            patch(
                "sys.argv",
                [
                    "cv-pipeline",
                    "infer",
                    "--image",
                    str(img_path),
                    "--output",
                    str(tmp_path / "out"),
                    "--model",
                    str(model_path),
                ],
            ),
            patch(
                "cv_pipeline.segmentation.SegmentationModel",
                side_effect=RuntimeError("corrupt checkpoint"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1


# ---- _run_train execution -------------------------------------------


@pytest.mark.unit
class TestRunTrain:
    """Tests for the full train command execution path."""

    def _make_data_dirs(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create empty but valid data and val directories."""
        data_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        data_dir.mkdir()
        val_dir.mkdir()
        return data_dir, val_dir

    def test_happy_path_exits_with_code_0(
        self,
        tmp_path: Path,
    ) -> None:
        """A successful training run should exit with code 0."""
        from cv_pipeline.cli import main
        from cv_pipeline.train import TrainingResult

        data_dir, val_dir = self._make_data_dirs(tmp_path)
        output_dir = tmp_path / "out"

        fake_result = TrainingResult(
            run_name="test-run",
            pipeline_version="0.1.0",
            best_epoch=1,
            best_val_f1=0.8,
            training_completed="2026-05-01T12:00:00Z",
        )

        with (
            patch(
                "sys.argv",
                [
                    "cv-pipeline",
                    "train",
                    "--data-dir",
                    str(data_dir),
                    "--val-dir",
                    str(val_dir),
                    "--output-dir",
                    str(output_dir),
                    "--epochs",
                    "1",
                    "--device",
                    "cpu",
                ],
            ),
            patch("cv_pipeline.train.train", return_value=fake_result),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 0

    def test_low_f1_exits_with_code_1(
        self,
        tmp_path: Path,
    ) -> None:
        """Training with best F1 below 0.5 should exit with code 1."""
        from cv_pipeline.cli import main
        from cv_pipeline.train import TrainingResult

        data_dir, val_dir = self._make_data_dirs(tmp_path)
        output_dir = tmp_path / "out"

        fake_result = TrainingResult(
            run_name="test-run",
            pipeline_version="0.1.0",
            best_epoch=1,
            best_val_f1=0.35,
            training_completed="2026-05-01T12:00:00Z",
        )

        with (
            patch(
                "sys.argv",
                [
                    "cv-pipeline",
                    "train",
                    "--data-dir",
                    str(data_dir),
                    "--val-dir",
                    str(val_dir),
                    "--output-dir",
                    str(output_dir),
                    "--epochs",
                    "1",
                    "--device",
                    "cpu",
                ],
            ),
            patch("cv_pipeline.train.train", return_value=fake_result),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1

    def test_missing_data_dir_exits_with_code_2(
        self,
        tmp_path: Path,
    ) -> None:
        """A nonexistent data directory should exit with code 2."""
        from cv_pipeline.cli import main

        val_dir = tmp_path / "val"
        val_dir.mkdir()

        with (
            patch(
                "sys.argv",
                [
                    "cv-pipeline",
                    "train",
                    "--data-dir",
                    str(tmp_path / "nonexistent"),
                    "--val-dir",
                    str(val_dir),
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--device",
                    "cpu",
                ],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 2

    def test_missing_val_dir_exits_with_code_2(
        self,
        tmp_path: Path,
    ) -> None:
        """A nonexistent validation directory should exit with code 2."""
        from cv_pipeline.cli import main

        data_dir = tmp_path / "train"
        data_dir.mkdir()

        with (
            patch(
                "sys.argv",
                [
                    "cv-pipeline",
                    "train",
                    "--data-dir",
                    str(data_dir),
                    "--val-dir",
                    str(tmp_path / "nonexistent"),
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--device",
                    "cpu",
                ],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 2

    def test_runtime_error_exits_with_code_2(
        self,
        tmp_path: Path,
    ) -> None:
        """A RuntimeError during training should exit with code 2."""
        from cv_pipeline.cli import main

        data_dir, val_dir = self._make_data_dirs(tmp_path)

        with (
            patch(
                "sys.argv",
                [
                    "cv-pipeline",
                    "train",
                    "--data-dir",
                    str(data_dir),
                    "--val-dir",
                    str(val_dir),
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--device",
                    "cpu",
                ],
            ),
            patch(
                "cv_pipeline.train.train",
                side_effect=RuntimeError("CUDA OOM"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 2
