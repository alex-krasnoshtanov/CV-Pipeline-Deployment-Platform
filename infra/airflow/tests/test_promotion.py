"""Unit tests for the champion-challenger promotion path.

Exercises the decision + traffic logic without Azure or the ML stack:
- promotion.resolve_champion_version / resolve_candidate_version (mocked client)
- promotion.compute_traffic_split (pure canary arithmetic)
- smoke_eval.f1_from_counts / decide_promote / pair_images_and_masks

These cover the parts that can cause a wrong promotion or a rejected traffic
split -- version resolution, the F1 maths, the margin gate, and the canary
split that must always sum to 100. The Azure-touching functions
(build_smoke_job, _health_check, deploy_candidate) are integration-tested by
scripts/azure/run_smoke_eval.py against the live workspace.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# promotion.py is in dags/, smoke_eval.py is in smoke_code/ -- both siblings of
# this tests/ dir under infra/airflow/. Add them like the DAG does.
_AIRFLOW = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_AIRFLOW / "dags"))
sys.path.insert(0, str(_AIRFLOW / "smoke_code"))

import promotion  # noqa: E402
import smoke_eval  # noqa: E402


def _model(version: str) -> MagicMock:
    """A stand-in registered-model object exposing only .version."""
    m = MagicMock()
    m.version = version
    return m


# ---- resolve_champion_version ----------------------------------------


def test_resolve_champion_picks_max_traffic() -> None:
    client = MagicMock()
    endpoint = MagicMock()
    endpoint.traffic = {"unet-v6": 0, "unet-v7": 0, "unet-v9": 100}
    client.online_endpoints.get.return_value = endpoint
    assert promotion.resolve_champion_version(client) == "9"


def test_resolve_champion_raises_without_live_traffic() -> None:
    client = MagicMock()
    endpoint = MagicMock()
    endpoint.traffic = {"unet-v9": 0}
    client.online_endpoints.get.return_value = endpoint
    with pytest.raises(RuntimeError):
        promotion.resolve_champion_version(client)


# ---- resolve_candidate_version ---------------------------------------


def test_resolve_candidate_picks_highest_version() -> None:
    client = MagicMock()
    client.models.list.return_value = [_model("7"), _model("10"), _model("9")]
    assert promotion.resolve_candidate_version(client) == "10"


def test_resolve_candidate_raises_when_empty() -> None:
    client = MagicMock()
    client.models.list.return_value = []
    with pytest.raises(RuntimeError):
        promotion.resolve_candidate_version(client)


# ---- compute_traffic_split (canary arithmetic) -----------------------


def test_traffic_full_cutover() -> None:
    assert promotion.compute_traffic_split({"unet-v9": 100}, "unet-v10", 100) == {
        "unet-v10": 100
    }


def test_traffic_canary_against_single_champion() -> None:
    split = promotion.compute_traffic_split({"unet-v9": 100}, "unet-v10", 20)
    assert split == {"unet-v9": 80, "unet-v10": 20}
    assert sum(split.values()) == 100


def test_traffic_canary_scales_multiple_existing() -> None:
    split = promotion.compute_traffic_split(
        {"unet-v8": 50, "unet-v9": 50}, "unet-v10", 10
    )
    assert sum(split.values()) == 100
    assert split["unet-v10"] >= 10  # gets at least the requested share


def test_traffic_no_existing_deployments_gives_full() -> None:
    # Nothing to canary against -> the new deployment must take everything.
    assert promotion.compute_traffic_split({}, "unet-v10", 30) == {"unet-v10": 100}


def test_traffic_redeploy_same_name_does_not_double_count() -> None:
    # Re-deploying an existing name: its old share is dropped, not scaled.
    split = promotion.compute_traffic_split(
        {"unet-v9": 50, "unet-v10": 50}, "unet-v10", 25
    )
    assert sum(split.values()) == 100
    assert split["unet-v9"] == 75


def test_traffic_always_sums_to_100_under_rounding() -> None:
    # Awkward existing shares that do not divide evenly must still total 100.
    for traffic in (1, 7, 33, 50, 99):
        split = promotion.compute_traffic_split(
            {"a": 1, "b": 1, "c": 1, "d": 1}, "new", traffic
        )
        assert sum(split.values()) == 100, (traffic, split)


# ---- _parse_invoke_response (invoke double-encoding) -----------------


def test_parse_invoke_response_plain_json() -> None:
    raw = '{"mask_b64": "abc", "landmark_count": 0}'
    parsed = promotion._parse_invoke_response(raw)
    assert isinstance(parsed, dict) and parsed["mask_b64"] == "abc"


def test_parse_invoke_response_double_encoded() -> None:
    # The shape that broke the health check: a JSON string containing JSON.
    inner = '{"mask_b64": "abc", "landmark_count": 0}'
    raw = json.dumps(inner)
    parsed = promotion._parse_invoke_response(raw)
    assert isinstance(parsed, dict) and parsed["mask_b64"] == "abc"


def test_parse_invoke_response_bytes() -> None:
    parsed = promotion._parse_invoke_response(b'{"mask_b64": "abc"}')
    assert parsed["mask_b64"] == "abc"


# ---- f1_from_counts --------------------------------------------------


def test_f1_perfect() -> None:
    assert smoke_eval.f1_from_counts(10, 0, 0) == 1.0


def test_f1_zero_denominator_is_zero() -> None:
    assert smoke_eval.f1_from_counts(0, 0, 0) == 0.0


def test_f1_known_value() -> None:
    # tp=4, fp=2, fn=2 -> 2*4 / (2*4 + 2 + 2) = 8/12
    assert smoke_eval.f1_from_counts(4, 2, 2) == pytest.approx(2 / 3)


# ---- decide_promote --------------------------------------------------


def test_decide_promote_beats_margin() -> None:
    assert smoke_eval.decide_promote(0.80, 0.81, 0.005) is True


def test_decide_promote_within_margin_is_false() -> None:
    assert smoke_eval.decide_promote(0.80, 0.803, 0.005) is False


def test_decide_promote_exact_boundary_promotes() -> None:
    assert smoke_eval.decide_promote(0.80, 0.805, 0.005) is True


def test_decide_promote_zero_margin_ties_promote() -> None:
    assert smoke_eval.decide_promote(0.80, 0.80, 0.0) is True


# ---- pair_images_and_masks -------------------------------------------


def test_pair_matches_by_stem_and_drops_unmatched(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    for name in ("a", "b", "c"):
        (tmp_path / "images" / f"{name}.png").write_bytes(b"x")
    for name in ("a", "b"):  # 'c' has no mask -> unmatched, dropped
        (tmp_path / "masks" / f"{name}.png").write_bytes(b"x")

    pairs = smoke_eval.pair_images_and_masks(tmp_path)
    assert sorted(img.stem for img, _ in pairs) == ["a", "b"]


def test_pair_allows_different_extensions(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    (tmp_path / "images" / "plate.tif").write_bytes(b"x")
    (tmp_path / "masks" / "plate.png").write_bytes(b"x")

    pairs = smoke_eval.pair_images_and_masks(tmp_path)
    assert len(pairs) == 1


def test_pair_matches_root_mask_convention(tmp_path: Path) -> None:
    # The real hades-smoke / test-set convention: <stem>_root_mask.<ext>.
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    (tmp_path / "images" / "plate_001.png").write_bytes(b"x")
    (tmp_path / "masks" / "plate_001_root_mask.png").write_bytes(b"x")

    pairs = smoke_eval.pair_images_and_masks(tmp_path)
    assert len(pairs) == 1
    assert pairs[0][1].name == "plate_001_root_mask.png"


def test_pair_root_mask_glob_fallback(tmp_path: Path) -> None:
    # A *root_mask* variant that is not an exact pattern still matches via glob.
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    (tmp_path / "images" / "plate_002.tif").write_bytes(b"x")
    (tmp_path / "masks" / "plate_002_root_mask_v2.png").write_bytes(b"x")

    pairs = smoke_eval.pair_images_and_masks(tmp_path)
    assert len(pairs) == 1


def test_pair_missing_subdir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        smoke_eval.pair_images_and_masks(tmp_path)
