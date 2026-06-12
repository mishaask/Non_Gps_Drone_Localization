from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from gps_ex1.tools.stage9_path_consistency_filter import (
    _apply_filtered_fill,
    _apply_path_rules,
    _make_points,
    _read_grid_centers,
)


def _write_grid(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["grid_region_id", "row", "col", "center_latitude", "center_longitude"])
        writer.writeheader()
        writer.writerow({"grid_region_id": "1", "row": "1", "col": "1", "center_latitude": "32.000000", "center_longitude": "35.000000"})
        writer.writerow({"grid_region_id": "2", "row": "1", "col": "2", "center_latitude": "32.000000", "center_longitude": "35.010000"})


def _row(frame: int, lat: float, lon: float, accepted: str = "1") -> dict[str, str]:
    return {
        "query_frame_index": str(frame),
        "query_time_s": str(frame / 30.0),
        "pred_latitude": f"{lat:.8f}",
        "pred_longitude": f"{lon:.8f}",
        "filter_accepted": accepted,
        "filter_reason": "accepted" if accepted == "1" else "not_enough_good_matches",
        "orb_good_matches": "40",
        "homography_inliers": "20",
        "verification_inlier_ratio": "0.5",
        "retrieval_similarity": "0.9",
        "projected_center_inside": "1",
        "homography_geometry_ok": "1",
    }


def _args(**overrides):
    base = dict(
        max_grid_center_distance_m=2000.0,
        min_confidence=0.0,
        context_window=5,
        support_window=5,
        teleport_threshold_m=250.0,
        bridge_threshold_m=250.0,
        min_same_grid_support=1,
        high_confidence_keep_threshold=999.0,
        trusted_grid_regions="",
        banned_grid_regions="",
        iterations=3,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_isolated_teleport_between_same_grid_context_is_rejected(tmp_path: Path) -> None:
    grid_path = tmp_path / "grid.csv"
    _write_grid(grid_path)
    centers = _read_grid_centers(grid_path)
    rows = [
        _row(0, 32.000000, 35.000000),
        _row(45, 32.000000, 35.010000),
        _row(90, 32.000050, 35.000050),
    ]
    points = _make_points(rows, centers, _args())
    _apply_path_rules(points, _args())

    assert points[0].path_accepted
    assert not points[1].path_accepted
    assert points[1].reason == "isolated_grid_teleport_between_same_context"
    assert points[2].path_accepted


def test_supported_same_grid_sequence_is_kept(tmp_path: Path) -> None:
    grid_path = tmp_path / "grid.csv"
    _write_grid(grid_path)
    centers = _read_grid_centers(grid_path)
    rows = [
        _row(0, 32.000000, 35.000000),
        _row(45, 32.000000, 35.010000),
        _row(90, 32.000020, 35.010020),
        _row(135, 32.000040, 35.010040),
    ]
    points = _make_points(rows, centers, _args(min_same_grid_support=1))
    _apply_path_rules(points, _args(min_same_grid_support=1))

    assert points[1].path_accepted
    assert points[2].path_accepted
    assert points[3].path_accepted


def test_interpolation_fills_rejected_row(tmp_path: Path) -> None:
    grid_path = tmp_path / "grid.csv"
    _write_grid(grid_path)
    centers = _read_grid_centers(grid_path)
    rows = [
        _row(0, 32.000000, 35.000000),
        _row(45, 32.000000, 35.010000),
        _row(90, 32.000100, 35.000100),
    ]
    points = _make_points(rows, centers, _args())
    _apply_path_rules(points, _args())
    _apply_filtered_fill(points, "interpolate")

    assert not points[1].path_accepted
    assert points[1].filtered_lat is not None
    assert points[1].filtered_lon is not None
    assert points[1].filtered_method == "interpolate_previous_next"
