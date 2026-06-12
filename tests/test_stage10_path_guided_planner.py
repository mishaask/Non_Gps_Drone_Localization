from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from gps_ex1.tools.stage10_path_guided_planner import build_stage10_plan, filter_seed_path, interpolate_expected_path, make_seed_points


def _write_prediction_csv(path: Path) -> None:
    fields = [
        "query_frame_index",
        "query_time_s",
        "pred_latitude",
        "pred_longitude",
        "filter_accepted",
        "filter_reason",
        "orb_good_matches",
        "homography_inliers",
        "verification_inlier_ratio",
        "retrieval_similarity",
        "projected_center_inside",
        "homography_geometry_ok",
        "reference_inlier_bbox_area_frac",
        "reference_inlier_bbox_width_frac",
        "reference_inlier_bbox_height_frac",
    ]
    rows = [
        # Good green anchor.
        [0, 0.0, 32.000000, 35.000000, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
        # Red teleport false anchor between two green anchors.
        [45, 1.5, 32.004000, 35.004000, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
        # Good green anchor close to first.
        [90, 3.0, 32.000010, 35.000010, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(fields)
        writer.writerows(rows)


def _args(**kwargs):
    values = dict(
        seed_min_confidence=0.0,
        seed_min_inliers=1,
        seed_min_good_matches=1,
        seed_max_ref_area_frac=1.0,
        seed_max_ref_width_frac=1.0,
        seed_max_ref_height_frac=1.0,
        teleport_threshold_m=100.0,
        bridge_threshold_m=50.0,
        max_anchor_speed_mps=80.0,
        high_confidence_seed_keep=999.0,
        seed_context_window=5,
        seed_filter_iterations=4,
        use_dominant_seed_component=False,
        dominant_edge_max_distance_m=180.0,
        dominant_edge_max_speed_mps=25.0,
        dominant_min_component_size=4,
        use_seed_support_validation=False,
        seed_support_radius_m=140.0,
        seed_support_max_speed_mps=30.0,
        seed_support_order_window=8,
        seed_min_support=1,
        seed_support_min_component_size=4,
        use_seed_tail_rescue=True,
        seed_tail_rescue_min_points=1,
        seed_tail_rescue_min_confidence=8.0,
        seed_tail_rescue_max_distance_m=420.0,
        seed_tail_rescue_max_speed_mps=35.0,
        seed_tail_rescue_max_gap_s=120.0,
    )
    values.update(kwargs)
    return argparse.Namespace(**values)


def test_seed_filter_rejects_isolated_teleport(tmp_path: Path) -> None:
    csv_path = tmp_path / "pred.csv"
    _write_prediction_csv(csv_path)

    with csv_path.open("r", newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    points = make_seed_points(rows, _args())
    filter_seed_path(points, _args())
    interpolate_expected_path(points)

    assert points[0].seed_trusted is True
    assert points[1].seed_trusted is False
    assert "teleport" in points[1].seed_reject_reason
    assert points[2].seed_trusted is True
    assert points[1].expected_lat is not None
    assert abs(points[1].expected_lat - 32.000005) < 1e-5


def test_build_stage10_plan_writes_windows(tmp_path: Path) -> None:
    pred_csv = tmp_path / "pred.csv"
    _write_prediction_csv(pred_csv)
    ref_index = tmp_path / "ref.npz"
    descriptors = np.ones((4, 3), dtype=np.float32)
    np.savez_compressed(
        ref_index,
        descriptors=descriptors,
        image_paths=np.array(["a.jpg", "b.jpg", "c.jpg", "d.jpg"]),
        flight_ids=np.array(["A", "A", "A", "A"]),
        frame_indices=np.array([0, 1, 2, 3]),
        video_times_s=np.array([0.0, 1.0, 2.0, 3.0]),
        drone_lats=np.array([32.0, 32.0001, 32.0002, 32.0003]),
        drone_lons=np.array([35.0, 35.0001, 35.0002, 35.0003]),
        center_lats=np.array([32.0, 32.0001, 32.0002, 32.0003]),
        center_lons=np.array([35.0, 35.0001, 35.0002, 35.0003]),
        headings_deg=np.array([0.0, 0.0, 0.0, 0.0]),
        center_offsets_m=np.array([0.0, 0.0, 0.0, 0.0]),
    )
    args = argparse.Namespace(
        prediction_csv=pred_csv,
        reference_index=ref_index,
        out_dir=tmp_path / "out",
        reference_coordinate="center",
        seed_min_confidence=0.0,
        seed_min_inliers=1,
        seed_min_good_matches=1,
        seed_max_ref_area_frac=1.0,
        seed_max_ref_width_frac=1.0,
        seed_max_ref_height_frac=1.0,
        teleport_threshold_m=100.0,
        bridge_threshold_m=50.0,
        max_anchor_speed_mps=80.0,
        high_confidence_seed_keep=999.0,
        seed_context_window=5,
        seed_filter_iterations=4,
        use_dominant_seed_component=False,
        dominant_edge_max_distance_m=180.0,
        dominant_edge_max_speed_mps=25.0,
        dominant_min_component_size=4,
        use_seed_support_validation=False,
        seed_support_radius_m=140.0,
        seed_support_max_speed_mps=30.0,
        seed_support_order_window=8,
        seed_min_support=1,
        seed_support_min_component_size=4,
        use_seed_tail_rescue=True,
        seed_tail_rescue_min_points=1,
        seed_tail_rescue_min_confidence=8.0,
        seed_tail_rescue_max_distance_m=420.0,
        seed_tail_rescue_max_speed_mps=35.0,
        seed_tail_rescue_max_gap_s=120.0,
        rerun_mode="all-non-seed",
        local_radius_m=200.0,
        max_local_radius_m=300.0,
        min_local_references=1,
        window_spacing_m=200.0,
        max_frames_per_window=10,
    )
    points, windows = build_stage10_plan(args)
    assert sum(p.will_rerun for p in points) == 1
    assert len(windows) == 1
    assert (args.out_dir / "stage10_path_rerun_plan.csv").exists()
    assert (args.out_dir / "local_reference_index_path_window_001.npz").exists()


def test_dominant_component_rejects_disconnected_seed_island(tmp_path: Path) -> None:
    fields = [
        "query_frame_index",
        "query_time_s",
        "pred_latitude",
        "pred_longitude",
        "filter_accepted",
        "filter_reason",
        "orb_good_matches",
        "homography_inliers",
        "verification_inlier_ratio",
        "retrieval_similarity",
        "projected_center_inside",
        "homography_geometry_ok",
        "reference_inlier_bbox_area_frac",
        "reference_inlier_bbox_width_frac",
        "reference_inlier_bbox_height_frac",
    ]
    rows = [
        [0, 0.0, 32.00000, 35.00000, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
        [45, 1.5, 32.00005, 35.00005, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
        [90, 3.0, 32.00010, 35.00010, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
        [135, 4.5, 32.00015, 35.00015, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
        # A later accepted false island far away. It should not become part of the seed path.
        [180, 6.0, 32.00400, 35.00400, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
        [225, 7.5, 32.00405, 35.00405, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
    ]
    pred_csv = tmp_path / "pred.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(fields)
        writer.writerows(rows)

    with pred_csv.open("r", newline="", encoding="utf-8") as fp:
        points = make_seed_points(list(csv.DictReader(fp)), _args(
            use_dominant_seed_component=True,
            dominant_edge_max_distance_m=60.0,
            dominant_edge_max_speed_mps=50.0,
            dominant_min_component_size=3,
        ))
    from gps_ex1.tools.stage10_path_guided_planner import apply_dominant_seed_component
    apply_dominant_seed_component(points, _args(
        use_dominant_seed_component=True,
        dominant_edge_max_distance_m=60.0,
        dominant_edge_max_speed_mps=50.0,
        dominant_min_component_size=3,
    ))

    assert sum(p.seed_trusted for p in points) == 4
    assert points[4].seed_trusted is False
    assert points[5].seed_trusted is False
    assert points[4].seed_reject_reason == "seed_rejected_not_in_dominant_component"



def test_seed_support_validation_rejects_small_bad_prefix_island(tmp_path: Path) -> None:
    from gps_ex1.tools.stage10_path_guided_planner import apply_dominant_seed_component, apply_seed_support_validation

    fields = [
        "query_frame_index",
        "query_time_s",
        "pred_latitude",
        "pred_longitude",
        "filter_accepted",
        "filter_reason",
        "orb_good_matches",
        "homography_inliers",
        "verification_inlier_ratio",
        "retrieval_similarity",
        "projected_center_inside",
        "homography_geometry_ok",
        "reference_inlier_bbox_area_frac",
        "reference_inlier_bbox_width_frac",
        "reference_inlier_bbox_height_frac",
    ]
    rows = [
        # Small false prefix island, internally consistent but not supported by the main path.
        [540, 18.0, 32.10320, 35.20763, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
        [1170, 39.0, 32.10422, 35.20787, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
        [1260, 42.0, 32.10415, 35.20746, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
        # Main supported path island.
        [1710, 57.0, 32.10629, 35.20673, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
        [1800, 60.0, 32.10629, 35.20673, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
        [1935, 64.5, 32.10622, 35.20683, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
        [1980, 66.0, 32.10570, 35.20741, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
        [2025, 67.5, 32.10570, 35.20741, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
    ]
    pred_csv = tmp_path / "pred.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(fields)
        writer.writerows(rows)

    with pred_csv.open("r", newline="", encoding="utf-8") as fp:
        points = make_seed_points(list(csv.DictReader(fp)), _args(
            use_dominant_seed_component=False,
            use_seed_support_validation=True,
            seed_support_radius_m=140.0,
            seed_support_max_speed_mps=30.0,
            seed_support_min_component_size=4,
            seed_min_support=1,
        ))
    apply_dominant_seed_component(points, _args(use_dominant_seed_component=False))
    apply_seed_support_validation(points, _args(
        use_seed_support_validation=True,
        seed_support_radius_m=140.0,
        seed_support_max_speed_mps=30.0,
        seed_support_order_window=8,
        seed_support_min_component_size=4,
        seed_min_support=1,
    ))

    assert points[0].seed_trusted is False
    assert points[1].seed_trusted is False
    assert points[2].seed_trusted is False
    assert points[0].seed_reject_reason == "seed_rejected_not_in_supported_component"
    assert sum(p.seed_trusted for p in points) == 5
    assert all(points[i].seed_trusted for i in range(3, 8))



def test_seed_tail_rescue_keeps_sparse_landing_suffix() -> None:
    from gps_ex1.tools.stage10_path_guided_planner import apply_seed_support_validation, apply_seed_tail_rescue

    fields = [
        "query_frame_index", "query_time_s", "pred_latitude", "pred_longitude",
        "filter_accepted", "filter_reason", "orb_good_matches", "homography_inliers",
        "verification_inlier_ratio", "retrieval_similarity", "projected_center_inside",
        "homography_geometry_ok", "reference_inlier_bbox_area_frac",
        "reference_inlier_bbox_width_frac", "reference_inlier_bbox_height_frac",
    ]
    # Five dense main-route anchors, then two sparse but physically plausible
    # landing-tail anchors.  The normal support validation rejects the tail
    # because it is a small component, but the asymmetric suffix rescue should
    # bring it back.
    rows = [
        [0, 0.0, 32.00000, 35.00000, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
        [45, 1.5, 32.00005, 35.00005, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
        [90, 3.0, 32.00010, 35.00010, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
        [135, 4.5, 32.00015, 35.00015, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
        [180, 6.0, 32.00020, 35.00020, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
        # Sparse suffix, far enough to be outside the support radius but slow/plausible.
        [2385, 85.0, 32.00270, 35.00270, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
        [2655, 94.0, 32.00320, 35.00320, 1, "accepted", 80, 40, 0.5, 0.8, 1, 1, 0.03, 0.2, 0.2],
    ]
    dict_rows = [dict(zip(fields, map(str, row))) for row in rows]
    args = _args(
        use_seed_support_validation=True,
        seed_support_radius_m=40.0,
        seed_support_max_speed_mps=30.0,
        seed_support_order_window=8,
        seed_support_min_component_size=4,
        seed_min_support=1,
        use_seed_tail_rescue=True,
        seed_tail_rescue_min_points=1,
        seed_tail_rescue_min_confidence=0.0,
        seed_tail_rescue_max_distance_m=450.0,
        seed_tail_rescue_max_speed_mps=35.0,
        seed_tail_rescue_max_gap_s=120.0,
    )
    points = make_seed_points(dict_rows, args)
    apply_seed_support_validation(points, args)
    assert points[5].seed_trusted is False
    assert points[6].seed_trusted is False

    apply_seed_tail_rescue(points, args)
    assert points[5].seed_trusted is True
    assert points[6].seed_trusted is True
    assert "tail_rescued" in points[5].seed_support_status
    assert "tail_rescued" in points[6].seed_support_status
