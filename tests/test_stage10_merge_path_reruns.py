from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gps_ex1.tools.stage10_merge_path_reruns import merge_stage10


def _write_csv(path: Path, fields: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(fields)
        writer.writerows(rows)


def test_merge_uses_local_candidate_and_expected_fallback(tmp_path: Path) -> None:
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
    ]
    base = tmp_path / "base.csv"
    _write_csv(base, fields, [[0, 0.0, 32.0, 35.0, 1, "accepted", 50, 20, 0.4, 0.8, 1, 1], [45, 1.5, 33.0, 36.0, 0, "bad", 0, 0, 0.0, 0.0, 0, 0]])

    plan = tmp_path / "plan.csv"
    plan_fields = ["query_frame_index", "query_time_s", "seed_trusted", "expected_latitude", "expected_longitude", "expected_method", "will_rerun", "path_window_id"]
    _write_csv(plan, plan_fields, [[0, 0.0, 1, 32.0, 35.0, "seed_anchor", 0, ""], [45, 1.5, 0, 32.0001, 35.0001, "interpolate_seed_anchors", 1, "001"]])

    rerun = tmp_path / "stage10_rerun_path_window_001_top8.csv"
    _write_csv(rerun, fields, [[45, 1.5, 32.00011, 35.00010, 1, "accepted", 70, 25, 0.5, 0.8, 1, 1]])

    out = tmp_path / "out.csv"
    args = argparse.Namespace(
        base_csv=base,
        plan_csv=plan,
        rerun_dir=tmp_path,
        rerun_glob="stage10_rerun_path_window_*_top8.csv",
        out=out,
        max_distance_from_expected_m=50.0,
        min_candidate_confidence=0.0,
        clear_untrusted_matches=True,
        rerun_seed_anchors=False,
    )
    replaced, seed_kept, fallback = merge_stage10(args)
    assert replaced == 1
    assert seed_kept == 1
    assert fallback == 0

    with out.open("r", newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    assert rows[1]["stage10_final_source"] == "local_path_rerun"
    assert rows[1]["filter_accepted"] == "1"
