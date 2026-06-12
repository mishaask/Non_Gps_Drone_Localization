"""Stage 10.4: local-best cluster lookpoint refinement for fallback frames.

This is the safe way to add the "best returned match in the expected area" idea.
It does NOT run a new global search. It uses the already-created Stage 10.3 local
rerun CSVs, which were produced from local reference indexes around the expected path.

Typical flow:
  1) Stage 10.3 planner creates seed path + local windows.
  2) Stage 10.3 local reruns search only inside those windows.
  3) Stage 10.3 merge creates a full path with visual matches + path_expected_fallback rows.
  4) This tool takes the best local rerun candidate for fallback rows and tries to compute
     a cluster lookpoint from the candidate's RANSAC/LightGlue inlier cluster.

Output labels are intentionally honest:
  - cluster_refined_visual_match: an already accepted Stage 10.3 visual match refined by cluster.
  - local_best_cluster_guess: a fallback row replaced by the best local candidate's cluster lookpoint.
  - path_expected_fallback: still no usable local candidate; keep the Stage 10.3 expected path.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path
from typing import Iterable

from gps_ex1.tools.stage10_cluster_lookpoint_refine import (
    ClusterResult,
    _accepted,
    _float_or_none,
    _fmt,
    _int_or_none,
    _read_rows,
    _refine_one_row,
    _require_cv2,
    _write_rows,
    _build_reference_lookup,
)
from gps_ex1.preprocess.reference_index import load_reference_index


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _expected_latlon(row: dict[str, str]) -> tuple[float, float] | None:
    # Prefer the final Stage 10.3 filtered coordinate. That is the path prior we trust.
    for lat_key, lon_key in [
        ("filtered_latitude", "filtered_longitude"),
        ("pred_latitude", "pred_longitude"),
        ("expected_latitude", "expected_longitude"),
    ]:
        lat = _float_or_none(row.get(lat_key))
        lon = _float_or_none(row.get(lon_key))
        if lat is not None and lon is not None and abs(lat) > 1e-12 and abs(lon) > 1e-12:
            return lat, lon
    return None


def _score_candidate(row: dict[str, str]) -> float:
    conf = _float_or_none(row.get("confidence")) or _float_or_none(row.get("match_confidence")) or 0.0
    inliers = _float_or_none(row.get("homography_inliers")) or _float_or_none(row.get("inliers")) or 0.0
    good = _float_or_none(row.get("good_matches")) or _float_or_none(row.get("matches")) or 0.0
    ratio = _float_or_none(row.get("inlier_ratio")) or 0.0
    accepted_bonus = 5.0 if _accepted(row) else 0.0
    # Keep confidence primary, but prefer candidates that at least had some geometric support.
    return conf + 0.20 * inliers + 0.05 * good + 2.0 * ratio + accepted_bonus


def _row_has_reference(row: dict[str, str]) -> bool:
    if (row.get("matched_reference_image") or "").strip():
        return True
    if (row.get("matched_reference_flight") or "").strip() and (row.get("matched_reference_frame_index") or "").strip():
        return True
    return False


def _build_candidate_map(rerun_dir: Path, rerun_glob: str) -> dict[int, dict[str, str]]:
    candidate_by_frame: dict[int, dict[str, str]] = {}
    pattern = str(rerun_dir / rerun_glob)
    for csv_path in sorted(glob.glob(pattern)):
        path = Path(csv_path)
        try:
            _, rows = _read_rows(path)
        except FileNotFoundError:
            continue
        for row in rows:
            frame = _int_or_none(row.get("query_frame_index") or row.get("query_frame") or row.get("frame"))
            if frame is None or not _row_has_reference(row):
                continue
            row = dict(row)
            row["local_best_candidate_csv"] = str(path)
            existing = candidate_by_frame.get(frame)
            if existing is None or _score_candidate(row) > _score_candidate(existing):
                candidate_by_frame[frame] = row
    return candidate_by_frame


def _ensure_fields(header: list[str]) -> list[str]:
    extra = [
        "local_best_status",
        "local_best_used",
        "local_best_kind",
        "local_best_candidate_reason",
        "local_best_candidate_confidence",
        "local_best_candidate_csv",
        "local_best_distance_from_expected_m",
        "local_best_cluster_inliers",
        "local_best_cluster_total_matches",
        "local_best_cluster_reference_x",
        "local_best_cluster_reference_y",
        "local_best_cluster_latitude",
        "local_best_cluster_longitude",
        "local_best_original_filtered_latitude",
        "local_best_original_filtered_longitude",
        "local_best_original_filtered_method",
    ]
    # Keep cluster columns from stage10_cluster_lookpoint_refine too if that tool writes them.
    cluster_extra = [
        "cluster_status",
        "cluster_used",
        "cluster_inliers",
        "cluster_total_matches",
        "cluster_reference_x",
        "cluster_reference_y",
        "cluster_reference_area_frac",
        "cluster_reference_width_frac",
        "cluster_reference_height_frac",
        "cluster_offset_east_m",
        "cluster_offset_north_m",
        "cluster_latitude",
        "cluster_longitude",
    ]
    return list(dict.fromkeys(header + extra + cluster_extra))


def _copy_cluster_columns(output: dict[str, str], result: ClusterResult) -> None:
    output["cluster_status"] = result.status
    output["cluster_used"] = "1" if result.used else "0"
    output["cluster_inliers"] = str(result.inliers)
    output["cluster_total_matches"] = str(result.total_matches)
    output["cluster_reference_x"] = _fmt(result.cluster_x, 3)
    output["cluster_reference_y"] = _fmt(result.cluster_y, 3)
    output["cluster_reference_area_frac"] = _fmt(result.cluster_area_frac, 6)
    output["cluster_reference_width_frac"] = _fmt(result.cluster_width_frac, 6)
    output["cluster_reference_height_frac"] = _fmt(result.cluster_height_frac, 6)
    output["cluster_offset_east_m"] = _fmt(result.offset_east_m, 3)
    output["cluster_offset_north_m"] = _fmt(result.offset_north_m, 3)
    output["cluster_latitude"] = _fmt(result.lat, 8)
    output["cluster_longitude"] = _fmt(result.lon, 8)


class _RefineArgs(argparse.Namespace):
    pass


def refine_local_best(args: argparse.Namespace) -> tuple[int, int, int, int]:
    header, merged_rows = _read_rows(args.merged_csv)
    candidate_by_frame = _build_candidate_map(args.rerun_dir, args.rerun_glob)

    index = load_reference_index(args.reference_index)
    lookup = _build_reference_lookup(index, args.project_root)
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(args.query_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open query video: {args.query_video}")

    # Reuse the cluster-lookpoint implementation with a small namespace object.
    refine_args = _RefineArgs()
    refine_args.project_root = args.project_root
    refine_args.debug_dir = args.debug_dir
    refine_args.query_scale_fill = args.query_scale_fill
    refine_args.match_backend = args.match_backend
    refine_args.orb_nfeatures = args.orb_nfeatures
    refine_args.ratio_test = args.ratio_test
    refine_args.min_inliers = args.min_inliers
    refine_args.ransac_reproj_threshold = args.ransac_reproj_threshold
    refine_args.max_cluster_area_frac = args.max_cluster_area_frac
    refine_args.max_cluster_width_frac = args.max_cluster_width_frac
    refine_args.max_cluster_height_frac = args.max_cluster_height_frac
    refine_args.camera_angle_deg = args.camera_angle_deg
    refine_args.angle_convention = args.angle_convention
    refine_args.horizontal_fov_deg = args.horizontal_fov_deg
    refine_args.fallback_altitude_m = args.fallback_altitude_m

    out_rows: list[dict[str, str]] = []
    attempted = 0
    used = 0
    fallback_attempted = 0
    fallback_used = 0

    for row in merged_rows:
        output = dict(row)
        output["local_best_original_filtered_latitude"] = output.get("filtered_latitude", "")
        output["local_best_original_filtered_longitude"] = output.get("filtered_longitude", "")
        output["local_best_original_filtered_method"] = output.get("filtered_method", "")
        frame = _int_or_none(row.get("query_frame_index") or row.get("query_frame") or row.get("frame"))
        source = (row.get("stage10_final_source") or row.get("filtered_method") or "").strip()
        is_fallback = source == "path_expected_fallback" or "fallback" in source

        candidate: dict[str, str] | None = None
        kind = ""
        if not is_fallback and _row_has_reference(row) and args.refine_existing_visual:
            candidate = dict(row)
            kind = "cluster_refined_visual_match"
        elif is_fallback and frame is not None and args.refine_fallbacks:
            candidate = candidate_by_frame.get(frame)
            kind = "local_best_cluster_guess"

        result = ClusterResult(status="not_attempted")
        distance_m: float | None = None
        if candidate is not None:
            attempted += 1
            if is_fallback:
                fallback_attempted += 1
            debug_counter = attempted if args.debug_dir is not None and (args.max_debug_images is None or attempted <= args.max_debug_images) else None
            _, result = _refine_one_row(candidate, cap, index, lookup, refine_args, debug_counter)
            expected = _expected_latlon(row)
            if result.lat is not None and result.lon is not None and expected is not None:
                distance_m = _haversine_m(expected[0], expected[1], result.lat, result.lon)

            allowed_by_distance = distance_m is None or distance_m <= args.max_distance_from_expected_m
            allowed_by_status = result.used or (args.allow_cluster_too_scattered and result.lat is not None and result.lon is not None and result.status.startswith("cluster_too_"))
            if allowed_by_status and allowed_by_distance:
                used += 1
                if is_fallback:
                    fallback_used += 1
                output["filtered_latitude"] = _fmt(result.lat, 8)
                output["filtered_longitude"] = _fmt(result.lon, 8)
                output["filtered_method"] = kind
                output["local_best_used"] = "1"
                output["local_best_kind"] = kind
                # Do not claim this is a hard accepted visual match unless it was already accepted.
                if is_fallback:
                    output["filter_accepted"] = output.get("filter_accepted", "0")
            else:
                output["local_best_used"] = "0"
                output["local_best_kind"] = kind
        else:
            output["local_best_used"] = "0"
            output["local_best_kind"] = "no_candidate"

        output["local_best_status"] = result.status
        output["local_best_candidate_reason"] = (candidate or {}).get("match_reason", (candidate or {}).get("reason", ""))
        output["local_best_candidate_confidence"] = (candidate or {}).get("confidence", "")
        output["local_best_candidate_csv"] = (candidate or {}).get("local_best_candidate_csv", "")
        output["local_best_distance_from_expected_m"] = _fmt(distance_m, 3)
        output["local_best_cluster_inliers"] = str(result.inliers)
        output["local_best_cluster_total_matches"] = str(result.total_matches)
        output["local_best_cluster_reference_x"] = _fmt(result.cluster_x, 3)
        output["local_best_cluster_reference_y"] = _fmt(result.cluster_y, 3)
        output["local_best_cluster_latitude"] = _fmt(result.lat, 8)
        output["local_best_cluster_longitude"] = _fmt(result.lon, 8)
        _copy_cluster_columns(output, result)
        out_rows.append(output)

    cap.release()
    _write_rows(args.out, _ensure_fields(header), out_rows)
    return attempted, used, fallback_attempted, fallback_used


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 10.4: replace fallback path points with best local candidate cluster-lookpoints when safe.")
    parser.add_argument("--merged-csv", required=True, type=Path, help="Stage 10.3 merged CSV.")
    parser.add_argument("--rerun-dir", required=True, type=Path, help="Directory containing local rerun CSVs.")
    parser.add_argument("--rerun-glob", required=True, help="Glob for local rerun CSVs, e.g. DJI_0010_stage10_3_rerun_path_window_*_top8.csv")
    parser.add_argument("--query-video", required=True, type=Path)
    parser.add_argument("--reference-index", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--refine-existing-visual", action="store_true", default=True)
    parser.add_argument("--no-refine-existing-visual", dest="refine_existing_visual", action="store_false")
    parser.add_argument("--refine-fallbacks", action="store_true", default=True)
    parser.add_argument("--no-refine-fallbacks", dest="refine_fallbacks", action="store_false")
    parser.add_argument("--match-backend", choices=["lightglue", "orb"], default="lightglue")
    parser.add_argument("--orb-nfeatures", type=int, default=4000)
    parser.add_argument("--ratio-test", type=float, default=0.75)
    parser.add_argument("--min-inliers", type=int, default=4)
    parser.add_argument("--ransac-reproj-threshold", type=float, default=6.0)
    parser.add_argument("--max-cluster-area-frac", type=float, default=0.20)
    parser.add_argument("--max-cluster-width-frac", type=float, default=0.60)
    parser.add_argument("--max-cluster-height-frac", type=float, default=0.60)
    parser.add_argument("--allow-cluster-too-scattered", action="store_true", default=False)
    parser.add_argument("--max-distance-from-expected-m", type=float, default=180.0)
    parser.add_argument("--camera-angle-deg", type=float, default=60.0)
    parser.add_argument("--angle-convention", choices=["from-horizon", "from-nadir"], default="from-horizon")
    parser.add_argument("--horizontal-fov-deg", type=float, default=73.0)
    parser.add_argument("--fallback-altitude-m", type=float, default=119.0)
    parser.add_argument("--query-scale-fill", choices=["blur", "median", "gray", "black"], default="blur")
    parser.add_argument("--max-debug-images", type=int, default=200)
    args = parser.parse_args()

    attempted, used, fallback_attempted, fallback_used = refine_local_best(args)
    print("\nStage 10.4 local-best cluster-lookpoint summary")
    print("=" * 80)
    print(f"Candidate rows attempted: {attempted}")
    print(f"Candidate rows used: {used}")
    print(f"Fallback rows attempted: {fallback_attempted}")
    print(f"Fallback rows upgraded to local-best cluster guesses: {fallback_used}")
    print(f"Wrote: {args.out}")
    if args.debug_dir is not None:
        print(f"Wrote debug images to: {args.debug_dir}")


if __name__ == "__main__":
    main()
