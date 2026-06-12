"""Stage 10.5: conservative local candidate upgrade for Stage 10.3 fallback frames.

This tool implements the corrected version of the "best match in the expected area" idea.

It is intentionally more conservative than Stage 10.4:
  * Stage 10.3 remains the trusted backbone.
  * Existing accepted visual matches can be cluster-refined.
  * Fallback rows are upgraded ONLY when a local rerun candidate is good enough.
  * A candidate is NOT accepted merely because it is the best candidate in the local window.
  * Rows without a good candidate stay as path_expected_fallback.

Recommended flow:
  1) Run Stage 10.3 path-guided local reruns.
  2) Run Stage 10.3 merge.
  3) Run this tool on the merged CSV and the rerun candidate CSVs.
  4) Export KML from this tool's output.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path

from gps_ex1.preprocess.reference_index import load_reference_index
from gps_ex1.tools.stage10_cluster_lookpoint_refine import (
    ClusterResult,
    _accepted,
    _build_reference_lookup,
    _float_or_none,
    _fmt,
    _int_or_none,
    _read_rows,
    _refine_one_row,
    _require_cv2,
    _write_rows,
)
from gps_ex1.tools.stage10_local_best_cluster_refine import (
    _copy_cluster_columns,
    _expected_latlon,
    _haversine_m,
    _row_has_reference,
    _score_candidate,
)


VISUAL_METHOD_KEYWORDS = (
    "visual",
    "seed",
    "local_rerun",
    "cluster_refined_visual_match",
    "accepted",
)


def _source_text(row: dict[str, str]) -> str:
    return " ".join(
        str(row.get(k, ""))
        for k in ["stage10_final_source", "filtered_method", "match_reason", "reason", "local_best_kind"]
    ).strip()


def _is_fallback(row: dict[str, str]) -> bool:
    text = _source_text(row).lower()
    return "fallback" in text or "path_expected" in text


def _is_existing_visual(row: dict[str, str]) -> bool:
    if _is_fallback(row):
        return False
    if not _row_has_reference(row):
        return False
    if _accepted(row):
        return True
    text = _source_text(row).lower()
    return any(word in text for word in VISUAL_METHOD_KEYWORDS)


def _candidate_reason(row: dict[str, str]) -> str:
    return (row.get("match_reason") or row.get("reason") or "").strip()


# Candidate CSVs come from several pipeline generations.  The older Stage 10.5
# code only checked generic names like ``good_matches`` and ``inlier_ratio``.
# Our localize_video outputs commonly use names such as ``orb_good_matches``
# and ``verification_inlier_ratio`` instead.  Missing those fields caused every
# fallback candidate to look like good=0 / ratio=0 and be rejected as low quality.
CONFIDENCE_FIELDS = (
    "confidence",
    "match_confidence",
    "retrieval_similarity",
    "descriptor_similarity",
    "similarity",
    "score",
)
INLIER_FIELDS = (
    "homography_inliers",
    "inliers",
    "verification_inliers",
    "ransac_inliers",
    "num_inliers",
)
GOOD_MATCH_FIELDS = (
    "good_matches",
    "matches",
    "orb_good_matches",
    "verification_good_matches",
    "num_good_matches",
    "total_good_matches",
)
INLIER_RATIO_FIELDS = (
    "inlier_ratio",
    "verification_inlier_ratio",
    "homography_inlier_ratio",
    "ransac_inlier_ratio",
)


def _first_float_with_source(row: dict[str, str], fields: tuple[str, ...]) -> tuple[float | None, str]:
    for field in fields:
        parsed = _float_or_none(row.get(field))
        if parsed is not None:
            return parsed, field
    return None, ""


def _candidate_confidence_with_source(row: dict[str, str]) -> tuple[float, str]:
    parsed, source = _first_float_with_source(row, CONFIDENCE_FIELDS)
    return (float(parsed), source) if parsed is not None else (0.0, "")


def _candidate_inliers_with_source(row: dict[str, str]) -> tuple[int, str]:
    parsed, source = _first_float_with_source(row, INLIER_FIELDS)
    return (int(round(parsed)), source) if parsed is not None else (0, "")


def _candidate_good_matches_with_source(row: dict[str, str]) -> tuple[int, str]:
    parsed, source = _first_float_with_source(row, GOOD_MATCH_FIELDS)
    return (int(round(parsed)), source) if parsed is not None else (0, "")


def _candidate_inlier_ratio_with_source(row: dict[str, str]) -> tuple[float, str]:
    parsed, source = _first_float_with_source(row, INLIER_RATIO_FIELDS)
    if parsed is not None:
        return float(parsed), source
    inliers = _candidate_inliers(row)
    good = _candidate_good_matches(row)
    if good > 0:
        return float(inliers) / float(good), "computed_inliers_over_good_matches"
    return 0.0, ""


def _candidate_confidence(row: dict[str, str]) -> float:
    return _candidate_confidence_with_source(row)[0]


def _candidate_inliers(row: dict[str, str]) -> int:
    return _candidate_inliers_with_source(row)[0]


def _candidate_good_matches(row: dict[str, str]) -> int:
    return _candidate_good_matches_with_source(row)[0]


def _candidate_inlier_ratio(row: dict[str, str]) -> float:
    return _candidate_inlier_ratio_with_source(row)[0]


def _candidate_quality_ok(row: dict[str, str], args: argparse.Namespace) -> tuple[bool, str]:
    if not _row_has_reference(row):
        return False, "no_reference"
    if args.require_candidate_accepted and not _accepted(row):
        return False, "candidate_not_accepted"
    reason = _candidate_reason(row).lower()
    if args.require_reason_accepted and reason and reason != "accepted":
        return False, f"candidate_reason_{reason}"
    conf = _candidate_confidence(row)
    inliers = _candidate_inliers(row)
    good = _candidate_good_matches(row)
    ratio = _candidate_inlier_ratio(row)
    if conf < args.min_candidate_confidence:
        return False, f"candidate_confidence_lt_{args.min_candidate_confidence:g}"
    if inliers < args.min_candidate_inliers:
        return False, f"candidate_inliers_lt_{args.min_candidate_inliers}"
    if good < args.min_candidate_good_matches:
        return False, f"candidate_good_matches_lt_{args.min_candidate_good_matches}"
    if ratio < args.min_candidate_inlier_ratio:
        return False, f"candidate_inlier_ratio_lt_{args.min_candidate_inlier_ratio:g}"
    return True, "candidate_quality_ok"


def _candidate_sort_key(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        1.0 if _accepted(row) else 0.0,
        _candidate_confidence(row),
        float(_candidate_inliers(row)),
        _score_candidate(row),
    )


def _build_candidate_lists(rerun_dir: Path, rerun_glob: str) -> dict[int, list[dict[str, str]]]:
    candidates: dict[int, list[dict[str, str]]] = {}
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
            row["stage10_5_candidate_csv"] = str(path)
            candidates.setdefault(frame, []).append(row)
    for frame, rows in candidates.items():
        rows.sort(key=_candidate_sort_key, reverse=True)
    return candidates


def _ensure_fields(header: list[str]) -> list[str]:
    extra = [
        "stage10_5_status",
        "stage10_5_used",
        "stage10_5_kind",
        "stage10_5_reject_reason",
        "stage10_5_candidate_reason",
        "stage10_5_candidate_confidence",
        "stage10_5_candidate_confidence_source",
        "stage10_5_candidate_inliers",
        "stage10_5_candidate_inliers_source",
        "stage10_5_candidate_good_matches",
        "stage10_5_candidate_good_matches_source",
        "stage10_5_candidate_inlier_ratio",
        "stage10_5_candidate_inlier_ratio_source",
        "stage10_5_candidate_csv",
        "stage10_5_distance_from_expected_m",
        "stage10_5_original_filtered_latitude",
        "stage10_5_original_filtered_longitude",
        "stage10_5_original_filtered_method",
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
    return list(dict.fromkeys(header + extra))


class _RefineArgs(argparse.Namespace):
    pass


def _make_refine_args(args: argparse.Namespace) -> _RefineArgs:
    refine_args = _RefineArgs()
    refine_args.project_root = args.project_root
    refine_args.debug_dir = args.debug_dir
    refine_args.query_scale_fill = args.query_scale_fill
    refine_args.match_backend = args.match_backend
    refine_args.orb_nfeatures = args.orb_nfeatures
    refine_args.ratio_test = args.ratio_test
    refine_args.min_inliers = args.min_cluster_inliers
    refine_args.ransac_reproj_threshold = args.ransac_reproj_threshold
    refine_args.max_cluster_area_frac = args.max_cluster_area_frac
    refine_args.max_cluster_width_frac = args.max_cluster_width_frac
    refine_args.max_cluster_height_frac = args.max_cluster_height_frac
    refine_args.camera_angle_deg = args.camera_angle_deg
    refine_args.angle_convention = args.angle_convention
    refine_args.horizontal_fov_deg = args.horizontal_fov_deg
    refine_args.fallback_altitude_m = args.fallback_altitude_m
    return refine_args


def _blank_result(status: str) -> ClusterResult:
    return ClusterResult(status=status, used=False)


def _apply_result_to_output(output: dict[str, str], result: ClusterResult, kind: str, distance_m: float | None) -> None:
    output["filtered_latitude"] = _fmt(result.lat, 8)
    output["filtered_longitude"] = _fmt(result.lon, 8)
    output["filtered_method"] = kind
    output["stage10_5_used"] = "1"
    output["stage10_5_kind"] = kind
    output["stage10_5_distance_from_expected_m"] = _fmt(distance_m, 3)
    _copy_cluster_columns(output, result)


def _candidate_metric_snapshot(row: dict[str, str]) -> dict[str, str]:
    conf, conf_src = _candidate_confidence_with_source(row)
    inliers, inliers_src = _candidate_inliers_with_source(row)
    good, good_src = _candidate_good_matches_with_source(row)
    ratio, ratio_src = _candidate_inlier_ratio_with_source(row)
    return {
        "candidate_reason": _candidate_reason(row),
        "candidate_filter_accepted": str(row.get("filter_accepted", "")),
        "candidate_confidence": _fmt(conf, 4),
        "candidate_confidence_source": conf_src,
        "candidate_inliers": str(inliers),
        "candidate_inliers_source": inliers_src,
        "candidate_good_matches": str(good),
        "candidate_good_matches_source": good_src,
        "candidate_inlier_ratio": _fmt(ratio, 4),
        "candidate_inlier_ratio_source": ratio_src,
        "candidate_csv": row.get("stage10_5_candidate_csv", ""),
        "candidate_ref_video": str(row.get("reference_video") or row.get("matched_reference_video") or row.get("ref_video") or ""),
        "candidate_ref_frame": str(row.get("reference_frame_index") or row.get("matched_reference_frame") or row.get("ref_frame") or ""),
        "candidate_ref_image": str(row.get("matched_reference_image") or row.get("reference_image") or row.get("image_path") or ""),
    }


def _write_diagnostics(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=["query_frame_index", "candidate_rank", "decision", "reject_reason"])
            writer.writeheader()
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def run_conservative_upgrade(args: argparse.Namespace) -> dict[str, int]:
    header, merged_rows = _read_rows(args.merged_csv)
    candidates_by_frame = _build_candidate_lists(args.rerun_dir, args.rerun_glob)

    index = load_reference_index(args.reference_index)
    lookup = _build_reference_lookup(index, args.project_root)
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(args.query_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open query video: {args.query_video}")

    refine_args = _make_refine_args(args)
    out_rows: list[dict[str, str]] = []
    diagnostic_rows: list[dict[str, str]] = []

    stats = {
        "rows_total": 0,
        "existing_visual_attempted": 0,
        "existing_visual_refined": 0,
        "fallback_rows": 0,
        "fallback_with_candidate": 0,
        "fallback_candidates_quality_ok": 0,
        "fallback_upgraded": 0,
        "fallback_kept": 0,
        "candidate_rejected_quality": 0,
        "candidate_rejected_cluster": 0,
        "candidate_rejected_distance": 0,
    }

    debug_counter = 0
    for row in merged_rows:
        stats["rows_total"] += 1
        output = dict(row)
        output["stage10_5_original_filtered_latitude"] = output.get("filtered_latitude", "")
        output["stage10_5_original_filtered_longitude"] = output.get("filtered_longitude", "")
        output["stage10_5_original_filtered_method"] = output.get("filtered_method", "")
        output["stage10_5_used"] = "0"
        output["stage10_5_kind"] = "kept_original"
        output["stage10_5_status"] = "not_attempted"
        output["stage10_5_reject_reason"] = ""

        frame = _int_or_none(row.get("query_frame_index") or row.get("query_frame") or row.get("frame"))
        candidate: dict[str, str] | None = None
        kind = ""
        reject_reason = ""

        if args.refine_existing_visual and _is_existing_visual(row):
            candidate = dict(row)
            kind = "cluster_refined_visual_match"
            stats["existing_visual_attempted"] += 1
        elif _is_fallback(row):
            stats["fallback_rows"] += 1
            frame_candidates = candidates_by_frame.get(frame or -1, [])
            if frame_candidates:
                stats["fallback_with_candidate"] += 1
            for rank, cand in enumerate(frame_candidates, start=1):
                ok, reason = _candidate_quality_ok(cand, args)
                diag = {
                    "query_frame_index": str(frame or ""),
                    "candidate_rank": str(rank),
                    "decision": "quality_ok" if ok else "rejected_quality",
                    "reject_reason": "" if ok else reason,
                }
                diag.update(_candidate_metric_snapshot(cand))
                diagnostic_rows.append(diag)
                if ok:
                    candidate = cand
                    kind = "candidate_upgrade_visual_match"
                    stats["fallback_candidates_quality_ok"] += 1
                    break
                reject_reason = reason
            if candidate is None:
                output["stage10_5_kind"] = "path_expected_fallback"
                output["stage10_5_reject_reason"] = reject_reason or "no_candidate"
                if frame_candidates:
                    stats["candidate_rejected_quality"] += 1
                stats["fallback_kept"] += 1

        result = _blank_result(output.get("stage10_5_reject_reason") or "not_attempted")
        distance_m: float | None = None
        if candidate is not None:
            debug_counter += 1
            debug_id = debug_counter if args.debug_dir is not None and (args.max_debug_images is None or debug_counter <= args.max_debug_images) else None
            _, result = _refine_one_row(candidate, cap, index, lookup, refine_args, debug_id)
            expected = _expected_latlon(row)
            if result.lat is not None and result.lon is not None and expected is not None:
                distance_m = _haversine_m(expected[0], expected[1], result.lat, result.lon)

            output["stage10_5_status"] = result.status
            output["stage10_5_candidate_reason"] = _candidate_reason(candidate)
            conf, conf_src = _candidate_confidence_with_source(candidate)
            inliers, inliers_src = _candidate_inliers_with_source(candidate)
            good, good_src = _candidate_good_matches_with_source(candidate)
            ratio, ratio_src = _candidate_inlier_ratio_with_source(candidate)
            output["stage10_5_candidate_confidence"] = _fmt(conf, 4)
            output["stage10_5_candidate_confidence_source"] = conf_src
            output["stage10_5_candidate_inliers"] = str(inliers)
            output["stage10_5_candidate_inliers_source"] = inliers_src
            output["stage10_5_candidate_good_matches"] = str(good)
            output["stage10_5_candidate_good_matches_source"] = good_src
            output["stage10_5_candidate_inlier_ratio"] = _fmt(ratio, 4)
            output["stage10_5_candidate_inlier_ratio_source"] = ratio_src
            output["stage10_5_candidate_csv"] = candidate.get("stage10_5_candidate_csv", "")
            output["stage10_5_distance_from_expected_m"] = _fmt(distance_m, 3)
            _copy_cluster_columns(output, result)

            allowed_by_cluster = result.used and result.inliers >= args.min_cluster_inliers
            allowed_by_distance = distance_m is None or distance_m <= args.max_distance_from_expected_m
            if allowed_by_cluster and allowed_by_distance:
                _apply_result_to_output(output, result, kind, distance_m)
                if kind == "cluster_refined_visual_match":
                    stats["existing_visual_refined"] += 1
                else:
                    stats["fallback_upgraded"] += 1
            else:
                output["stage10_5_used"] = "0"
                output["stage10_5_kind"] = "kept_original" if kind == "cluster_refined_visual_match" else "path_expected_fallback"
                if not allowed_by_cluster:
                    output["stage10_5_reject_reason"] = f"cluster_not_strong_enough_{result.status}"
                    if kind != "cluster_refined_visual_match":
                        stats["candidate_rejected_cluster"] += 1
                elif not allowed_by_distance:
                    output["stage10_5_reject_reason"] = f"distance_gt_{args.max_distance_from_expected_m:g}m"
                    if kind != "cluster_refined_visual_match":
                        stats["candidate_rejected_distance"] += 1
                if kind != "cluster_refined_visual_match":
                    stats["fallback_kept"] += 1

        out_rows.append(output)

    cap.release()
    _write_rows(args.out, _ensure_fields(header), out_rows)
    if args.diagnostics_csv is not None:
        _write_diagnostics(args.diagnostics_csv, diagnostic_rows)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 10.5: conservative candidate upgrade. Upgrade fallback rows only when the local rerun candidate is good enough."
    )
    parser.add_argument("--merged-csv", required=True, type=Path)
    parser.add_argument("--rerun-dir", required=True, type=Path)
    parser.add_argument("--rerun-glob", required=True)
    parser.add_argument("--query-video", required=True, type=Path)
    parser.add_argument("--reference-index", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--refine-existing-visual", action="store_true", default=True)
    parser.add_argument("--no-refine-existing-visual", dest="refine_existing_visual", action="store_false")

    parser.add_argument("--match-backend", choices=["lightglue", "orb"], default="lightglue")
    parser.add_argument("--orb-nfeatures", type=int, default=4000)
    parser.add_argument("--ratio-test", type=float, default=0.75)
    parser.add_argument("--min-cluster-inliers", type=int, default=8)
    parser.add_argument("--ransac-reproj-threshold", type=float, default=5.0)
    parser.add_argument("--max-cluster-area-frac", type=float, default=0.14)
    parser.add_argument("--max-cluster-width-frac", type=float, default=0.45)
    parser.add_argument("--max-cluster-height-frac", type=float, default=0.45)
    parser.add_argument("--max-distance-from-expected-m", type=float, default=140.0)

    parser.add_argument("--require-candidate-accepted", action="store_true", default=True)
    parser.add_argument("--allow-nonaccepted-candidates", dest="require_candidate_accepted", action="store_false")
    parser.add_argument("--require-reason-accepted", action="store_true", default=False)
    parser.add_argument("--min-candidate-confidence", type=float, default=5.0)
    parser.add_argument("--min-candidate-inliers", type=int, default=8)
    parser.add_argument("--min-candidate-good-matches", type=int, default=15)
    parser.add_argument("--min-candidate-inlier-ratio", type=float, default=0.18)

    parser.add_argument("--camera-angle-deg", type=float, default=60.0)
    parser.add_argument("--angle-convention", choices=["from-horizon", "from-nadir"], default="from-horizon")
    parser.add_argument("--horizontal-fov-deg", type=float, default=73.0)
    parser.add_argument("--fallback-altitude-m", type=float, default=119.0)
    parser.add_argument("--query-scale-fill", choices=["blur", "median", "gray", "black"], default="blur")
    parser.add_argument("--max-debug-images", type=int, default=150)
    parser.add_argument("--diagnostics-csv", type=Path, default=None, help="Optional CSV explaining each candidate quality decision and parsed metric sources.")
    args = parser.parse_args()

    stats = run_conservative_upgrade(args)
    print("\nStage 10.5 conservative candidate-upgrade summary")
    print("=" * 80)
    for key in [
        "rows_total",
        "existing_visual_attempted",
        "existing_visual_refined",
        "fallback_rows",
        "fallback_with_candidate",
        "fallback_candidates_quality_ok",
        "fallback_upgraded",
        "fallback_kept",
        "candidate_rejected_quality",
        "candidate_rejected_cluster",
        "candidate_rejected_distance",
    ]:
        print(f"{key}: {stats[key]}")
    print(f"Wrote: {args.out}")
    if args.diagnostics_csv is not None:
        print(f"Wrote diagnostics CSV: {args.diagnostics_csv}")
    if args.debug_dir is not None:
        print(f"Wrote debug images to: {args.debug_dir}")


if __name__ == "__main__":
    main()
