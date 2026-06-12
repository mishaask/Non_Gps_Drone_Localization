"""Merge Stage 10 path-guided local reruns.

For every non-seed frame, this tool can ignore the old global first-pass match
and choose the best accepted local rerun that is close to the expected path
coordinate.  If no local visual match is good enough, it writes the expected
path coordinate into filtered_latitude/filtered_longitude for KML export while
leaving filter_accepted=0.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from dataclasses import dataclass
from pathlib import Path

from gps_ex1.geometry.geo import haversine_m
from gps_ex1.tools.stage7_temporal_consensus import (
    _dedup_fieldnames,
    _float_or_none,
    _int_or_zero,
    _safe_fieldnames,
    row_confidence,
)


@dataclass(frozen=True)
class PlanRow:
    query_frame_index: int
    query_time_s: float
    seed_trusted: bool
    expected_lat: float | None
    expected_lon: float | None
    expected_method: str
    will_rerun: bool
    path_window_id: int | None


@dataclass(frozen=True)
class Candidate:
    window_id: int
    row: dict[str, str]
    score: float
    dist_to_expected_m: float | None


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    raw_header = _safe_fieldnames(path)
    header = _dedup_fieldnames(raw_header)
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        next(reader)
        for values in reader:
            if len(values) < len(header):
                values = values + [""] * (len(header) - len(values))
            rows.append(dict(zip(header, values)))
    return header, rows


def _write_rows(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def _accepted(row: dict[str, str]) -> bool:
    return (row.get("filter_accepted") or "").strip() in {"1", "true", "True"}


def _parse_window_id(path: Path) -> int | None:
    match = re.search(r"path_window_(\d+)", path.name)
    if match:
        return int(match.group(1))
    return None


def _read_plan(path: Path) -> dict[int, PlanRow]:
    _, rows = _read_rows(path)
    out: dict[int, PlanRow] = {}
    for row in rows:
        frame = _int_or_zero(row.get("query_frame_index"))
        window_text = (row.get("path_window_id") or "").strip()
        out[frame] = PlanRow(
            query_frame_index=frame,
            query_time_s=_float_or_none(row.get("query_time_s")) or 0.0,
            seed_trusted=(row.get("seed_trusted") or "").strip() == "1",
            expected_lat=_float_or_none(row.get("expected_latitude")),
            expected_lon=_float_or_none(row.get("expected_longitude")),
            expected_method=(row.get("expected_method") or "").strip(),
            will_rerun=(row.get("will_rerun") or "").strip() == "1",
            path_window_id=None if not window_text else int(window_text),
        )
    return out


def _candidate_distance(row: dict[str, str], plan: PlanRow) -> float | None:
    lat = _float_or_none(row.get("pred_latitude"))
    lon = _float_or_none(row.get("pred_longitude"))
    if lat is None or lon is None or plan.expected_lat is None or plan.expected_lon is None:
        return None
    return float(haversine_m(lat, lon, plan.expected_lat, plan.expected_lon))


def _read_candidates(rerun_dir: Path, rerun_glob: str, plan_by_frame: dict[int, PlanRow], args: argparse.Namespace) -> dict[int, list[Candidate]]:
    candidates: dict[int, list[Candidate]] = {}
    pattern = str(rerun_dir / rerun_glob)
    for path_text in sorted(glob.glob(pattern)):
        path = Path(path_text)
        window_id = _parse_window_id(path)
        if window_id is None:
            print(f"Skipping rerun CSV with no path window id in name: {path.name}")
            continue
        _, rows = _read_rows(path)
        for row in rows:
            frame = _int_or_zero(row.get("query_frame_index"))
            plan = plan_by_frame.get(frame)
            if plan is None or not plan.will_rerun:
                continue
            if plan.path_window_id is not None and plan.path_window_id != window_id:
                continue
            if not _accepted(row):
                continue
            score = row_confidence(row)
            if score < args.min_candidate_confidence:
                continue
            dist = _candidate_distance(row, plan)
            if dist is None or dist > args.max_distance_from_expected_m:
                continue
            candidates.setdefault(frame, []).append(Candidate(window_id=window_id, row=row, score=score, dist_to_expected_m=dist))
    return candidates


def _clear_match_fields(row: dict[str, str]) -> None:
    for key in [
        "pred_latitude",
        "pred_longitude",
        "raw_pred_latitude",
        "raw_pred_longitude",
        "matched_reference_image",
        "matched_reference_flight",
        "matched_reference_frame_index",
        "matched_reference_time_s",
        "matched_reference_timecode",
        "retrieval_distance",
        "retrieval_similarity",
        "selected_query_scale",
        "orb_good_matches",
        "homography_inliers",
        "mean_match_distance",
        "verification_inlier_ratio",
        "projected_center_x",
        "projected_center_y",
        "projected_center_inside",
        "homography_quad_area_frac",
        "homography_geometry_ok",
        "verification_geometry_reason",
        "query_inlier_bbox_area_frac",
        "query_inlier_bbox_width_frac",
        "query_inlier_bbox_height_frac",
        "reference_inlier_bbox_area_frac",
        "reference_inlier_bbox_width_frac",
        "reference_inlier_bbox_height_frac",
    ]:
        if key in row:
            row[key] = ""


def _fmt(value: float | None, decimals: int = 8) -> str:
    if value is None:
        return ""
    return f"{float(value):.{decimals}f}"


def _best_candidate(candidates: list[Candidate]) -> Candidate | None:
    if not candidates:
        return None
    return max(candidates, key=lambda c: (c.score, -float(c.dist_to_expected_m or 1e9)))


def merge_stage10(args: argparse.Namespace) -> tuple[int, int, int]:
    base_header, base_rows = _read_rows(args.base_csv)
    plan_by_frame = _read_plan(args.plan_csv)
    candidates_by_frame = _read_candidates(args.rerun_dir, args.rerun_glob, plan_by_frame, args)

    extra_cols = [
        "stage10_original_filter_accepted",
        "stage10_original_filter_reason",
        "stage10_original_pred_latitude",
        "stage10_original_pred_longitude",
        "stage10_seed_trusted",
        "stage10_expected_latitude",
        "stage10_expected_longitude",
        "stage10_expected_method",
        "stage10_path_window_id",
        "stage10_local_rerun_replaced",
        "stage10_local_rerun_score",
        "stage10_local_rerun_dist_to_expected_m",
        "stage10_final_source",
        "filtered_latitude",
        "filtered_longitude",
        "filtered_method",
    ]
    out_header = list(dict.fromkeys(base_header + extra_cols))
    out_rows: list[dict[str, str]] = []
    replaced = 0
    seed_kept = 0
    expected_fallback = 0

    for base in base_rows:
        frame = _int_or_zero(base.get("query_frame_index"))
        plan = plan_by_frame.get(frame)
        output = dict(base)

        original_filter_accepted = output.get("filter_accepted", "")
        original_filter_reason = output.get("filter_reason", "")
        original_lat = output.get("pred_latitude", "")
        original_lon = output.get("pred_longitude", "")

        output["stage10_original_filter_accepted"] = original_filter_accepted
        output["stage10_original_filter_reason"] = original_filter_reason
        output["stage10_original_pred_latitude"] = original_lat
        output["stage10_original_pred_longitude"] = original_lon

        if plan is None:
            output["stage10_final_source"] = "no_plan_keep_original"
            out_rows.append(output)
            continue

        output["stage10_seed_trusted"] = "1" if plan.seed_trusted else "0"
        output["stage10_expected_latitude"] = _fmt(plan.expected_lat)
        output["stage10_expected_longitude"] = _fmt(plan.expected_lon)
        output["stage10_expected_method"] = plan.expected_method
        output["stage10_path_window_id"] = "" if plan.path_window_id is None else f"{plan.path_window_id:03d}"
        output["stage10_local_rerun_replaced"] = "0"
        output["stage10_local_rerun_score"] = ""
        output["stage10_local_rerun_dist_to_expected_m"] = ""

        if plan.seed_trusted and not args.rerun_seed_anchors:
            output["filtered_latitude"] = output.get("pred_latitude", "")
            output["filtered_longitude"] = output.get("pred_longitude", "")
            output["filtered_method"] = "seed_anchor_visual"
            output["stage10_final_source"] = "seed_anchor_keep_original"
            seed_kept += 1
            out_rows.append(output)
            continue

        candidate = _best_candidate(candidates_by_frame.get(frame, []))
        if candidate is not None:
            # Start from the winning rerun row, but keep original/expected metadata.
            winner = dict(candidate.row)
            for key in extra_cols:
                if key in output:
                    winner[key] = output[key]
            winner["stage10_local_rerun_replaced"] = "1"
            winner["stage10_local_rerun_score"] = f"{candidate.score:.6f}"
            winner["stage10_local_rerun_dist_to_expected_m"] = "" if candidate.dist_to_expected_m is None else f"{candidate.dist_to_expected_m:.3f}"
            winner["stage10_final_source"] = "local_path_rerun"
            winner["filtered_latitude"] = winner.get("pred_latitude", "")
            winner["filtered_longitude"] = winner.get("pred_longitude", "")
            winner["filtered_method"] = "local_path_rerun_visual"
            replaced += 1
            out_rows.append(winner)
            continue

        # No good local visual match.  Use the path estimate for KML, but do not
        # claim a visual match was accepted.
        output["filter_accepted"] = "0"
        output["filter_reason"] = "stage10_no_local_path_match"
        output["filtered_latitude"] = _fmt(plan.expected_lat)
        output["filtered_longitude"] = _fmt(plan.expected_lon)
        output["filtered_method"] = f"path_expected_{plan.expected_method}"
        output["stage10_final_source"] = "path_expected_fallback"
        if args.clear_untrusted_matches:
            _clear_match_fields(output)
        expected_fallback += 1
        out_rows.append(output)

    _write_rows(args.out, out_header, out_rows)
    return replaced, seed_kept, expected_fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Stage 10 path-guided local reruns into a final CSV.")
    parser.add_argument("--base-csv", required=True, type=Path)
    parser.add_argument("--plan-csv", required=True, type=Path)
    parser.add_argument("--rerun-dir", required=True, type=Path)
    parser.add_argument("--rerun-glob", default="stage10_rerun_path_window_*_top*.csv")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-distance-from-expected-m", type=float, default=140.0)
    parser.add_argument("--min-candidate-confidence", type=float, default=6.0)
    parser.add_argument("--clear-untrusted-matches", action="store_true", help="Clear reference/match fields when no path-local visual match is trusted.")
    parser.add_argument("--rerun-seed-anchors", action="store_true", help="Allow reruns to replace trusted seed anchors too.")
    args = parser.parse_args()

    replaced, seed_kept, expected_fallback = merge_stage10(args)
    print("\nStage 10 path-guided merge summary")
    print("=" * 80)
    print(f"Seed anchors kept: {seed_kept}")
    print(f"Local rerun visual replacements: {replaced}")
    print(f"Path expected fallbacks: {expected_fallback}")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
