"""Merge Stage 8 context-approved reruns safely.

Unlike Stage 7 merging, this tool only considers a rerun row if its region is
listed as an approved candidate for that frame in stage8_context_rerun_plan.csv.
It also only replaces rows with accepted rerun results that improve confidence.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gps_ex1.tools.stage7_temporal_consensus import _dedup_fieldnames, _safe_fieldnames, row_confidence


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


def _frame(row: dict[str, str]) -> int:
    return int(float((row.get("query_frame_index") or "0").strip() or 0))


def _accepted(row: dict[str, str]) -> bool:
    return (row.get("filter_accepted") or "").strip() == "1"


def _read_plan(path: Path) -> dict[int, set[int]]:
    approved: dict[int, set[int]] = {}
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if (row.get("will_rerun") or "").strip() != "1":
                continue
            frame = int(float((row.get("query_frame_index") or "0").strip() or 0))
            regions: set[int] = set()
            for part in (row.get("candidate_regions") or "").replace(",", ";").split(";"):
                token = part.strip()
                if token:
                    regions.add(int(token))
            approved[frame] = regions
    return approved


def _copy_prediction_fields(output: dict[str, str], rerun: dict[str, str]) -> None:
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
        "filter_accepted",
        "filter_reason",
        "filter_visual_score",
        "filter_fused_score",
    ]:
        if key in rerun:
            output[key] = rerun[key]


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely merge Stage 8 context reruns.")
    parser.add_argument("--base-csv", required=True, type=Path)
    parser.add_argument("--plan-csv", required=True, type=Path)
    parser.add_argument("--rerun", required=True, nargs="*", help="Pairs: REGION_ID CSV_PATH REGION_ID CSV_PATH ...")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-score-improvement", type=float, default=0.50)
    args = parser.parse_args()

    if len(args.rerun) % 2 != 0:
        raise SystemExit("--rerun must be supplied as pairs: REGION_ID CSV_PATH")

    base_header, base_rows = _read_rows(args.base_csv)
    base_by_frame = {_frame(row): row for row in base_rows}
    approved_regions = _read_plan(args.plan_csv)
    replacements: dict[int, tuple[int, dict[str, str], float, float]] = {}

    pairs: list[tuple[int, Path]] = []
    for i in range(0, len(args.rerun), 2):
        pairs.append((int(args.rerun[i]), Path(args.rerun[i + 1])))

    for region_id, rerun_path in pairs:
        _, rerun_rows = _read_rows(rerun_path)
        for rerun in rerun_rows:
            frame = _frame(rerun)
            if region_id not in approved_regions.get(frame, set()):
                continue
            base = base_by_frame.get(frame)
            if base is None:
                continue
            if not _accepted(rerun):
                continue
            base_score = row_confidence(base)
            rerun_score = row_confidence(rerun)
            if rerun_score < base_score + args.min_score_improvement and _accepted(base):
                continue
            existing = replacements.get(frame)
            if existing is None or rerun_score > existing[2]:
                replacements[frame] = (region_id, rerun, rerun_score, base_score)

    extra_cols = [
        "stage8_context_rerun_replaced",
        "stage8_context_rerun_region_id",
        "stage8_context_rerun_score",
        "stage8_context_before_score",
        "stage8_context_rerun_reason",
    ]
    out_header = list(dict.fromkeys(base_header + extra_cols))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=out_header)
        writer.writeheader()
        for base in base_rows:
            frame = _frame(base)
            output = dict(base)
            if frame in replacements:
                region_id, rerun, rerun_score, base_score = replacements[frame]
                _copy_prediction_fields(output, rerun)
                output["stage8_context_rerun_replaced"] = "1"
                output["stage8_context_rerun_region_id"] = str(region_id)
                output["stage8_context_rerun_score"] = f"{rerun_score:.6f}"
                output["stage8_context_before_score"] = f"{base_score:.6f}"
                output["stage8_context_rerun_reason"] = rerun.get("filter_reason", "")
            else:
                output["stage8_context_rerun_replaced"] = "0"
                output["stage8_context_rerun_region_id"] = ""
                output["stage8_context_rerun_score"] = ""
                output["stage8_context_before_score"] = ""
                output["stage8_context_rerun_reason"] = ""
            writer.writerow({k: output.get(k, "") for k in out_header})

    print(f"Stage 8 merged {len(replacements)} context-approved rerun replacements into: {args.out}")


if __name__ == "__main__":
    main()
