"""Merge Stage 7 local rerun CSVs back into the temporal corrected CSV."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge local Stage 7 rerun CSVs into the temporal-corrected CSV.")
    parser.add_argument("--base-csv", required=True, type=Path, help="Stage 7 temporal corrected CSV or original raw CSV.")
    parser.add_argument("--rerun-csv", required=True, nargs="+", type=Path, help="One or more local rerun CSVs.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-score-improvement", type=float, default=0.50)
    args = parser.parse_args()

    base_header, base_rows = _read_rows(args.base_csv)
    base_by_frame = {_frame(row): row for row in base_rows}
    replacements: dict[int, tuple[dict[str, str], float]] = {}

    for rerun_path in args.rerun_csv:
        _, rerun_rows = _read_rows(rerun_path)
        for rerun in rerun_rows:
            frame = _frame(rerun)
            base = base_by_frame.get(frame)
            if base is None:
                continue
            base_score = row_confidence(base)
            rerun_score = row_confidence(rerun)
            replace = False
            if _accepted(rerun) and not _accepted(base):
                replace = True
            elif _accepted(rerun) and rerun_score >= base_score + args.min_score_improvement:
                replace = True
            if replace:
                existing = replacements.get(frame)
                if existing is None or rerun_score > existing[1]:
                    replacements[frame] = (rerun, rerun_score)

    extra_cols = [
        "stage7_local_rerun_replaced",
        "stage7_local_rerun_score",
        "stage7_local_rerun_reason",
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
                rerun, score = replacements[frame]
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
                output["stage7_local_rerun_replaced"] = "1"
                output["stage7_local_rerun_score"] = f"{score:.6f}"
                output["stage7_local_rerun_reason"] = rerun.get("filter_reason", "")
            else:
                output["stage7_local_rerun_replaced"] = "0"
                output["stage7_local_rerun_score"] = ""
                output["stage7_local_rerun_reason"] = ""
            writer.writerow({k: output.get(k, "") for k in out_header})

    print(f"Merged {len(replacements)} local-rerun replacements into: {args.out}")


if __name__ == "__main__":
    main()
