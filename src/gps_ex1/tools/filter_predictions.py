"""Filter an existing prediction CSV using temporal continuity.

This tool is useful when the expensive video localization already ran. It does
not re-read the video or reference images; it cleans the CSV by rejecting weak
visual matches, rejecting impossible jumps, and optionally holding/smoothing the
last accepted position.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gps_ex1.localization.temporal_filter import (
    CandidateObservation,
    OnlineTrackState,
    TemporalFilterConfig,
    exponential_smooth_state,
    hold_state_at_time,
    select_temporal_candidate,
)


EXTRA_COLUMNS = [
    "filtered_latitude",
    "filtered_longitude",
    "filter_accepted",
    "filter_reason",
    "filter_visual_score",
    "filter_temporal_distance_m",
    "filter_fused_score",
    "filter_smoothed",
]


def _float_or_none(text: str | None) -> float | None:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    return float(text)


def _int_or_zero(text: str | None) -> int:
    value = _float_or_none(text)
    return 0 if value is None else int(value)


def _candidate_from_row(row: dict[str, str]) -> CandidateObservation | None:
    lat = _float_or_none(row.get("pred_latitude"))
    lon = _float_or_none(row.get("pred_longitude"))
    if lat is None or lon is None:
        return None
    return CandidateObservation(
        index=-1,
        latitude=lat,
        longitude=lon,
        matched_flight=(row.get("matched_reference_flight") or "").strip(),
        matched_frame_index=_int_or_zero(row.get("matched_reference_frame_index")),
        matched_time_s=_float_or_none(row.get("matched_reference_time_s")) or 0.0,
        retrieval_distance=_float_or_none(row.get("retrieval_distance")) or 0.0,
        retrieval_similarity=_float_or_none(row.get("retrieval_similarity")) or 0.0,
        good_matches=_int_or_zero(row.get("orb_good_matches")),
        homography_inliers=_int_or_zero(row.get("homography_inliers")),
        mean_match_distance=_float_or_none(row.get("mean_match_distance")) or float("inf"),
    )


def filter_prediction_csv(
    prediction_csv: Path,
    output_csv: Path,
    cfg: TemporalFilterConfig,
    hold_last_on_reject: bool = True,
    overwrite_prediction_columns: bool = False,
) -> dict[str, int]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with prediction_csv.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
        original_fieldnames = list(reader.fieldnames or [])

    fieldnames = list(original_fieldnames)
    for column in EXTRA_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)

    previous: OnlineTrackState | None = None
    accepted = 0
    held = 0
    rejected = 0
    written_rows: list[dict[str, str]] = []

    for row in rows:
        query_time_s = _float_or_none(row.get("query_time_s")) or 0.0
        candidate = _candidate_from_row(row)

        if candidate is None:
            rejected += 1
            row.update({column: "" for column in EXTRA_COLUMNS})
            row["filter_accepted"] = "0"
            row["filter_reason"] = "missing_prediction"
            written_rows.append(row)
            continue

        selection = select_temporal_candidate([candidate], previous, query_time_s, cfg)
        if selection.accepted and selection.candidate is not None:
            previous = exponential_smooth_state(previous, selection.candidate, query_time_s, cfg)
            accepted += 1
            filtered_lat = previous.latitude
            filtered_lon = previous.longitude
            smoothed = "1"
            accepted_text = "1"
            reason = selection.reason
        elif hold_last_on_reject and previous is not None:
            held += 1
            # Keep the last physically plausible estimate. This creates a
            # realistic low-confidence track instead of a huge KML jump.
            filtered_lat = previous.latitude
            filtered_lon = previous.longitude
            smoothed = "1"
            accepted_text = "0"
            reason = selection.reason + "_held_last"
        else:
            rejected += 1
            filtered_lat = None
            filtered_lon = None
            smoothed = "0"
            accepted_text = "0"
            reason = selection.reason

        row["filtered_latitude"] = "" if filtered_lat is None else f"{filtered_lat:.8f}"
        row["filtered_longitude"] = "" if filtered_lon is None else f"{filtered_lon:.8f}"
        row["filter_accepted"] = accepted_text
        row["filter_reason"] = reason
        row["filter_visual_score"] = f"{selection.visual_score:.6f}"
        row["filter_temporal_distance_m"] = "" if selection.temporal_distance_m is None else f"{selection.temporal_distance_m:.3f}"
        row["filter_fused_score"] = f"{selection.fused_score:.6f}"
        row["filter_smoothed"] = smoothed

        if overwrite_prediction_columns:
            row["pred_latitude"] = "" if filtered_lat is None else f"{filtered_lat:.8f}"
            row["pred_longitude"] = "" if filtered_lon is None else f"{filtered_lon:.8f}"

        written_rows.append(row)

    with output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(written_rows)

    return {"rows": len(rows), "accepted": accepted, "held": held, "rejected": rejected}


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter a prediction CSV with temporal continuity constraints.")
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-inliers", type=int, default=6)
    parser.add_argument("--min-good-matches", type=int, default=8)
    parser.add_argument("--max-speed-mps", type=float, default=18.0)
    parser.add_argument("--base-gate-m", type=float, default=55.0)
    parser.add_argument("--hard-jump-m", type=float, default=180.0)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--no-hold-last", action="store_true", help="Leave rejected rows blank instead of holding the last valid position.")
    parser.add_argument("--overwrite-prediction-columns", action="store_true", help="Write filtered coordinates into pred_latitude/pred_longitude too.")
    args = parser.parse_args()

    cfg = TemporalFilterConfig(
        min_inliers=args.min_inliers,
        min_good_matches=args.min_good_matches,
        max_speed_mps=args.max_speed_mps,
        base_gate_m=args.base_gate_m,
        hard_jump_m=args.hard_jump_m,
        ema_alpha=args.ema_alpha,
    )
    summary = filter_prediction_csv(
        prediction_csv=args.prediction_csv,
        output_csv=args.out,
        cfg=cfg,
        hold_last_on_reject=not args.no_hold_last,
        overwrite_prediction_columns=args.overwrite_prediction_columns,
    )
    print(f"Wrote filtered predictions: {args.out}")
    print(f"Rows: {summary['rows']}")
    print(f"Accepted new visual positions: {summary['accepted']}")
    print(f"Held previous position after rejection: {summary['held']}")
    print(f"Rejected without output: {summary['rejected']}")


if __name__ == "__main__":
    main()
