"""Derive center/drone prediction CSVs from the selected reference matches.

The localizer first decides which reference frame each query frame matches. This
script reuses those decisions and rewrites the coordinates to either:

* center - estimated ground point in the center of the camera image
* drone  - actual drone GNSS coordinate of the matched reference frame

It can also reapply the same temporal filter to the rewritten coordinates, which
is useful for creating a realistic Google Earth drone-position path without
rerunning the expensive video matcher.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gps_ex1.io.timecode import seconds_to_timecode
from gps_ex1.localization.temporal_filter import (
    CandidateObservation,
    OnlineTrackState,
    TemporalFilterConfig,
    exponential_smooth_state,
    hold_state_at_time,
    select_temporal_candidate,
)
from gps_ex1.preprocess.reference_index import load_reference_index

TIME_COLUMNS = ["query_timecode", "matched_reference_timecode"]
FILTER_COLUMNS = [
    "filter_accepted",
    "filter_reason",
    "filter_visual_score",
    "filter_temporal_distance_m",
    "filter_fused_score",
    "filter_smoothed",
]
RAW_COLUMNS = ["raw_pred_latitude", "raw_pred_longitude"]


def _float_or_none(text: str | None) -> float | None:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _int_or_none(text: str | None) -> int | None:
    value = _float_or_none(text)
    return None if value is None else int(value)


def _build_lookup(reference_index_path: Path, target: str) -> dict[tuple[str, int], tuple[float, float]]:
    index = load_reference_index(reference_index_path)
    lookup: dict[tuple[str, int], tuple[float, float]] = {}
    for i, flight_id in enumerate(index.flight_ids):
        key = (str(flight_id), int(index.frame_indices[i]))
        if target == "center":
            lookup[key] = (float(index.center_lats[i]), float(index.center_lons[i]))
        elif target == "drone":
            lookup[key] = (float(index.drone_lats[i]), float(index.drone_lons[i]))
        else:
            raise ValueError(f"Unsupported target: {target}")
    return lookup


def _candidate_from_row(row: dict[str, str], coord: tuple[float, float]) -> CandidateObservation:
    lat, lon = coord
    return CandidateObservation(
        index=-1,
        latitude=lat,
        longitude=lon,
        matched_flight=(row.get("matched_reference_flight") or "").strip(),
        matched_frame_index=_int_or_none(row.get("matched_reference_frame_index")) or 0,
        matched_time_s=_float_or_none(row.get("matched_reference_time_s")) or 0.0,
        retrieval_distance=_float_or_none(row.get("retrieval_distance")) or 0.0,
        retrieval_similarity=_float_or_none(row.get("retrieval_similarity")) or 0.0,
        good_matches=_int_or_none(row.get("orb_good_matches")) or 0,
        homography_inliers=_int_or_none(row.get("homography_inliers")) or 0,
        mean_match_distance=_float_or_none(row.get("mean_match_distance")) or float("inf"),
    )


def _ensure_fieldnames(fieldnames: list[str]) -> list[str]:
    output = list(fieldnames)

    if "query_timecode" not in output:
        insert_at = output.index("query_time_s") + 1 if "query_time_s" in output else len(output)
        output.insert(insert_at, "query_timecode")
    if "matched_reference_timecode" not in output:
        insert_at = output.index("matched_reference_time_s") + 1 if "matched_reference_time_s" in output else len(output)
        output.insert(insert_at, "matched_reference_timecode")

    for column in RAW_COLUMNS + FILTER_COLUMNS:
        if column not in output:
            output.append(column)
    return output


def derive_target_predictions(
    prediction_csv: Path,
    reference_index_path: Path,
    output_csv: Path,
    target: str,
    cfg: TemporalFilterConfig,
    apply_temporal_filter: bool = True,
    hold_last_on_reject: bool = True,
) -> dict[str, int]:
    lookup = _build_lookup(reference_index_path, target=target)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with prediction_csv.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
        fieldnames = _ensure_fieldnames(list(reader.fieldnames or []))

    previous: OnlineTrackState | None = None
    accepted = 0
    held = 0
    rejected = 0
    missing = 0
    written_rows: list[dict[str, str]] = []

    for row in rows:
        row = dict(row)
        row["prediction_target"] = target
        row["query_timecode"] = seconds_to_timecode(row.get("query_time_s"))
        row["matched_reference_timecode"] = seconds_to_timecode(row.get("matched_reference_time_s"))

        flight = (row.get("matched_reference_flight") or "").strip()
        frame_idx = _int_or_none(row.get("matched_reference_frame_index"))
        coord = lookup.get((flight, frame_idx)) if flight and frame_idx is not None else None

        if coord is None:
            missing += 1
            row["raw_pred_latitude"] = ""
            row["raw_pred_longitude"] = ""
            row["pred_latitude"] = ""
            row["pred_longitude"] = ""
            row["filter_accepted"] = "0"
            row["filter_reason"] = "missing_reference_match"
            row["filter_visual_score"] = ""
            row["filter_temporal_distance_m"] = ""
            row["filter_fused_score"] = ""
            row["filter_smoothed"] = "0"
            rejected += 1
            written_rows.append(row)
            continue

        candidate = _candidate_from_row(row, coord)
        raw_lat, raw_lon = coord
        row["raw_pred_latitude"] = f"{raw_lat:.8f}"
        row["raw_pred_longitude"] = f"{raw_lon:.8f}"

        if apply_temporal_filter:
            query_time_s = _float_or_none(row.get("query_time_s")) or 0.0
            selection = select_temporal_candidate([candidate], previous, query_time_s, cfg)
            if selection.accepted and selection.candidate is not None:
                previous = exponential_smooth_state(previous, selection.candidate, query_time_s, cfg)
                pred_lat = previous.latitude
                pred_lon = previous.longitude
                accepted_text = "1"
                smoothed = "1"
                reason = selection.reason
                accepted += 1
            elif hold_last_on_reject and previous is not None:
                pred_lat = previous.latitude
                pred_lon = previous.longitude
                previous = hold_state_at_time(previous, query_time_s)
                accepted_text = "0"
                smoothed = "1"
                reason = selection.reason + "_held_last"
                held += 1
            else:
                pred_lat = None
                pred_lon = None
                accepted_text = "0"
                smoothed = "0"
                reason = selection.reason
                rejected += 1

            row["filter_accepted"] = accepted_text
            row["filter_reason"] = reason
            row["filter_visual_score"] = f"{selection.visual_score:.6f}"
            row["filter_temporal_distance_m"] = "" if selection.temporal_distance_m is None else f"{selection.temporal_distance_m:.3f}"
            row["filter_fused_score"] = f"{selection.fused_score:.6f}"
            row["filter_smoothed"] = smoothed
        else:
            pred_lat = raw_lat
            pred_lon = raw_lon
            row["filter_accepted"] = row.get("filter_accepted", "")
            row["filter_reason"] = row.get("filter_reason", "")
            row["filter_visual_score"] = row.get("filter_visual_score", "")
            row["filter_temporal_distance_m"] = row.get("filter_temporal_distance_m", "")
            row["filter_fused_score"] = row.get("filter_fused_score", "")
            row["filter_smoothed"] = row.get("filter_smoothed", "")
            accepted += 1

        row["pred_latitude"] = "" if pred_lat is None else f"{pred_lat:.8f}"
        row["pred_longitude"] = "" if pred_lon is None else f"{pred_lon:.8f}"
        written_rows.append(row)

    with output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(written_rows)

    return {"rows": len(rows), "accepted": accepted, "held": held, "rejected": rejected, "missing": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive center/drone prediction coordinates from matched reference frames.")
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--reference-index", required=True, type=Path)
    parser.add_argument("--target", choices=["center", "drone"], required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--no-temporal-filter", action="store_true", help="Use raw matched-frame coordinates without smoothing/holding.")
    parser.add_argument("--no-hold-last", action="store_true", help="Leave rejected rows blank instead of holding last valid point.")
    parser.add_argument("--min-inliers", type=int, default=6)
    parser.add_argument("--min-good-matches", type=int, default=8)
    parser.add_argument("--max-speed-mps", type=float, default=18.0)
    parser.add_argument("--base-gate-m", type=float, default=55.0)
    parser.add_argument("--hard-jump-m", type=float, default=120.0, help="Strict default; avoids >200m Google Earth jumps.")
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    args = parser.parse_args()

    cfg = TemporalFilterConfig(
        min_inliers=args.min_inliers,
        min_good_matches=args.min_good_matches,
        max_speed_mps=args.max_speed_mps,
        base_gate_m=args.base_gate_m,
        hard_jump_m=args.hard_jump_m,
        ema_alpha=args.ema_alpha,
    )
    summary = derive_target_predictions(
        prediction_csv=args.prediction_csv,
        reference_index_path=args.reference_index,
        output_csv=args.out,
        target=args.target,
        cfg=cfg,
        apply_temporal_filter=not args.no_temporal_filter,
        hold_last_on_reject=not args.no_hold_last,
    )
    print(f"Wrote {args.target} predictions: {args.out}")
    print(f"Rows: {summary['rows']}")
    print(f"Accepted new visual positions: {summary['accepted']}")
    print(f"Held previous position after rejection: {summary['held']}")
    print(f"Rejected without output: {summary['rejected']}")
    if summary["missing"]:
        print(f"Rows without reference-index match: {summary['missing']}")


if __name__ == "__main__":
    main()
