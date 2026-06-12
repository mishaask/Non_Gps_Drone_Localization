"""Stage 8 temporal-context rerun planner.

This tool fixes the Stage 7 limitation where a bad frame could create or follow
its own weak region.  Stage 8 uses ONLY high-confidence anchor frames to define
stable GPS regions, then prepares reruns for weak / rejected / suspicious-jump
frames against the regions supported by nearby anchors in the timeline.

The output is debug-friendly: it writes a context plan CSV, per-region frame
lists, and local reference indexes.  The actual LightGlue reruns are still run
with the normal localize_video pipeline, so every proposed fix remains visually
verifiable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from gps_ex1.geometry.geo import haversine_m
from gps_ex1.preprocess.reference_index import ReferenceIndex, load_reference_index
from gps_ex1.tools.stage7_temporal_consensus import (
    PredictionRow,
    Region,
    _cluster_anchors,
    _dedup_fieldnames,
    _float_or_none,
    _int_or_zero,
    _make_predictions,
    _read_csv_rows,
    _safe_fieldnames,
    row_confidence,
)


def _accepted(row: dict[str, str]) -> bool:
    return (row.get("filter_accepted") or "").strip() == "1"


def _write_frame_list(path: Path, frames: Iterable[int]) -> int:
    unique = sorted(set(int(f) for f in frames))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for frame in unique:
            fp.write(f"{frame}\n")
    return len(unique)


def _nearest_anchor(predictions: list[PredictionRow], i: int, direction: int, max_steps: int) -> PredictionRow | None:
    j = i + direction
    steps = 0
    while 0 <= j < len(predictions) and steps < max_steps:
        p = predictions[j]
        if p.is_anchor and p.region_id is not None:
            return p
        j += direction
        steps += 1
    return None


def _region_by_id(regions: list[Region]) -> dict[int, Region]:
    return {r.region_id: r for r in regions}


def _nearest_region_for_prediction(pred: PredictionRow, regions: list[Region]) -> tuple[int | None, float | None]:
    if pred.lat is None or pred.lon is None:
        return None, None
    best_id: int | None = None
    best_dist: float | None = None
    for region in regions:
        d = haversine_m(pred.lat, pred.lon, region.centroid_lat, region.centroid_lon)
        if best_dist is None or d < best_dist:
            best_dist = d
            best_id = region.region_id
    return best_id, best_dist


def _candidate_regions_for_frame(
    predictions: list[PredictionRow],
    i: int,
    regions: list[Region],
    global_region_ids: list[int],
    args: argparse.Namespace,
) -> tuple[list[int], str, int | None, int | None]:
    prev_anchor = _nearest_anchor(predictions, i, -1, args.context_window)
    next_anchor = _nearest_anchor(predictions, i, 1, args.context_window)
    prev_region = None if prev_anchor is None else prev_anchor.region_id
    next_region = None if next_anchor is None else next_anchor.region_id

    candidates: list[int] = []
    reason = ""

    if prev_region is not None and next_region is not None and prev_region == next_region:
        candidates.append(prev_region)
        reason = "between_same_region_anchors"
    elif prev_region is not None and next_region is not None:
        candidates.extend([prev_region, next_region])
        reason = "between_different_region_anchors"
    elif prev_region is not None:
        candidates.append(prev_region)
        reason = "after_previous_region_anchor"
    elif next_region is not None:
        candidates.append(next_region)
        reason = "before_next_region_anchor"
    else:
        reason = "no_nearby_anchor_context"

    if args.add_global_top_regions and len(candidates) < args.max_candidate_regions_per_frame:
        for rid in global_region_ids:
            if rid not in candidates:
                candidates.append(rid)
            if len(candidates) >= args.max_candidate_regions_per_frame:
                break

    # Keep stable ordering and cap candidate count.
    deduped: list[int] = []
    for rid in candidates:
        if rid is not None and rid not in deduped:
            deduped.append(rid)
    return deduped[: args.max_candidate_regions_per_frame], reason, prev_region, next_region


def _is_bad_frame(pred: PredictionRow, nearest_region_id: int | None, nearest_region_dist: float | None, candidate_regions: list[int], args: argparse.Namespace) -> tuple[bool, str]:
    # Strong anchors are trusted; they define context rather than get rerun.
    if pred.is_anchor:
        return False, "anchor_keep"

    if not candidate_regions:
        return False, "no_context_skip"

    if not pred.accepted:
        return True, "rejected_or_weak_in_context"

    # Accepted but not high-confidence: rerun if it is far from the context region
    # or if its nearest region is not one of the context-approved regions.
    if nearest_region_id is not None and nearest_region_id not in candidate_regions:
        return True, "accepted_jump_to_non_context_region"
    if nearest_region_dist is not None and nearest_region_dist > args.jump_threshold_m:
        return True, "accepted_far_from_context_region"

    # Accepted but weak confidence should not be allowed to create a path jump.
    if pred.confidence < args.low_confidence_threshold:
        return True, "accepted_low_confidence_in_context"

    return False, "accepted_keep"


def _save_reference_subset(index: ReferenceIndex, indices: list[int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(indices, dtype=np.int64)
    np.savez_compressed(
        out_path,
        descriptors=index.descriptors[arr],
        image_paths=index.image_paths[arr],
        flight_ids=index.flight_ids[arr],
        frame_indices=index.frame_indices[arr],
        video_times_s=index.video_times_s[arr],
        drone_lats=index.drone_lats[arr],
        drone_lons=index.drone_lons[arr],
        center_lats=index.center_lats[arr],
        center_lons=index.center_lons[arr],
        headings_deg=index.headings_deg[arr],
        center_offsets_m=index.center_offsets_m[arr],
    )


def _reference_lats_lons(index: ReferenceIndex, coordinate: str) -> tuple[np.ndarray, np.ndarray]:
    if coordinate == "drone":
        return index.drone_lats.astype(float), index.drone_lons.astype(float)
    return index.center_lats.astype(float), index.center_lons.astype(float)


def _write_local_reference_indexes(args: argparse.Namespace, regions: list[Region], used_region_ids: set[int]) -> None:
    index = load_reference_index(args.reference_index)
    lats, lons = _reference_lats_lons(index, args.reference_coordinate)
    all_indices = np.arange(len(lats))
    regions_by_id = _region_by_id(regions)

    for rid in sorted(used_region_ids):
        region = regions_by_id.get(rid)
        if region is None:
            continue
        radius = args.local_reference_radius_m
        selected: list[int] = []
        while True:
            selected = [
                int(i)
                for i in all_indices
                if np.isfinite(lats[i])
                and np.isfinite(lons[i])
                and haversine_m(float(lats[i]), float(lons[i]), region.centroid_lat, region.centroid_lon) <= radius
            ]
            if len(selected) >= args.min_local_references or radius >= args.max_local_reference_radius_m:
                break
            radius *= 1.5
        out_npz = args.out_dir / f"local_reference_index_context_region_{rid:03d}.npz"
        _save_reference_subset(index, selected, out_npz)
        print(f"Wrote context local reference index for region {rid}: {out_npz} ({len(selected)} refs, radius≈{radius:.1f}m)")


def _write_regions_csv(path: Path, regions: list[Region]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["region_id", "anchor_count", "centroid_latitude", "centroid_longitude", "median_confidence", "min_query_frame", "max_query_frame"])
        for r in regions:
            frames = [p.query_frame_index for p in r.members]
            writer.writerow([r.region_id, len(r.members), f"{r.centroid_lat:.8f}", f"{r.centroid_lon:.8f}", f"{r.median_confidence:.6f}", min(frames), max(frames)])


def _write_plan_csv(path: Path, plan_rows: list[dict[str, str]]) -> None:
    fields = [
        "query_frame_index",
        "query_time_s",
        "original_accepted",
        "original_reason",
        "original_confidence",
        "original_nearest_region_id",
        "original_distance_to_nearest_region_m",
        "prev_anchor_region_id",
        "next_anchor_region_id",
        "context_reason",
        "bad_frame_reason",
        "candidate_regions",
        "will_rerun",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in plan_rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 8 temporal-context rerun planner for GPS_EX1.")
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--reference-index", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--reference-coordinate", choices=["center", "drone"], default="center")
    parser.add_argument("--region-radius-m", type=float, default=120.0)
    parser.add_argument("--context-window", type=int, default=8, help="Rows before/after used to find stable anchor context.")
    parser.add_argument("--jump-threshold-m", type=float, default=180.0)
    parser.add_argument("--low-confidence-threshold", type=float, default=8.0)
    parser.add_argument("--anchor-min-confidence", type=float, default=6.0)
    parser.add_argument("--anchor-min-inliers", type=int, default=10)
    parser.add_argument("--anchor-min-good-matches", type=int, default=20)
    parser.add_argument("--anchor-max-ref-area-frac", type=float, default=0.12)
    parser.add_argument("--anchor-max-ref-width-frac", type=float, default=0.45)
    parser.add_argument("--anchor-max-ref-height-frac", type=float, default=0.45)
    parser.add_argument("--global-top-regions", type=int, default=2)
    parser.add_argument("--add-global-top-regions", action="store_true", help="Also try the globally most popular anchor regions when local context is missing/ambiguous.")
    parser.add_argument("--max-candidate-regions-per-frame", type=int, default=2)
    parser.add_argument("--local-reference-radius-m", type=float, default=180.0)
    parser.add_argument("--max-local-reference-radius-m", type=float, default=500.0)
    parser.add_argument("--min-local-references", type=int, default=120)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    header, raw_rows = _read_csv_rows(args.prediction_csv)
    predictions = _make_predictions(raw_rows, args)
    regions = _cluster_anchors(predictions, args.region_radius_m)
    regions = regions[: max(1, args.global_top_regions if args.add_global_top_regions else len(regions))]
    global_region_ids = [r.region_id for r in regions[: args.global_top_regions]]

    plan_rows: list[dict[str, str]] = []
    frames_by_region: dict[int, list[int]] = {}
    used_region_ids: set[int] = set()

    for i, pred in enumerate(predictions):
        candidate_regions, context_reason, prev_region, next_region = _candidate_regions_for_frame(predictions, i, regions, global_region_ids, args)
        nearest_region_id, nearest_region_dist = _nearest_region_for_prediction(pred, regions)
        should_rerun, bad_reason = _is_bad_frame(pred, nearest_region_id, nearest_region_dist, candidate_regions, args)

        if should_rerun:
            for rid in candidate_regions:
                frames_by_region.setdefault(rid, []).append(pred.query_frame_index)
                used_region_ids.add(rid)

        plan_rows.append(
            {
                "query_frame_index": str(pred.query_frame_index),
                "query_time_s": f"{pred.query_time_s:.3f}",
                "original_accepted": "1" if pred.accepted else "0",
                "original_reason": pred.reason,
                "original_confidence": f"{pred.confidence:.6f}",
                "original_nearest_region_id": "" if nearest_region_id is None else str(nearest_region_id),
                "original_distance_to_nearest_region_m": "" if nearest_region_dist is None else f"{nearest_region_dist:.3f}",
                "prev_anchor_region_id": "" if prev_region is None else str(prev_region),
                "next_anchor_region_id": "" if next_region is None else str(next_region),
                "context_reason": context_reason,
                "bad_frame_reason": bad_reason,
                "candidate_regions": ";".join(str(rid) for rid in candidate_regions),
                "will_rerun": "1" if should_rerun else "0",
            }
        )

    _write_regions_csv(args.out_dir / "stage8_regions.csv", regions)
    _write_plan_csv(args.out_dir / "stage8_context_rerun_plan.csv", plan_rows)

    total_rerun_assignments = 0
    for rid, frames in sorted(frames_by_region.items()):
        count = _write_frame_list(args.out_dir / f"rerun_context_region_{rid:03d}_frames.txt", frames)
        total_rerun_assignments += count
        print(f"Wrote context rerun frame list for region {rid}: {count} frames")

    _write_local_reference_indexes(args, regions, used_region_ids)

    bad_frames_unique = sorted({int(row["query_frame_index"]) for row in plan_rows if row.get("will_rerun") == "1"})
    _write_frame_list(args.out_dir / "stage8_bad_frames_all.txt", bad_frames_unique)

    print("\nStage 8 temporal-context planner summary")
    print("=" * 80)
    print(f"Input rows: {len(predictions)}")
    print(f"Anchors used for regions: {sum(1 for p in predictions if p.is_anchor)}")
    print(f"Regions found: {len(regions)}")
    for r in regions:
        print(f"  region {r.region_id}: anchors={len(r.members)}, centroid=({r.centroid_lat:.8f},{r.centroid_lon:.8f}), median_conf={r.median_confidence:.2f}")
    print(f"Unique bad/suspicious frames selected for rerun: {len(bad_frames_unique)}")
    print(f"Per-region rerun assignments: {total_rerun_assignments}")
    print(f"Wrote plan: {args.out_dir / 'stage8_context_rerun_plan.csv'}")


if __name__ == "__main__":
    main()
