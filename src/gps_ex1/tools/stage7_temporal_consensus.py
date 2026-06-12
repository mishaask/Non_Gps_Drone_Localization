"""Stage 7 temporal consensus and local-rerun preparation.

This is intentionally debug-friendly: it does not hide the raw per-frame
predictions. Instead it adds temporal-consensus columns, writes region reports,
and prepares frame lists / local reference indexes for rechecking suspicious
frames near the most likely GPS regions.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np

from gps_ex1.geometry.geo import haversine_m
from gps_ex1.preprocess.reference_index import ReferenceIndex, load_reference_index


@dataclass
class PredictionRow:
    order: int
    row: dict[str, str]
    query_frame_index: int
    query_time_s: float
    lat: float | None
    lon: float | None
    accepted: bool
    reason: str
    confidence: float
    is_anchor: bool = False
    region_id: int | None = None
    consensus_region_id: int | None = None
    temporal_status: str = "unclassified"
    distance_to_consensus_m: float | None = None
    corrected_lat: float | None = None
    corrected_lon: float | None = None
    correction_method: str = "raw"


@dataclass
class Region:
    region_id: int
    members: list[PredictionRow] = field(default_factory=list)
    centroid_lat: float = 0.0
    centroid_lon: float = 0.0
    median_confidence: float = 0.0

    def recompute(self) -> None:
        if not self.members:
            return
        self.centroid_lat = float(sum(p.lat for p in self.members if p.lat is not None) / len(self.members))
        self.centroid_lon = float(sum(p.lon for p in self.members if p.lon is not None) / len(self.members))
        self.median_confidence = float(median([p.confidence for p in self.members]))


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int_or_zero(value: str | None) -> int:
    parsed = _float_or_none(value)
    return 0 if parsed is None else int(parsed)


def _safe_fieldnames(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        return next(reader)


def _dedup_fieldnames(names: Iterable[str]) -> list[str]:
    counts: dict[str, int] = {}
    output: list[str] = []
    for name in names:
        base = name.strip() or "unnamed"
        count = counts.get(base, 0)
        counts[base] = count + 1
        output.append(base if count == 0 else f"{base}__dup{count}")
    return output


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
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


def row_confidence(row: dict[str, str]) -> float:
    """Practical uncalibrated confidence score; larger is better."""

    accepted = (row.get("filter_accepted") or "").strip() == "1"
    reason = (row.get("filter_reason") or "").strip()
    good = _int_or_zero(row.get("orb_good_matches"))
    inliers = _int_or_zero(row.get("homography_inliers"))
    ratio = _float_or_none(row.get("verification_inlier_ratio")) or 0.0
    sim = _float_or_none(row.get("retrieval_similarity")) or 0.0
    ref_area = _float_or_none(row.get("reference_inlier_bbox_area_frac"))
    ref_width = _float_or_none(row.get("reference_inlier_bbox_width_frac"))
    ref_height = _float_or_none(row.get("reference_inlier_bbox_height_frac"))
    inside = (row.get("projected_center_inside") or "").strip() == "1"
    geom_ok = (row.get("homography_geometry_ok") or "").strip() == "1"

    score = 0.0
    score += 4.0 if accepted else -1.5
    score += min(good, 80) / 12.0
    score += min(inliers, 60) / 6.0
    score += 5.0 * max(0.0, min(ratio, 1.0))
    score += 2.0 * sim
    score += 0.8 if inside else -1.5
    score += 0.8 if geom_ok else -1.5

    if ref_area is not None:
        score -= max(0.0, ref_area - 0.08) * 8.0
    if ref_width is not None:
        score -= max(0.0, ref_width - 0.35) * 4.0
    if ref_height is not None:
        score -= max(0.0, ref_height - 0.35) * 4.0

    if reason in {"homography_not_found", "not_enough_good_matches", "not_enough_inliers"}:
        score -= 2.0
    if reason in {"weak_inlier_ratio", "bad_homography_geometry"}:
        score -= 1.2
    if "scattered" in reason:
        score -= 2.0
    return float(score)


def _make_predictions(rows: list[dict[str, str]], args: argparse.Namespace) -> list[PredictionRow]:
    predictions: list[PredictionRow] = []
    for i, row in enumerate(rows):
        lat = _float_or_none(row.get("pred_latitude"))
        lon = _float_or_none(row.get("pred_longitude"))
        accepted = (row.get("filter_accepted") or "").strip() == "1"
        reason = (row.get("filter_reason") or "").strip() or "unknown"
        confidence = row_confidence(row)
        query_frame_index = _int_or_zero(row.get("query_frame_index"))
        query_time_s = _float_or_none(row.get("query_time_s")) or 0.0
        pred = PredictionRow(
            order=i,
            row=row,
            query_frame_index=query_frame_index,
            query_time_s=query_time_s,
            lat=lat,
            lon=lon,
            accepted=accepted,
            reason=reason,
            confidence=confidence,
        )
        ref_area = _float_or_none(row.get("reference_inlier_bbox_area_frac"))
        ref_width = _float_or_none(row.get("reference_inlier_bbox_width_frac"))
        ref_height = _float_or_none(row.get("reference_inlier_bbox_height_frac"))
        pred.is_anchor = bool(
            lat is not None
            and lon is not None
            and accepted
            and confidence >= args.anchor_min_confidence
            and _int_or_zero(row.get("homography_inliers")) >= args.anchor_min_inliers
            and _int_or_zero(row.get("orb_good_matches")) >= args.anchor_min_good_matches
            and (ref_area is None or ref_area <= args.anchor_max_ref_area_frac)
            and (ref_width is None or ref_width <= args.anchor_max_ref_width_frac)
            and (ref_height is None or ref_height <= args.anchor_max_ref_height_frac)
        )
        predictions.append(pred)
    return predictions


def _cluster_anchors(predictions: list[PredictionRow], radius_m: float) -> list[Region]:
    regions: list[Region] = []
    next_region_id = 1
    anchors = [p for p in predictions if p.is_anchor and p.lat is not None and p.lon is not None]
    # Strongest anchors seed regions first.
    anchors.sort(key=lambda p: (-p.confidence, p.query_time_s))

    for pred in anchors:
        best: Region | None = None
        best_dist = float("inf")
        for region in regions:
            dist = haversine_m(pred.lat, pred.lon, region.centroid_lat, region.centroid_lon)
            if dist < best_dist:
                best_dist = dist
                best = region
        if best is not None and best_dist <= radius_m:
            best.members.append(pred)
            pred.region_id = best.region_id
            best.recompute()
        else:
            region = Region(region_id=next_region_id, members=[pred], centroid_lat=pred.lat, centroid_lon=pred.lon)
            pred.region_id = next_region_id
            region.recompute()
            regions.append(region)
            next_region_id += 1

    # Stable ordering by size then confidence. Rename IDs to 1..N.
    regions.sort(key=lambda r: (-len(r.members), -r.median_confidence, r.region_id))
    old_to_new: dict[int, int] = {}
    for new_id, region in enumerate(regions, start=1):
        old_to_new[region.region_id] = new_id
        region.region_id = new_id
    for pred in predictions:
        if pred.region_id in old_to_new:
            pred.region_id = old_to_new[pred.region_id]
    return regions


def _nearest_anchor(predictions: list[PredictionRow], row_index: int, direction: int, max_steps: int) -> PredictionRow | None:
    j = row_index + direction
    steps = 0
    while 0 <= j < len(predictions) and steps < max_steps:
        p = predictions[j]
        if p.is_anchor and p.region_id is not None:
            return p
        j += direction
        steps += 1
    return None


def _consensus_region_for_row(predictions: list[PredictionRow], i: int, max_steps: int) -> int | None:
    prev_anchor = _nearest_anchor(predictions, i, -1, max_steps)
    next_anchor = _nearest_anchor(predictions, i, 1, max_steps)
    cur = predictions[i]

    if prev_anchor is not None and next_anchor is not None and prev_anchor.region_id == next_anchor.region_id:
        return prev_anchor.region_id
    if cur.is_anchor and cur.region_id is not None:
        return cur.region_id
    if prev_anchor is not None and next_anchor is None:
        return prev_anchor.region_id
    if next_anchor is not None and prev_anchor is None:
        return next_anchor.region_id
    return None


def _region_by_id(regions: list[Region]) -> dict[int, Region]:
    return {r.region_id: r for r in regions}


def _interpolate_between_same_region(predictions: list[PredictionRow], i: int, region_id: int, max_steps: int) -> tuple[float, float, str] | None:
    prev_anchor = _nearest_anchor(predictions, i, -1, max_steps)
    next_anchor = _nearest_anchor(predictions, i, 1, max_steps)
    if (
        prev_anchor is not None
        and next_anchor is not None
        and prev_anchor.region_id == region_id
        and next_anchor.region_id == region_id
        and prev_anchor.lat is not None
        and prev_anchor.lon is not None
        and next_anchor.lat is not None
        and next_anchor.lon is not None
        and next_anchor.query_time_s > prev_anchor.query_time_s
    ):
        alpha = (predictions[i].query_time_s - prev_anchor.query_time_s) / (next_anchor.query_time_s - prev_anchor.query_time_s)
        alpha = max(0.0, min(1.0, alpha))
        lat = prev_anchor.lat * (1.0 - alpha) + next_anchor.lat * alpha
        lon = prev_anchor.lon * (1.0 - alpha) + next_anchor.lon * alpha
        return lat, lon, "interpolated_between_same_region_anchors"
    return None


def _classify_and_correct(predictions: list[PredictionRow], regions: list[Region], args: argparse.Namespace) -> None:
    regions_by_id = _region_by_id(regions)
    for i, pred in enumerate(predictions):
        pred.consensus_region_id = _consensus_region_for_row(predictions, i, args.neighbor_window)
        pred.corrected_lat = pred.lat
        pred.corrected_lon = pred.lon
        pred.correction_method = "raw"

        if pred.consensus_region_id is None:
            pred.temporal_status = "no_consensus_region"
            continue

        region = regions_by_id.get(pred.consensus_region_id)
        if region is None:
            pred.temporal_status = "no_consensus_region"
            continue

        if pred.lat is not None and pred.lon is not None:
            pred.distance_to_consensus_m = haversine_m(pred.lat, pred.lon, region.centroid_lat, region.centroid_lon)

        if pred.is_anchor and pred.region_id == pred.consensus_region_id:
            pred.temporal_status = "anchor_consistent"
        elif pred.accepted and pred.distance_to_consensus_m is not None and pred.distance_to_consensus_m <= args.jump_threshold_m:
            pred.temporal_status = "accepted_near_consensus"
        elif pred.accepted and pred.distance_to_consensus_m is not None and pred.distance_to_consensus_m > args.jump_threshold_m:
            pred.temporal_status = "suspicious_jump_from_consensus"
        elif not pred.accepted:
            pred.temporal_status = "rejected_inside_consensus_context"
        else:
            pred.temporal_status = "low_confidence_consensus_context"

        should_fill = args.fill_gaps and pred.temporal_status in {
            "rejected_inside_consensus_context",
            "suspicious_jump_from_consensus",
            "low_confidence_consensus_context",
        }
        if should_fill:
            interpolated = _interpolate_between_same_region(predictions, i, pred.consensus_region_id, args.neighbor_window)
            if interpolated is not None:
                pred.corrected_lat, pred.corrected_lon, pred.correction_method = interpolated
            else:
                pred.corrected_lat = region.centroid_lat
                pred.corrected_lon = region.centroid_lon
                pred.correction_method = "filled_with_region_centroid"


def _write_corrected_csv(path: Path, header: list[str], predictions: list[PredictionRow]) -> None:
    extra = [
        "stage7_confidence",
        "stage7_is_anchor",
        "stage7_region_id",
        "stage7_consensus_region_id",
        "stage7_temporal_status",
        "stage7_distance_to_consensus_m",
        "stage7_corrected_latitude",
        "stage7_corrected_longitude",
        "stage7_correction_method",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=header + extra)
        writer.writeheader()
        for p in predictions:
            row = dict(p.row)
            row.update(
                {
                    "stage7_confidence": f"{p.confidence:.6f}",
                    "stage7_is_anchor": "1" if p.is_anchor else "0",
                    "stage7_region_id": "" if p.region_id is None else str(p.region_id),
                    "stage7_consensus_region_id": "" if p.consensus_region_id is None else str(p.consensus_region_id),
                    "stage7_temporal_status": p.temporal_status,
                    "stage7_distance_to_consensus_m": "" if p.distance_to_consensus_m is None else f"{p.distance_to_consensus_m:.3f}",
                    "stage7_corrected_latitude": "" if p.corrected_lat is None else f"{p.corrected_lat:.8f}",
                    "stage7_corrected_longitude": "" if p.corrected_lon is None else f"{p.corrected_lon:.8f}",
                    "stage7_correction_method": p.correction_method,
                }
            )
            writer.writerow(row)


def _write_regions_csv(path: Path, regions: list[Region]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["region_id", "anchor_count", "centroid_latitude", "centroid_longitude", "median_confidence", "min_query_frame", "max_query_frame"])
        for r in regions:
            frames = [p.query_frame_index for p in r.members]
            writer.writerow([r.region_id, len(r.members), f"{r.centroid_lat:.8f}", f"{r.centroid_lon:.8f}", f"{r.median_confidence:.6f}", min(frames), max(frames)])


def _write_frame_list(path: Path, frames: Iterable[int]) -> int:
    unique = sorted(set(int(f) for f in frames))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for frame in unique:
            fp.write(f"{frame}\n")
    return len(unique)


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


def _write_local_reference_indexes(args: argparse.Namespace, regions: list[Region]) -> None:
    if args.reference_index is None:
        return
    index = load_reference_index(args.reference_index)
    lats, lons = _reference_lats_lons(index, args.reference_coordinate)
    all_indices = np.arange(len(lats))

    for region in regions[: args.max_regions]:
        radius = args.local_reference_radius_m
        selected: list[int] = []
        while True:
            mask = [
                int(i)
                for i in all_indices
                if np.isfinite(lats[i])
                and np.isfinite(lons[i])
                and haversine_m(float(lats[i]), float(lons[i]), region.centroid_lat, region.centroid_lon) <= radius
            ]
            selected = mask
            if len(selected) >= args.min_local_references or radius >= args.max_local_reference_radius_m:
                break
            radius *= 1.5
        out_npz = args.out_dir / f"local_reference_index_region_{region.region_id:03d}.npz"
        _save_reference_subset(index, selected, out_npz)
        print(f"Wrote local reference index for region {region.region_id}: {out_npz} ({len(selected)} refs, radius≈{radius:.1f}m)")


def _write_rerun_artifacts(args: argparse.Namespace, predictions: list[PredictionRow], regions: list[Region]) -> None:
    suspicious_statuses = {
        "rejected_inside_consensus_context",
        "suspicious_jump_from_consensus",
        "low_confidence_consensus_context",
    }
    all_suspicious = [p.query_frame_index for p in predictions if p.temporal_status in suspicious_statuses]
    count = _write_frame_list(args.out_dir / "stage7_suspicious_or_gap_frames.txt", all_suspicious)
    print(f"Wrote suspicious/gap frame list: {count} frames")

    for region in regions[: args.max_regions]:
        frames = [
            p.query_frame_index
            for p in predictions
            if p.consensus_region_id == region.region_id and p.temporal_status in suspicious_statuses
        ]
        if not frames:
            continue
        path = args.out_dir / f"rerun_frames_region_{region.region_id:03d}.txt"
        _write_frame_list(path, frames)

    # Debug-friendly batch template; user can edit thresholds/scales later.
    bat_path = args.out_dir / "README_run_local_reruns.bat.txt"
    with bat_path.open("w", encoding="utf-8") as fp:
        fp.write("REM Copy one block at a time into CMD from the project root.\n")
        fp.write("REM These commands rerun suspicious/gap frames only against local reference indexes.\n\n")
        for region in regions[: args.max_regions]:
            frames_path = args.out_dir / f"rerun_frames_region_{region.region_id:03d}.txt"
            ref_path = args.out_dir / f"local_reference_index_region_{region.region_id:03d}.npz"
            if not frames_path.exists() or not ref_path.exists():
                continue
            fp.write(f"REM Region {region.region_id}: {len(region.members)} anchors around {region.centroid_lat:.8f},{region.centroid_lon:.8f}\n")
            fp.write("python -m gps_ex1.pipeline.localize_video ^\n")
            fp.write("  --video data/raw/DJI_0011.mp4 ^\n")
            fp.write(f"  --reference-index {ref_path.as_posix()} ^\n")
            fp.write(f"  --out {(args.out_dir / f'stage7_rerun_region_{region.region_id:03d}.csv').as_posix()} ^\n")
            fp.write(f"  --query-frame-list {frames_path.as_posix()} ^\n")
            fp.write("  --top-k 50 ^\n")
            fp.write("  --query-scales 0.2,0.3 ^\n")
            fp.write("  --query-scale-fill blur ^\n")
            fp.write("  --prediction-target center ^\n")
            fp.write("  --descriptor-backend anyloc-gem ^\n")
            fp.write("  --descriptor-image-size 322 ^\n")
            fp.write("  --verification-backend lightglue ^\n")
            fp.write("  --min-inliers 10 ^\n")
            fp.write("  --min-good-matches 20 ^\n")
            fp.write("  --min-inlier-ratio 0.25 ^\n")
            fp.write("  --max-reference-inlier-area-frac 0.10 ^\n")
            fp.write("  --max-reference-inlier-width-frac 0.35 ^\n")
            fp.write("  --max-reference-inlier-height-frac 0.35 ^\n")
            fp.write("  --mask-dynamic-objects ^\n")
            fp.write('  --mask-model "models/yolov8-s-p2-mixup=0.4/weights/best.pt" ^\n')
            fp.write("  --mask-classes 3,4,5,8,9 ^\n")
            fp.write("  --mask-confidence 0.25 ^\n")
            fp.write("  --mask-imgsz 640 ^\n")
            fp.write("  --mask-source box ^\n")
            fp.write("  --mask-max-area-frac 0.015 ^\n")
            fp.write("  --mask-dilate-px 0 ^\n")
            fp.write("  --mask-fill gray\n\n")
    print(f"Wrote local rerun command template: {bat_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 7 temporal consensus and local rerun preparation for GPS_EX1.")
    parser.add_argument("--prediction-csv", required=True, type=Path, help="Raw Stage 6 prediction CSV.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output folder for Stage 7 CSVs/reports/frame lists.")
    parser.add_argument("--reference-index", type=Path, default=None, help="Optional full reference index. Required for local reference-index writing.")
    parser.add_argument("--reference-coordinate", choices=["center", "drone"], default="center")
    parser.add_argument("--region-radius-m", type=float, default=120.0, help="Anchor clustering radius in meters.")
    parser.add_argument("--neighbor-window", type=int, default=8, help="How many sampled query rows before/after to use for local consensus.")
    parser.add_argument("--jump-threshold-m", type=float, default=180.0, help="Accepted predictions farther than this from local consensus are suspicious.")
    parser.add_argument("--fill-gaps", action="store_true", help="Write corrected coordinates for rejected/suspicious rows using neighboring same-region anchors or region centroid.")
    parser.add_argument("--anchor-min-confidence", type=float, default=6.0)
    parser.add_argument("--anchor-min-inliers", type=int, default=10)
    parser.add_argument("--anchor-min-good-matches", type=int, default=20)
    parser.add_argument("--anchor-max-ref-area-frac", type=float, default=0.12)
    parser.add_argument("--anchor-max-ref-width-frac", type=float, default=0.45)
    parser.add_argument("--anchor-max-ref-height-frac", type=float, default=0.45)
    parser.add_argument("--write-local-reference-indexes", action="store_true")
    parser.add_argument("--local-reference-radius-m", type=float, default=180.0)
    parser.add_argument("--max-local-reference-radius-m", type=float, default=500.0)
    parser.add_argument("--min-local-references", type=int, default=120)
    parser.add_argument("--max-regions", type=int, default=5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    header, raw_rows = _read_csv_rows(args.prediction_csv)
    predictions = _make_predictions(raw_rows, args)
    regions = _cluster_anchors(predictions, args.region_radius_m)
    _classify_and_correct(predictions, regions, args)

    corrected_csv = args.out_dir / "stage7_temporal_corrected.csv"
    regions_csv = args.out_dir / "stage7_regions.csv"
    _write_corrected_csv(corrected_csv, header, predictions)
    _write_regions_csv(regions_csv, regions)
    _write_rerun_artifacts(args, predictions, regions)
    if args.write_local_reference_indexes:
        _write_local_reference_indexes(args, regions)

    status_counts: dict[str, int] = {}
    for p in predictions:
        status_counts[p.temporal_status] = status_counts.get(p.temporal_status, 0) + 1

    print("\nStage 7 temporal consensus summary")
    print("=" * 80)
    print(f"Input rows: {len(predictions)}")
    print(f"Anchors: {sum(1 for p in predictions if p.is_anchor)}")
    print(f"Regions: {len(regions)}")
    for region in regions[: args.max_regions]:
        print(
            f"  region {region.region_id}: anchors={len(region.members)}, "
            f"centroid=({region.centroid_lat:.8f},{region.centroid_lon:.8f}), "
            f"median_conf={region.median_confidence:.2f}"
        )
    print(f"Status counts: {sorted(status_counts.items())}")
    print(f"Wrote corrected CSV: {corrected_csv}")
    print(f"Wrote region summary: {regions_csv}")


if __name__ == "__main__":
    main()
