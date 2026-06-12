"""Stage 9 fixed-grid temporal-context rerun planner.

Stage 8 built regions from first-pass predictions.  That can be too broad or
biased by weak guesses.  Stage 9 instead builds fixed map-grid regions from the
reference index itself, then uses only high-confidence first-pass anchors to
choose which grid cells are plausible in the query timeline.

The output is intentionally debug-friendly:
- a fixed grid report for the reference footprint;
- one local reference index per grid cell, with overlap/halo;
- a context rerun plan for weak / rejected / sudden-jump query frames;
- one frame-list per candidate grid cell.

The actual reruns still use gps_ex1.pipeline.localize_video so every proposed
fix remains inspectable with the normal LightGlue debug tools.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from gps_ex1.geometry.geo import haversine_m, latlon_to_local_xy, local_xy_to_latlon
from gps_ex1.preprocess.reference_index import ReferenceIndex, load_reference_index
from gps_ex1.tools.stage7_temporal_consensus import (
    PredictionRow,
    _float_or_none,
    _int_or_zero,
    _make_predictions,
    _read_csv_rows,
)


@dataclass(frozen=True)
class GridCell:
    region_id: int
    row: int
    col: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    center_lat: float
    center_lon: float
    base_ref_count: int = 0
    halo_ref_count: int = 0
    halo_m: float = 0.0


def _reference_lats_lons(index: ReferenceIndex, coordinate: str) -> tuple[np.ndarray, np.ndarray]:
    if coordinate == "drone":
        return index.drone_lats.astype(float), index.drone_lons.astype(float)
    return index.center_lats.astype(float), index.center_lons.astype(float)


def _finite_reference_xy(index: ReferenceIndex, coordinate: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    lats, lons = _reference_lats_lons(index, coordinate)
    finite = np.isfinite(lats) & np.isfinite(lons)
    if not np.any(finite):
        raise RuntimeError("Reference index has no finite coordinates for grid building.")
    origin_lat = float(np.mean(lats[finite]))
    origin_lon = float(np.mean(lons[finite]))
    xs = np.full_like(lats, np.nan, dtype=float)
    ys = np.full_like(lats, np.nan, dtype=float)
    for i in np.where(finite)[0]:
        p = latlon_to_local_xy(float(lats[i]), float(lons[i]), origin_lat, origin_lon)
        xs[i] = p.east_m
        ys[i] = p.north_m
    return xs, ys, finite, origin_lat, origin_lon


def _make_grid_cells(
    xs: np.ndarray,
    ys: np.ndarray,
    finite: np.ndarray,
    origin_lat: float,
    origin_lon: float,
    rows: int,
    cols: int,
    padding_m: float,
) -> list[GridCell]:
    min_x = float(np.nanmin(xs[finite])) - padding_m
    max_x = float(np.nanmax(xs[finite])) + padding_m
    min_y = float(np.nanmin(ys[finite])) - padding_m
    max_y = float(np.nanmax(ys[finite])) + padding_m
    if max_x <= min_x or max_y <= min_y:
        raise RuntimeError("Reference grid bounds collapsed; check reference coordinates.")

    cell_w = (max_x - min_x) / cols
    cell_h = (max_y - min_y) / rows
    cells: list[GridCell] = []
    for r in range(rows):
        for c in range(cols):
            x0 = min_x + c * cell_w
            x1 = min_x + (c + 1) * cell_w
            y0 = min_y + r * cell_h
            y1 = min_y + (r + 1) * cell_h
            center_lat, center_lon = local_xy_to_latlon((x0 + x1) / 2.0, (y0 + y1) / 2.0, origin_lat, origin_lon)
            region_id = r * cols + c + 1
            cells.append(GridCell(region_id, r + 1, c + 1, x0, x1, y0, y1, center_lat, center_lon))
    return cells


def _grid_cell_for_xy(x: float, y: float, cells: list[GridCell]) -> int | None:
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    # Include the max edge in the last row/col via a tiny epsilon.
    eps = 1e-9
    for cell in cells:
        if cell.x_min - eps <= x <= cell.x_max + eps and cell.y_min - eps <= y <= cell.y_max + eps:
            return cell.region_id
    return None


def _grid_cell_for_latlon(lat: float | None, lon: float | None, cells: list[GridCell], origin_lat: float, origin_lon: float) -> int | None:
    if lat is None or lon is None:
        return None
    p = latlon_to_local_xy(lat, lon, origin_lat, origin_lon)
    return _grid_cell_for_xy(p.east_m, p.north_m, cells)


def _indices_for_cell(xs: np.ndarray, ys: np.ndarray, finite: np.ndarray, cell: GridCell, halo_m: float) -> list[int]:
    x0 = cell.x_min - halo_m
    x1 = cell.x_max + halo_m
    y0 = cell.y_min - halo_m
    y1 = cell.y_max + halo_m
    mask = finite & (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    return [int(i) for i in np.where(mask)[0]]


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


def _write_frame_list(path: Path, frames: Iterable[int]) -> int:
    unique = sorted(set(int(f) for f in frames))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for frame in unique:
            fp.write(f"{frame}\n")
    return len(unique)


def _write_grid_csv(path: Path, cells: list[GridCell]) -> None:
    fields = [
        "grid_region_id",
        "row",
        "col",
        "center_latitude",
        "center_longitude",
        "x_min_m",
        "x_max_m",
        "y_min_m",
        "y_max_m",
        "base_ref_count",
        "halo_ref_count",
        "halo_m",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            writer.writerow(
                {
                    "grid_region_id": cell.region_id,
                    "row": cell.row,
                    "col": cell.col,
                    "center_latitude": f"{cell.center_lat:.8f}",
                    "center_longitude": f"{cell.center_lon:.8f}",
                    "x_min_m": f"{cell.x_min:.3f}",
                    "x_max_m": f"{cell.x_max:.3f}",
                    "y_min_m": f"{cell.y_min:.3f}",
                    "y_max_m": f"{cell.y_max:.3f}",
                    "base_ref_count": cell.base_ref_count,
                    "halo_ref_count": cell.halo_ref_count,
                    "halo_m": f"{cell.halo_m:.3f}",
                }
            )


def _write_reference_assignments(path: Path, index: ReferenceIndex, xs: np.ndarray, ys: np.ndarray, finite: np.ndarray, cells: list[GridCell]) -> None:
    fields = ["ref_index", "flight_id", "frame_index", "video_time_s", "image_path", "latitude", "longitude", "grid_region_id", "x_m", "y_m"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for i in range(len(index.image_paths)):
            if not finite[i]:
                continue
            rid = _grid_cell_for_xy(float(xs[i]), float(ys[i]), cells)
            writer.writerow(
                {
                    "ref_index": i,
                    "flight_id": str(index.flight_ids[i]),
                    "frame_index": int(index.frame_indices[i]),
                    "video_time_s": f"{float(index.video_times_s[i]):.3f}",
                    "image_path": str(index.image_paths[i]),
                    "latitude": f"{float(index.center_lats[i]):.8f}",
                    "longitude": f"{float(index.center_lons[i]):.8f}",
                    "grid_region_id": "" if rid is None else rid,
                    "x_m": f"{float(xs[i]):.3f}",
                    "y_m": f"{float(ys[i]):.3f}",
                }
            )


def _build_grid_and_indexes(args: argparse.Namespace) -> tuple[list[GridCell], float, float]:
    index = load_reference_index(args.reference_index)
    xs, ys, finite, origin_lat, origin_lon = _finite_reference_xy(index, args.reference_coordinate)
    raw_cells = _make_grid_cells(xs, ys, finite, origin_lat, origin_lon, args.grid_rows, args.grid_cols, args.grid_padding_m)

    cells: list[GridCell] = []
    for cell in raw_cells:
        base_indices = _indices_for_cell(xs, ys, finite, cell, 0.0)
        halo = args.grid_halo_m
        selected = _indices_for_cell(xs, ys, finite, cell, halo)
        while len(selected) < args.min_local_references and halo < args.max_grid_halo_m:
            halo = min(args.max_grid_halo_m, halo * 1.5)
            selected = _indices_for_cell(xs, ys, finite, cell, halo)
        full_cell = GridCell(
            region_id=cell.region_id,
            row=cell.row,
            col=cell.col,
            x_min=cell.x_min,
            x_max=cell.x_max,
            y_min=cell.y_min,
            y_max=cell.y_max,
            center_lat=cell.center_lat,
            center_lon=cell.center_lon,
            base_ref_count=len(base_indices),
            halo_ref_count=len(selected),
            halo_m=halo,
        )
        cells.append(full_cell)
        _save_reference_subset(index, selected, args.out_dir / f"local_reference_index_grid_{cell.region_id:03d}.npz")

    _write_grid_csv(args.out_dir / "stage9_grid_regions.csv", cells)
    _write_reference_assignments(args.out_dir / "stage9_reference_frame_regions.csv", index, xs, ys, finite, cells)
    return cells, origin_lat, origin_lon


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


def _candidate_regions_for_frame(predictions: list[PredictionRow], i: int, global_region_ids: list[int], args: argparse.Namespace) -> tuple[list[int], str, int | None, int | None]:
    prev_anchor = _nearest_anchor(predictions, i, -1, args.context_window)
    next_anchor = _nearest_anchor(predictions, i, 1, args.context_window)
    prev_region = None if prev_anchor is None else prev_anchor.region_id
    next_region = None if next_anchor is None else next_anchor.region_id

    candidates: list[int] = []
    if prev_region is not None and next_region is not None and prev_region == next_region:
        candidates.append(prev_region)
        reason = "between_same_grid_anchors"
    elif prev_region is not None and next_region is not None:
        candidates.extend([prev_region, next_region])
        reason = "between_different_grid_anchors"
    elif prev_region is not None:
        candidates.append(prev_region)
        reason = "after_previous_grid_anchor"
    elif next_region is not None:
        candidates.append(next_region)
        reason = "before_next_grid_anchor"
    else:
        reason = "no_nearby_anchor_context"

    if args.add_global_top_regions and len(candidates) < args.max_candidate_regions_per_frame:
        for rid in global_region_ids:
            if rid not in candidates:
                candidates.append(rid)
            if len(candidates) >= args.max_candidate_regions_per_frame:
                break

    deduped: list[int] = []
    for rid in candidates:
        if rid is not None and rid not in deduped:
            deduped.append(rid)
    return deduped[: args.max_candidate_regions_per_frame], reason, prev_region, next_region


def _nearest_candidate_center_distance(pred: PredictionRow, cells_by_id: dict[int, GridCell], candidate_regions: list[int]) -> float | None:
    if pred.lat is None or pred.lon is None or not candidate_regions:
        return None
    distances = []
    for rid in candidate_regions:
        cell = cells_by_id.get(rid)
        if cell is not None:
            distances.append(haversine_m(pred.lat, pred.lon, cell.center_lat, cell.center_lon))
    if not distances:
        return None
    return float(min(distances))


def _is_bad_frame(pred: PredictionRow, current_region: int | None, candidate_regions: list[int], dist_to_candidate: float | None, args: argparse.Namespace) -> tuple[bool, str]:
    if pred.is_anchor:
        return False, "anchor_keep"
    if not candidate_regions:
        return False, "no_context_skip"
    if not pred.accepted:
        return True, "rejected_or_weak_in_grid_context"
    if current_region is not None and current_region not in candidate_regions:
        return True, "accepted_jump_to_non_context_grid"
    if dist_to_candidate is not None and dist_to_candidate > args.jump_threshold_m:
        return True, "accepted_far_from_context_grid"
    if pred.confidence < args.low_confidence_threshold:
        return True, "accepted_low_confidence_in_grid_context"
    return False, "accepted_keep"


def _write_plan_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "query_frame_index",
        "query_time_s",
        "original_accepted",
        "original_reason",
        "original_confidence",
        "original_grid_region_id",
        "prev_anchor_grid_region_id",
        "next_anchor_grid_region_id",
        "context_reason",
        "bad_frame_reason",
        "distance_to_candidate_grid_center_m",
        "candidate_grid_regions",
        "will_rerun",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 9 fixed-grid temporal-context rerun planner for GPS_EX1.")
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--reference-index", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--reference-coordinate", choices=["center", "drone"], default="center")
    parser.add_argument("--grid-rows", type=int, default=3)
    parser.add_argument("--grid-cols", type=int, default=5)
    parser.add_argument("--grid-padding-m", type=float, default=30.0)
    parser.add_argument("--grid-halo-m", type=float, default=100.0)
    parser.add_argument("--max-grid-halo-m", type=float, default=260.0)
    parser.add_argument("--min-local-references", type=int, default=120)
    parser.add_argument("--context-window", type=int, default=8)
    parser.add_argument("--jump-threshold-m", type=float, default=180.0)
    parser.add_argument("--low-confidence-threshold", type=float, default=8.0)
    parser.add_argument("--anchor-min-confidence", type=float, default=6.0)
    parser.add_argument("--anchor-min-inliers", type=int, default=10)
    parser.add_argument("--anchor-min-good-matches", type=int, default=20)
    parser.add_argument("--anchor-max-ref-area-frac", type=float, default=0.12)
    parser.add_argument("--anchor-max-ref-width-frac", type=float, default=0.45)
    parser.add_argument("--anchor-max-ref-height-frac", type=float, default=0.45)
    parser.add_argument("--global-top-regions", type=int, default=2)
    parser.add_argument("--add-global-top-regions", action="store_true")
    parser.add_argument("--max-candidate-regions-per-frame", type=int, default=2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells, origin_lat, origin_lon = _build_grid_and_indexes(args)
    cells_by_id = {cell.region_id: cell for cell in cells}

    _, raw_rows = _read_csv_rows(args.prediction_csv)
    predictions = _make_predictions(raw_rows, args)

    anchor_counts: dict[int, int] = {}
    for pred in predictions:
        rid = _grid_cell_for_latlon(pred.lat, pred.lon, cells, origin_lat, origin_lon)
        if pred.is_anchor and rid is not None:
            pred.region_id = rid
            anchor_counts[rid] = anchor_counts.get(rid, 0) + 1
        else:
            pred.region_id = None

    global_region_ids = [rid for rid, _ in sorted(anchor_counts.items(), key=lambda kv: (-kv[1], kv[0]))[: args.global_top_regions]]

    plan_rows: list[dict[str, str]] = []
    frames_by_region: dict[int, list[int]] = {}
    for i, pred in enumerate(predictions):
        current_region = _grid_cell_for_latlon(pred.lat, pred.lon, cells, origin_lat, origin_lon)
        candidate_regions, context_reason, prev_region, next_region = _candidate_regions_for_frame(predictions, i, global_region_ids, args)
        dist_to_candidate = _nearest_candidate_center_distance(pred, cells_by_id, candidate_regions)
        should_rerun, bad_reason = _is_bad_frame(pred, current_region, candidate_regions, dist_to_candidate, args)

        if should_rerun:
            for rid in candidate_regions:
                frames_by_region.setdefault(rid, []).append(pred.query_frame_index)

        plan_rows.append(
            {
                "query_frame_index": str(pred.query_frame_index),
                "query_time_s": f"{pred.query_time_s:.3f}",
                "original_accepted": "1" if pred.accepted else "0",
                "original_reason": pred.reason,
                "original_confidence": f"{pred.confidence:.6f}",
                "original_grid_region_id": "" if current_region is None else str(current_region),
                "prev_anchor_grid_region_id": "" if prev_region is None else str(prev_region),
                "next_anchor_grid_region_id": "" if next_region is None else str(next_region),
                "context_reason": context_reason,
                "bad_frame_reason": bad_reason,
                "distance_to_candidate_grid_center_m": "" if dist_to_candidate is None else f"{dist_to_candidate:.3f}",
                "candidate_grid_regions": ";".join(str(rid) for rid in candidate_regions),
                "will_rerun": "1" if should_rerun else "0",
            }
        )

    _write_plan_csv(args.out_dir / "stage9_grid_context_rerun_plan.csv", plan_rows)
    total_rerun_assignments = 0
    for rid, frames in sorted(frames_by_region.items()):
        count = _write_frame_list(args.out_dir / f"rerun_grid_region_{rid:03d}_frames.txt", frames)
        total_rerun_assignments += count
        print(f"Wrote grid rerun frame list for region {rid}: {count} frames")
    bad_frames_unique = sorted({int(row["query_frame_index"]) for row in plan_rows if row.get("will_rerun") == "1"})
    _write_frame_list(args.out_dir / "stage9_bad_frames_all.txt", bad_frames_unique)

    print("\nStage 9 fixed-grid temporal planner summary")
    print("=" * 80)
    print(f"Input rows: {len(predictions)}")
    print(f"Grid: {args.grid_cols} columns × {args.grid_rows} rows = {len(cells)} cells")
    print(f"Reference coordinate: {args.reference_coordinate}")
    print(f"Anchors used for grid context: {sum(anchor_counts.values())}")
    print("Anchor counts by grid region:")
    for rid, count in sorted(anchor_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        cell = cells_by_id[rid]
        print(f"  grid {rid:03d} (row={cell.row}, col={cell.col}): anchors={count}, refs={cell.halo_ref_count}, center=({cell.center_lat:.8f},{cell.center_lon:.8f})")
    print(f"Global top grid regions: {','.join(str(r) for r in global_region_ids) if global_region_ids else 'none'}")
    print(f"Unique bad/suspicious frames selected for rerun: {len(bad_frames_unique)}")
    print(f"Per-grid rerun assignments: {total_rerun_assignments}")
    print(f"Wrote grid regions: {args.out_dir / 'stage9_grid_regions.csv'}")
    print(f"Wrote plan: {args.out_dir / 'stage9_grid_context_rerun_plan.csv'}")


if __name__ == "__main__":
    main()
