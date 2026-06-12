"""Stage 9.1 trajectory/path-consistency filter.

Stage 9 restricts reruns to fixed grid regions, but the merge is still mostly
frame-by-frame.  A visually plausible single-frame match can therefore jump to a
wrong campus area and then jump back.  This tool adds the missing trajectory
sanity layer: it rejects isolated accepted predictions that are inconsistent
with nearby accepted predictions, and optionally writes an interpolated/held
filtered path for visualization.

Typical usage::

    python -m gps_ex1.tools.stage9_path_consistency_filter ^
      --prediction-csv data/processed/stage9_every45_grid15/stage9_merged_grid_008_013_top8.csv ^
      --grid-regions-csv data/processed/stage9_every45_grid15/stage9_grid_regions.csv ^
      --out data/processed/stage9_every45_grid15/stage9_merged_grid_008_013_top8_path_filtered.csv ^
      --override-filter-accepted ^
      --filtered-fill interpolate

The original prediction fields are preserved in stage9_path_original_* columns.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from gps_ex1.geometry.geo import haversine_m
from gps_ex1.tools.stage7_temporal_consensus import (
    _dedup_fieldnames,
    _float_or_none,
    _int_or_zero,
    _safe_fieldnames,
    row_confidence,
)


@dataclass(frozen=True)
class GridCenter:
    region_id: int
    row: int
    col: int
    center_lat: float
    center_lon: float


@dataclass
class PathPoint:
    order: int
    row: dict[str, str]
    query_frame_index: int
    query_time_s: float
    lat: float | None
    lon: float | None
    accepted: bool
    confidence: float
    grid_region_id: int | None
    path_accepted: bool
    status: str = "kept"
    reason: str = "kept"
    previous_trusted_order: int | None = None
    next_trusted_order: int | None = None
    previous_distance_m: float | None = None
    next_distance_m: float | None = None
    same_grid_support: int = 0
    filtered_lat: float | None = None
    filtered_lon: float | None = None
    filtered_method: str = "raw"


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


def _read_grid_centers(path: Path) -> list[GridCenter]:
    centers: list[GridCenter] = []
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rid = _int_or_zero(row.get("grid_region_id"))
            lat = _float_or_none(row.get("center_latitude"))
            lon = _float_or_none(row.get("center_longitude"))
            if rid <= 0 or lat is None or lon is None:
                continue
            centers.append(
                GridCenter(
                    region_id=rid,
                    row=_int_or_zero(row.get("row")),
                    col=_int_or_zero(row.get("col")),
                    center_lat=lat,
                    center_lon=lon,
                )
            )
    if not centers:
        raise RuntimeError(f"No grid centers were found in {path}")
    return centers


def _nearest_grid_region(lat: float | None, lon: float | None, centers: list[GridCenter], max_distance_m: float) -> tuple[int | None, float | None]:
    if lat is None or lon is None:
        return None, None
    best_id: int | None = None
    best_dist = float("inf")
    for center in centers:
        dist = haversine_m(lat, lon, center.center_lat, center.center_lon)
        if dist < best_dist:
            best_dist = dist
            best_id = center.region_id
    if best_dist > max_distance_m:
        return None, best_dist
    return best_id, best_dist


def _parse_region_set(text: str | None) -> set[int]:
    if text is None:
        return set()
    regions: set[int] = set()
    for part in text.replace(";", ",").split(","):
        token = part.strip()
        if token:
            regions.add(int(token))
    return regions


def _make_points(rows: list[dict[str, str]], centers: list[GridCenter], args: argparse.Namespace) -> list[PathPoint]:
    points: list[PathPoint] = []
    for order, row in enumerate(rows):
        lat = _float_or_none(row.get("pred_latitude"))
        lon = _float_or_none(row.get("pred_longitude"))
        grid_id, _ = _nearest_grid_region(lat, lon, centers, args.max_grid_center_distance_m)
        accepted = _accepted(row)
        confidence = row_confidence(row)
        path_accepted = bool(accepted and lat is not None and lon is not None and confidence >= args.min_confidence)
        points.append(
            PathPoint(
                order=order,
                row=row,
                query_frame_index=_int_or_zero(row.get("query_frame_index")),
                query_time_s=_float_or_none(row.get("query_time_s")) or 0.0,
                lat=lat,
                lon=lon,
                accepted=accepted,
                confidence=confidence,
                grid_region_id=grid_id,
                path_accepted=path_accepted,
                filtered_lat=lat if path_accepted else None,
                filtered_lon=lon if path_accepted else None,
            )
        )
    return points


def _trusted_orders(points: list[PathPoint]) -> list[int]:
    return [point.order for point in points if point.path_accepted and point.lat is not None and point.lon is not None]


def _nearest_trusted(points: list[PathPoint], order: int, direction: int, max_steps: int) -> PathPoint | None:
    j = order + direction
    steps = 0
    while 0 <= j < len(points) and steps < max_steps:
        candidate = points[j]
        if candidate.path_accepted and candidate.lat is not None and candidate.lon is not None:
            return candidate
        j += direction
        steps += 1
    return None


def _distance(a: PathPoint | None, b: PathPoint | None) -> float | None:
    if a is None or b is None or a.lat is None or a.lon is None or b.lat is None or b.lon is None:
        return None
    return float(haversine_m(a.lat, a.lon, b.lat, b.lon))


def _same_grid_support(points: list[PathPoint], order: int, window: int) -> int:
    current = points[order]
    if current.grid_region_id is None:
        return 0
    start = max(0, order - window)
    end = min(len(points), order + window + 1)
    support = 0
    for j in range(start, end):
        if j == order:
            continue
        candidate = points[j]
        if candidate.path_accepted and candidate.grid_region_id == current.grid_region_id:
            support += 1
    return support


def _reject(point: PathPoint, reason: str) -> None:
    point.path_accepted = False
    point.status = "rejected"
    point.reason = reason
    point.filtered_lat = None
    point.filtered_lon = None
    point.filtered_method = "rejected"


def _apply_path_rules(points: list[PathPoint], args: argparse.Namespace) -> None:
    trusted_regions = _parse_region_set(args.trusted_grid_regions)
    banned_regions = _parse_region_set(args.banned_grid_regions)

    # Iterate a few times because rejecting one teleport can expose another one.
    for _ in range(max(1, args.iterations)):
        changed = False
        for point in points:
            if not point.path_accepted:
                continue

            point.same_grid_support = _same_grid_support(points, point.order, args.support_window)
            prev_point = _nearest_trusted(points, point.order, -1, args.context_window)
            next_point = _nearest_trusted(points, point.order, 1, args.context_window)
            point.previous_trusted_order = None if prev_point is None else prev_point.order
            point.next_trusted_order = None if next_point is None else next_point.order
            point.previous_distance_m = _distance(point, prev_point)
            point.next_distance_m = _distance(point, next_point)

            if trusted_regions and point.grid_region_id not in trusted_regions:
                _reject(point, "outside_trusted_grid_regions")
                changed = True
                continue
            if banned_regions and point.grid_region_id in banned_regions:
                _reject(point, "inside_banned_grid_region")
                changed = True
                continue

            prev_dist = point.previous_distance_m
            next_dist = point.next_distance_m
            prev_grid = None if prev_point is None else prev_point.grid_region_id
            next_grid = None if next_point is None else next_point.grid_region_id
            cur_grid = point.grid_region_id

            # Green -> red -> green: previous and next context agree, current cell disagrees,
            # and current location is far from both sides.
            if (
                prev_point is not None
                and next_point is not None
                and prev_grid is not None
                and prev_grid == next_grid
                and cur_grid is not None
                and cur_grid != prev_grid
                and prev_dist is not None
                and next_dist is not None
                and prev_dist >= args.teleport_threshold_m
                and next_dist >= args.teleport_threshold_m
            ):
                _reject(point, "isolated_grid_teleport_between_same_context")
                changed = True
                continue

            # Single accepted point far from its local path with no same-cell support.
            far_from_prev = prev_dist is not None and prev_dist >= args.teleport_threshold_m
            far_from_next = next_dist is not None and next_dist >= args.teleport_threshold_m
            has_any_near_context = prev_point is not None or next_point is not None
            if has_any_near_context and (far_from_prev or far_from_next) and point.same_grid_support < args.min_same_grid_support:
                # If both neighbors exist, be strict. If only one exists, allow very high confidence.
                if prev_point is not None and next_point is not None:
                    _reject(point, "far_jump_with_low_same_grid_support")
                    changed = True
                    continue
                if point.confidence < args.high_confidence_keep_threshold:
                    _reject(point, "one_sided_far_jump_with_low_same_grid_support")
                    changed = True
                    continue

            # If both neighboring trusted points are themselves close to each other, current should not be far away.
            bridge_dist = _distance(prev_point, next_point)
            if (
                prev_point is not None
                and next_point is not None
                and bridge_dist is not None
                and bridge_dist <= args.bridge_threshold_m
                and prev_dist is not None
                and next_dist is not None
                and min(prev_dist, next_dist) >= args.teleport_threshold_m
            ):
                _reject(point, "far_from_short_bridge_context")
                changed = True
                continue

            point.status = "kept"
            point.reason = "path_consistent"

        if not changed:
            break


def _find_previous_kept(points: list[PathPoint], order: int) -> PathPoint | None:
    for j in range(order - 1, -1, -1):
        if points[j].path_accepted and points[j].lat is not None and points[j].lon is not None:
            return points[j]
    return None


def _find_next_kept(points: list[PathPoint], order: int) -> PathPoint | None:
    for j in range(order + 1, len(points)):
        if points[j].path_accepted and points[j].lat is not None and points[j].lon is not None:
            return points[j]
    return None


def _apply_filtered_fill(points: list[PathPoint], fill_mode: str) -> None:
    for point in points:
        if point.path_accepted:
            point.filtered_lat = point.lat
            point.filtered_lon = point.lon
            point.filtered_method = "raw_kept"
            continue

        if fill_mode == "none":
            point.filtered_lat = None
            point.filtered_lon = None
            point.filtered_method = "none"
            continue

        prev_point = _find_previous_kept(points, point.order)
        next_point = _find_next_kept(points, point.order)
        if fill_mode == "hold":
            if prev_point is not None:
                point.filtered_lat = prev_point.lat
                point.filtered_lon = prev_point.lon
                point.filtered_method = "hold_previous"
            elif next_point is not None:
                point.filtered_lat = next_point.lat
                point.filtered_lon = next_point.lon
                point.filtered_method = "hold_next"
            else:
                point.filtered_lat = None
                point.filtered_lon = None
                point.filtered_method = "none"
            continue

        if fill_mode == "interpolate":
            if prev_point is not None and next_point is not None and next_point.query_time_s != prev_point.query_time_s:
                alpha = (point.query_time_s - prev_point.query_time_s) / (next_point.query_time_s - prev_point.query_time_s)
                alpha = max(0.0, min(1.0, alpha))
                point.filtered_lat = (1.0 - alpha) * float(prev_point.lat) + alpha * float(next_point.lat)
                point.filtered_lon = (1.0 - alpha) * float(prev_point.lon) + alpha * float(next_point.lon)
                point.filtered_method = "interpolate_previous_next"
            elif prev_point is not None:
                point.filtered_lat = prev_point.lat
                point.filtered_lon = prev_point.lon
                point.filtered_method = "hold_previous"
            elif next_point is not None:
                point.filtered_lat = next_point.lat
                point.filtered_lon = next_point.lon
                point.filtered_method = "hold_next"
            else:
                point.filtered_lat = None
                point.filtered_lon = None
                point.filtered_method = "none"
            continue

        raise ValueError(f"Unsupported fill mode: {fill_mode}")


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
    ]:
        if key in row:
            row[key] = ""


def _format_float(value: float | None, decimals: int = 8) -> str:
    if value is None:
        return ""
    return f"{float(value):.{decimals}f}"


def _augment_rows(points: list[PathPoint], header: list[str], args: argparse.Namespace) -> tuple[list[str], list[dict[str, str]]]:
    extra_cols = [
        "stage9_path_original_filter_accepted",
        "stage9_path_original_filter_reason",
        "stage9_path_original_pred_latitude",
        "stage9_path_original_pred_longitude",
        "stage9_path_original_matched_reference_flight",
        "stage9_path_original_matched_reference_frame_index",
        "stage9_path_grid_region_id",
        "stage9_path_confidence",
        "stage9_path_filter_accepted",
        "stage9_path_filter_status",
        "stage9_path_filter_reason",
        "stage9_path_prev_trusted_frame",
        "stage9_path_next_trusted_frame",
        "stage9_path_jump_prev_m",
        "stage9_path_jump_next_m",
        "stage9_path_same_grid_support",
        "filtered_latitude",
        "filtered_longitude",
        "filtered_method",
    ]
    out_header = list(dict.fromkeys(header + extra_cols))
    out_rows: list[dict[str, str]] = []
    for point in points:
        row = dict(point.row)
        original_filter_accepted = row.get("filter_accepted", "")
        original_filter_reason = row.get("filter_reason", "")
        original_lat = row.get("pred_latitude", "")
        original_lon = row.get("pred_longitude", "")
        original_ref_flight = row.get("matched_reference_flight", "")
        original_ref_frame = row.get("matched_reference_frame_index", "")

        row["stage9_path_original_filter_accepted"] = original_filter_accepted
        row["stage9_path_original_filter_reason"] = original_filter_reason
        row["stage9_path_original_pred_latitude"] = original_lat
        row["stage9_path_original_pred_longitude"] = original_lon
        row["stage9_path_original_matched_reference_flight"] = original_ref_flight
        row["stage9_path_original_matched_reference_frame_index"] = original_ref_frame
        row["stage9_path_grid_region_id"] = "" if point.grid_region_id is None else str(point.grid_region_id)
        row["stage9_path_confidence"] = f"{point.confidence:.6f}"
        row["stage9_path_filter_accepted"] = "1" if point.path_accepted else "0"
        row["stage9_path_filter_status"] = point.status
        row["stage9_path_filter_reason"] = point.reason
        row["stage9_path_prev_trusted_frame"] = "" if point.previous_trusted_order is None else str(points[point.previous_trusted_order].query_frame_index)
        row["stage9_path_next_trusted_frame"] = "" if point.next_trusted_order is None else str(points[point.next_trusted_order].query_frame_index)
        row["stage9_path_jump_prev_m"] = "" if point.previous_distance_m is None else f"{point.previous_distance_m:.3f}"
        row["stage9_path_jump_next_m"] = "" if point.next_distance_m is None else f"{point.next_distance_m:.3f}"
        row["stage9_path_same_grid_support"] = str(point.same_grid_support)
        row["filtered_latitude"] = _format_float(point.filtered_lat)
        row["filtered_longitude"] = _format_float(point.filtered_lon)
        row["filtered_method"] = point.filtered_method

        if args.override_filter_accepted:
            row["filter_accepted"] = "1" if point.path_accepted else "0"
            if point.path_accepted:
                # Preserve the original accepted reason.
                row["filter_reason"] = original_filter_reason or "accepted"
            else:
                row["filter_reason"] = f"path_rejected_{point.reason}"

        if (args.clear_path_rejected_matches and not point.path_accepted and point.accepted) or (args.clear_all_rejected_matches and not point.path_accepted):
            _clear_match_fields(row)

        out_rows.append(row)
    return out_header, out_rows


def _write_summary(path: Path, points: list[PathPoint]) -> None:
    kept = [p for p in points if p.path_accepted]
    rejected = [p for p in points if p.accepted and not p.path_accepted]
    by_reason: dict[str, int] = {}
    by_grid: dict[str, int] = {}
    for p in points:
        key = p.reason if not p.path_accepted else "kept"
        by_reason[key] = by_reason.get(key, 0) + 1
        grid_key = "" if p.grid_region_id is None else str(p.grid_region_id)
        by_grid[grid_key] = by_grid.get(grid_key, 0) + 1
    with path.open("w", encoding="utf-8") as fp:
        fp.write("Stage 9.1 path-consistency filter summary\n")
        fp.write("=" * 80 + "\n")
        fp.write(f"Input rows: {len(points)}\n")
        fp.write(f"Path-kept accepted rows: {len(kept)}\n")
        fp.write(f"Previously accepted rows rejected by path filter: {len(rejected)}\n")
        fp.write("\nReason counts:\n")
        for reason, count in sorted(by_reason.items(), key=lambda kv: (-kv[1], kv[0])):
            fp.write(f"  {reason}: {count}\n")
        fp.write("\nNearest-grid counts over all rows with coordinates:\n")
        for grid, count in sorted(by_grid.items(), key=lambda kv: (kv[0] == "", kv[0])):
            fp.write(f"  {grid or 'none'}: {count}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 9.1 reject impossible green->red->green visual-localization jumps.")
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--grid-regions-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--teleport-threshold-m", type=float, default=160.0)
    parser.add_argument("--bridge-threshold-m", type=float, default=120.0)
    parser.add_argument("--context-window", type=int, default=10)
    parser.add_argument("--support-window", type=int, default=8)
    parser.add_argument("--min-same-grid-support", type=int, default=1)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--high-confidence-keep-threshold", type=float, default=18.0)
    parser.add_argument("--max-grid-center-distance-m", type=float, default=450.0)
    parser.add_argument("--trusted-grid-regions", default="", help="Optional comma-separated grid IDs to keep, e.g. 7,8. Others are path-rejected.")
    parser.add_argument("--banned-grid-regions", default="", help="Optional comma-separated grid IDs to reject, e.g. 13.")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--filtered-fill", choices=["none", "hold", "interpolate"], default="interpolate")
    parser.add_argument("--override-filter-accepted", action="store_true", help="Overwrite filter_accepted/filter_reason using path-filter result.")
    parser.add_argument("--clear-path-rejected-matches", action="store_true", help="Clear match/reference fields for accepted rows rejected by path filter, so debug export skips them.")
    parser.add_argument("--clear-all-rejected-matches", action="store_true", help="Clear match/reference fields for every row rejected by the path filter.")
    args = parser.parse_args()

    if args.context_window < 1 or args.support_window < 1:
        raise ValueError("context/support windows must be >= 1")

    header, rows = _read_rows(args.prediction_csv)
    centers = _read_grid_centers(args.grid_regions_csv)
    points = _make_points(rows, centers, args)
    _apply_path_rules(points, args)
    _apply_filtered_fill(points, args.filtered_fill)
    out_header, out_rows = _augment_rows(points, header, args)
    _write_rows(args.out, out_header, out_rows)

    summary_path = args.summary_out or args.out.with_suffix(".summary.txt")
    _write_summary(summary_path, points)

    original_accepted = sum(1 for p in points if p.accepted)
    kept = sum(1 for p in points if p.path_accepted)
    rejected_original_accepted = sum(1 for p in points if p.accepted and not p.path_accepted)
    print("Stage 9.1 path-consistency filter")
    print("=" * 80)
    print(f"Input rows: {len(points)}")
    print(f"Original accepted rows: {original_accepted}")
    print(f"Path-kept accepted rows: {kept}")
    print(f"Previously accepted rows rejected by path filter: {rejected_original_accepted}")
    print(f"Wrote filtered CSV: {args.out}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
