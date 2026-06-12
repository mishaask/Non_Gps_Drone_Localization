"""Stage 10 path-guided local-rerun planner.

This stage is meant to become the main GPS_EX1 pipeline after Stage 9.1.
It uses the first global run only to build a physically plausible seed path,
then creates small local reference indexes around the expected path location.

Why this is different from Stage 9:
- Stage 9 asks: which fixed grid cells are plausible?
- Stage 10 asks: where should the drone be at this timestamp?

That lets us rerun bad/uncertain frames only around the expected coordinate,
with a higher local top-k, without exploding runtime or giving every bad frame
many chances to match a visually similar but physically impossible region.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from gps_ex1.geometry.geo import haversine_m, latlon_to_local_xy
from gps_ex1.preprocess.reference_index import ReferenceIndex, load_reference_index
from gps_ex1.tools.stage7_temporal_consensus import (
    _float_or_none,
    _int_or_zero,
    _read_csv_rows,
    row_confidence,
)


@dataclass
class SeedPoint:
    order: int
    row: dict[str, str]
    query_frame_index: int
    query_time_s: float
    lat: float | None
    lon: float | None
    original_accepted: bool
    confidence: float
    initial_seed: bool = False
    seed_trusted: bool = False
    seed_reject_reason: str = ""
    expected_lat: float | None = None
    expected_lon: float | None = None
    expected_method: str = "none"
    will_rerun: bool = False
    path_window_id: int | None = None
    local_radius_m: float | None = None
    local_reference_count: int = 0
    dominant_support_count: int = 0
    dominant_component_id: int | None = None
    seed_support_count: int = 0
    seed_support_component_id: int | None = None
    seed_support_component_size: int = 0
    seed_support_status: str = ""


@dataclass
class PathWindow:
    window_id: int
    orders: list[int]
    center_lat: float
    center_lon: float
    local_radius_m: float
    reference_count: int
    expected_min_frame: int
    expected_max_frame: int


def _accepted(row: dict[str, str]) -> bool:
    return (row.get("filter_accepted") or "").strip() in {"1", "true", "True"}


def _reference_lats_lons(index: ReferenceIndex, coordinate: str) -> tuple[np.ndarray, np.ndarray]:
    if coordinate == "drone":
        return index.drone_lats.astype(float), index.drone_lons.astype(float)
    return index.center_lats.astype(float), index.center_lons.astype(float)


def _reference_xy(index: ReferenceIndex, coordinate: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    lats, lons = _reference_lats_lons(index, coordinate)
    finite = np.isfinite(lats) & np.isfinite(lons)
    if not np.any(finite):
        raise RuntimeError("Reference index has no finite coordinates.")
    origin_lat = float(np.mean(lats[finite]))
    origin_lon = float(np.mean(lons[finite]))
    xs = np.full_like(lats, np.nan, dtype=float)
    ys = np.full_like(lats, np.nan, dtype=float)
    for i in np.where(finite)[0]:
        p = latlon_to_local_xy(float(lats[i]), float(lons[i]), origin_lat, origin_lon)
        xs[i] = p.east_m
        ys[i] = p.north_m
    return xs, ys, finite, origin_lat, origin_lon


def _prediction_quality_ok(row: dict[str, str], args: argparse.Namespace) -> bool:
    if not _accepted(row):
        return False
    if row_confidence(row) < args.seed_min_confidence:
        return False
    if _int_or_zero(row.get("homography_inliers")) < args.seed_min_inliers:
        return False
    if _int_or_zero(row.get("orb_good_matches")) < args.seed_min_good_matches:
        return False
    ref_area = _float_or_none(row.get("reference_inlier_bbox_area_frac"))
    ref_width = _float_or_none(row.get("reference_inlier_bbox_width_frac"))
    ref_height = _float_or_none(row.get("reference_inlier_bbox_height_frac"))
    if ref_area is not None and ref_area > args.seed_max_ref_area_frac:
        return False
    if ref_width is not None and ref_width > args.seed_max_ref_width_frac:
        return False
    if ref_height is not None and ref_height > args.seed_max_ref_height_frac:
        return False
    return True


def make_seed_points(rows: list[dict[str, str]], args: argparse.Namespace) -> list[SeedPoint]:
    points: list[SeedPoint] = []
    for order, row in enumerate(rows):
        lat = _float_or_none(row.get("pred_latitude"))
        lon = _float_or_none(row.get("pred_longitude"))
        accepted = _accepted(row)
        initial = bool(lat is not None and lon is not None and _prediction_quality_ok(row, args))
        points.append(
            SeedPoint(
                order=order,
                row=row,
                query_frame_index=_int_or_zero(row.get("query_frame_index")),
                query_time_s=_float_or_none(row.get("query_time_s")) or float(order),
                lat=lat,
                lon=lon,
                original_accepted=accepted,
                confidence=row_confidence(row),
                initial_seed=initial,
                seed_trusted=initial,
                seed_reject_reason="" if initial else "not_seed_candidate",
            )
        )
    return points


def _trusted_before(points: list[SeedPoint], order: int, max_steps: int | None = None) -> SeedPoint | None:
    steps = 0
    for j in range(order - 1, -1, -1):
        if max_steps is not None and steps >= max_steps:
            return None
        if points[j].seed_trusted and points[j].lat is not None and points[j].lon is not None:
            return points[j]
        steps += 1
    return None


def _trusted_after(points: list[SeedPoint], order: int, max_steps: int | None = None) -> SeedPoint | None:
    steps = 0
    for j in range(order + 1, len(points)):
        if max_steps is not None and steps >= max_steps:
            return None
        if points[j].seed_trusted and points[j].lat is not None and points[j].lon is not None:
            return points[j]
        steps += 1
    return None


def _distance(a: SeedPoint | None, b: SeedPoint | None) -> float | None:
    if a is None or b is None or a.lat is None or a.lon is None or b.lat is None or b.lon is None:
        return None
    return float(haversine_m(a.lat, a.lon, b.lat, b.lon))


def _speed_mps(a: SeedPoint | None, b: SeedPoint | None) -> float | None:
    dist = _distance(a, b)
    if dist is None or a is None or b is None:
        return None
    dt = abs(b.query_time_s - a.query_time_s)
    if dt <= 1e-6:
        return None
    return dist / dt



def _initial_seed_indices(points: list[SeedPoint]) -> list[int]:
    return [i for i, p in enumerate(points) if p.initial_seed and p.lat is not None and p.lon is not None]


def _component_score(points: list[SeedPoint], indices: list[int]) -> tuple[int, float, float]:
    """Rank components: more anchors first, then confidence, then time span."""
    if not indices:
        return (0, 0.0, 0.0)
    confidence = float(sum(points[i].confidence for i in indices))
    times = [points[i].query_time_s for i in indices]
    return (len(indices), confidence, max(times) - min(times))


def _build_seed_adjacency(points: list[SeedPoint], indices: list[int], args: argparse.Namespace) -> dict[int, set[int]]:
    """Build a graph of mutually plausible seed anchors.

    A single accepted visual match can be wrong.  A seed is much more believable
    if it belongs to a cluster/chain of nearby accepted anchors.  Edges are kept
    only when two anchors are close enough in space and do not imply excessive
    speed.  This deliberately rejects large spatial leaps even when the time gap
    is long enough to make the speed look plausible.
    """
    adjacency: dict[int, set[int]] = {i: set() for i in indices}
    for pos, i in enumerate(indices):
        a = points[i]
        for j in indices[pos + 1:]:
            b = points[j]
            dist = _distance(a, b)
            speed = _speed_mps(a, b)
            if dist is None:
                continue
            if dist > args.dominant_edge_max_distance_m:
                continue
            if speed is not None and speed > args.dominant_edge_max_speed_mps:
                continue
            adjacency[i].add(j)
            adjacency[j].add(i)
    return adjacency


def _connected_components(adjacency: dict[int, set[int]]) -> list[list[int]]:
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in adjacency:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for nxt in adjacency[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(sorted(component, key=lambda i: points_order_key_placeholder(i)))
    return components


def points_order_key_placeholder(index: int) -> int:
    # Tiny helper used only to keep mypy/older linters simple inside _connected_components.
    return index


def apply_dominant_seed_component(points: list[SeedPoint], args: argparse.Namespace) -> None:
    """Keep only the dominant continuous seed component.

    This is the Stage 10.1 fix for the issue visible in Google Earth:
    the old Stage 10 could keep several disconnected accepted-match islands,
    producing a white path that teleports between campus areas.  Here we keep
    the largest/highest-confidence physically connected island and mark the
    other accepted anchors as rejected before interpolation/local rerun planning.
    """
    if not args.use_dominant_seed_component:
        return

    indices = _initial_seed_indices(points)
    if len(indices) < max(1, args.dominant_min_component_size):
        return

    adjacency = _build_seed_adjacency(points, indices, args)
    components = _connected_components(adjacency)
    if not components:
        return

    # Store support diagnostics before rejecting anything.
    for component_id, component in enumerate(components, start=1):
        for idx in component:
            points[idx].dominant_component_id = component_id
            points[idx].dominant_support_count = len(adjacency.get(idx, set()))

    components.sort(key=lambda comp: _component_score(points, comp), reverse=True)
    best = components[0]
    best_set = set(best)

    # If the best component is too small, do not pretend we have a reliable path.
    if len(best_set) < args.dominant_min_component_size:
        for idx in indices:
            points[idx].seed_trusted = False
            points[idx].seed_reject_reason = "seed_rejected_no_dominant_component"
        return

    for idx in indices:
        if idx in best_set:
            points[idx].seed_trusted = True
            points[idx].seed_reject_reason = "seed_kept_dominant_component"
        else:
            points[idx].seed_trusted = False
            points[idx].seed_reject_reason = "seed_rejected_not_in_dominant_component"


def apply_seed_support_validation(points: list[SeedPoint], args: argparse.Namespace) -> None:
    """Reject accepted seed anchors that do not belong to a locally supported island.

    Stage 10.2 fix: a wrong early match can be locally accepted and can even be
    connected to the main component by a single loose edge.  We therefore run a
    second, stricter support graph over the currently trusted seed anchors.  A
    seed is kept only if it belongs to the largest locally dense support island.

    This is intentionally not a manual blacklist.  It is a temporal-support rule:
    an accepted visual match becomes a seed only when nearby accepted matches also
    support roughly the same physical area.  Small prefix/suffix islands such as
    three early red-region false anchors are rejected automatically.
    """
    if not args.use_seed_support_validation:
        for point in points:
            if point.seed_trusted:
                point.seed_support_status = "support_validation_disabled"
        return

    indices = [i for i, p in enumerate(points) if p.seed_trusted and p.lat is not None and p.lon is not None]
    if len(indices) < max(1, args.seed_support_min_component_size):
        for idx in indices:
            points[idx].seed_trusted = False
            points[idx].seed_reject_reason = "seed_rejected_no_supported_component"
            points[idx].seed_support_status = "rejected_no_supported_component"
        return

    adjacency: dict[int, set[int]] = {i: set() for i in indices}
    for pos, i in enumerate(indices):
        a = points[i]
        for j in indices[pos + 1:]:
            b = points[j]
            dist = _distance(a, b)
            speed = _speed_mps(a, b)
            if dist is None:
                continue
            if dist > args.seed_support_radius_m:
                continue
            if speed is not None and speed > args.seed_support_max_speed_mps:
                continue
            # Optional local-neighborhood guard: support should come from
            # relatively nearby accepted anchors in the sequence, not from one
            # random point far away in time.
            if args.seed_support_order_window > 0:
                seed_rank_i = indices.index(i)
                seed_rank_j = indices.index(j)
                if abs(seed_rank_i - seed_rank_j) > args.seed_support_order_window:
                    continue
            adjacency[i].add(j)
            adjacency[j].add(i)

    components = _connected_components(adjacency)
    if not components:
        for idx in indices:
            points[idx].seed_trusted = False
            points[idx].seed_reject_reason = "seed_rejected_no_support_edges"
            points[idx].seed_support_status = "rejected_no_support_edges"
        return

    # Diagnostics first.
    for component_id, component in enumerate(components, start=1):
        for idx in component:
            points[idx].seed_support_component_id = component_id
            points[idx].seed_support_component_size = len(component)
            points[idx].seed_support_count = len(adjacency.get(idx, set()))

    components.sort(key=lambda comp: _component_score(points, comp), reverse=True)
    best = components[0]
    best_set = set(best)

    if len(best_set) < args.seed_support_min_component_size:
        for idx in indices:
            points[idx].seed_trusted = False
            points[idx].seed_reject_reason = "seed_rejected_support_component_too_small"
            points[idx].seed_support_status = "rejected_support_component_too_small"
        return

    for idx in indices:
        p = points[idx]
        if idx not in best_set:
            p.seed_trusted = False
            p.seed_reject_reason = "seed_rejected_not_in_supported_component"
            p.seed_support_status = "rejected_not_in_supported_component"
            continue
        if p.seed_support_count < args.seed_min_support:
            p.seed_trusted = False
            p.seed_reject_reason = "seed_rejected_low_local_support"
            p.seed_support_status = "rejected_low_local_support"
            continue
        p.seed_trusted = True
        p.seed_reject_reason = "seed_kept_supported_component"
        p.seed_support_status = "kept_supported_component"


def _tail_edge_ok(a: SeedPoint | None, b: SeedPoint | None, args: argparse.Namespace) -> bool:
    """Return True if two late seed anchors form a plausible sparse tail edge."""
    dist = _distance(a, b)
    speed = _speed_mps(a, b)
    if dist is None:
        return False
    if dist > args.seed_tail_rescue_max_distance_m:
        return False
    if speed is not None and speed > args.seed_tail_rescue_max_speed_mps:
        return False
    if a is not None and b is not None:
        gap_s = abs(b.query_time_s - a.query_time_s)
        if gap_s > args.seed_tail_rescue_max_gap_s:
            return False
    return True


def apply_seed_tail_rescue(points: list[SeedPoint], args: argparse.Namespace) -> None:
    """Rescue a sparse but plausible suffix/landing tail.

    Stage 10.2 intentionally rejects small unsupported islands.  That fixed the
    false prefix anchors, but it can also remove a valid landing tail because
    the end of the flight naturally has little or no *future* support.  This
    function is asymmetric on purpose:

    * unsupported prefix islands remain rejected;
    * only a suffix after the current dominant/supported path can be rescued;
    * the rescued suffix must be strong enough and physically plausible from
      the last trusted seed.
    """
    if not args.use_seed_tail_rescue:
        return

    trusted = [
        i for i, p in enumerate(points)
        if p.seed_trusted and p.initial_seed and p.lat is not None and p.lon is not None
    ]
    if not trusted:
        return

    last_trusted = max(trusted, key=lambda i: points[i].query_time_s)
    late_candidates = [
        i for i, p in enumerate(points)
        if (
            p.initial_seed
            and not p.seed_trusted
            and p.lat is not None
            and p.lon is not None
            and p.query_time_s > points[last_trusted].query_time_s
            and p.confidence >= args.seed_tail_rescue_min_confidence
        )
    ]
    if not late_candidates:
        return

    # Build the latest plausible suffix chain backwards.  Starting from the
    # newest rejected seed means we rescue the landing tail, not a random middle
    # island.
    late_candidates = sorted(late_candidates, key=lambda i: points[i].query_time_s)
    suffix = [late_candidates[-1]]
    current = late_candidates[-1]
    for candidate in reversed(late_candidates[:-1]):
        if _tail_edge_ok(points[candidate], points[current], args):
            suffix.append(candidate)
            current = candidate

    suffix = sorted(suffix, key=lambda i: points[i].query_time_s)
    if len(suffix) < args.seed_tail_rescue_min_points:
        return

    prev = _trusted_before(points, suffix[0], None)
    if not _tail_edge_ok(prev, points[suffix[0]], args):
        return

    for idx in suffix:
        p = points[idx]
        p.seed_trusted = True
        p.seed_reject_reason = "seed_kept_sparse_suffix_tail_rescue"
        if p.seed_support_status:
            p.seed_support_status += "+tail_rescued"
        else:
            p.seed_support_status = "tail_rescued"



def filter_seed_path(points: list[SeedPoint], args: argparse.Namespace) -> None:
    """Reject impossible first-pass accepted anchors.

    The goal is not to create a perfect path.  The goal is to remove obvious
    red-region false anchors before they can guide local reruns.
    """

    for _ in range(max(1, args.seed_filter_iterations)):
        changed = False
        for point in points:
            if not point.seed_trusted:
                continue
            prev_point = _trusted_before(points, point.order, args.seed_context_window)
            next_point = _trusted_after(points, point.order, args.seed_context_window)
            prev_dist = _distance(prev_point, point)
            next_dist = _distance(point, next_point)
            bridge_dist = _distance(prev_point, next_point)
            prev_speed = _speed_mps(prev_point, point)
            next_speed = _speed_mps(point, next_point)
            bridge_speed = _speed_mps(prev_point, next_point)

            far_prev = prev_dist is not None and prev_dist >= args.teleport_threshold_m
            far_next = next_dist is not None and next_dist >= args.teleport_threshold_m
            fast_prev = prev_speed is not None and prev_speed >= args.max_anchor_speed_mps
            fast_next = next_speed is not None and next_speed >= args.max_anchor_speed_mps
            bridge_is_reasonable = (
                bridge_dist is not None
                and bridge_dist <= args.bridge_threshold_m
            ) or (
                bridge_speed is not None
                and bridge_speed <= args.max_anchor_speed_mps
            )

            # Classic green -> red -> green: previous and next are mutually plausible,
            # current is far from both.
            if prev_point is not None and next_point is not None and bridge_is_reasonable and far_prev and far_next:
                point.seed_trusted = False
                point.seed_reject_reason = "seed_rejected_isolated_teleport_between_reasonable_neighbors"
                changed = True
                continue

            # If it requires physically impossible speed on both sides, reject it.
            if prev_point is not None and next_point is not None and fast_prev and fast_next:
                point.seed_trusted = False
                point.seed_reject_reason = "seed_rejected_speed_impossible_between_neighbors"
                changed = True
                continue

            # Do not reject one-sided beginning/end anchors here.  A bad early
            # neighbor can otherwise cause the true first/last anchor to be removed.
            # Isolated one-sided points can still be ignored later if they do not
            # support any useful local-rerun path.

            point.seed_reject_reason = "seed_kept"
        if not changed:
            break


def interpolate_expected_path(points: list[SeedPoint]) -> None:
    for point in points:
        if point.seed_trusted and point.lat is not None and point.lon is not None:
            point.expected_lat = point.lat
            point.expected_lon = point.lon
            point.expected_method = "seed_anchor"
            continue
        prev_point = _trusted_before(points, point.order, None)
        next_point = _trusted_after(points, point.order, None)
        if prev_point is not None and next_point is not None and next_point.query_time_s != prev_point.query_time_s:
            alpha = (point.query_time_s - prev_point.query_time_s) / (next_point.query_time_s - prev_point.query_time_s)
            alpha = max(0.0, min(1.0, alpha))
            point.expected_lat = (1.0 - alpha) * float(prev_point.lat) + alpha * float(next_point.lat)
            point.expected_lon = (1.0 - alpha) * float(prev_point.lon) + alpha * float(next_point.lon)
            point.expected_method = "interpolate_seed_anchors"
        elif prev_point is not None:
            point.expected_lat = prev_point.lat
            point.expected_lon = prev_point.lon
            point.expected_method = "hold_previous_seed"
        elif next_point is not None:
            point.expected_lat = next_point.lat
            point.expected_lon = next_point.lon
            point.expected_method = "hold_next_seed"
        else:
            point.expected_lat = None
            point.expected_lon = None
            point.expected_method = "no_seed_path"


def mark_rerun_frames(points: list[SeedPoint], mode: str) -> None:
    for point in points:
        if point.expected_lat is None or point.expected_lon is None:
            point.will_rerun = False
            continue
        if mode == "all-frames":
            point.will_rerun = True
        elif mode == "all-non-seed":
            point.will_rerun = not point.seed_trusted
        elif mode == "rejected-only":
            point.will_rerun = not point.original_accepted or not point.seed_trusted
        else:
            raise ValueError(f"Unsupported rerun mode: {mode}")


def _expected_xy(point: SeedPoint, origin_lat: float, origin_lon: float) -> tuple[float, float] | None:
    if point.expected_lat is None or point.expected_lon is None:
        return None
    p = latlon_to_local_xy(point.expected_lat, point.expected_lon, origin_lat, origin_lon)
    return p.east_m, p.north_m


def _make_path_windows(points: list[SeedPoint], origin_lat: float, origin_lon: float, args: argparse.Namespace) -> list[list[int]]:
    rerun_orders = [p.order for p in points if p.will_rerun and p.expected_lat is not None and p.expected_lon is not None]
    if not rerun_orders:
        return []

    windows: list[list[int]] = []
    current: list[int] = []
    current_xy: list[tuple[float, float]] = []

    for order in rerun_orders:
        xy = _expected_xy(points[order], origin_lat, origin_lon)
        if xy is None:
            continue
        start_new = False
        if current and len(current) >= args.max_frames_per_window:
            start_new = True
        if current_xy:
            cx = float(np.mean([p[0] for p in current_xy]))
            cy = float(np.mean([p[1] for p in current_xy]))
            if float(np.hypot(xy[0] - cx, xy[1] - cy)) > args.window_spacing_m:
                start_new = True
        if start_new:
            windows.append(current)
            current = []
            current_xy = []
        current.append(order)
        current_xy.append(xy)
    if current:
        windows.append(current)
    return windows


def _indices_near_expected_points(
    ref_xs: np.ndarray,
    ref_ys: np.ndarray,
    finite: np.ndarray,
    expected_xys: list[tuple[float, float]],
    radius_m: float,
) -> list[int]:
    if not expected_xys:
        return []
    mask = np.zeros_like(finite, dtype=bool)
    finite_indices = np.where(finite)[0]
    for ex, ey in expected_xys:
        dist = np.hypot(ref_xs[finite_indices] - ex, ref_ys[finite_indices] - ey)
        mask[finite_indices[dist <= radius_m]] = True
    return [int(i) for i in np.where(mask & finite)[0]]


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


def build_local_windows(points: list[SeedPoint], index: ReferenceIndex, args: argparse.Namespace) -> list[PathWindow]:
    ref_xs, ref_ys, finite, origin_lat, origin_lon = _reference_xy(index, args.reference_coordinate)
    order_windows = _make_path_windows(points, origin_lat, origin_lon, args)
    path_windows: list[PathWindow] = []

    for window_id, orders in enumerate(order_windows, start=1):
        expected_xys = [_expected_xy(points[o], origin_lat, origin_lon) for o in orders]
        expected_xys = [xy for xy in expected_xys if xy is not None]
        radius = float(args.local_radius_m)
        selected = _indices_near_expected_points(ref_xs, ref_ys, finite, expected_xys, radius)
        while len(selected) < args.min_local_references and radius < args.max_local_radius_m:
            radius = min(args.max_local_radius_m, radius * 1.5)
            selected = _indices_near_expected_points(ref_xs, ref_ys, finite, expected_xys, radius)

        out_index = args.out_dir / f"local_reference_index_path_window_{window_id:03d}.npz"
        _save_reference_subset(index, selected, out_index)
        frame_list = args.out_dir / f"rerun_path_window_{window_id:03d}_frames.txt"
        _write_frame_list(frame_list, [points[o].query_frame_index for o in orders])

        center_lat = float(np.mean([points[o].expected_lat for o in orders if points[o].expected_lat is not None]))
        center_lon = float(np.mean([points[o].expected_lon for o in orders if points[o].expected_lon is not None]))
        for order in orders:
            points[order].path_window_id = window_id
            points[order].local_radius_m = radius
            points[order].local_reference_count = len(selected)

        path_windows.append(
            PathWindow(
                window_id=window_id,
                orders=orders,
                center_lat=center_lat,
                center_lon=center_lon,
                local_radius_m=radius,
                reference_count=len(selected),
                expected_min_frame=min(points[o].query_frame_index for o in orders),
                expected_max_frame=max(points[o].query_frame_index for o in orders),
            )
        )
    return path_windows


def _write_seed_path(path: Path, points: list[SeedPoint]) -> None:
    fields = [
        "query_frame_index",
        "query_time_s",
        "pred_latitude",
        "pred_longitude",
        "original_accepted",
        "original_reason",
        "confidence",
        "initial_seed",
        "seed_trusted",
        "seed_reject_reason",
        "dominant_component_id",
        "dominant_support_count",
        "seed_support_component_id",
        "seed_support_component_size",
        "seed_support_count",
        "seed_support_status",
        "expected_latitude",
        "expected_longitude",
        "expected_method",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for point in points:
            if not point.initial_seed and not point.seed_trusted:
                # Keep the CSV focused on anchors/expected path, not every rejected visual attempt.
                continue
            writer.writerow(
                {
                    "query_frame_index": point.query_frame_index,
                    "query_time_s": f"{point.query_time_s:.3f}",
                    "pred_latitude": "" if point.lat is None else f"{point.lat:.8f}",
                    "pred_longitude": "" if point.lon is None else f"{point.lon:.8f}",
                    "original_accepted": "1" if point.original_accepted else "0",
                    "original_reason": point.row.get("filter_reason", ""),
                    "confidence": f"{point.confidence:.6f}",
                    "initial_seed": "1" if point.initial_seed else "0",
                    "seed_trusted": "1" if point.seed_trusted else "0",
                    "seed_reject_reason": point.seed_reject_reason,
                    "dominant_component_id": "" if point.dominant_component_id is None else point.dominant_component_id,
                    "dominant_support_count": point.dominant_support_count,
                    "seed_support_component_id": "" if point.seed_support_component_id is None else point.seed_support_component_id,
                    "seed_support_component_size": point.seed_support_component_size,
                    "seed_support_count": point.seed_support_count,
                    "seed_support_status": point.seed_support_status,
                    "expected_latitude": "" if point.expected_lat is None else f"{point.expected_lat:.8f}",
                    "expected_longitude": "" if point.expected_lon is None else f"{point.expected_lon:.8f}",
                    "expected_method": point.expected_method,
                }
            )


def _write_plan(path: Path, points: list[SeedPoint]) -> None:
    fields = [
        "query_frame_index",
        "query_time_s",
        "original_accepted",
        "original_reason",
        "original_confidence",
        "initial_seed",
        "seed_trusted",
        "seed_reject_reason",
        "dominant_component_id",
        "dominant_support_count",
        "seed_support_component_id",
        "seed_support_component_size",
        "seed_support_count",
        "seed_support_status",
        "expected_latitude",
        "expected_longitude",
        "expected_method",
        "will_rerun",
        "path_window_id",
        "local_radius_m",
        "local_reference_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "query_frame_index": point.query_frame_index,
                    "query_time_s": f"{point.query_time_s:.3f}",
                    "original_accepted": "1" if point.original_accepted else "0",
                    "original_reason": point.row.get("filter_reason", ""),
                    "original_confidence": f"{point.confidence:.6f}",
                    "initial_seed": "1" if point.initial_seed else "0",
                    "seed_trusted": "1" if point.seed_trusted else "0",
                    "seed_reject_reason": point.seed_reject_reason,
                    "dominant_component_id": "" if point.dominant_component_id is None else point.dominant_component_id,
                    "dominant_support_count": point.dominant_support_count,
                    "seed_support_component_id": "" if point.seed_support_component_id is None else point.seed_support_component_id,
                    "seed_support_component_size": point.seed_support_component_size,
                    "seed_support_count": point.seed_support_count,
                    "seed_support_status": point.seed_support_status,
                    "expected_latitude": "" if point.expected_lat is None else f"{point.expected_lat:.8f}",
                    "expected_longitude": "" if point.expected_lon is None else f"{point.expected_lon:.8f}",
                    "expected_method": point.expected_method,
                    "will_rerun": "1" if point.will_rerun else "0",
                    "path_window_id": "" if point.path_window_id is None else f"{point.path_window_id:03d}",
                    "local_radius_m": "" if point.local_radius_m is None else f"{point.local_radius_m:.3f}",
                    "local_reference_count": point.local_reference_count,
                }
            )


def _write_window_csv(path: Path, windows: list[PathWindow]) -> None:
    fields = ["path_window_id", "frame_count", "min_query_frame", "max_query_frame", "center_latitude", "center_longitude", "local_radius_m", "reference_count"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for window in windows:
            writer.writerow(
                {
                    "path_window_id": f"{window.window_id:03d}",
                    "frame_count": len(window.orders),
                    "min_query_frame": window.expected_min_frame,
                    "max_query_frame": window.expected_max_frame,
                    "center_latitude": f"{window.center_lat:.8f}",
                    "center_longitude": f"{window.center_lon:.8f}",
                    "local_radius_m": f"{window.local_radius_m:.3f}",
                    "reference_count": window.reference_count,
                }
            )


def build_stage10_plan(args: argparse.Namespace) -> tuple[list[SeedPoint], list[PathWindow]]:
    header, rows = _read_csv_rows(args.prediction_csv)
    index = load_reference_index(args.reference_index)
    points = make_seed_points(rows, args)
    apply_dominant_seed_component(points, args)
    apply_seed_support_validation(points, args)
    apply_seed_tail_rescue(points, args)
    filter_seed_path(points, args)
    interpolate_expected_path(points)
    mark_rerun_frames(points, args.rerun_mode)
    windows = build_local_windows(points, index, args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_seed_path(args.out_dir / "stage10_seed_path.csv", points)
    _write_plan(args.out_dir / "stage10_path_rerun_plan.csv", points)
    _write_window_csv(args.out_dir / "stage10_path_windows.csv", windows)
    return points, windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a path-guided local-rerun plan from first-pass accepted anchors.")
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--reference-index", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--reference-coordinate", choices=["center", "drone"], default="center")
    parser.add_argument("--seed-min-confidence", type=float, default=6.0)
    parser.add_argument("--seed-min-inliers", type=int, default=10)
    parser.add_argument("--seed-min-good-matches", type=int, default=20)
    parser.add_argument("--seed-max-ref-area-frac", type=float, default=0.12)
    parser.add_argument("--seed-max-ref-width-frac", type=float, default=0.45)
    parser.add_argument("--seed-max-ref-height-frac", type=float, default=0.45)
    parser.add_argument("--teleport-threshold-m", type=float, default=160.0)
    parser.add_argument("--bridge-threshold-m", type=float, default=120.0)
    parser.add_argument("--max-anchor-speed-mps", type=float, default=45.0)
    parser.add_argument("--high-confidence-seed-keep", type=float, default=18.0)
    parser.add_argument("--seed-context-window", type=int, default=10)
    parser.add_argument("--seed-filter-iterations", type=int, default=4)
    parser.add_argument("--use-dominant-seed-component", action="store_true", default=True)
    parser.add_argument("--no-dominant-seed-component", dest="use_dominant_seed_component", action="store_false")
    parser.add_argument("--dominant-edge-max-distance-m", type=float, default=180.0)
    parser.add_argument("--dominant-edge-max-speed-mps", type=float, default=25.0)
    parser.add_argument("--dominant-min-component-size", type=int, default=4)
    parser.add_argument("--use-seed-support-validation", action="store_true", default=True)
    parser.add_argument("--no-seed-support-validation", dest="use_seed_support_validation", action="store_false")
    parser.add_argument("--seed-support-radius-m", type=float, default=140.0)
    parser.add_argument("--seed-support-max-speed-mps", type=float, default=30.0)
    parser.add_argument("--seed-support-order-window", type=int, default=8)
    parser.add_argument("--seed-min-support", type=int, default=1)
    parser.add_argument("--seed-support-min-component-size", type=int, default=4)
    parser.add_argument("--use-seed-tail-rescue", action="store_true", default=True)
    parser.add_argument("--no-seed-tail-rescue", dest="use_seed_tail_rescue", action="store_false")
    parser.add_argument("--seed-tail-rescue-min-points", type=int, default=1)
    parser.add_argument("--seed-tail-rescue-min-confidence", type=float, default=8.0)
    parser.add_argument("--seed-tail-rescue-max-distance-m", type=float, default=420.0)
    parser.add_argument("--seed-tail-rescue-max-speed-mps", type=float, default=35.0)
    parser.add_argument("--seed-tail-rescue-max-gap-s", type=float, default=120.0)
    parser.add_argument("--rerun-mode", choices=["all-non-seed", "rejected-only", "all-frames"], default="all-non-seed")
    parser.add_argument("--local-radius-m", type=float, default=120.0)
    parser.add_argument("--max-local-radius-m", type=float, default=260.0)
    parser.add_argument("--min-local-references", type=int, default=100)
    parser.add_argument("--window-spacing-m", type=float, default=120.0)
    parser.add_argument("--max-frames-per-window", type=int, default=45)
    args = parser.parse_args()

    points, windows = build_stage10_plan(args)
    initial_seed = sum(1 for p in points if p.initial_seed)
    trusted_seed = sum(1 for p in points if p.seed_trusted)
    will_rerun = sum(1 for p in points if p.will_rerun)

    print("\nStage 10 path-guided planner summary")
    print("=" * 80)
    print(f"Input rows: {len(points)}")
    print(f"Initial accepted seed anchors: {initial_seed}")
    print(f"Trusted seed anchors after dominant/jump filtering: {trusted_seed}")
    print(f"Rejected seed anchors: {initial_seed - trusted_seed}")
    tail_rescued = sum(1 for p in points if p.initial_seed and "tail_rescued" in (p.seed_support_status or ""))
    if tail_rescued:
        print(f"Sparse suffix tail rescued anchors: {tail_rescued}")
    component_counts: dict[str, int] = {}
    for p in points:
        if p.initial_seed:
            key = "" if p.dominant_component_id is None else str(p.dominant_component_id)
            component_counts[key] = component_counts.get(key, 0) + 1
    if component_counts:
        print("Seed component sizes: " + ", ".join(f"{k or 'none'}={v}" for k, v in sorted(component_counts.items())))
    support_counts: dict[str, int] = {}
    support_status_counts: dict[str, int] = {}
    for p in points:
        if p.initial_seed:
            key = "" if p.seed_support_component_id is None else str(p.seed_support_component_id)
            support_counts[key] = support_counts.get(key, 0) + 1
            status = p.seed_support_status or "none"
            support_status_counts[status] = support_status_counts.get(status, 0) + 1
    if support_counts:
        print("Seed support component sizes: " + ", ".join(f"{k or 'none'}={v}" for k, v in sorted(support_counts.items())))
    if support_status_counts:
        print("Seed support statuses: " + ", ".join(f"{k}={v}" for k, v in sorted(support_status_counts.items())))
    print(f"Frames selected for local path rerun: {will_rerun}")
    print(f"Path local windows: {len(windows)}")
    for window in windows:
        print(
            f"  window {window.window_id:03d}: frames={len(window.orders)}, "
            f"range={window.expected_min_frame}-{window.expected_max_frame}, "
            f"refs={window.reference_count}, radius≈{window.local_radius_m:.1f}m"
        )
    print(f"Wrote seed path: {args.out_dir / 'stage10_seed_path.csv'}")
    print(f"Wrote rerun plan: {args.out_dir / 'stage10_path_rerun_plan.csv'}")


if __name__ == "__main__":
    main()
