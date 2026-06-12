"""Refine Stage 10.3 visual-match coordinates using the matched inlier cluster.

This is the safe version of the earlier Stage 11 idea.
It does NOT run global localization again and it does NOT choose new reference frames.
It only takes the already-trusted Stage 10.3 matched query/reference pairs and moves
accepted visual-match coordinates from the broad reference-frame center toward the
actual RANSAC inlier cluster / landmark area in the matched reference image.

Use this AFTER:
    stage10_path_guided_planner -> local reruns -> stage10_merge_path_reruns

Typical input:
    data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_3_merged_tail_rescue_top8.csv

Typical output:
    data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_3_cluster_lookpoint_refined.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from gps_ex1.geometry.geo import local_xy_to_latlon
from gps_ex1.localization.verification import lightglue_match_points
from gps_ex1.pipeline.localize_video import _scaled_query_canvas
from gps_ex1.preprocess.reference_index import ReferenceIndex, load_reference_index, resolve_image_path


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV is required. Install it with: python -m pip install opencv-python") from exc
    return cv2


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _int_or_none(value: object) -> int | None:
    parsed = _float_or_none(value)
    if parsed is None:
        return None
    return int(round(parsed))


def _fmt(value: float | None, decimals: int = 8) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{decimals}f}"


def _accepted(row: dict[str, str]) -> bool:
    return (row.get("filter_accepted") or "").strip() in {"1", "true", "True"}


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        header = list(reader.fieldnames or [])
        return header, [dict(row) for row in reader]


def _write_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _read_query_frame(cap, frame_index: int):
    cv2 = _require_cv2()
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def _compute_orb_matches(query_bgr, reference_bgr, nfeatures: int, ratio_test: float):
    cv2 = _require_cv2()
    query_gray = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kq, dq = orb.detectAndCompute(query_gray, None)
    kr, dr = orb.detectAndCompute(reference_gray, None)
    if dq is None or dr is None or len(kq) < 4 or len(kr) < 4:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = matcher.knnMatch(dq, dr, k=2)
    q_pts: list[tuple[float, float]] = []
    r_pts: list[tuple[float, float]] = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        best, second = pair
        if best.distance < ratio_test * second.distance:
            q_pts.append(kq[best.queryIdx].pt)
            r_pts.append(kr[best.trainIdx].pt)
    return np.asarray(q_pts, dtype=np.float32), np.asarray(r_pts, dtype=np.float32)


def _compute_matches(query_bgr, reference_bgr, reference_path: Path, backend: str, orb_nfeatures: int, ratio_test: float):
    normalized = backend.strip().lower()
    if normalized == "lightglue":
        return lightglue_match_points(query_bgr, reference_bgr, reference_path)
    if normalized == "orb":
        return _compute_orb_matches(query_bgr, reference_bgr, orb_nfeatures, ratio_test)
    raise ValueError("--match-backend must be 'lightglue' or 'orb'")


@dataclass(frozen=True)
class ClusterResult:
    status: str
    inliers: int = 0
    total_matches: int = 0
    cluster_x: float | None = None
    cluster_y: float | None = None
    cluster_area_frac: float | None = None
    cluster_width_frac: float | None = None
    cluster_height_frac: float | None = None
    lat: float | None = None
    lon: float | None = None
    offset_east_m: float | None = None
    offset_north_m: float | None = None
    used: bool = False


def _cluster_metrics(points: np.ndarray, shape_hw: tuple[int, int]) -> tuple[float | None, float | None, float | None]:
    if len(points) < 2:
        return None, None, None
    h, w = shape_hw
    if h <= 0 or w <= 0:
        return None, None, None
    x0 = float(np.min(points[:, 0]))
    x1 = float(np.max(points[:, 0]))
    y0 = float(np.min(points[:, 1]))
    y1 = float(np.max(points[:, 1]))
    width_frac = max(0.0, min(1.0, (x1 - x0) / max(float(w), 1.0)))
    height_frac = max(0.0, min(1.0, (y1 - y0) / max(float(h), 1.0)))
    return width_frac * height_frac, width_frac, height_frac


def _reference_path_key(path_text: str, project_root: Path) -> str:
    path = resolve_image_path(path_text, project_root)
    try:
        return str(path.resolve()).replace("\\", "/").lower()
    except OSError:
        return str(path).replace("\\", "/").lower()


def _build_reference_lookup(index: ReferenceIndex, project_root: Path) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for i, path_text in enumerate(index.image_paths):
        key = _reference_path_key(str(path_text), project_root)
        lookup[key] = i
        lookup[Path(str(path_text).replace("\\", "/")).name.lower()] = i
    return lookup


def _find_reference_index(row: dict[str, str], index: ReferenceIndex, lookup: dict[str, int], project_root: Path) -> int | None:
    ref_image = (row.get("matched_reference_image") or "").strip()
    if ref_image:
        key = _reference_path_key(ref_image, project_root)
        if key in lookup:
            return lookup[key]
        name = Path(ref_image.replace("\\", "/")).name.lower()
        if name in lookup:
            return lookup[name]

    flight = (row.get("matched_reference_flight") or "").strip()
    frame = _int_or_none(row.get("matched_reference_frame_index"))
    if flight and frame is not None:
        for i in range(len(index.frame_indices)):
            if str(index.flight_ids[i]) == flight and int(index.frame_indices[i]) == frame:
                return i
    return None


def _valid_ref_center(index: ReferenceIndex, ref_idx: int) -> tuple[float, float] | None:
    center_lat = float(index.center_lats[ref_idx])
    center_lon = float(index.center_lons[ref_idx])
    if math.isfinite(center_lat) and math.isfinite(center_lon) and abs(center_lat) > 1e-12 and abs(center_lon) > 1e-12:
        return center_lat, center_lon
    drone_lat = float(index.drone_lats[ref_idx])
    drone_lon = float(index.drone_lons[ref_idx])
    if math.isfinite(drone_lat) and math.isfinite(drone_lon):
        return drone_lat, drone_lon
    return None


def _pixel_to_ground_offset(
    cluster_x: float,
    cluster_y: float,
    image_w: int,
    image_h: int,
    heading_deg: float,
    camera_angle_deg: float,
    angle_convention: str,
    horizontal_fov_deg: float,
    fallback_altitude_m: float,
) -> tuple[float, float]:
    """Approximate pixel offset from reference-image center as local east/north meters.

    This is intentionally conservative: it only nudges accepted Stage 10.3 visual
    matches toward the actual matched landmark cluster. It does not choose a new
    global location or create a new path by itself.
    """

    # Interpret camera angle consistently with the rest of the project.
    angle_rad = math.radians(camera_angle_deg)
    altitude = abs(float(fallback_altitude_m))
    if angle_convention == "from-horizon":
        slant_range = altitude / max(math.sin(angle_rad), 1e-6)
    else:  # from-nadir
        slant_range = altitude / max(math.cos(angle_rad), 1e-6)

    hfov_rad = math.radians(horizontal_fov_deg)
    vfov_rad = 2.0 * math.atan(math.tan(hfov_rad / 2.0) * (float(image_h) / max(float(image_w), 1.0)))
    footprint_width_m = 2.0 * slant_range * math.tan(hfov_rad / 2.0)
    footprint_height_m = 2.0 * slant_range * math.tan(vfov_rad / 2.0)

    right_m = ((cluster_x - image_w / 2.0) / max(float(image_w), 1.0)) * footprint_width_m
    forward_m = ((cluster_y - image_h / 2.0) / max(float(image_h), 1.0)) * footprint_height_m

    theta = math.radians(heading_deg)
    right_theta = math.radians(heading_deg + 90.0)
    east = forward_m * math.sin(theta) + right_m * math.sin(right_theta)
    north = forward_m * math.cos(theta) + right_m * math.cos(right_theta)
    return east, north


def _draw_debug(debug_path: Path, query_bgr, reference_bgr, q_pts: np.ndarray, r_pts: np.ndarray, inlier_mask: np.ndarray | None, cluster: ClusterResult, row: dict[str, str]) -> None:
    cv2 = _require_cv2()
    height = 520
    qh, qw = query_bgr.shape[:2]
    rh, rw = reference_bgr.shape[:2]
    qscale = height / max(float(qh), 1.0)
    rscale = height / max(float(rh), 1.0)
    query_small = cv2.resize(query_bgr, (max(1, int(qw * qscale)), height), interpolation=cv2.INTER_AREA)
    ref_small = cv2.resize(reference_bgr, (max(1, int(rw * rscale)), height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, query_small.shape[1] + ref_small.shape[1], 3), dtype=np.uint8)
    canvas[:, : query_small.shape[1]] = query_small
    canvas[:, query_small.shape[1] :] = ref_small

    offset_x = query_small.shape[1]
    if len(q_pts) and len(r_pts):
        mask = np.ones(len(q_pts), dtype=bool) if inlier_mask is None else inlier_mask.astype(bool).reshape(-1)
        for i, (qp, rp) in enumerate(zip(q_pts, r_pts)):
            inlier = bool(mask[i]) if i < len(mask) else False
            color = (0, 220, 0) if inlier else (0, 0, 220)
            qpt = (int(round(qp[0] * qscale)), int(round(qp[1] * qscale)))
            rpt = (int(round(offset_x + rp[0] * rscale)), int(round(rp[1] * rscale)))
            cv2.line(canvas, qpt, rpt, color, 1, cv2.LINE_AA)
            if inlier:
                cv2.circle(canvas, rpt, 3, (0, 255, 255), -1, cv2.LINE_AA)
    if cluster.cluster_x is not None and cluster.cluster_y is not None:
        cx = int(round(offset_x + cluster.cluster_x * rscale))
        cy = int(round(cluster.cluster_y * rscale))
        cv2.drawMarker(canvas, (cx, cy), (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)

    header = canvas.copy()
    cv2.rectangle(header, (0, 0), (canvas.shape[1], 80), (0, 0, 0), -1)
    cv2.addWeighted(header[:80], 0.30, canvas[:80], 0.70, 0, canvas[:80])
    lines = [
        f"frame={row.get('query_frame_index','')} status={cluster.status} used={1 if cluster.used else 0} inliers={cluster.inliers}/{cluster.total_matches}",
        f"cluster=({_fmt(cluster.cluster_x,1)},{_fmt(cluster.cluster_y,1)}) latlon=({_fmt(cluster.lat,8)},{_fmt(cluster.lon,8)})",
        f"ref={row.get('matched_reference_flight','')} frame={row.get('matched_reference_frame_index','')} source={row.get('stage10_final_source','')}",
    ]
    y = 22
    for line in lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_path), canvas)


def _refine_one_row(row: dict[str, str], cap, index: ReferenceIndex, lookup: dict[str, int], args: argparse.Namespace, debug_counter: int | None) -> tuple[dict[str, str], ClusterResult]:
    cv2 = _require_cv2()
    out = dict(row)
    frame_idx = _int_or_none(row.get("query_frame_index"))
    if frame_idx is None:
        return out, ClusterResult(status="missing_query_frame")
    ref_idx = _find_reference_index(row, index, lookup, args.project_root)
    if ref_idx is None:
        return out, ClusterResult(status="missing_reference_index")

    ref_image_path = resolve_image_path(str(index.image_paths[ref_idx]), args.project_root)
    reference_bgr = cv2.imread(str(ref_image_path), cv2.IMREAD_COLOR)
    if reference_bgr is None:
        return out, ClusterResult(status="missing_reference_image")
    query_bgr_original = _read_query_frame(cap, frame_idx)
    if query_bgr_original is None:
        return out, ClusterResult(status="missing_query_image")

    scale = _float_or_none(row.get("selected_query_scale")) or 1.0
    query_bgr = _scaled_query_canvas(query_bgr_original, scale, args.query_scale_fill)
    q_pts, r_pts = _compute_matches(query_bgr, reference_bgr, ref_image_path, args.match_backend, args.orb_nfeatures, args.ratio_test)
    if len(q_pts) < 4 or len(r_pts) < 4:
        return out, ClusterResult(status="not_enough_matches", total_matches=int(len(q_pts)))

    H, mask = cv2.findHomography(q_pts.reshape(-1, 1, 2), r_pts.reshape(-1, 1, 2), cv2.RANSAC, args.ransac_reproj_threshold)
    if H is None or mask is None:
        return out, ClusterResult(status="homography_not_found", total_matches=int(len(q_pts)))
    inlier_mask = mask.reshape(-1).astype(bool)
    inlier_count = int(np.sum(inlier_mask))
    if inlier_count < args.min_inliers:
        result = ClusterResult(status="weak_cluster", inliers=inlier_count, total_matches=int(len(q_pts)))
        if args.debug_dir is not None and debug_counter is not None:
            _draw_debug(args.debug_dir / f"cluster_refine_{debug_counter:04d}_frame_{frame_idx:06d}_weak.jpg", query_bgr, reference_bgr, q_pts, r_pts, inlier_mask, result, row)
        return out, result

    ref_inliers = r_pts[inlier_mask].reshape(-1, 2)
    area_frac, width_frac, height_frac = _cluster_metrics(ref_inliers, reference_bgr.shape[:2])
    if area_frac is not None and area_frac > args.max_cluster_area_frac:
        status = "cluster_too_scattered"
    elif width_frac is not None and width_frac > args.max_cluster_width_frac:
        status = "cluster_too_wide"
    elif height_frac is not None and height_frac > args.max_cluster_height_frac:
        status = "cluster_too_tall"
    else:
        status = "cluster_refined"

    cluster_x = float(np.median(ref_inliers[:, 0]))
    cluster_y = float(np.median(ref_inliers[:, 1]))
    ref_center = _valid_ref_center(index, ref_idx)
    if ref_center is None:
        return out, ClusterResult(status="missing_reference_geo", inliers=inlier_count, total_matches=int(len(q_pts)), cluster_x=cluster_x, cluster_y=cluster_y, cluster_area_frac=area_frac, cluster_width_frac=width_frac, cluster_height_frac=height_frac)

    heading = float(index.headings_deg[ref_idx]) if math.isfinite(float(index.headings_deg[ref_idx])) else 0.0
    east_m, north_m = _pixel_to_ground_offset(
        cluster_x=cluster_x,
        cluster_y=cluster_y,
        image_w=reference_bgr.shape[1],
        image_h=reference_bgr.shape[0],
        heading_deg=heading,
        camera_angle_deg=args.camera_angle_deg,
        angle_convention=args.angle_convention,
        horizontal_fov_deg=args.horizontal_fov_deg,
        fallback_altitude_m=args.fallback_altitude_m,
    )
    lat, lon = local_xy_to_latlon(east_m, north_m, ref_center[0], ref_center[1])
    used = status == "cluster_refined"
    result = ClusterResult(
        status=status,
        inliers=inlier_count,
        total_matches=int(len(q_pts)),
        cluster_x=cluster_x,
        cluster_y=cluster_y,
        cluster_area_frac=area_frac,
        cluster_width_frac=width_frac,
        cluster_height_frac=height_frac,
        lat=lat,
        lon=lon,
        offset_east_m=east_m,
        offset_north_m=north_m,
        used=used,
    )

    if args.debug_dir is not None and debug_counter is not None:
        suffix = "used" if used else status
        _draw_debug(args.debug_dir / f"cluster_refine_{debug_counter:04d}_frame_{frame_idx:06d}_{suffix}.jpg", query_bgr, reference_bgr, q_pts, r_pts, inlier_mask, result, row)
    return out, result


def refine_csv(args: argparse.Namespace) -> tuple[int, int, int]:
    header, rows = _read_rows(args.prediction_csv)
    index = load_reference_index(args.reference_index)
    lookup = _build_reference_lookup(index, args.project_root)
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(args.query_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open query video: {args.query_video}")

    extra = [
        "cluster_original_pred_latitude",
        "cluster_original_pred_longitude",
        "cluster_original_filtered_latitude",
        "cluster_original_filtered_longitude",
        "cluster_status",
        "cluster_used",
        "cluster_inliers",
        "cluster_total_matches",
        "cluster_reference_x",
        "cluster_reference_y",
        "cluster_reference_area_frac",
        "cluster_reference_width_frac",
        "cluster_reference_height_frac",
        "cluster_offset_east_m",
        "cluster_offset_north_m",
        "cluster_latitude",
        "cluster_longitude",
    ]
    out_header = list(dict.fromkeys(header + extra))
    out_rows: list[dict[str, str]] = []
    attempted = 0
    used = 0
    skipped = 0

    for row in rows:
        output = dict(row)
        output["cluster_original_pred_latitude"] = output.get("pred_latitude", "")
        output["cluster_original_pred_longitude"] = output.get("pred_longitude", "")
        output["cluster_original_filtered_latitude"] = output.get("filtered_latitude", "")
        output["cluster_original_filtered_longitude"] = output.get("filtered_longitude", "")

        should_refine = True
        if args.accepted_only and not _accepted(row):
            should_refine = False
        if args.visual_only and (row.get("stage10_final_source") or "") == "path_expected_fallback":
            should_refine = False
        if not (row.get("matched_reference_image") or "").strip():
            should_refine = False

        result = ClusterResult(status="skipped")
        if should_refine:
            attempted += 1
            debug_counter = attempted if args.debug_dir is not None and (args.max_debug_images is None or attempted <= args.max_debug_images) else None
            output, result = _refine_one_row(output, cap, index, lookup, args, debug_counter)
        else:
            skipped += 1

        output["cluster_status"] = result.status
        output["cluster_used"] = "1" if result.used else "0"
        output["cluster_inliers"] = str(result.inliers)
        output["cluster_total_matches"] = str(result.total_matches)
        output["cluster_reference_x"] = _fmt(result.cluster_x, 3)
        output["cluster_reference_y"] = _fmt(result.cluster_y, 3)
        output["cluster_reference_area_frac"] = _fmt(result.cluster_area_frac, 6)
        output["cluster_reference_width_frac"] = _fmt(result.cluster_width_frac, 6)
        output["cluster_reference_height_frac"] = _fmt(result.cluster_height_frac, 6)
        output["cluster_offset_east_m"] = _fmt(result.offset_east_m, 3)
        output["cluster_offset_north_m"] = _fmt(result.offset_north_m, 3)
        output["cluster_latitude"] = _fmt(result.lat, 8)
        output["cluster_longitude"] = _fmt(result.lon, 8)

        if result.used:
            used += 1
            if args.replace_prediction:
                output["pred_latitude"] = _fmt(result.lat, 8)
                output["pred_longitude"] = _fmt(result.lon, 8)
                if "raw_pred_latitude" in output:
                    output["raw_pred_latitude"] = _fmt(result.lat, 8)
                if "raw_pred_longitude" in output:
                    output["raw_pred_longitude"] = _fmt(result.lon, 8)
            if args.replace_filtered:
                output["filtered_latitude"] = _fmt(result.lat, 8)
                output["filtered_longitude"] = _fmt(result.lon, 8)
                output["filtered_method"] = "cluster_refined_visual_match"
        out_rows.append(output)

    cap.release()
    _write_rows(args.out, out_header, out_rows)
    return attempted, used, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine trusted Stage 10.3 visual-match coordinates using matched inlier clusters.")
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--query-video", required=True, type=Path)
    parser.add_argument("--reference-index", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--accepted-only", action="store_true", default=True, help="Only refine rows with filter_accepted=1. Default: on.")
    parser.add_argument("--all-rows", dest="accepted_only", action="store_false", help="Try every row that still has a matched_reference_image.")
    parser.add_argument("--visual-only", action="store_true", default=True, help="Skip Stage 10 path_expected_fallback rows. Default: on.")
    parser.add_argument("--include-fallbacks", dest="visual_only", action="store_false", help="Also try rows marked as path_expected_fallback if they still contain a reference match.")
    parser.add_argument("--match-backend", choices=["lightglue", "orb"], default="lightglue")
    parser.add_argument("--orb-nfeatures", type=int, default=4000)
    parser.add_argument("--ratio-test", type=float, default=0.75)
    parser.add_argument("--min-inliers", type=int, default=10)
    parser.add_argument("--ransac-reproj-threshold", type=float, default=5.0)
    parser.add_argument("--max-cluster-area-frac", type=float, default=0.12)
    parser.add_argument("--max-cluster-width-frac", type=float, default=0.45)
    parser.add_argument("--max-cluster-height-frac", type=float, default=0.45)
    parser.add_argument("--camera-angle-deg", type=float, default=60.0)
    parser.add_argument("--angle-convention", choices=["from-horizon", "from-nadir"], default="from-horizon")
    parser.add_argument("--horizontal-fov-deg", type=float, default=73.0)
    parser.add_argument("--fallback-altitude-m", type=float, default=119.0)
    parser.add_argument("--query-scale-fill", choices=["blur", "median", "gray", "black"], default="blur")
    parser.add_argument("--replace-prediction", action="store_true", help="Replace pred_latitude/pred_longitude when cluster refinement succeeds.")
    parser.add_argument("--replace-filtered", action="store_true", default=True, help="Replace filtered_latitude/filtered_longitude when cluster refinement succeeds. Default: on.")
    parser.add_argument("--no-replace-filtered", dest="replace_filtered", action="store_false")
    parser.add_argument("--max-debug-images", type=int, default=None)
    args = parser.parse_args()

    attempted, used, skipped = refine_csv(args)
    print("\nStage 10.3 cluster-lookpoint refinement summary")
    print("=" * 80)
    print(f"Rows attempted: {attempted}")
    print(f"Rows refined/used: {used}")
    print(f"Rows skipped before matching: {skipped}")
    print(f"Wrote: {args.out}")
    if args.debug_dir is not None:
        print(f"Wrote debug images to: {args.debug_dir}")


if __name__ == "__main__":
    main()
