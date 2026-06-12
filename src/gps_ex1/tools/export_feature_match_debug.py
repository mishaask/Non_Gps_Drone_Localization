"""Export article-style feature match debug images.

The normal export_match_debug tool creates a clean side-by-side visual check.
This tool creates images closer to the WildNav/article figures:

    query/test frame | matched reference frame

with local feature correspondences drawn as colored lines, RANSAC inliers
highlighted, and the homography-projected query image footprint shown on the
reference image when a valid homography is found.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from gps_ex1.io.timecode import seconds_to_timecode
from gps_ex1.segmentation.dynamic_masks import DEFAULT_DYNAMIC_CLASSES, create_dynamic_object_masker
from gps_ex1.localization.verification import lightglue_match_points
from gps_ex1.pipeline.localize_video import _scaled_query_canvas


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV is required. Install it with: python -m pip install opencv-python") from exc
    return cv2


def _resolve_path(text: str, project_root: Path) -> Path:
    raw = Path(text)
    normalized = Path(str(raw).replace("\\", "/"))
    candidates = [raw, normalized, project_root / raw, project_root / normalized]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _read_query_frame(cap, frame_index: int):
    cv2 = _require_cv2()
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def _resize_to_height(image, height: int):
    cv2 = _require_cv2()
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Invalid image dimensions")
    scale = height / h
    resized = cv2.resize(image, (max(1, int(round(w * scale))), height), interpolation=cv2.INTER_AREA)
    return resized, scale


def _put_text_lines(image, lines: list[str], origin=(12, 24), font_scale: float = 0.55):
    cv2 = _require_cv2()
    x, y = origin
    for line in lines:
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += int(round(24 * font_scale / 0.55))


def _safe_float(text: str | None) -> float | None:
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _compute_orb_matches(query_bgr, reference_bgr, nfeatures: int, ratio_test: float, query_ignore_mask=None, reference_ignore_mask=None):
    cv2 = _require_cv2()
    query_gray = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=nfeatures)
    allowed_q = None
    allowed_r = None
    if query_ignore_mask is not None and np.any(query_ignore_mask):
        allowed_q = cv2.bitwise_not(query_ignore_mask.astype(np.uint8))
    if reference_ignore_mask is not None and np.any(reference_ignore_mask):
        allowed_r = cv2.bitwise_not(reference_ignore_mask.astype(np.uint8))

    keypoints_q, descriptors_q = orb.detectAndCompute(query_gray, allowed_q)
    keypoints_r, descriptors_r = orb.detectAndCompute(reference_gray, allowed_r)

    if descriptors_q is None or descriptors_r is None or len(keypoints_q) < 4 or len(keypoints_r) < 4:
        return keypoints_q or [], keypoints_r or [], [], None, None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(descriptors_q, descriptors_r, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        best, second = pair
        if best.distance < ratio_test * second.distance:
            good.append(best)

    homography = None
    mask = None
    if len(good) >= 4:
        src_pts = np.float32([keypoints_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_r[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return keypoints_q, keypoints_r, good, homography, mask




def _compute_lightglue_matches(query_bgr, reference_bgr, reference_path: Path):
    """Compute LightGlue/SuperPoint matches for visualization.

    The returned keypoints/matches mimic OpenCV's ORB output so the same drawing
    code can be used for ORB and LightGlue.
    """
    cv2 = _require_cv2()
    points_q, points_r = lightglue_match_points(query_bgr, reference_bgr, reference_path)
    if len(points_q) == 0:
        return [], [], [], None, None

    keypoints_q = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in points_q]
    keypoints_r = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in points_r]
    good = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0) for i in range(len(points_q))]

    homography = None
    mask = None
    if len(good) >= 4:
        src_pts = points_q.reshape(-1, 1, 2).astype(np.float32)
        dst_pts = points_r.reshape(-1, 1, 2).astype(np.float32)
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return keypoints_q, keypoints_r, good, homography, mask


def _apply_translucent_header(canvas, height: int, alpha: float):
    """Darken the header area without fully hiding the image underneath."""
    cv2 = _require_cv2()
    alpha = max(0.0, min(1.0, float(alpha)))
    if alpha <= 0.0 or height <= 0:
        return
    height = min(height, canvas.shape[0])
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], height), (0, 0, 0), -1)
    cv2.addWeighted(overlay[:height, :], alpha, canvas[:height, :], 1.0 - alpha, 0, canvas[:height, :])

def _draw_center_cross(image, point: tuple[int, int], color: tuple[int, int, int], size: int = 12):
    cv2 = _require_cv2()
    x, y = point
    cv2.line(image, (x - size, y), (x + size, y), color, 2, cv2.LINE_AA)
    cv2.line(image, (x, y - size), (x, y + size), color, 2, cv2.LINE_AA)
    cv2.circle(image, (x, y), 4, color, -1, cv2.LINE_AA)


def _draw_article_style_match(
    query_bgr,
    reference_bgr,
    row: dict[str, str],
    query_video_name: str,
    reference_path: Path,
    display_height: int,
    nfeatures: int,
    ratio_test: float,
    max_lines: int,
    show_outliers: bool,
    match_backend: str,
    header_alpha: float,
    query_ignore_mask=None,
    reference_ignore_mask=None,
):
    cv2 = _require_cv2()

    backend_norm = match_backend.strip().lower()
    if backend_norm in {"lightglue", "superpoint-lightglue", "sp-lightglue"}:
        # LightGlue does not use the ORB ignore masks. If dynamic-object masking is
        # enabled, the displayed/filled images are already passed in here.
        kq, kr, good, homography, mask = _compute_lightglue_matches(query_bgr, reference_bgr, reference_path)
        match_label = "LightGlue"
    else:
        kq, kr, good, homography, mask = _compute_orb_matches(query_bgr, reference_bgr, nfeatures, ratio_test, query_ignore_mask=query_ignore_mask, reference_ignore_mask=reference_ignore_mask)
        match_label = "ORB"
    inlier_mask = mask.ravel().astype(bool).tolist() if mask is not None else [False] * len(good)
    inlier_count = int(sum(inlier_mask)) if inlier_mask else 0

    query_display, sq = _resize_to_height(query_bgr, display_height)
    ref_display, sr = _resize_to_height(reference_bgr, display_height)

    h = max(query_display.shape[0], ref_display.shape[0])
    wq = query_display.shape[1]
    wr = ref_display.shape[1]
    canvas = np.zeros((h, wq + wr, 3), dtype=np.uint8)
    canvas[: query_display.shape[0], :wq] = query_display
    canvas[: ref_display.shape[0], wq : wq + wr] = ref_display

    # Query and reference center markers. These are helpful for our assignment,
    # because the requested output is the coordinate of the center of the screen.
    qh, qw = query_bgr.shape[:2]
    rh, rw = reference_bgr.shape[:2]
    _draw_center_cross(canvas, (int(round((qw / 2) * sq)), int(round((qh / 2) * sq))), (255, 255, 255), 14)

    projected_center_text = ""
    if homography is not None:
        corners_q = np.float32([[0, 0], [qw - 1, 0], [qw - 1, qh - 1], [0, qh - 1]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners_q, homography).reshape(-1, 2)
        projected_scaled = np.array([[wq + x * sr, y * sr] for x, y in projected], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [projected_scaled], True, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.polylines(canvas, [projected_scaled], True, (0, 255, 0), 2, cv2.LINE_AA)

        center_q = np.float32([[[qw / 2, qh / 2]]])
        projected_center = cv2.perspectiveTransform(center_q, homography).reshape(2)
        cx_ref = int(round(wq + projected_center[0] * sr))
        cy_ref = int(round(projected_center[1] * sr))
        if 0 <= projected_center[0] < rw and 0 <= projected_center[1] < rh:
            _draw_center_cross(canvas, (cx_ref, cy_ref), (0, 255, 255), 14)
            projected_center_text = f"projected center px=({projected_center[0]:.1f},{projected_center[1]:.1f})"
        else:
            projected_center_text = "projected center outside reference image"

    # Choose matches to draw. Prefer inliers, because they are the matches that
    # RANSAC says agree with one geometric transform. If there are too few, draw
    # the best good matches so failures are still visible.
    indexed_matches = list(enumerate(good))
    inlier_matches = [(i, m) for i, m in indexed_matches if i < len(inlier_mask) and inlier_mask[i]]
    outlier_matches = [(i, m) for i, m in indexed_matches if not (i < len(inlier_mask) and inlier_mask[i])]

    if inlier_matches:
        matches_to_draw = inlier_matches[:max_lines]
        if show_outliers:
            remaining = max(0, max_lines - len(matches_to_draw))
            matches_to_draw += outlier_matches[:remaining]
    else:
        matches_to_draw = indexed_matches[:max_lines]

    palette = [
        (0, 255, 255),   # yellow
        (0, 255, 0),     # green
        (0, 165, 255),   # orange
        (255, 255, 0),   # cyan
        (255, 0, 255),   # magenta
        (255, 255, 255), # white
    ]
    for draw_i, (match_i, match) in enumerate(matches_to_draw):
        q_pt = kq[match.queryIdx].pt
        r_pt = kr[match.trainIdx].pt
        x1 = int(round(q_pt[0] * sq))
        y1 = int(round(q_pt[1] * sq))
        x2 = int(round(wq + r_pt[0] * sr))
        y2 = int(round(r_pt[1] * sr))
        is_inlier = match_i < len(inlier_mask) and inlier_mask[match_i]
        color = palette[draw_i % len(palette)] if is_inlier else (0, 0, 255)
        thickness = 2 if is_inlier else 1
        cv2.circle(canvas, (x1, y1), 4, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x2, y2), 4, color, -1, cv2.LINE_AA)
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    # Translucent text strip at the top for readability without hiding the image.
    header_h = 154
    _apply_translucent_header(canvas, header_h, header_alpha)

    query_frame_index = row.get("query_frame_index", "")
    query_time = row.get("query_timecode") or seconds_to_timecode(row.get("query_time_s"))
    reference_flight = row.get("matched_reference_flight", "")
    reference_frame_index = row.get("matched_reference_frame_index", "")
    reference_time = row.get("matched_reference_timecode") or seconds_to_timecode(row.get("matched_reference_time_s"))
    reason = row.get("filter_reason", "")
    accepted = row.get("filter_accepted", "")
    raw_lat = row.get("pred_latitude", "")
    raw_lon = row.get("pred_longitude", "")
    filt_lat = row.get("filtered_latitude", "")
    filt_lon = row.get("filtered_longitude", "")
    dist = row.get("filter_temporal_distance_m", "")

    _put_text_lines(
        canvas,
        [
            f"QUERY: {query_video_name} frame={query_frame_index} t={query_time}",
            f"REFERENCE: {reference_flight} frame={reference_frame_index} t={reference_time}",
            f"{match_label} matches={len(good)} RANSAC inliers={inlier_count} row_reason={reason} accepted={accepted}",
            f"selected_query_scale={row.get('selected_query_scale', '')} debug_match_backend={match_label}",
            f"ref_inlier_bbox area={row.get('reference_inlier_bbox_area_frac', '')} width={row.get('reference_inlier_bbox_width_frac', '')} height={row.get('reference_inlier_bbox_height_frac', '')}",
            f"raw=({raw_lat},{raw_lon}) filtered=({filt_lat},{filt_lon}) temporal_dist_m={dist}",
            projected_center_text,
        ],
        origin=(12, 22),
        font_scale=0.48,
    )

    # Captions similar to the article figure.
    cv2.putText(canvas, "(a) Query / real-time drone frame", (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, "(a) Query / real-time drone frame", (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "(b) Matched reference frame", (wq + 20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, "(b) Matched reference frame", (wq + 20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    diagnostics = {
        "debug_match_backend": match_label,
        "debug_good_matches_recomputed": str(len(good)),
        "debug_inliers_recomputed": str(inlier_count),
        "debug_homography_found": "1" if homography is not None else "0",
        "debug_reference_path": str(reference_path),
    }
    return canvas, diagnostics


def export_feature_match_debug(
    prediction_csv: Path,
    query_video: Path,
    out_dir: Path,
    project_root: Path,
    max_rows: int = 40,
    accepted_only: bool = False,
    rejected_only: bool = False,
    every_n_rows: int = 1,
    display_height: int = 540,
    nfeatures: int = 2500,
    ratio_test: float = 0.75,
    max_lines: int = 80,
    show_outliers: bool = False,
    mask_dynamic_objects: bool = False,
    mask_model: str = "yolov8n-seg.pt",
    mask_classes: str = DEFAULT_DYNAMIC_CLASSES,
    mask_confidence: float = 0.35,
    mask_iou: float = 0.7,
    mask_dilate_px: int = 4,
    mask_fill: str = "blur",
    mask_imgsz: int = 960,
    mask_device: str | None = None,
    mask_min_area_px: int = 20,
    mask_max_area_frac: float = 0.015,
    mask_max_width_frac: float = 0.20,
    mask_max_height_frac: float = 0.20,
    mask_source: str = "box",
    mask_display: str = "filled",
    match_backend: str = "orb",
    query_scale_mode: str = "selected",
    query_scale_fill: str = "blur",
    header_alpha: float = 0.35,
) -> int:
    cv2 = _require_cv2()
    out_dir.mkdir(parents=True, exist_ok=True)
    masker = create_dynamic_object_masker(
        enabled=mask_dynamic_objects,
        model_name=mask_model,
        classes=mask_classes,
        confidence=mask_confidence,
        iou=mask_iou,
        dilate_px=mask_dilate_px,
        fill=mask_fill,
        imgsz=mask_imgsz,
        device=mask_device,
        min_area_px=mask_min_area_px,
        max_area_frac=mask_max_area_frac,
        max_width_frac=mask_max_width_frac,
        max_height_frac=mask_max_height_frac,
        mask_source=mask_source,
    )
    if masker is not None:
        print(f"Dynamic-object masking enabled for debug images: model={mask_model}, classes={mask_classes}, source={mask_source}, display={mask_display}")

    with prediction_csv.open("r", newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))

    cap = cv2.VideoCapture(str(query_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open query video: {query_video}")

    written = 0
    manifest_rows: list[dict[str, str]] = []

    for row_i, row in enumerate(rows):
        if row_i % max(1, every_n_rows) != 0:
            continue

        accepted_value = (row.get("filter_accepted") or "").strip()
        is_accepted = accepted_value in {"1", "true", "True"}
        if accepted_only and not is_accepted:
            continue
        if rejected_only and is_accepted:
            continue
        if written >= max_rows:
            break

        frame_text = (row.get("query_frame_index") or "").strip()
        ref_text = (row.get("matched_reference_image") or "").strip()
        if not frame_text or not ref_text:
            continue

        frame_index_float = _safe_float(frame_text)
        if frame_index_float is None:
            continue
        frame_index = int(round(frame_index_float))

        query_frame = _read_query_frame(cap, frame_index)
        if query_frame is None:
            continue

        selected_scale = _safe_float(row.get("selected_query_scale")) or 1.0
        if query_scale_mode == "selected":
            query_frame = _scaled_query_canvas(query_frame, selected_scale, query_scale_fill)
        elif query_scale_mode == "original":
            selected_scale = 1.0
        else:
            raise ValueError("query_scale_mode must be one of: selected, original")

        ref_path = _resolve_path(ref_text, project_root)
        reference_frame = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        if reference_frame is None:
            continue

        query_mask = None
        reference_mask = None
        query_detection_count = 0
        reference_detection_count = 0
        if masker is not None:
            query_mask = masker.dynamic_mask_bgr(query_frame)
            reference_mask = masker.dynamic_mask_bgr(reference_frame, cache_key=str(ref_path))
            query_detection_count = sum(1 for d in masker.detections_bgr(query_frame) if d.accepted)
            reference_detection_count = sum(1 for d in masker.detections_bgr(reference_frame, cache_key=str(ref_path)) if d.accepted)
            if mask_display == "filled":
                query_frame = masker.mask_bgr(query_frame)
                reference_frame = masker.mask_bgr(reference_frame, cache_key=str(ref_path))
            elif mask_display == "overlay":
                query_frame = masker.debug_overlay_bgr(query_frame)
                reference_frame = masker.debug_overlay_bgr(reference_frame, cache_key=str(ref_path))
            elif mask_display == "original":
                pass
            else:
                raise ValueError("mask_display must be one of: filled, overlay, original")

        debug_image, diagnostics = _draw_article_style_match(
            query_bgr=query_frame,
            reference_bgr=reference_frame,
            row=row,
            query_video_name=query_video.name,
            reference_path=ref_path,
            display_height=display_height,
            nfeatures=nfeatures,
            ratio_test=ratio_test,
            max_lines=max_lines,
            show_outliers=show_outliers,
            match_backend=match_backend,
            header_alpha=header_alpha,
            query_ignore_mask=query_mask,
            reference_ignore_mask=reference_mask,
        )
        diagnostics["debug_selected_query_scale_used"] = f"{selected_scale:.4f}"
        diagnostics["debug_query_accepted_masks"] = str(query_detection_count)
        diagnostics["debug_reference_accepted_masks"] = str(reference_detection_count)

        safe_ref_flight = row.get("matched_reference_flight", "ref") or "ref"
        safe_ref_frame = row.get("matched_reference_frame_index", "") or ""
        safe_reason = (row.get("filter_reason", "") or "no_reason").replace(" ", "_").replace("/", "_")
        out_name = f"feature_match_{written:04d}_query_{frame_index:06d}_ref_{safe_ref_flight}_{safe_ref_frame}_{safe_reason}.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), debug_image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        manifest_row = dict(row)
        manifest_row.update(
            {
                "debug_image": str(out_path),
                "resolved_reference_image": str(ref_path),
                **diagnostics,
            }
        )
        manifest_rows.append(manifest_row)
        written += 1

    cap.release()

    manifest_path = out_dir / "feature_match_debug_manifest.csv"
    fieldnames: list[str] = []
    for row in manifest_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    if fieldnames:
        with manifest_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export article-style ORB/LightGlue/RANSAC feature-match debug images.")
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--query-video", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root used to resolve relative reference image paths.")
    parser.add_argument("--max-rows", type=int, default=40)
    parser.add_argument("--accepted-only", action="store_true")
    parser.add_argument("--rejected-only", action="store_true")
    parser.add_argument("--every-n-rows", type=int, default=1)
    parser.add_argument("--display-height", type=int, default=540)
    parser.add_argument("--nfeatures", type=int, default=2500)
    parser.add_argument("--ratio-test", type=float, default=0.75)
    parser.add_argument("--max-lines", type=int, default=80)
    parser.add_argument("--show-outliers", action="store_true", help="Draw some rejected/outlier matches in red when space is available.")
    parser.add_argument("--mask-dynamic-objects", action="store_true", help="Use YOLO to suppress people/vehicles before drawing feature-match lines.")
    parser.add_argument("--mask-model", default="yolov8n-seg.pt", help="Ultralytics model name/path for dynamic-object masking.")
    parser.add_argument("--mask-classes", default=DEFAULT_DYNAMIC_CLASSES, help="Comma-separated COCO class names/IDs to mask.")
    parser.add_argument("--mask-confidence", type=float, default=0.35, help="YOLO confidence threshold for dynamic-object masking.")
    parser.add_argument("--mask-iou", type=float, default=0.7, help="YOLO NMS IoU threshold for dynamic-object masking.")
    parser.add_argument("--mask-dilate-px", type=int, default=4, help="Dilate dynamic object masks by this many pixels.")
    parser.add_argument("--mask-fill", choices=["median", "gray", "black", "blur"], default="blur", help="How to fill masked dynamic-object regions for debug display.")
    parser.add_argument("--mask-imgsz", type=int, default=960, help="YOLO inference image size. Larger values catch small aerial cars better.")
    parser.add_argument("--mask-device", default=None, help="Optional Ultralytics device, e.g. cpu or 0.")
    parser.add_argument("--mask-min-area-px", type=int, default=20, help="Reject detections smaller than this many pixels.")
    parser.add_argument("--mask-max-area-frac", type=float, default=0.015, help="Reject detections covering more than this fraction of the image; avoids rooftop false positives.")
    parser.add_argument("--mask-max-width-frac", type=float, default=0.20, help="Reject detections wider than this fraction of the image.")
    parser.add_argument("--mask-max-height-frac", type=float, default=0.20, help="Reject detections taller than this fraction of the image.")
    parser.add_argument("--mask-source", choices=["box", "segmentation", "auto"], default="box", help="Use YOLO boxes, segmentation masks, or auto fallback. Box is safest for aerial cars.")
    parser.add_argument("--mask-display", choices=["filled", "overlay", "original"], default="filled", help="How to display masked areas in the debug image. ORB keypoints are ignored in masked areas regardless.")
    parser.add_argument("--match-backend", choices=["orb", "lightglue"], default="orb", help="Feature lines to draw in the debug image. Use lightglue after installing requirements-stage5.txt.")
    parser.add_argument("--query-scale-mode", choices=["selected", "original"], default="selected", help="Draw matches on the selected scaled query canvas from the CSV, or on the original query frame.")
    parser.add_argument("--query-scale-fill", choices=["blur", "median", "gray", "black"], default="blur", help="Canvas fill used when drawing selected scaled query frames.")
    parser.add_argument("--header-alpha", type=float, default=0.35, help="Transparency of the dark header overlay: 0=no header background, 1=solid black.")
    args = parser.parse_args()

    if args.accepted_only and args.rejected_only:
        raise ValueError("Use only one of --accepted-only or --rejected-only.")

    written = export_feature_match_debug(
        prediction_csv=args.prediction_csv,
        query_video=args.query_video,
        out_dir=args.out_dir,
        project_root=args.project_root,
        max_rows=args.max_rows,
        accepted_only=args.accepted_only,
        rejected_only=args.rejected_only,
        every_n_rows=args.every_n_rows,
        display_height=args.display_height,
        nfeatures=args.nfeatures,
        ratio_test=args.ratio_test,
        max_lines=args.max_lines,
        show_outliers=args.show_outliers,
        mask_dynamic_objects=args.mask_dynamic_objects,
        mask_model=args.mask_model,
        mask_classes=args.mask_classes,
        mask_confidence=args.mask_confidence,
        mask_iou=args.mask_iou,
        mask_dilate_px=args.mask_dilate_px,
        mask_fill=args.mask_fill,
        mask_imgsz=args.mask_imgsz,
        mask_device=args.mask_device,
        mask_min_area_px=args.mask_min_area_px,
        mask_max_area_frac=args.mask_max_area_frac,
        mask_max_width_frac=args.mask_max_width_frac,
        mask_max_height_frac=args.mask_max_height_frac,
        mask_source=args.mask_source,
        mask_display=args.mask_display,
        match_backend=args.match_backend,
        query_scale_mode=args.query_scale_mode,
        query_scale_fill=args.query_scale_fill,
        header_alpha=args.header_alpha,
    )
    print(f"Wrote {written} article-style feature match debug images to: {args.out_dir}")
    print(f"Manifest: {args.out_dir / 'feature_match_debug_manifest.csv'}")


if __name__ == "__main__":
    main()
