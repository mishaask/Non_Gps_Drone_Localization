"""Create before/after visual comparison sheets for Stage 7 local reruns.

This debug tool is intentionally simple and transparent:

1. Read a previous/raw CSV and a new/rerun CSV.
2. Keep the query frames that appear in both CSVs, or the frames listed in
   --frame-list.
3. Reuse export_feature_match_debug to render each row from both CSVs.
4. Stack the BEFORE and AFTER debug images into one comparison image per frame.

The result is useful for checking whether a local temporal rerun actually
improved the visual match before accepting/merging it.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Iterable

from gps_ex1.geometry.geo import haversine_m
from gps_ex1.segmentation.dynamic_masks import DEFAULT_DYNAMIC_CLASSES
from gps_ex1.tools.export_feature_match_debug import _require_cv2, export_feature_match_debug
from gps_ex1.tools.stage7_temporal_consensus import row_confidence


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _frame(row: dict[str, str]) -> int | None:
    value = _safe_float(row.get("query_frame_index"))
    if value is None:
        return None
    return int(round(value))


def _accepted(row: dict[str, str]) -> bool:
    return (row.get("filter_accepted") or "").strip() in {"1", "true", "True"}


def _lat_lon(row: dict[str, str]) -> tuple[float | None, float | None]:
    lat = _safe_float(row.get("pred_latitude") or row.get("raw_pred_latitude"))
    lon = _safe_float(row.get("pred_longitude") or row.get("raw_pred_longitude"))
    return lat, lon


def _distance_between_rows_m(a: dict[str, str], b: dict[str, str]) -> float | None:
    lat_a, lon_a = _lat_lon(a)
    lat_b, lon_b = _lat_lon(b)
    if lat_a is None or lon_a is None or lat_b is None or lon_b is None:
        return None
    return haversine_m(lat_a, lon_a, lat_b, lon_b)


def _parse_frame_list(path: Path) -> list[int]:
    frames: list[int] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for part in line.split(","):
            token = part.strip()
            if not token:
                continue
            if ":" in token and "-" in token:
                range_part, step_part = token.split(":", 1)
                start_text, end_text = range_part.split("-", 1)
                start = int(start_text.strip())
                end = int(end_text.strip())
                step = int(step_part.strip())
                frames.extend(list(range(start, end + 1, step)))
            elif "-" in token:
                start_text, end_text = token.split("-", 1)
                start = int(start_text.strip())
                end = int(end_text.strip())
                frames.extend(list(range(start, end + 1)))
            else:
                frames.append(int(token))
    # Preserve order while removing duplicates.
    seen: set[int] = set()
    output: list[int] = []
    for frame in frames:
        if frame not in seen:
            seen.add(frame)
            output.append(frame)
    return output


def _write_filtered_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_fields = list(fieldnames)
    for row in rows:
        for key in row.keys():
            if key not in all_fields:
                all_fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=all_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in all_fields})


def _manifest_by_frame(path: Path) -> dict[int, dict[str, str]]:
    _, rows = _read_rows(path)
    by_frame: dict[int, dict[str, str]] = {}
    for row in rows:
        frame = _frame(row)
        if frame is not None and frame not in by_frame:
            by_frame[frame] = row
    return by_frame


def _resize_to_width(image, width: int):
    cv2 = _require_cv2()
    h, w = image.shape[:2]
    if w == width:
        return image
    scale = width / max(1, w)
    return cv2.resize(image, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def _resize_to_height(image, height: int):
    cv2 = _require_cv2()
    h, w = image.shape[:2]
    if h == height:
        return image
    scale = height / max(1, h)
    return cv2.resize(image, (max(1, int(round(w * scale))), height), interpolation=cv2.INTER_AREA)


def _put_text_lines(image, lines: Iterable[str], x: int, y: int, font_scale: float = 0.55) -> None:
    cv2 = _require_cv2()
    yy = y
    for line in lines:
        cv2.putText(image, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        yy += int(round(24 * font_scale / 0.55))


def _short_row_summary(prefix: str, row: dict[str, str], score: float) -> list[str]:
    return [
        f"{prefix}: accepted={row.get('filter_accepted','')} reason={row.get('filter_reason','')} score={score:.2f}",
        f"{prefix}: ref={row.get('matched_reference_flight','')} frame={row.get('matched_reference_frame_index','')} scale={row.get('selected_query_scale','')}",
        f"{prefix}: matches={row.get('orb_good_matches','')} inliers={row.get('homography_inliers','')} ratio={row.get('verification_inlier_ratio','')} ref_bbox={row.get('reference_inlier_bbox_area_frac','')}",
    ]


def _make_comparison_image(before_img_path: Path, after_img_path: Path, before_row: dict[str, str], after_row: dict[str, str], layout: str):
    cv2 = _require_cv2()
    import numpy as np

    before_img = cv2.imread(str(before_img_path), cv2.IMREAD_COLOR)
    after_img = cv2.imread(str(after_img_path), cv2.IMREAD_COLOR)
    if before_img is None:
        raise RuntimeError(f"Could not read BEFORE image: {before_img_path}")
    if after_img is None:
        raise RuntimeError(f"Could not read AFTER image: {after_img_path}")

    before_score = row_confidence(before_row)
    after_score = row_confidence(after_row)
    delta_score = after_score - before_score
    before_ok = _accepted(before_row)
    after_ok = _accepted(after_row)
    gps_delta = _distance_between_rows_m(before_row, after_row)
    frame = _frame(after_row) or _frame(before_row) or 0

    extra = f"frame={frame} delta_score={delta_score:+.2f} accepted {int(before_ok)}->{int(after_ok)}"
    if gps_delta is not None:
        extra += f" gps_delta={gps_delta:.1f}m"
    header_lines = [extra]
    header_lines.extend(_short_row_summary("BEFORE", before_row, before_score))
    header_lines.extend(_short_row_summary("AFTER", after_row, after_score))

    if layout == "horizontal":
        target_h = max(before_img.shape[0], after_img.shape[0])
        before_img = _resize_to_height(before_img, target_h)
        after_img = _resize_to_height(after_img, target_h)
        separator = np.full((target_h, 8, 3), 30, dtype=np.uint8)
        body = np.hstack([before_img, separator, after_img])
    else:
        target_w = max(before_img.shape[1], after_img.shape[1])
        before_img = _resize_to_width(before_img, target_w)
        after_img = _resize_to_width(after_img, target_w)
        separator = np.full((8, target_w, 3), 30, dtype=np.uint8)
        body = np.vstack([before_img, separator, after_img])

    header_h = 150
    header = np.zeros((header_h, body.shape[1], 3), dtype=np.uint8)
    _put_text_lines(header, header_lines, 12, 24, font_scale=0.55)
    return np.vstack([header, body])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create side-by-side/stacked BEFORE vs AFTER Stage 7 debug comparisons.")
    parser.add_argument("--before-csv", required=True, type=Path, help="Original/raw prediction CSV, e.g. Stage 6 top50.")
    parser.add_argument("--after-csv", required=True, type=Path, help="New rerun or merged prediction CSV.")
    parser.add_argument("--query-video", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--frame-list", type=Path, default=None, help="Optional frame list. Defaults to frames present in --after-csv.")
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--layout", choices=["vertical", "horizontal"], default="vertical")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary before/after debug folders.")
    parser.add_argument("--temp-dir", type=Path, default=None)

    # Most options mirror export_feature_match_debug.
    parser.add_argument("--display-height", type=int, default=540)
    parser.add_argument("--nfeatures", type=int, default=2500)
    parser.add_argument("--ratio-test", type=float, default=0.75)
    parser.add_argument("--max-lines", type=int, default=80)
    parser.add_argument("--show-outliers", action="store_true")
    parser.add_argument("--match-backend", choices=["orb", "lightglue"], default="lightglue")
    parser.add_argument("--query-scale-mode", choices=["selected", "original"], default="selected")
    parser.add_argument("--query-scale-fill", choices=["blur", "median", "gray", "black"], default="blur")
    parser.add_argument("--header-alpha", type=float, default=0.15)

    parser.add_argument("--mask-dynamic-objects", action="store_true")
    parser.add_argument("--mask-model", default="yolov8n-seg.pt")
    parser.add_argument("--mask-classes", default=DEFAULT_DYNAMIC_CLASSES)
    parser.add_argument("--mask-confidence", type=float, default=0.35)
    parser.add_argument("--mask-iou", type=float, default=0.7)
    parser.add_argument("--mask-dilate-px", type=int, default=4)
    parser.add_argument("--mask-fill", choices=["median", "gray", "black", "blur"], default="blur")
    parser.add_argument("--mask-imgsz", type=int, default=960)
    parser.add_argument("--mask-device", default=None)
    parser.add_argument("--mask-min-area-px", type=int, default=20)
    parser.add_argument("--mask-max-area-frac", type=float, default=0.015)
    parser.add_argument("--mask-max-width-frac", type=float, default=0.20)
    parser.add_argument("--mask-max-height-frac", type=float, default=0.20)
    parser.add_argument("--mask-source", choices=["box", "segmentation", "auto"], default="box")
    parser.add_argument("--mask-display", choices=["filled", "overlay", "original"], default="filled")
    args = parser.parse_args()

    before_header, before_rows = _read_rows(args.before_csv)
    after_header, after_rows = _read_rows(args.after_csv)
    before_by_frame: dict[int, dict[str, str]] = {}
    after_by_frame: dict[int, dict[str, str]] = {}
    for row in before_rows:
        frame = _frame(row)
        if frame is not None and frame not in before_by_frame:
            before_by_frame[frame] = row
    for row in after_rows:
        frame = _frame(row)
        if frame is not None and frame not in after_by_frame:
            after_by_frame[frame] = row

    if args.frame_list is not None:
        candidate_frames = _parse_frame_list(args.frame_list)
    else:
        candidate_frames = [_frame(row) for row in after_rows]
        candidate_frames = [frame for frame in candidate_frames if frame is not None]

    frames: list[int] = []
    seen: set[int] = set()
    for frame in candidate_frames:
        if frame in seen:
            continue
        if frame in before_by_frame and frame in after_by_frame:
            frames.append(frame)
            seen.add(frame)
        if len(frames) >= args.max_rows:
            break

    if not frames:
        raise RuntimeError("No shared query_frame_index values found between before/after CSVs.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    temp_root = args.temp_dir or (args.out_dir / "_tmp_before_after_debug")
    if temp_root.exists():
        shutil.rmtree(temp_root)
    before_dir = temp_root / "before"
    after_dir = temp_root / "after"
    before_csv = temp_root / "before_filtered.csv"
    after_csv = temp_root / "after_filtered.csv"

    _write_filtered_csv(before_csv, before_header, [before_by_frame[f] for f in frames])
    _write_filtered_csv(after_csv, after_header, [after_by_frame[f] for f in frames])

    common_kwargs = dict(
        query_video=args.query_video,
        project_root=args.project_root,
        max_rows=len(frames),
        accepted_only=False,
        rejected_only=False,
        every_n_rows=1,
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
    print(f"Rendering BEFORE debug images for {len(frames)} frames...")
    export_feature_match_debug(prediction_csv=before_csv, out_dir=before_dir, **common_kwargs)
    print(f"Rendering AFTER debug images for {len(frames)} frames...")
    export_feature_match_debug(prediction_csv=after_csv, out_dir=after_dir, **common_kwargs)

    before_manifest = _manifest_by_frame(before_dir / "feature_match_debug_manifest.csv")
    after_manifest = _manifest_by_frame(after_dir / "feature_match_debug_manifest.csv")

    cv2 = _require_cv2()
    manifest_rows: list[dict[str, str]] = []
    written = 0
    for frame in frames:
        bman = before_manifest.get(frame)
        aman = after_manifest.get(frame)
        if bman is None or aman is None:
            continue
        before_img = Path(bman["debug_image"])
        after_img = Path(aman["debug_image"])
        before_row = before_by_frame[frame]
        after_row = after_by_frame[frame]
        comp = _make_comparison_image(before_img, after_img, before_row, after_row, layout=args.layout)
        before_ok = _accepted(before_row)
        after_ok = _accepted(after_row)
        before_score = row_confidence(before_row)
        after_score = row_confidence(after_row)
        safe_status = "improved" if after_score > before_score else "not_improved"
        safe_accept = f"{int(before_ok)}to{int(after_ok)}"
        out_name = f"before_after_{written:04d}_query_{frame:06d}_{safe_accept}_{safe_status}.jpg"
        out_path = args.out_dir / out_name
        cv2.imwrite(str(out_path), comp, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        manifest_rows.append(
            {
                "query_frame_index": str(frame),
                "comparison_image": str(out_path),
                "before_debug_image": str(before_img),
                "after_debug_image": str(after_img),
                "before_accepted": "1" if before_ok else "0",
                "after_accepted": "1" if after_ok else "0",
                "before_reason": before_row.get("filter_reason", ""),
                "after_reason": after_row.get("filter_reason", ""),
                "before_score": f"{before_score:.6f}",
                "after_score": f"{after_score:.6f}",
                "delta_score": f"{after_score - before_score:.6f}",
                "before_ref": before_row.get("matched_reference_image", ""),
                "after_ref": after_row.get("matched_reference_image", ""),
                "before_scale": before_row.get("selected_query_scale", ""),
                "after_scale": after_row.get("selected_query_scale", ""),
            }
        )
        written += 1

    manifest_path = args.out_dir / "before_after_debug_manifest.csv"
    if manifest_rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    if not args.keep_temp and temp_root.exists():
        shutil.rmtree(temp_root)

    print(f"Wrote {written} BEFORE/AFTER comparison images to: {args.out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
