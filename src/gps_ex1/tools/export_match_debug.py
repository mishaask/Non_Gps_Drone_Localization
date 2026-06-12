"""Create side-by-side debug images for query frames and matched reference frames.

This lets you manually check whether a prediction is visually logical:
left  = frame from the query/test video
right = matched reference keyframe chosen by the localizer
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gps_ex1.io.timecode import seconds_to_timecode


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install it with: pip install opencv-python") from exc
    return cv2


def _resolve_path(text: str, project_root: Path) -> Path:
    raw = Path(text)
    candidates = [raw, Path(str(raw).replace("\\", "/")), project_root / raw, project_root / Path(str(raw).replace("\\", "/"))]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _put_text_lines(image, lines: list[str], origin=(12, 24)):
    cv2 = _require_cv2()
    x, y = origin
    for line in lines:
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


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
    if h == height:
        return image
    scale = height / max(h, 1)
    return cv2.resize(image, (max(1, int(round(w * scale))), height), interpolation=cv2.INTER_AREA)


def export_match_debug(
    prediction_csv: Path,
    query_video: Path,
    out_dir: Path,
    project_root: Path,
    max_rows: int = 40,
    accepted_only: bool = False,
    every_n_rows: int = 1,
) -> int:
    cv2 = _require_cv2()
    out_dir.mkdir(parents=True, exist_ok=True)

    with prediction_csv.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)

    cap = cv2.VideoCapture(str(query_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open query video: {query_video}")

    written = 0
    manifest_rows: list[dict[str, str]] = []

    for row_i, row in enumerate(rows):
        if row_i % max(1, every_n_rows) != 0:
            continue
        if accepted_only and (row.get("filter_accepted") or "").strip() not in {"1", "true", "True"}:
            continue
        if written >= max_rows:
            break

        frame_text = (row.get("query_frame_index") or "").strip()
        ref_text = (row.get("matched_reference_image") or "").strip()
        if not frame_text or not ref_text:
            continue

        frame_index = int(float(frame_text))
        query_frame = _read_query_frame(cap, frame_index)
        if query_frame is None:
            continue

        ref_path = _resolve_path(ref_text, project_root)
        reference_frame = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        if reference_frame is None:
            continue

        height = 540
        left = _resize_to_height(query_frame, height)
        right = _resize_to_height(reference_frame, height)

        _put_text_lines(
            left,
            [
                f"QUERY {query_video.name}",
                f"frame {frame_index}",
                f"t={row.get('query_timecode') or seconds_to_timecode(row.get('query_time_s'))}",
            ],
        )
        _put_text_lines(
            right,
            [
                f"MATCH {row.get('matched_reference_flight','')}",
                f"frame {row.get('matched_reference_frame_index','')}",
                f"t={row.get('matched_reference_timecode') or seconds_to_timecode(row.get('matched_reference_time_s'))}",
                f"inliers={row.get('homography_inliers','')} good={row.get('orb_good_matches','')}",
                f"reason={row.get('filter_reason','')}",
            ],
        )

        combined = cv2.hconcat([left, right]) if left.shape[0] == right.shape[0] else None
        if combined is None:
            continue

        out_name = f"match_{written:04d}_query_{frame_index:06d}_ref_{row.get('matched_reference_flight','ref')}_{row.get('matched_reference_frame_index','')}.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), combined)
        manifest_rows.append(
            {
                "debug_image": str(out_path),
                "query_frame_index": str(frame_index),
                "query_timecode": row.get("query_timecode") or seconds_to_timecode(row.get("query_time_s")),
                "matched_reference_image": str(ref_path),
                "matched_reference_flight": row.get("matched_reference_flight", ""),
                "matched_reference_frame_index": row.get("matched_reference_frame_index", ""),
                "matched_reference_timecode": row.get("matched_reference_timecode") or seconds_to_timecode(row.get("matched_reference_time_s")),
                "filter_reason": row.get("filter_reason", ""),
                "homography_inliers": row.get("homography_inliers", ""),
                "orb_good_matches": row.get("orb_good_matches", ""),
            }
        )
        written += 1

    cap.release()

    manifest_path = out_dir / "match_debug_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        fieldnames = [
            "debug_image",
            "query_frame_index",
            "query_timecode",
            "matched_reference_image",
            "matched_reference_flight",
            "matched_reference_frame_index",
            "matched_reference_timecode",
            "filter_reason",
            "homography_inliers",
            "orb_good_matches",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export side-by-side query/reference match debug images.")
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--query-video", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root used to resolve relative reference image paths.")
    parser.add_argument("--max-rows", type=int, default=40)
    parser.add_argument("--accepted-only", action="store_true")
    parser.add_argument("--every-n-rows", type=int, default=1)
    args = parser.parse_args()

    written = export_match_debug(
        prediction_csv=args.prediction_csv,
        query_video=args.query_video,
        out_dir=args.out_dir,
        project_root=args.project_root,
        max_rows=args.max_rows,
        accepted_only=args.accepted_only,
        every_n_rows=args.every_n_rows,
    )
    print(f"Wrote {written} debug match images to: {args.out_dir}")
    print(f"Manifest: {args.out_dir / 'match_debug_manifest.csv'}")


if __name__ == "__main__":
    main()
