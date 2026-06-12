"""Audit prediction CSV matches using simple scene compatibility checks.

This tool is meant for the exact situation where KML paths look plausible after
smoothing but side-by-side debug images show that many accepted matches are
visually wrong. It does not use GNSS. It only asks: "does the query image look
compatible with the reference image that was selected?"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gps_ex1.io.timecode import seconds_to_timecode
from gps_ex1.quality.scene_signature import (
    compute_scene_signature_bgr,
    compute_scene_signature_path,
    scene_compatibility_reason,
)


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install it with: pip install opencv-python") from exc
    return cv2


def _read_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        text = (row.get(key) or "").strip()
        return default if not text else float(text)
    except ValueError:
        return default


def _read_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        text = (row.get(key) or "").strip()
        return default if not text else int(float(text))
    except ValueError:
        return default


def _grab_video_frame(cap, frame_index: int):
    cv2 = _require_cv2()
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def audit_match_quality(
    prediction_csv: Path,
    query_video: Path,
    out_csv: Path,
    min_inliers: int,
    min_good_matches: int,
    max_scene_distance: float,
    max_top_sky_delta: float,
) -> None:
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(query_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open query video: {query_video}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, object] = {}

    with prediction_csv.open("r", newline="", encoding="utf-8") as in_fp, out_csv.open("w", newline="", encoding="utf-8") as out_fp:
        reader = csv.DictReader(in_fp)
        extra_fields = [
            "qa_pass",
            "qa_reason",
            "qa_scene_distance",
            "query_top_sky_ratio",
            "reference_top_sky_ratio",
            "query_low_altitude_proxy",
            "reference_low_altitude_proxy",
            "qa_min_inliers",
            "qa_min_good_matches",
        ]
        fieldnames = list(reader.fieldnames or []) + [name for name in extra_fields if name not in (reader.fieldnames or [])]
        writer = csv.DictWriter(out_fp, fieldnames=fieldnames)
        writer.writeheader()

        total = kept = rejected = missing = 0
        for row in reader:
            total += 1
            frame_index = _read_int(row, "query_frame_index", -1)
            ref_path = (row.get("matched_reference_image") or "").strip()
            inliers = _read_int(row, "homography_inliers", 0)
            good_matches = _read_int(row, "orb_good_matches", 0)

            qa_pass = True
            reason = "qa_pass"
            scene_dist = 999.0
            query_sig = None
            ref_sig = None

            if frame_index < 0 or not ref_path:
                qa_pass = False
                reason = "missing_match_fields"
                missing += 1
            elif inliers < min_inliers:
                qa_pass = False
                reason = "too_few_inliers"
            elif good_matches < min_good_matches:
                qa_pass = False
                reason = "too_few_good_matches"
            else:
                frame = _grab_video_frame(cap, frame_index)
                if frame is None:
                    qa_pass = False
                    reason = "missing_query_frame"
                    missing += 1
                else:
                    query_sig = compute_scene_signature_bgr(frame)
                    if ref_path not in cache:
                        try:
                            cache[ref_path] = compute_scene_signature_path(ref_path)
                        except RuntimeError:
                            cache[ref_path] = None
                    ref_sig = cache[ref_path]
                    if ref_sig is None:
                        qa_pass = False
                        reason = "missing_reference_image"
                        missing += 1
                    else:
                        compatible, reason, scene_dist = scene_compatibility_reason(
                            query_sig,
                            ref_sig,  # type: ignore[arg-type]
                            max_scene_distance=max_scene_distance,
                            max_top_sky_delta=max_top_sky_delta,
                        )
                        qa_pass = compatible

            if qa_pass:
                kept += 1
            else:
                rejected += 1

            row["qa_pass"] = "1" if qa_pass else "0"
            row["qa_reason"] = reason
            row["qa_scene_distance"] = "" if scene_dist == 999.0 else f"{scene_dist:.6f}"
            row["query_top_sky_ratio"] = "" if query_sig is None else f"{query_sig.top_sky_ratio:.6f}"
            row["reference_top_sky_ratio"] = "" if ref_sig is None else f"{ref_sig.top_sky_ratio:.6f}"
            row["query_low_altitude_proxy"] = "" if query_sig is None else f"{query_sig.low_altitude_proxy:.6f}"
            row["reference_low_altitude_proxy"] = "" if ref_sig is None else f"{ref_sig.low_altitude_proxy:.6f}"
            row["qa_min_inliers"] = str(min_inliers)
            row["qa_min_good_matches"] = str(min_good_matches)
            writer.writerow(row)

    cap.release()
    print(f"Wrote QA-audited CSV: {out_csv}")
    print(f"Rows: {total} | kept: {kept} | rejected: {rejected} | missing assets: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit visual match quality without GNSS.")
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--query-video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-inliers", type=int, default=18)
    parser.add_argument("--min-good-matches", type=int, default=25)
    parser.add_argument("--max-scene-distance", type=float, default=0.22)
    parser.add_argument("--max-top-sky-delta", type=float, default=0.22)
    args = parser.parse_args()

    audit_match_quality(
        prediction_csv=args.prediction_csv,
        query_video=args.query_video,
        out_csv=args.out,
        min_inliers=args.min_inliers,
        min_good_matches=args.min_good_matches,
        max_scene_distance=args.max_scene_distance,
        max_top_sky_delta=args.max_top_sky_delta,
    )


if __name__ == "__main__":
    main()
