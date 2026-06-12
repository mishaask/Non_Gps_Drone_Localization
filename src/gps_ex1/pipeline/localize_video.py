"""Online-style visual localization of a query video against the reference index.

This script processes the query video sequentially. It does not use query GNSS
for prediction. If a query SRT is supplied, it is used only for evaluation.

Version 2 adds an optional temporal filter. The first baseline chose each frame
independently, which made the KML jump between visually similar but geographically
distant reference frames. The temporal mode keeps the solution physically
plausible while still processing frames in online order.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from gps_ex1.geometry.geo import haversine_m
from gps_ex1.io.srt_parser import nearest_record_by_time, parse_srt_file, valid_gps_records
from gps_ex1.io.timecode import seconds_to_timecode
from gps_ex1.features.global_descriptors import BasicDescriptorExtractor, create_descriptor_extractor
from gps_ex1.segmentation.dynamic_masks import DEFAULT_DYNAMIC_CLASSES, create_dynamic_object_masker
from gps_ex1.localization.retrieval import retrieve_candidates_from_descriptor
from gps_ex1.localization.verification import VerificationScore, score_candidate_alignment
from gps_ex1.localization.temporal_filter import (
    CandidateObservation,
    OnlineTrackState,
    TemporalFilterConfig,
    TemporalSelection,
    exponential_smooth_state,
    hold_state_at_time,
    select_temporal_candidate,
    select_visual_candidate,
    visual_quality_score,
)
from gps_ex1.preprocess.reference_index import load_reference_index


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install it with: pip install opencv-python") from exc
    return cv2




def _parse_query_scales(scales_text: str | None) -> tuple[float, ...]:
    """Parse comma-separated query scale factors while preserving order.

    A scale smaller than 1.0 shrinks the query image into a same-size canvas.
    This simulates the lower-altitude query occupying less of the descriptor
    frame, which is useful when matching against higher-altitude reference
    flights. Example: --query-scales 1.0,0.8,0.6,0.5
    """

    if scales_text is None or not scales_text.strip():
        return (1.0,)

    scales: list[float] = []
    seen: set[float] = set()
    for raw in scales_text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            scale = float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid query scale '{raw}'. Use comma-separated floats, e.g. 1.0,0.8,0.6,0.5") from exc
        if not (0.10 <= scale <= 1.0):
            raise ValueError("query scales must be in the range [0.10, 1.0]. This experiment only shrinks query frames.")
        key = round(scale, 4)
        if key not in seen:
            scales.append(scale)
            seen.add(key)

    if not scales:
        return (1.0,)
    return tuple(scales)




def _parse_query_frame_list(path: Path | None) -> tuple[int, ...] | None:
    """Parse a debug-friendly query-frame list.

    Accepted formats inside the text file:
    - one frame per line: 1800
    - comma/space separated frames: 1800, 4200 5400
    - inclusive ranges: 1800-2400
    - inclusive ranges with step: 1800-2400:150
    - comments after #: 1800  # circular building
    """

    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Query frame list not found: {path}")

    frames: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].replace(",", " ").strip()
        if not line:
            continue
        for token in line.split():
            if "-" in token:
                span, _, step_text = token.partition(":")
                start_text, end_text = span.split("-", 1)
                start = int(start_text)
                end = int(end_text)
                step = int(step_text) if step_text else 1
                if step <= 0:
                    raise ValueError(f"Invalid non-positive step in query frame token: {token}")
                if end < start:
                    raise ValueError(f"Frame range end before start in token: {token}")
                for frame_idx in range(start, end + 1, step):
                    frames.add(frame_idx)
            else:
                frames.add(int(token))

    if not frames:
        raise ValueError(f"Query frame list is empty: {path}")
    return tuple(sorted(frames))


def _fmt_optional_float(value: float | None, precision: int = 6) -> str:
    if value is None:
        return ""
    return f"{float(value):.{precision}f}"

def _scaled_query_canvas(image_bgr: np.ndarray, scale: float, fill: str = "blur") -> np.ndarray:
    """Shrink a query image into a same-size canvas for scale-aware retrieval.

    The descriptor extractor always resizes its input to a fixed DINOv2 size.
    Therefore, simply resizing the query image would not change the effective
    object scale. Instead, we shrink the image content and paste it into a
    same-size canvas. This makes buildings/trees appear smaller to the global
    descriptor, approximating a higher-altitude footprint.
    """

    scale = float(scale)
    if abs(scale - 1.0) < 1e-6:
        return image_bgr

    cv2 = _require_cv2()
    h, w = image_bgr.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    fill_mode = fill.strip().lower()
    if fill_mode == "gray":
        canvas = np.full_like(image_bgr, 127)
    elif fill_mode == "black":
        canvas = np.zeros_like(image_bgr)
    elif fill_mode == "median":
        median_color = np.median(image_bgr.reshape(-1, image_bgr.shape[2]), axis=0).astype(np.uint8)
        canvas = np.empty_like(image_bgr)
        canvas[:, :] = median_color
    elif fill_mode == "blur":
        # Keep only very low-frequency context outside the shrunken view. This
        # avoids a harsh artificial border while still suppressing tiny details.
        kernel = max(31, (min(h, w) // 12) | 1)
        canvas = cv2.GaussianBlur(image_bgr, (kernel, kernel), 0)
    else:
        raise ValueError("query scale fill must be one of: blur, median, gray, black")

    x0 = (w - new_w) // 2
    y0 = (h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _retrieve_multiscale_candidates(
    frame_for_descriptor: np.ndarray,
    descriptor_extractor,
    reference_index,
    top_k: int,
    query_scales: tuple[float, ...],
    query_scale_fill: str,
):
    """Retrieve candidates from several shrunken query versions and merge them."""

    best_by_ref: dict[int, object] = {}
    for scale in query_scales:
        scaled = _scaled_query_canvas(frame_for_descriptor, scale, query_scale_fill)
        query_descriptor = descriptor_extractor.describe_bgr(scaled)
        candidates = retrieve_candidates_from_descriptor(query_descriptor, reference_index, top_k=top_k, query_scale=scale)
        for candidate in candidates:
            previous = best_by_ref.get(candidate.index)
            if previous is None or candidate.similarity > previous.similarity:
                best_by_ref[candidate.index] = candidate

    merged = sorted(best_by_ref.values(), key=lambda c: c.similarity, reverse=True)
    return merged[:top_k]

def _prediction_coordinate(index, candidate_idx: int, target: str) -> tuple[float, float]:
    if target == "center":
        return float(index.center_lats[candidate_idx]), float(index.center_lons[candidate_idx])
    if target == "drone":
        return float(index.drone_lats[candidate_idx]), float(index.drone_lons[candidate_idx])
    raise ValueError(f"Unsupported prediction target: {target}")


def _write_header(writer: csv.writer) -> None:
    writer.writerow(
        [
            "query_frame_index",
            "query_time_s",
            "query_timecode",
            "prediction_target",
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
            "query_inlier_bbox_area_frac",
            "query_inlier_bbox_width_frac",
            "query_inlier_bbox_height_frac",
            "reference_inlier_bbox_area_frac",
            "reference_inlier_bbox_width_frac",
            "reference_inlier_bbox_height_frac",
            "temporal_filter_enabled",
            "filter_accepted",
            "filter_reason",
            "filter_visual_score",
            "filter_temporal_distance_m",
            "filter_fused_score",
            "truth_latitude_eval_only",
            "truth_longitude_eval_only",
            "error_m_eval_only",
            "processing_time_ms",
        ]
    )


def _empty_row(frame_index: int, query_time_s: float, target: str, processing_time_ms: float, reason: str) -> list[object]:
    return [
        frame_index,
        f"{query_time_s:.3f}",
        seconds_to_timecode(query_time_s),
        target,
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        0,
        0,
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "0",
        "0",
        reason,
        "",
        "",
        "",
        "",
        "",
        "",
        f"{processing_time_ms:.2f}",
    ]


def _candidate_from_index(index, candidate_idx: int, retrieval, score: VerificationScore, target: str) -> CandidateObservation:
    lat, lon = _prediction_coordinate(index, candidate_idx, target)
    return CandidateObservation(
        index=candidate_idx,
        latitude=lat,
        longitude=lon,
        matched_flight=str(index.flight_ids[candidate_idx]),
        matched_frame_index=int(index.frame_indices[candidate_idx]),
        matched_time_s=float(index.video_times_s[candidate_idx]),
        retrieval_distance=float(retrieval.distance),
        retrieval_similarity=float(retrieval.similarity),
        query_scale=float(getattr(retrieval, "query_scale", 1.0)),
        good_matches=int(score.good_matches),
        homography_inliers=int(score.homography_inliers),
        mean_match_distance=float(score.mean_match_distance),
        homography_found=bool(score.homography_found),
        inlier_ratio=float(score.inlier_ratio),
        projected_center_x=score.projected_center_x,
        projected_center_y=score.projected_center_y,
        projected_center_inside=bool(score.projected_center_inside),
        projected_quad_area_frac=score.projected_quad_area_frac,
        projected_quad_valid=bool(score.projected_quad_valid),
        geometry_reason=str(score.geometry_reason),
        query_inlier_bbox_area_frac=score.query_inlier_bbox_area_frac,
        query_inlier_bbox_width_frac=score.query_inlier_bbox_width_frac,
        query_inlier_bbox_height_frac=score.query_inlier_bbox_height_frac,
        reference_inlier_bbox_area_frac=score.reference_inlier_bbox_area_frac,
        reference_inlier_bbox_width_frac=score.reference_inlier_bbox_width_frac,
        reference_inlier_bbox_height_frac=score.reference_inlier_bbox_height_frac,
    )




def localize_video(
    video_path: Path,
    reference_index_path: Path,
    output_csv: Path,
    every_n_frames: int,
    top_k: int,
    prediction_target: str,
    srt_path: Path | None = None,
    max_frames: int | None = None,
    query_frame_list: tuple[int, ...] | None = None,
    temporal_filter_enabled: bool = False,
    temporal_config: TemporalFilterConfig | None = None,
    descriptor_backend: str = "basic",
    descriptor_model: str = "dinov2_vits14",
    descriptor_device: str | None = None,
    descriptor_image_size: int = 518,
    query_scales: tuple[float, ...] = (1.0,),
    query_scale_fill: str = "blur",
    verification_backend: str = "orb",
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
) -> None:
    cv2 = _require_cv2()
    index = load_reference_index(reference_index_path)
    temporal_config = temporal_config or TemporalFilterConfig()
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
    descriptor_extractor = create_descriptor_extractor(
        descriptor_backend,
        model_name=descriptor_model,
        device=descriptor_device,
        image_size=descriptor_image_size,
    )
    print(f"Descriptor backend: {descriptor_extractor.name}")
    print(f"Query scales: {','.join(f'{scale:g}' for scale in query_scales)} (fill={query_scale_fill})")
    query_frame_set = set(query_frame_list) if query_frame_list is not None else None
    query_frame_max = max(query_frame_set) if query_frame_set else None
    if query_frame_set is not None:
        preview = ','.join(str(v) for v in sorted(query_frame_set)[:12])
        suffix = '...' if len(query_frame_set) > 12 else ''
        print(f"Manual query-frame list enabled: {len(query_frame_set)} frames [{preview}{suffix}]")
    if len(query_scales) > 1:
        print("Multi-scale retrieval enabled: each sampled query frame is searched at multiple shrunken scales and merged before verification.")
    if masker is not None:
        print(f"Dynamic-object masking enabled for query frames: model={mask_model}, classes={mask_classes}, source={mask_source}, max_area_frac={mask_max_area_frac}")
    print(f"Verification backend: {verification_backend}")

    eval_records = None
    if srt_path is not None:
        eval_records = valid_gps_records(parse_srt_file(srt_path))
        print("Loaded query SRT for evaluation only. It will not be used for prediction.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open query video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    accepted = 0
    held_or_rejected = 0
    errors = []
    state: OnlineTrackState | None = None

    with output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        _write_header(writer)

        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and processed >= max_frames:
                break

            if query_frame_set is not None:
                if query_frame_max is not None and frame_index > query_frame_max:
                    break
                if frame_index not in query_frame_set:
                    frame_index += 1
                    continue
            elif frame_index % every_n_frames != 0:
                frame_index += 1
                continue

            start = time.perf_counter()
            query_time_s = frame_index / fps
            frame_for_matching = masker.mask_bgr(frame) if masker is not None else frame
            retrieval_candidates = _retrieve_multiscale_candidates(
                frame_for_descriptor=frame_for_matching,
                descriptor_extractor=descriptor_extractor,
                reference_index=index,
                top_k=top_k,
                query_scales=query_scales,
                query_scale_fill=query_scale_fill,
            )

            observations: list[CandidateObservation] = []
            image_by_index: dict[int, str] = {}
            verification_frame_by_scale: dict[float, np.ndarray] = {}
            for candidate in retrieval_candidates:
                scale = float(getattr(candidate, "query_scale", 1.0))
                if scale not in verification_frame_by_scale:
                    verification_frame_by_scale[scale] = _scaled_query_canvas(frame, scale, query_scale_fill)
                score = score_candidate_alignment(
                    verification_frame_by_scale[scale],
                    str(index.image_paths[candidate.index]),
                    backend=verification_backend,
                    dynamic_masker=masker,
                )
                obs = _candidate_from_index(index, candidate.index, candidate, score, prediction_target)
                observations.append(obs)
                image_by_index[candidate.index] = str(index.image_paths[candidate.index])

            if temporal_filter_enabled:
                selection = select_temporal_candidate(observations, state, query_time_s, temporal_config)
            else:
                selection = select_visual_candidate(observations, temporal_config)

            if selection.candidate is None:
                processing_time_ms = (time.perf_counter() - start) * 1000.0
                writer.writerow(_empty_row(frame_index, query_time_s, prediction_target, processing_time_ms, selection.reason))
                processed += 1
                frame_index += 1
                continue

            raw_lat = selection.candidate.latitude
            raw_lon = selection.candidate.longitude

            if temporal_filter_enabled:
                if selection.accepted:
                    state = exponential_smooth_state(state, selection.candidate, query_time_s, temporal_config)
                    pred_lat = state.latitude
                    pred_lon = state.longitude
                    accepted += 1
                elif state is not None:
                    # Online fallback: keep the last physically plausible point
                    # instead of drawing a huge jump in the final KML.
                    pred_lat = state.latitude
                    pred_lon = state.longitude
                    state = hold_state_at_time(state, query_time_s)
                    held_or_rejected += 1
                else:
                    pred_lat = raw_lat
                    pred_lon = raw_lon
                    held_or_rejected += 1
            else:
                pred_lat = raw_lat
                pred_lon = raw_lon
                if selection.accepted:
                    accepted += 1
                else:
                    held_or_rejected += 1

            truth_lat = ""
            truth_lon = ""
            error_m = ""
            if eval_records is not None:
                truth = nearest_record_by_time(eval_records, query_time_s)
                if truth is not None and truth.latitude is not None and truth.longitude is not None:
                    truth_lat = f"{truth.latitude:.8f}"
                    truth_lon = f"{truth.longitude:.8f}"
                    # Note: if prediction_target='center', this compares center prediction to drone GNSS.
                    # It is useful as a diagnostic, but not a perfect ground truth for the center point.
                    err = haversine_m(pred_lat, pred_lon, truth.latitude, truth.longitude)
                    error_m = f"{err:.3f}"
                    errors.append(err)

            processing_time_ms = (time.perf_counter() - start) * 1000.0
            selected_idx = selection.candidate.index
            writer.writerow(
                [
                    frame_index,
                    f"{query_time_s:.3f}",
                    seconds_to_timecode(query_time_s),
                    prediction_target,
                    f"{pred_lat:.8f}",
                    f"{pred_lon:.8f}",
                    f"{raw_lat:.8f}",
                    f"{raw_lon:.8f}",
                    image_by_index.get(selected_idx, str(index.image_paths[selected_idx])),
                    selection.candidate.matched_flight,
                    selection.candidate.matched_frame_index,
                    f"{selection.candidate.matched_time_s:.3f}",
                    seconds_to_timecode(selection.candidate.matched_time_s),
                    f"{selection.candidate.retrieval_distance:.6f}",
                    f"{selection.candidate.retrieval_similarity:.6f}",
                    f"{selection.candidate.query_scale:.4f}",
                    selection.candidate.good_matches,
                    selection.candidate.homography_inliers,
                    "" if not np.isfinite(selection.candidate.mean_match_distance) else f"{selection.candidate.mean_match_distance:.3f}",
                    f"{selection.candidate.inlier_ratio:.6f}",
                    "" if selection.candidate.projected_center_x is None else f"{selection.candidate.projected_center_x:.3f}",
                    "" if selection.candidate.projected_center_y is None else f"{selection.candidate.projected_center_y:.3f}",
                    "1" if selection.candidate.projected_center_inside else "0",
                    "" if selection.candidate.projected_quad_area_frac is None else f"{selection.candidate.projected_quad_area_frac:.6f}",
                    "1" if selection.candidate.projected_quad_valid else "0",
                    selection.candidate.geometry_reason,
                    _fmt_optional_float(selection.candidate.query_inlier_bbox_area_frac),
                    _fmt_optional_float(selection.candidate.query_inlier_bbox_width_frac),
                    _fmt_optional_float(selection.candidate.query_inlier_bbox_height_frac),
                    _fmt_optional_float(selection.candidate.reference_inlier_bbox_area_frac),
                    _fmt_optional_float(selection.candidate.reference_inlier_bbox_width_frac),
                    _fmt_optional_float(selection.candidate.reference_inlier_bbox_height_frac),
                    "1" if temporal_filter_enabled else "0",
                    "1" if selection.accepted else "0",
                    selection.reason,
                    f"{selection.visual_score:.6f}",
                    "" if selection.temporal_distance_m is None else f"{selection.temporal_distance_m:.3f}",
                    f"{selection.fused_score:.6f}",
                    truth_lat,
                    truth_lon,
                    error_m,
                    f"{processing_time_ms:.2f}",
                ]
            )
            processed += 1
            if processed % 10 == 0:
                print(
                    f"Processed query frames: {processed}, latest match: "
                    f"{selection.candidate.matched_flight} frame {selection.candidate.matched_frame_index}, "
                    f"reason={selection.reason}"
                )

            frame_index += 1

    cap.release()
    print(f"Wrote predictions: {output_csv}")
    print(f"Processed sampled query frames: {processed}")
    print(f"Frames with accepted/new visual prediction: {accepted}")
    if temporal_filter_enabled:
        print(f"Frames held/rejected by temporal filter: {held_or_rejected}")
    if errors:
        print(f"Eval-only mean error against query SRT drone path: {float(np.mean(errors)):.2f} m")
        print(f"Eval-only median error against query SRT drone path: {float(np.median(errors)):.2f} m")


def compute_global_descriptor_from_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """Same descriptor as reference_index.compute_global_descriptor, but for an in-memory frame."""

    cv2 = _require_cv2()
    return BasicDescriptorExtractor().describe_bgr(image_bgr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Localize a query video against the offline reference index.")
    parser.add_argument("--video", required=True, type=Path, help="Query/test video. GNSS is not used for prediction.")
    parser.add_argument("--reference-index", required=True, type=Path, help="Reference index created by build_reference_index")
    parser.add_argument("--out", required=True, type=Path, help="Output predictions CSV")
    parser.add_argument("--srt", type=Path, default=None, help="Optional query SRT for evaluation only")
    parser.add_argument("--every-n-frames", type=int, default=30, help="Sequential sampling stride")
    parser.add_argument("--top-k", type=int, default=10, help="Reference candidates to rerank with ORB")
    parser.add_argument("--prediction-target", choices=["center", "drone"], default="center", help="Coordinate stored from matched reference frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional quick debug limit on sampled frames")
    parser.add_argument("--query-frame-list", type=Path, default=None, help="Optional text file of exact query frame indices to process. Overrides --every-n-frames for controlled debugging.")
    parser.add_argument("--temporal-filter", action="store_true", help="Use visual confidence + physical continuity to avoid impossible jumps")
    parser.add_argument("--min-inliers", type=int, default=12, help="Minimum RANSAC homography inliers required for accepting a candidate")
    parser.add_argument("--min-good-matches", type=int, default=12, help="Minimum ORB/LightGlue good matches required for accepting a candidate")
    parser.add_argument("--min-inlier-ratio", type=float, default=0.25, help="Minimum RANSAC inlier ratio required for accepting a candidate")
    parser.add_argument("--max-reference-inlier-area-frac", type=float, default=1.0, help="Reject candidate if RANSAC inliers cover more than this fraction of the reference image bbox area. 1.0 disables this cluster gate.")
    parser.add_argument("--max-reference-inlier-width-frac", type=float, default=1.0, help="Reject candidate if reference inlier bbox is wider than this image fraction. 1.0 disables this cluster gate.")
    parser.add_argument("--max-reference-inlier-height-frac", type=float, default=1.0, help="Reject candidate if reference inlier bbox is taller than this image fraction. 1.0 disables this cluster gate.")
    parser.add_argument("--max-query-inlier-area-frac", type=float, default=1.0, help="Reject candidate if query inliers are too scattered in the scaled-query canvas. 1.0 disables this cluster gate.")
    parser.add_argument("--allow-center-outside", action="store_true", help="Debug option: allow homographies whose projected query center falls outside the reference image")
    parser.add_argument("--allow-bad-homography-geometry", action="store_true", help="Debug option: allow very small/large/skewed homography polygons")
    parser.add_argument("--max-speed-mps", type=float, default=18.0, help="Approximate drone speed limit for continuity gating")
    parser.add_argument("--base-gate-m", type=float, default=55.0, help="Minimum allowed point-to-point movement gate")
    parser.add_argument("--hard-jump-m", type=float, default=180.0, help="Reject jumps larger than this unless visual evidence is very strong")
    parser.add_argument("--ema-alpha", type=float, default=0.35, help="Coordinate smoothing factor when temporal filter is enabled")
    parser.add_argument(
        "--descriptor-backend",
        choices=["basic", "dinov2", "anyloc", "anyloc-gem"],
        default="basic",
        help="Global descriptor backend. Must match the backend used to build the reference index.",
    )
    parser.add_argument("--descriptor-model", default="dinov2_vits14", help="DINOv2 torch.hub model name used by dinov2/anyloc-gem backends.")
    parser.add_argument("--descriptor-device", default=None, help="Optional descriptor device, e.g. cpu, cuda, or 0. Defaults to cuda if available.")
    parser.add_argument("--descriptor-image-size", type=int, default=518, help="Input size for DINOv2/AnyLoc descriptors. Must match the index; rounded to multiple of 14.")
    parser.add_argument("--query-scales", default="1.0", help="Comma-separated query shrink scales for multi-scale retrieval, e.g. 1.0,0.8,0.6,0.5")
    parser.add_argument("--query-scale-fill", choices=["blur", "median", "gray", "black"], default="blur", help="Canvas fill used outside the shrunken query image.")
    parser.add_argument(
        "--verification-backend",
        choices=["orb", "lightglue"],
        default="orb",
        help="Candidate verification backend. ORB is lightweight; LightGlue is optional and stronger.",
    )
    parser.add_argument("--mask-dynamic-objects", action="store_true", help="Use YOLO to suppress people/vehicles in query frames before retrieval and verification.")
    parser.add_argument("--mask-model", default="yolov8n-seg.pt", help="Ultralytics model name/path for dynamic-object masking.")
    parser.add_argument("--mask-classes", default=DEFAULT_DYNAMIC_CLASSES, help="Comma-separated COCO class names/IDs to mask.")
    parser.add_argument("--mask-confidence", type=float, default=0.35, help="YOLO confidence threshold for dynamic-object masking.")
    parser.add_argument("--mask-iou", type=float, default=0.7, help="YOLO NMS IoU threshold for dynamic-object masking.")
    parser.add_argument("--mask-dilate-px", type=int, default=4, help="Dilate dynamic object masks by this many pixels.")
    parser.add_argument("--mask-fill", choices=["median", "gray", "black", "blur"], default="blur", help="How to fill masked dynamic-object regions for descriptors/debug display.")
    parser.add_argument("--mask-imgsz", type=int, default=960, help="YOLO inference image size. Larger values catch small aerial cars better.")
    parser.add_argument("--mask-device", default=None, help="Optional Ultralytics device, e.g. cpu or 0.")
    parser.add_argument("--mask-min-area-px", type=int, default=20, help="Reject detections smaller than this many pixels.")
    parser.add_argument("--mask-max-area-frac", type=float, default=0.015, help="Reject detections covering more than this fraction of the image; avoids rooftop false positives.")
    parser.add_argument("--mask-max-width-frac", type=float, default=0.20, help="Reject detections wider than this fraction of the image.")
    parser.add_argument("--mask-max-height-frac", type=float, default=0.20, help="Reject detections taller than this fraction of the image.")
    parser.add_argument("--mask-source", choices=["box", "segmentation", "auto"], default="box", help="Use YOLO boxes, segmentation masks, or auto fallback. Box is safest for aerial cars.")
    args = parser.parse_args()

    temporal_config = TemporalFilterConfig(
        min_inliers=args.min_inliers,
        min_good_matches=args.min_good_matches,
        min_inlier_ratio=args.min_inlier_ratio,
        require_center_inside=not args.allow_center_outside,
        require_valid_homography_quad=not args.allow_bad_homography_geometry,
        max_reference_inlier_bbox_area_frac=args.max_reference_inlier_area_frac,
        max_reference_inlier_bbox_width_frac=args.max_reference_inlier_width_frac,
        max_reference_inlier_bbox_height_frac=args.max_reference_inlier_height_frac,
        max_query_inlier_bbox_area_frac=args.max_query_inlier_area_frac,
        max_speed_mps=args.max_speed_mps,
        base_gate_m=args.base_gate_m,
        hard_jump_m=args.hard_jump_m,
        ema_alpha=args.ema_alpha,
    )

    query_scales = _parse_query_scales(args.query_scales)
    query_frame_list = _parse_query_frame_list(args.query_frame_list)

    localize_video(
        video_path=args.video,
        reference_index_path=args.reference_index,
        output_csv=args.out,
        srt_path=args.srt,
        every_n_frames=args.every_n_frames,
        top_k=args.top_k,
        prediction_target=args.prediction_target,
        max_frames=args.max_frames,
        query_frame_list=query_frame_list,
        temporal_filter_enabled=args.temporal_filter,
        temporal_config=temporal_config,
        descriptor_backend=args.descriptor_backend,
        descriptor_model=args.descriptor_model,
        descriptor_device=args.descriptor_device,
        descriptor_image_size=args.descriptor_image_size,
        query_scales=query_scales,
        query_scale_fill=args.query_scale_fill,
        verification_backend=args.verification_backend,
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
    )


if __name__ == "__main__":
    main()
