"""Export YOLO dynamic-object mask debug images with labels/confidence.

This tool is meant to answer questions like: "why did the masker remove a roof
but not a car?" Each output image contains:

    original | detector overlay with labels | masked image used for descriptors

Accepted detections are green. Rejected detections are red with a reason such as
``too_large_area``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from gps_ex1.segmentation.dynamic_masks import DEFAULT_DYNAMIC_CLASSES, create_dynamic_object_masker


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV is required. Install it with: python -m pip install opencv-python") from exc
    return cv2


def _resize_to_height(image, height: int):
    cv2 = _require_cv2()
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Invalid image dimensions")
    scale = height / h
    return cv2.resize(image, (max(1, int(round(w * scale))), height), interpolation=cv2.INTER_AREA)


def _put_title(image, title: str) -> None:
    cv2 = _require_cv2()
    cv2.rectangle(image, (0, 0), (image.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(image, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)


def _compose(original, overlay, masked, height: int):
    original = _resize_to_height(original, height)
    overlay = _resize_to_height(overlay, height)
    masked = _resize_to_height(masked, height)
    _put_title(original, "original")
    _put_title(overlay, "YOLO overlay: green=accepted, red=rejected")
    _put_title(masked, "masked image used for descriptor/debug display")
    return np.hstack([original, overlay, masked])


def _iter_video_frames(video_path: Path, every_n_frames: int, max_frames: int | None):
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_index = 0
    yielded = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % every_n_frames == 0:
                yield f"frame_{frame_index:06d}", frame, frame_index
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            frame_index += 1
    finally:
        cap.release()


def _iter_image_files(frames_dir: Path, max_frames: int | None):
    cv2 = _require_cv2()
    files = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(sorted(frames_dir.glob(pattern)))
    for i, path in enumerate(files):
        if max_frames is not None and i >= max_frames:
            break
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            yield path.stem, image, i


def export_mask_debug(
    out_dir: Path,
    video: Path | None = None,
    frames_dir: Path | None = None,
    every_n_frames: int = 30,
    max_frames: int | None = 40,
    display_height: int = 360,
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
) -> int:
    cv2 = _require_cv2()
    if (video is None) == (frames_dir is None):
        raise ValueError("Provide exactly one of --video or --frames-dir")

    out_dir.mkdir(parents=True, exist_ok=True)
    masker = create_dynamic_object_masker(
        enabled=True,
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

    iterator = _iter_video_frames(video, every_n_frames, max_frames) if video is not None else _iter_image_files(frames_dir, max_frames)
    manifest: list[dict[str, object]] = []
    written = 0
    for label, image, frame_index in iterator:
        cache_key = str(video or frames_dir) + ":" + str(frame_index)
        overlay = masker.debug_overlay_bgr(image, cache_key=cache_key, show_rejected=True)
        masked = masker.mask_bgr(image, cache_key=cache_key)
        combined = _compose(image, overlay, masked, height=display_height)
        out_path = out_dir / f"mask_debug_{written:04d}_{label}.jpg"
        cv2.imwrite(str(out_path), combined, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        detections = masker.detections_bgr(image, cache_key=cache_key)
        for det in detections:
            manifest.append(
                {
                    "debug_image": str(out_path),
                    "source_label": label,
                    "frame_index": frame_index,
                    "class": det.class_name,
                    "confidence": f"{det.confidence:.4f}",
                    "xyxy": det.xyxy,
                    "box_area_frac": f"{det.box_area_frac:.6f}",
                    "mask_area_frac": f"{det.mask_area_frac:.6f}",
                    "accepted": int(det.accepted),
                    "reason": det.reason,
                }
            )
        written += 1

    if manifest:
        with (out_dir / "mask_debug_manifest.csv").open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(manifest[0].keys()))
            writer.writeheader()
            writer.writerows(manifest)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO mask debug images with class labels and rejection reasons.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path)
    group.add_argument("--frames-dir", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--every-n-frames", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--display-height", type=int, default=360)
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
    args = parser.parse_args()

    written = export_mask_debug(
        out_dir=args.out_dir,
        video=args.video,
        frames_dir=args.frames_dir,
        every_n_frames=args.every_n_frames,
        max_frames=args.max_frames,
        display_height=args.display_height,
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
    print(f"Wrote {written} mask debug images to: {args.out_dir}")
    print(f"Manifest: {args.out_dir / 'mask_debug_manifest.csv'}")


if __name__ == "__main__":
    main()
