"""Build the offline visual reference index from extracted reference frames."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from gps_ex1.segmentation.dynamic_masks import DEFAULT_DYNAMIC_CLASSES, create_dynamic_object_masker
from gps_ex1.preprocess.reference_index import (
    add_camera_center_coordinates,
    build_reference_index,
    estimate_headings,
    read_frame_metadata_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build visual reference index from extracted frame metadata CSVs.")
    parser.add_argument("--frames-csv", nargs="+", required=True, type=Path, help="Frame metadata CSVs from extract_keyframes")
    parser.add_argument("--out", required=True, type=Path, help="Output .npz reference index")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="Base directory for resolving relative image paths")
    parser.add_argument("--camera-angle-deg", type=float, default=60.0, help="Known camera angle from the assignment videos")
    parser.add_argument(
        "--angle-convention",
        choices=["from-horizon", "from-nadir"],
        default="from-horizon",
        help="Interpretation of camera angle. Default: 60 degrees below horizon.",
    )
    parser.add_argument("--default-altitude-m", type=float, default=None, help="Fallback altitude if rel_alt is missing")
    parser.add_argument("--limit", type=int, default=None, help="Optional small limit for quick debugging")
    parser.add_argument(
        "--descriptor-backend",
        choices=["basic", "dinov2", "anyloc", "anyloc-gem"],
        default="basic",
        help="Global descriptor backend. Use basic for quick CPU tests; dinov2 for direct DINOv2; anyloc-gem for DINOv2 patch GeM retrieval.",
    )
    parser.add_argument("--descriptor-model", default="dinov2_vits14", help="DINOv2 torch.hub model name used by dinov2/anyloc-gem backends.")
    parser.add_argument("--descriptor-device", default=None, help="Optional descriptor device, e.g. cpu, cuda, or 0. Defaults to cuda if available.")
    parser.add_argument("--descriptor-image-size", type=int, default=518, help="Input size for DINOv2/AnyLoc descriptors. 518 is accurate; 322 is faster. Rounded to multiple of 14.")

    parser.add_argument("--mask-dynamic-objects", action="store_true", help="Use YOLO segmentation/detection to suppress people/vehicles before indexing.")
    parser.add_argument("--mask-model", default="yolov8n-seg.pt", help="Ultralytics model name/path for dynamic-object masking.")
    parser.add_argument("--mask-classes", default=DEFAULT_DYNAMIC_CLASSES, help="Comma-separated COCO class names/IDs to mask.")
    parser.add_argument("--mask-confidence", type=float, default=0.35, help="YOLO confidence threshold for dynamic-object masking.")
    parser.add_argument("--mask-iou", type=float, default=0.7, help="YOLO NMS IoU threshold for dynamic-object masking.")
    parser.add_argument("--mask-dilate-px", type=int, default=4, help="Dilate dynamic object masks by this many pixels.")
    parser.add_argument("--mask-fill", choices=["median", "gray", "black", "blur"], default="blur", help="How to fill masked dynamic-object regions for descriptor images.")
    parser.add_argument("--mask-imgsz", type=int, default=960, help="YOLO inference image size. Larger values catch small aerial cars better.")
    parser.add_argument("--mask-device", default=None, help="Optional Ultralytics device, e.g. cpu or 0.")
    parser.add_argument("--mask-min-area-px", type=int, default=20, help="Reject detections smaller than this many pixels.")
    parser.add_argument("--mask-max-area-frac", type=float, default=0.015, help="Reject detections covering more than this fraction of the image; avoids rooftop false positives.")
    parser.add_argument("--mask-max-width-frac", type=float, default=0.20, help="Reject detections wider than this fraction of the image.")
    parser.add_argument("--mask-max-height-frac", type=float, default=0.20, help="Reject detections taller than this fraction of the image.")
    parser.add_argument("--mask-source", choices=["box", "segmentation", "auto"], default="box", help="Use YOLO boxes, segmentation masks, or auto fallback. Box is safest for aerial cars.")
    parser.add_argument("--masked-frame-dir", type=Path, default=None, help="Optional folder to write masked reference frames. Usually leave unset so verification still uses original reference frames with keypoint masks.")
    args = parser.parse_args()

    rows = []
    for csv_path in args.frames_csv:
        rows.extend(read_frame_metadata_csv(csv_path))

    if not rows:
        raise RuntimeError("No valid frame rows found. Check your --frames-csv paths.")

    print(f"Loaded {len(rows)} frame rows")
    rows = estimate_headings(rows)
    rows = add_camera_center_coordinates(
        rows,
        camera_angle_deg=args.camera_angle_deg,
        angle_convention=args.angle_convention,
        default_altitude_m=args.default_altitude_m,
    )

    masker = create_dynamic_object_masker(
        enabled=args.mask_dynamic_objects,
        model_name=args.mask_model,
        classes=args.mask_classes,
        confidence=args.mask_confidence,
        iou=args.mask_iou,
        dilate_px=args.mask_dilate_px,
        fill=args.mask_fill,
        imgsz=args.mask_imgsz,
        device=args.mask_device,
        min_area_px=args.mask_min_area_px,
        max_area_frac=args.mask_max_area_frac,
        max_width_frac=args.mask_max_width_frac,
        max_height_frac=args.mask_max_height_frac,
        mask_source=args.mask_source,
    )
    if masker is not None:
        print(f"Dynamic-object masking enabled: model={args.mask_model}, classes={args.mask_classes}")
        if args.masked_frame_dir is None:
            print("Note: --masked-frame-dir was not set. This is recommended: descriptors are masked, but original paths are kept so verification can ignore masked keypoints instead of matching gray blobs.")

    # tqdm is imported here so the dependency is exercised by the user-facing CLI.
    # The actual descriptor loop is inside build_reference_index, which prints missing files.
    list(tqdm(range(1), desc="Preparing index"))
    index = build_reference_index(
        rows,
        args.out,
        base_dir=args.base_dir,
        limit=args.limit,
        descriptor_backend=args.descriptor_backend,
        descriptor_model=args.descriptor_model,
        descriptor_device=args.descriptor_device,
        descriptor_image_size=args.descriptor_image_size,
        masker=masker,
        masked_frame_dir=args.masked_frame_dir,
    )
    print(f"Wrote reference index: {args.out}")
    print(f"Indexed frames: {len(index.image_paths)}")
    print(f"Descriptor shape: {index.descriptors.shape}")
    print("Prediction target stored in index: center_lats/center_lons")


if __name__ == "__main__":
    main()
