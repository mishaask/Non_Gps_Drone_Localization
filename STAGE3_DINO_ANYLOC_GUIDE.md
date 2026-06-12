# Stage 3: DINOv2 and AnyLoc-style retrieval

This version keeps the Stage-2 strict geometry gates and adds two stronger retrieval backends:

- `dinov2`: direct global descriptor from Meta DINOv2.
- `anyloc-gem` / `anyloc`: AnyLoc-style DINOv2 patch-token GeM aggregation.

The goal is to run both against the same reference frames and query video, then compare the CSV/KML/debug outputs.

## Install heavy dependencies

From the project root. For Stage 3, use `requirements-stage3.txt`; `requirements-modern.txt` is only needed later for LightGlue/FAISS experiments:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
python -m pip install -r requirements-stage3.txt
```

The first DINOv2/AnyLoc run may download DINOv2 weights through `torch.hub`.

## Quick smoke tests

Before running all 2037 reference frames, test only 40 frames:

```bat
python -m gps_ex1.pipeline.build_reference_index --frames-csv data/processed/DJI_0006_frames.csv data/processed/DJI_0007_frames.csv data/processed/DJI_0008_frames.csv data/processed/DJI_0009_frames.csv --out data/processed/test_index_dinov2.npz --limit 40 --camera-angle-deg 60 --angle-convention from-horizon --descriptor-backend dinov2 --descriptor-image-size 322 --mask-dynamic-objects --mask-model "models/yolov8-s-p2-mixup=0.4/weights/best.pt" --mask-classes 3,4,5,8,9 --mask-confidence 0.25 --mask-imgsz 640 --mask-source box --mask-max-area-frac 0.015 --mask-dilate-px 0 --mask-fill gray
```

```bat
python -m gps_ex1.pipeline.build_reference_index --frames-csv data/processed/DJI_0006_frames.csv data/processed/DJI_0007_frames.csv data/processed/DJI_0008_frames.csv data/processed/DJI_0009_frames.csv --out data/processed/test_index_anyloc_gem.npz --limit 40 --camera-angle-deg 60 --angle-convention from-horizon --descriptor-backend anyloc-gem --descriptor-image-size 322 --mask-dynamic-objects --mask-model "models/yolov8-s-p2-mixup=0.4/weights/best.pt" --mask-classes 3,4,5,8,9 --mask-confidence 0.25 --mask-imgsz 640 --mask-source box --mask-max-area-frac 0.015 --mask-dilate-px 0 --mask-fill gray
```

## Full DINOv2 run

```bat
python -m gps_ex1.pipeline.build_reference_index --frames-csv data/processed/DJI_0006_frames.csv data/processed/DJI_0007_frames.csv data/processed/DJI_0008_frames.csv data/processed/DJI_0009_frames.csv --out data/processed/reference_index_dinov2_masked.npz --camera-angle-deg 60 --angle-convention from-horizon --descriptor-backend dinov2 --descriptor-image-size 322 --mask-dynamic-objects --mask-model "models/yolov8-s-p2-mixup=0.4/weights/best.pt" --mask-classes 3,4,5,8,9 --mask-confidence 0.25 --mask-imgsz 640 --mask-source box --mask-max-area-frac 0.015 --mask-dilate-px 0 --mask-fill gray
```

```bat
python -m gps_ex1.pipeline.localize_video --video data/raw/DJI_0011.mp4 --reference-index data/processed/reference_index_dinov2_masked.npz --out data/processed/DJI_0011_predictions_dinov2_stage3.csv --every-n-frames 30 --top-k 50 --prediction-target center --temporal-filter --descriptor-backend dinov2 --descriptor-image-size 322 --verification-backend orb --min-inliers 12 --min-good-matches 12 --min-inlier-ratio 0.25 --mask-dynamic-objects --mask-model "models/yolov8-s-p2-mixup=0.4/weights/best.pt" --mask-classes 3,4,5,8,9 --mask-confidence 0.25 --mask-imgsz 640 --mask-source box --mask-max-area-frac 0.015 --mask-dilate-px 0 --mask-fill gray
```

## Full AnyLoc-style run

```bat
python -m gps_ex1.pipeline.build_reference_index --frames-csv data/processed/DJI_0006_frames.csv data/processed/DJI_0007_frames.csv data/processed/DJI_0008_frames.csv data/processed/DJI_0009_frames.csv --out data/processed/reference_index_anyloc_gem_masked.npz --camera-angle-deg 60 --angle-convention from-horizon --descriptor-backend anyloc-gem --descriptor-image-size 322 --mask-dynamic-objects --mask-model "models/yolov8-s-p2-mixup=0.4/weights/best.pt" --mask-classes 3,4,5,8,9 --mask-confidence 0.25 --mask-imgsz 640 --mask-source box --mask-max-area-frac 0.015 --mask-dilate-px 0 --mask-fill gray
```

```bat
python -m gps_ex1.pipeline.localize_video --video data/raw/DJI_0011.mp4 --reference-index data/processed/reference_index_anyloc_gem_masked.npz --out data/processed/DJI_0011_predictions_anyloc_gem_stage3.csv --every-n-frames 30 --top-k 50 --prediction-target center --temporal-filter --descriptor-backend anyloc-gem --descriptor-image-size 322 --verification-backend orb --min-inliers 12 --min-good-matches 12 --min-inlier-ratio 0.25 --mask-dynamic-objects --mask-model "models/yolov8-s-p2-mixup=0.4/weights/best.pt" --mask-classes 3,4,5,8,9 --mask-confidence 0.25 --mask-imgsz 640 --mask-source box --mask-max-area-frac 0.015 --mask-dilate-px 0 --mask-fill gray
```

## Compare the runs

```bat
python -m gps_ex1.tools.compare_descriptor_runs --prediction-csv data/processed/DJI_0011_predictions_stage2_strict.csv data/processed/DJI_0011_predictions_dinov2_stage3.csv data/processed/DJI_0011_predictions_anyloc_gem_stage3.csv
```

## Export KML/debug for a chosen run

```bat
python -m gps_ex1.tools.export_kml --prediction-csv data/processed/DJI_0011_predictions_anyloc_gem_stage3.csv --out data/processed/kml/DJI_0011_anyloc_gem_stage3.kml --name "DJI_0011 AnyLoc-GEM Stage 3"
```

```bat
python -m gps_ex1.tools.export_feature_match_debug --prediction-csv data/processed/DJI_0011_predictions_anyloc_gem_stage3.csv --query-video data/raw/DJI_0011.mp4 --out-dir data/processed/debug_matches/DJI_0011_anyloc_gem_stage3_feature_lines --project-root . --max-rows 80 --show-outliers --mask-dynamic-objects --mask-model "models/yolov8-s-p2-mixup=0.4/weights/best.pt" --mask-classes 3,4,5,8,9 --mask-confidence 0.25 --mask-imgsz 640 --mask-source box --mask-max-area-frac 0.015 --mask-dilate-px 0 --mask-fill gray
```
