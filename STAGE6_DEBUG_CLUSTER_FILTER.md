# Stage 6 — Debug-friendly controlled evaluation + cluster filtering

This stage is intentionally **not production-final**. It is for fast visual experiments while we tune the UAV localization pipeline.

## What Stage 6 adds

1. `--query-frame-list <txt>`
   - Process exact query frames instead of sampling the whole video.
   - Supports one frame per line, comma/space separated frames, ranges like `1800-2400`, and stepped ranges like `1800-2400:150`.

2. Inlier-cluster diagnostics in the prediction CSV:
   - `query_inlier_bbox_area_frac`
   - `query_inlier_bbox_width_frac`
   - `query_inlier_bbox_height_frac`
   - `reference_inlier_bbox_area_frac`
   - `reference_inlier_bbox_width_frac`
   - `reference_inlier_bbox_height_frac`

3. Optional cluster gates:
   - `--max-reference-inlier-area-frac`
   - `--max-reference-inlier-width-frac`
   - `--max-reference-inlier-height-frac`
   - `--max-query-inlier-area-frac`

4. Better debug overlays:
   - selected query scale is drawn
   - LightGlue/ORB lines are drawn on the selected scaled query canvas
   - reference inlier bbox metrics are printed in the transparent header

## Why cluster filtering helps

Good matches in our current run usually concentrate on a single distinctive structure, especially the circular/curved building area. Bad matches often scatter across unrelated roads/roofs/buildings. Stage 6 lets us reject candidates whose RANSAC inliers cover too much of the reference frame.

## Recommended controlled test

```bat
python -m gps_ex1.pipeline.localize_video ^
  --video data/raw/DJI_0011.mp4 ^
  --reference-index data/processed/reference_index_anyloc_gem_masked.npz ^
  --out data/processed/DJI_0011_stage6_debug_frames_lightglue_cluster.csv ^
  --query-frame-list configs/DJI_0011_debug_frames.txt ^
  --top-k 15 ^
  --query-scales 0.2,0.3,0.4 ^
  --query-scale-fill blur ^
  --prediction-target center ^
  --descriptor-backend anyloc-gem ^
  --descriptor-image-size 322 ^
  --verification-backend lightglue ^
  --min-inliers 10 ^
  --min-good-matches 20 ^
  --min-inlier-ratio 0.25 ^
  --max-reference-inlier-area-frac 0.18 ^
  --max-reference-inlier-width-frac 0.55 ^
  --max-reference-inlier-height-frac 0.45 ^
  --mask-dynamic-objects ^
  --mask-model "models/yolov8-s-p2-mixup=0.4/weights/best.pt" ^
  --mask-classes 3,4,5,8,9 ^
  --mask-confidence 0.25 ^
  --mask-imgsz 640 ^
  --mask-source box ^
  --mask-max-area-frac 0.015 ^
  --mask-dilate-px 0 ^
  --mask-fill gray
```

## Export accepted debug images

```bat
python -m gps_ex1.tools.export_feature_match_debug ^
  --prediction-csv data/processed/DJI_0011_stage6_debug_frames_lightglue_cluster.csv ^
  --query-video data/raw/DJI_0011.mp4 ^
  --out-dir data/processed/debug_matches/DJI_0011_stage6_accepted ^
  --project-root . ^
  --max-rows 100 ^
  --accepted-only ^
  --query-scale-mode selected ^
  --query-scale-fill blur ^
  --match-backend lightglue ^
  --header-alpha 0.15 ^
  --mask-dynamic-objects ^
  --mask-model "models/yolov8-s-p2-mixup=0.4/weights/best.pt" ^
  --mask-classes 3,4,5,8,9 ^
  --mask-confidence 0.25 ^
  --mask-imgsz 640 ^
  --mask-source box ^
  --mask-max-area-frac 0.015 ^
  --mask-dilate-px 0 ^
  --mask-fill gray
```

## Export rejected debug images

```bat
python -m gps_ex1.tools.export_feature_match_debug ^
  --prediction-csv data/processed/DJI_0011_stage6_debug_frames_lightglue_cluster.csv ^
  --query-video data/raw/DJI_0011.mp4 ^
  --out-dir data/processed/debug_matches/DJI_0011_stage6_rejected ^
  --project-root . ^
  --max-rows 100 ^
  --rejected-only ^
  --show-outliers ^
  --query-scale-mode selected ^
  --query-scale-fill blur ^
  --match-backend lightglue ^
  --header-alpha 0.15 ^
  --mask-dynamic-objects ^
  --mask-model "models/yolov8-s-p2-mixup=0.4/weights/best.pt" ^
  --mask-classes 3,4,5,8,9 ^
  --mask-confidence 0.25 ^
  --mask-imgsz 640 ^
  --mask-source box ^
  --mask-max-area-frac 0.015 ^
  --mask-dilate-px 0 ^
  --mask-fill gray
```

## Compare summary

```bat
python -m gps_ex1.tools.compare_descriptor_runs ^
  --prediction-csv data/processed/DJI_0011_stage6_debug_frames_lightglue_cluster.csv
```

The comparison output now includes `ref_bbox`, the median reference inlier bbox area fraction.
