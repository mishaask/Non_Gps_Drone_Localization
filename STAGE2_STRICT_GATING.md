# Stage 2: strict visual/geometric acceptance gates

This build keeps the memory-fixed YOLO/VisDrone masking pipeline and adds stricter match acceptance.

## What changed

The localization step now computes additional geometric diagnostics for every candidate:

- `verification_inlier_ratio`
- `projected_center_x`
- `projected_center_y`
- `projected_center_inside`
- `homography_quad_area_frac`
- `homography_geometry_ok`
- `verification_geometry_reason`

A candidate is now rejected when:

- it has too few matches,
- it has too few RANSAC/homography inliers,
- its inlier ratio is too weak,
- the projected center of the query frame falls outside the matched reference frame,
- the homography polygon is clearly degenerate or unrealistic,
- or the temporal jump is too large.

This is designed to stop the system from accepting visually wrong matches with only 4-6 inliers.

## Recommended run

After copying your `data` and `models` folders into this project and installing the venv:

```bat
python -m gps_ex1.pipeline.localize_video --video data/raw/DJI_0011.mp4 --reference-index data/processed/reference_index_visdrone_masked_fast.npz --out data/processed/DJI_0011_predictions_stage2_strict.csv --every-n-frames 30 --top-k 50 --prediction-target center --temporal-filter --descriptor-backend basic --verification-backend orb --min-inliers 12 --min-good-matches 12 --min-inlier-ratio 0.25 --mask-dynamic-objects --mask-model "models/yolov8-s-p2-mixup=0.4/weights/best.pt" --mask-classes 3,4,5,8,9 --mask-confidence 0.25 --mask-imgsz 640 --mask-source box --mask-max-area-frac 0.015 --mask-dilate-px 0 --mask-fill gray
```

Then export debug images:

```bat
python -m gps_ex1.tools.export_feature_match_debug --prediction-csv data/processed/DJI_0011_predictions_stage2_strict.csv --query-video data/raw/DJI_0011.mp4 --out-dir data/processed/debug_matches/DJI_0011_stage2_strict_feature_lines --project-root . --max-rows 80 --show-outliers --mask-dynamic-objects --mask-model "models/yolov8-s-p2-mixup=0.4/weights/best.pt" --mask-classes 3,4,5,8,9 --mask-confidence 0.25 --mask-imgsz 640 --mask-source box --mask-max-area-frac 0.015 --mask-dilate-px 0 --mask-fill gray
```

## Debug switches

For comparison only, you can relax the new gates:

- `--allow-center-outside`
- `--allow-bad-homography-geometry`

Do not use those for the final KML unless you are intentionally debugging.
