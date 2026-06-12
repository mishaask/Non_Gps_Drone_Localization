# Stage 5 — Multi-scale retrieval + LightGlue verification

This stage extends Stage 4 with:

1. `--query-scales` experiments such as `0.1,0.2,0.3,0.4`.
2. Optional `--verification-backend lightglue` for stronger learned local matching.
3. Scaled-query debug images: `export_feature_match_debug` can now draw matches on the `selected_query_scale` canvas instead of the original query frame.
4. Transparent header overlay: use `--header-alpha 0.25` or `--header-alpha 0.0` instead of a solid black header.
5. Optional LightGlue debug lines: use `--match-backend lightglue` in `export_feature_match_debug` after installing `requirements-stage5.txt`.

## Install

```bat
python -m pip install -r requirements-stage5.txt
```

## Recommended quick run

For DJI_0011, one frame every 5 seconds, max 50 sampled frames:

```bat
python -m gps_ex1.pipeline.localize_video ^
  --video data/raw/DJI_0011.mp4 ^
  --reference-index data/processed/reference_index_anyloc_gem_masked.npz ^
  --out data/processed/DJI_0011_anyloc_lightglue_scales_01_04_5s_50.csv ^
  --every-n-frames 150 ^
  --max-frames 50 ^
  --top-k 15 ^
  --query-scales 0.1,0.2,0.3,0.4 ^
  --query-scale-fill blur ^
  --prediction-target center ^
  --descriptor-backend anyloc-gem ^
  --descriptor-image-size 322 ^
  --verification-backend lightglue ^
  --min-inliers 8 ^
  --min-good-matches 15 ^
  --min-inlier-ratio 0.20 ^
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

`--top-k 15` is intentional. LightGlue is much heavier than ORB, so verifying 50 candidates per query frame can be slow.

## Debug images using selected scale and transparent header

```bat
python -m gps_ex1.tools.export_feature_match_debug ^
  --prediction-csv data/processed/DJI_0011_anyloc_lightglue_scales_01_04_5s_50.csv ^
  --query-video data/raw/DJI_0011.mp4 ^
  --out-dir data/processed/debug_matches/DJI_0011_lightglue_scales_01_04_5s_50 ^
  --project-root . ^
  --max-rows 50 ^
  --query-scale-mode selected ^
  --query-scale-fill blur ^
  --match-backend lightglue ^
  --header-alpha 0.25 ^
  --show-outliers ^
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

Use `--header-alpha 0.0` if you want only text outlines and no dark header background.
