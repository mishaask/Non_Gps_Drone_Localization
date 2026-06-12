# Stage 4 - Multi-scale query retrieval

This version adds a quick experiment for the lower-altitude DJI_0011 problem.
The query image can be searched at several shrunken scales before ORB/LightGlue
verification:

```text
query frame
-> scale 1.0, 0.8, 0.6, 0.5
-> extract descriptor for each scale
-> retrieve top-k for each scale
-> merge duplicate reference frames by best similarity
-> verify final top-k candidates
```

Why this exists:

* The reference flights are higher than the query flight.
* A building/tree that fills a large part of the query image may appear much
  smaller in the reference image.
* The descriptor extractor resizes every input to a fixed size, so ordinary
  resize alone does not simulate altitude. Instead, this code shrinks the query
  content into a same-size canvas.

New CLI options:

```text
--query-scales 1.0,0.8,0.6,0.5
--query-scale-fill blur|median|gray|black
```

The CSV now includes:

```text
selected_query_scale
```

Use this column to check whether accepted / visually correct matches come mostly
from 0.5 or 0.6. If yes, the scale mismatch hypothesis is likely correct and the
next step should be vehicle/building-based scale estimation.

Recommended quick test on DJI_0011, one frame every 5 seconds, max 50 sampled
frames:

```bat
python -m gps_ex1.pipeline.localize_video --video data/raw/DJI_0011.mp4 --reference-index data/processed/reference_index_anyloc_gem_masked.npz --out data/processed/DJI_0011_anyloc_multiscale_5s_50.csv --every-n-frames 150 --max-frames 50 --top-k 50 --query-scales 1.0,0.8,0.6,0.5 --query-scale-fill blur --prediction-target center --descriptor-backend anyloc-gem --descriptor-image-size 322 --verification-backend orb --min-inliers 6 --min-good-matches 8 --min-inlier-ratio 0.15 --mask-dynamic-objects --mask-model "models/yolov8-s-p2-mixup=0.4/weights/best.pt" --mask-classes 3,4,5,8,9 --mask-confidence 0.25 --mask-imgsz 640 --mask-source box --mask-max-area-frac 0.015 --mask-dilate-px 0 --mask-fill gray
```

For DINOv2 instead of AnyLoc-GEM, use:

```bat
python -m gps_ex1.pipeline.localize_video --video data/raw/DJI_0011.mp4 --reference-index data/processed/reference_index_dinov2_masked.npz --out data/processed/DJI_0011_dinov2_multiscale_5s_50.csv --every-n-frames 150 --max-frames 50 --top-k 50 --query-scales 1.0,0.8,0.6,0.5 --query-scale-fill blur --prediction-target center --descriptor-backend dinov2 --descriptor-image-size 322 --verification-backend orb --min-inliers 6 --min-good-matches 8 --min-inlier-ratio 0.15 --mask-dynamic-objects --mask-model "models/yolov8-s-p2-mixup=0.4/weights/best.pt" --mask-classes 3,4,5,8,9 --mask-confidence 0.25 --mask-imgsz 640 --mask-source box --mask-max-area-frac 0.015 --mask-dilate-px 0 --mask-fill gray
```
