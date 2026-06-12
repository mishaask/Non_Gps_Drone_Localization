# Safer car masking guide

This package includes a safer YOLO masking experiment for the GPS_EX1 drone localization pipeline.

## What changed from the previous masking zip

The earlier masking version used broad classes like `car,bus,truck,train` and allowed segmentation masks. On aerial footage, this caused large rooftops/trees/buildings to be masked while many tiny cars remained visible.

This version changes the defaults:

```text
mask classes: car only
mask source: YOLO bounding boxes by default
confidence: 0.35
image size: 960
max area fraction: 0.015
fill: blur
ORB verification: ignores keypoints inside masks instead of matching mask blobs
```

## First command to run

```bat
python -m gps_ex1.tools.export_mask_debug --video data/raw/DJI_0011.mp4 --out-dir data/processed/debug_matches/DJI_0011_mask_detector_debug --every-n-frames 30 --max-frames 40 --mask-classes car --mask-confidence 0.35 --mask-imgsz 960 --mask-source box --mask-max-area-frac 0.015
```

Open the output folder. Each image shows:

```text
original | detection overlay | masked image
```

Green boxes = accepted car masks. Red boxes = rejected detections with reason.

## Recommended full run on DJI_0011

```bat
python -m gps_ex1.pipeline.build_reference_index --frames-csv data/processed/DJI_0006_frames.csv data/processed/DJI_0007_frames.csv data/processed/DJI_0008_frames.csv data/processed/DJI_0009_frames.csv --out data/processed/reference_index_car_masked.npz --camera-angle-deg 60 --angle-convention from-horizon --descriptor-backend basic --mask-dynamic-objects --mask-classes car --mask-confidence 0.35 --mask-imgsz 960 --mask-source box --mask-max-area-frac 0.015
```

```bat
python -m gps_ex1.pipeline.localize_video --video data/raw/DJI_0011.mp4 --reference-index data/processed/reference_index_car_masked.npz --out data/processed/DJI_0011_predictions_car_masked.csv --every-n-frames 30 --top-k 10 --prediction-target center --temporal-filter --descriptor-backend basic --verification-backend orb --mask-dynamic-objects --mask-classes car --mask-confidence 0.35 --mask-imgsz 960 --mask-source box --mask-max-area-frac 0.015
```

```bat
python -m gps_ex1.tools.export_feature_match_debug --prediction-csv data/processed/DJI_0011_predictions_car_masked.csv --query-video data/raw/DJI_0011.mp4 --out-dir data/processed/debug_matches/DJI_0011_feature_lines_car_masked --project-root . --max-rows 80 --show-outliers --mask-dynamic-objects --mask-classes car --mask-confidence 0.35 --mask-imgsz 960 --mask-source box --mask-max-area-frac 0.015 --mask-display filled
```

```bat
python -m gps_ex1.tools.export_kml --prediction-csv data/processed/DJI_0011_predictions_car_masked.csv --out data/processed/kml/DJI_0011_car_masked.kml --name "DJI_0011 car-masked predicted center path"
```

## Important notes

Do not assume masking improved the result. Compare the old and new debug images/KML. Generic YOLO is not trained specifically for top-down drone cars, so it may still miss vehicles. If the detector debug shows missed cars or roof false positives, the real fix is an aerial vehicle detector or fine-tuning.
