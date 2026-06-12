# GPS_EX1 — GNSS-Denied Visual Navigation for Drones

This repo is a ready-to-test implementation for **Ex1: Visual Navigation for Drones**.

The project goal is:

1. Use reference drone flights that contain **video + SRT telemetry + GNSS**.
2. Preprocess those reference flights into a geo-tagged visual database.
3. Given a new query/test video **without using its GNSS**, estimate the geographic coordinate at the **center of the camera image**.
4. Export CSV/KML results for debugging and Google Earth visualization.

The repo includes the original lightweight baseline plus optional hooks for the stronger modern pipeline:

```text
SRT parser
→ reference frame extraction
→ camera-center GPS estimation from altitude/angle/heading
→ visual retrieval
→ geometric verification
→ temporal filtering
→ CSV/KML output
```

The default pipeline is CPU-friendly and testable immediately. The optional stronger mode uses **DINOv2** for global descriptors and **LightGlue** for image-pair verification.

---

## What is included

```text
src/gps_ex1/io/              SRT parsing, timecode, video frame extraction
src/gps_ex1/geometry/        geo math and camera-center projection
src/gps_ex1/features/        descriptor backends: basic and optional DINOv2
src/gps_ex1/preprocess/      reference index builder
src/gps_ex1/localization/    retrieval, ORB verifier, optional LightGlue verifier, temporal filter
src/gps_ex1/pipeline/        runnable pipeline commands
src/gps_ex1/tools/           KML export, analysis, QA/debug utilities
data/raw/                    provided DJI reference SRT files
(data/links/)                YouTube URL files for reference/test videos
data/processed/              existing CSV/KML/debug outputs from the current baseline
configs/                     example dataset config
docs/                        notes, assignment PDFs, and pipeline explanations
tests/                       unit tests
```

Large video files are **not** included in the zip. Put downloaded videos here:

```text
data/raw/DJI_0006.mp4
data/raw/DJI_0007.mp4
data/raw/DJI_0008.mp4
data/raw/DJI_0009.mp4
data/raw/DJI_0010.mp4
data/raw/DJI_0011.mp4
```

---

## Install

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

### Linux / macOS / WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

Optional stronger DINOv2/LightGlue mode:

```bash
pip install -r requirements-modern.txt
```

The first DINOv2 run may download model weights through `torch.hub`.

---

## Quick smoke test

Run the unit tests:

```bash
pytest -q
```

Analyze existing generated prediction CSVs:

```bash
python -m gps_ex1.tools.analyze_predictions \
  --prediction-csv data/processed/DJI_0010_predictions_filtered.csv data/processed/DJI_0011_predictions_filtered.csv
```

Inspect one SRT file:

```bash
python -m gps_ex1.tools.inspect_srt data/raw/DJI_0006.SRT --csv-dir data/processed
```

---

## Full baseline pipeline

### 1. Build reference telemetry metadata

```bash
python -m gps_ex1.pipeline.build_reference_metadata \
  --srt data/raw/DJI_0006.SRT data/raw/DJI_0007.SRT data/raw/DJI_0008.SRT data/raw/DJI_0009.SRT \
  --out data/processed/reference_telemetry.csv
```

### 2. Extract reference keyframes

Run once per reference video after putting the `.mp4` files in `data/raw/`:

```bash
python -m gps_ex1.pipeline.extract_keyframes \
  --video data/raw/DJI_0006.mp4 \
  --srt data/raw/DJI_0006.SRT \
  --flight-id DJI_0006 \
  --out-dir data/processed/reference_frames/DJI_0006 \
  --metadata-csv data/processed/DJI_0006_frames.csv \
  --every-n-frames 30
```

Repeat for DJI_0007, DJI_0008, DJI_0009.

### 3. Build the visual reference index

Lightweight baseline descriptor:

```bash
python -m gps_ex1.pipeline.build_reference_index \
  --frames-csv data/processed/DJI_0006_frames.csv data/processed/DJI_0007_frames.csv data/processed/DJI_0008_frames.csv data/processed/DJI_0009_frames.csv \
  --out data/processed/reference_index.npz \
  --camera-angle-deg 60 \
  --angle-convention from-horizon \
  --descriptor-backend basic
```

Optional DINOv2 descriptor:

```bash
python -m gps_ex1.pipeline.build_reference_index \
  --frames-csv data/processed/DJI_0006_frames.csv data/processed/DJI_0007_frames.csv data/processed/DJI_0008_frames.csv data/processed/DJI_0009_frames.csv \
  --out data/processed/reference_index_dinov2.npz \
  --camera-angle-deg 60 \
  --angle-convention from-horizon \
  --descriptor-backend dinov2
```

### 4. Localize a query video

Baseline mode:

```bash
python -m gps_ex1.pipeline.localize_video \
  --video data/raw/DJI_0010.mp4 \
  --reference-index data/processed/reference_index.npz \
  --out data/processed/DJI_0010_predictions_new.csv \
  --every-n-frames 30 \
  --top-k 10 \
  --prediction-target center \
  --temporal-filter \
  --descriptor-backend basic \
  --verification-backend orb
```

Fast debug run:

```bash
python -m gps_ex1.pipeline.localize_video \
  --video data/raw/DJI_0010.mp4 \
  --reference-index data/processed/reference_index.npz \
  --out data/processed/DJI_0010_debug.csv \
  --every-n-frames 30 \
  --top-k 5 \
  --max-frames 20 \
  --temporal-filter
```

Optional stronger mode:

```bash
python -m gps_ex1.pipeline.localize_video \
  --video data/raw/DJI_0010.mp4 \
  --reference-index data/processed/reference_index_dinov2.npz \
  --out data/processed/DJI_0010_predictions_dinov2_lightglue.csv \
  --every-n-frames 30 \
  --top-k 10 \
  --prediction-target center \
  --temporal-filter \
  --descriptor-backend dinov2 \
  --verification-backend lightglue
```

Important: the descriptor backend used for localization must match the backend used to build the reference index.

### 5. Export KML

```bash
python -m gps_ex1.tools.export_kml \
  --prediction-csv data/processed/DJI_0010_predictions_new.csv \
  --out data/processed/kml/DJI_0010_predictions_new.kml \
  --name "DJI_0010 predicted center path"
```

Open the `.kml` in Google Earth.

---

## Recommended project direction

For the final presentation/report, the recommended architecture is:

```text
DINOv2/AnyLoc-style retrieval
+ LightGlue/LoFTR-style verification
+ camera geometry for center-point GPS
+ temporal filtering / Kalman-style smoothing
```

The current zip is structured so that the baseline works immediately, while the stronger DINOv2/LightGlue path is already connected through command-line flags.

---

## Notes

- The query SRT, if supplied, is used only for evaluation/debugging. It is not used for prediction.
- The current baseline can still make wrong visual matches in repetitive aerial scenes. Use the debug match images and KML jumps to inspect quality.
- Full SLAM / 3D reconstruction is intentionally not the main path because it is heavier and riskier for this assignment.


### Article-style feature match debug images

To generate debug images like the GNSS-free UAV localization paper figures, with colored feature-match lines, RANSAC inliers, and the projected query footprint on the matched reference frame:

```bash
python -m gps_ex1.tools.export_feature_match_debug \
  --prediction-csv data/processed/DJI_0011_predictions_from_scratch.csv \
  --query-video data/raw/DJI_0011.mp4 \
  --out-dir data/processed/debug_matches/DJI_0011_feature_lines \
  --project-root . \
  --max-rows 80 \
  --show-outliers
```

For only rejected temporal jumps, useful when debugging false rejections:

```bash
python -m gps_ex1.tools.export_feature_match_debug \
  --prediction-csv data/processed/DJI_0011_predictions_from_scratch.csv \
  --query-video data/raw/DJI_0011.mp4 \
  --out-dir data/processed/debug_matches/DJI_0011_rejected_feature_lines \
  --project-root . \
  --max-rows 80 \
  --rejected-only \
  --show-outliers
```

---

## Safer YOLO dynamic-object masking experiment

This updated zip changes the masking strategy because generic YOLO/SAM can misbehave on top-down drone footage. The safer defaults are now:

```text
--mask-classes car
--mask-source box
--mask-confidence 0.35
--mask-imgsz 960
--mask-max-area-frac 0.015
--mask-fill blur
```

Why: tiny aerial cars are hard for COCO-trained YOLO, and large roof/vegetation regions can be falsely detected as vehicle-like objects. Area/size filtering rejects detections that are too large to plausibly be cars.

Install optional YOLO dependency:

```bash
python -m pip install ultralytics
```

### 0. First debug the detector itself

Before rebuilding the whole index, generate detector overlays. This shows original | YOLO overlay | masked image. Green boxes are accepted, red boxes are rejected with a reason such as `too_large_area`.

```bash
python -m gps_ex1.tools.export_mask_debug \
  --video data/raw/DJI_0011.mp4 \
  --out-dir data/processed/debug_matches/DJI_0011_mask_detector_debug \
  --every-n-frames 30 \
  --max-frames 40 \
  --mask-classes car \
  --mask-confidence 0.35 \
  --mask-imgsz 960 \
  --mask-source box \
  --mask-max-area-frac 0.015
```

### 1. Rebuild masked reference descriptors

Recommended: **do not use `--masked-frame-dir`** for this experiment. The index descriptors are computed from masked images, but original reference paths are kept. During ORB verification, the code now ignores keypoints inside detected car masks instead of matching against gray/blurred blobs.

```bash
python -m gps_ex1.pipeline.build_reference_index \
  --frames-csv data/processed/DJI_0006_frames.csv data/processed/DJI_0007_frames.csv data/processed/DJI_0008_frames.csv data/processed/DJI_0009_frames.csv \
  --out data/processed/reference_index_car_masked.npz \
  --camera-angle-deg 60 \
  --angle-convention from-horizon \
  --descriptor-backend basic \
  --mask-dynamic-objects \
  --mask-classes car \
  --mask-confidence 0.35 \
  --mask-imgsz 960 \
  --mask-source box \
  --mask-max-area-frac 0.015
```

### 2. Run DJI_0011 with the masked index

```bash
python -m gps_ex1.pipeline.localize_video \
  --video data/raw/DJI_0011.mp4 \
  --reference-index data/processed/reference_index_car_masked.npz \
  --out data/processed/DJI_0011_predictions_car_masked.csv \
  --every-n-frames 30 \
  --top-k 10 \
  --prediction-target center \
  --temporal-filter \
  --descriptor-backend basic \
  --verification-backend orb \
  --mask-dynamic-objects \
  --mask-classes car \
  --mask-confidence 0.35 \
  --mask-imgsz 960 \
  --mask-source box \
  --mask-max-area-frac 0.015
```

### 3. Export article-style feature debug images with cars removed/ignored

```bash
python -m gps_ex1.tools.export_feature_match_debug \
  --prediction-csv data/processed/DJI_0011_predictions_car_masked.csv \
  --query-video data/raw/DJI_0011.mp4 \
  --out-dir data/processed/debug_matches/DJI_0011_feature_lines_car_masked \
  --project-root . \
  --max-rows 80 \
  --show-outliers \
  --mask-dynamic-objects \
  --mask-classes car \
  --mask-confidence 0.35 \
  --mask-imgsz 960 \
  --mask-source box \
  --mask-max-area-frac 0.015 \
  --mask-display filled
```

For detector labels in the feature-match images instead of filled masks, use:

```bash
--mask-display overlay
```

### 4. Export KML

```bash
python -m gps_ex1.tools.export_kml \
  --prediction-csv data/processed/DJI_0011_predictions_car_masked.csv \
  --out data/processed/kml/DJI_0011_car_masked.kml \
  --name "DJI_0011 car-masked predicted center path"
```

Compare:

```text
data/processed/DJI_0011_predictions_from_scratch.csv
data/processed/DJI_0011_predictions_car_masked.csv
```

and compare the old/new debug folders. If the new detector debug still misses most cars, the next step is an aerial-vehicle-specific detector rather than generic COCO YOLO.


## Stage 4: multi-scale query retrieval

See `STAGE4_MULTISCALE_RETRIEVAL.md` for the lower-altitude query experiment using `--query-scales 1.0,0.8,0.6,0.5`.


## Stage 9 fixed-grid regions

See `STAGE9_FIXED_GRID_REGIONS.md`.
