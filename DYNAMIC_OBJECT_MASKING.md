# Dynamic Object Masking: YOLO/SAM-style car/person removal

This project can optionally suppress dynamic objects before retrieval, ORB/LightGlue verification, and article-style debug visualization.

Why: cars, people, buses, trucks, bicycles, and motorcycles may appear in different positions between reference flights and query flights. ORB can accidentally match a white car in the query to a different white car in the reference, producing a wrong location.

The implementation uses Ultralytics YOLO segmentation/detection. A segmentation model such as `yolov8n-seg.pt` masks instance pixels; a detection-only model falls back to masking boxes. This is a practical YOLO/SAM-like step without making the baseline depend on heavy packages.

## Install optional dependency

```bat
python -m pip install ultralytics
```

or:

```bat
python -m pip install -r requirements-modern.txt
```

## Rebuild masked reference index

```bat
python -m gps_ex1.pipeline.build_reference_index --frames-csv data/processed/DJI_0006_frames.csv data/processed/DJI_0007_frames.csv data/processed/DJI_0008_frames.csv data/processed/DJI_0009_frames.csv --out data/processed/reference_index_masked.npz --camera-angle-deg 60 --angle-convention from-horizon --descriptor-backend basic --mask-dynamic-objects --masked-frame-dir data/processed/reference_frames_masked
```

This writes masked copies of reference frames to:

```text
data/processed/reference_frames_masked/
```

and stores those masked paths inside `reference_index_masked.npz`.

## Run DJI_0011 with masked query frames

```bat
python -m gps_ex1.pipeline.localize_video --video data/raw/DJI_0011.mp4 --reference-index data/processed/reference_index_masked.npz --out data/processed/DJI_0011_predictions_masked.csv --every-n-frames 30 --top-k 10 --prediction-target center --temporal-filter --descriptor-backend basic --verification-backend orb --mask-dynamic-objects
```

## Export article-style debug images with masked cars/people

```bat
python -m gps_ex1.tools.export_feature_match_debug --prediction-csv data/processed/DJI_0011_predictions_masked.csv --query-video data/raw/DJI_0011.mp4 --out-dir data/processed/debug_matches/DJI_0011_feature_lines_masked --project-root . --max-rows 80 --show-outliers --mask-dynamic-objects
```

## Notes

- `--top-k 10` already retrieves multiple candidates and reranks them by verification/temporal score. It is not simply taking the first retrieved match.
- YOLO masking can help with cars/people, but it will not fix wrong matches caused by repeated roads, roofs, trees, or fields. For that, the stronger next step is DINOv2/AnyLoc retrieval and LightGlue/LoFTR verification.
- First YOLO run may download model weights. Use `--mask-device cpu` if GPU setup causes problems.
