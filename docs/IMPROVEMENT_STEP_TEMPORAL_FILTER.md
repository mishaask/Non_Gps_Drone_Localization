# Improvement Step: Temporal Continuity Filter

## Why this was added

The first visual-localization baseline chose the best-looking reference frame independently for every query frame. That produced valid CSV/KML files, but the predicted paths were physically unrealistic because many drone frames contain visually repetitive roads, roofs, parking lots, and trees.

The new version adds a temporal consistency layer. The localizer still performs visual retrieval and ORB/homography scoring, but it now also checks whether the next estimate is physically plausible relative to the previous accepted estimate.

## What changed

Added modules:

- `src/gps_ex1/localization/temporal_filter.py`
- `src/gps_ex1/tools/filter_predictions.py`
- `src/gps_ex1/tools/analyze_predictions.py`

Updated modules:

- `src/gps_ex1/pipeline/localize_video.py`
- `src/gps_ex1/tools/export_kml.py`

Generated improved outputs:

- `data/processed/DJI_0010_predictions_filtered.csv`
- `data/processed/DJI_0011_predictions_filtered.csv`
- `data/processed/DJI_0010_drone_predictions_filtered.csv`
- `data/processed/DJI_0011_drone_predictions_filtered.csv`
- `data/processed/kml/DJI_0010_filtered_center_path.kml`
- `data/processed/kml/DJI_0011_filtered_center_path.kml`
- `data/processed/kml/DJI_0010_filtered_center_and_drone_paths.kml`
- `data/processed/kml/DJI_0011_filtered_center_and_drone_paths.kml`

## How to improve existing predictions without rerunning the videos

```bash
python -m gps_ex1.tools.filter_predictions ^
  --prediction-csv data/processed/DJI_0010_predictions.csv ^
  --out data/processed/DJI_0010_predictions_filtered.csv ^
  --overwrite-prediction-columns

python -m gps_ex1.tools.filter_predictions ^
  --prediction-csv data/processed/DJI_0011_predictions.csv ^
  --out data/processed/DJI_0011_predictions_filtered.csv ^
  --overwrite-prediction-columns
```

Then export KML:

```bash
python -m gps_ex1.tools.export_kml ^
  --prediction-csv data/processed/DJI_0010_predictions_filtered.csv ^
  --out data/processed/kml/DJI_0010_filtered_center_path.kml ^
  --name "DJI_0010 filtered predicted center path"

python -m gps_ex1.tools.export_kml ^
  --prediction-csv data/processed/DJI_0011_predictions_filtered.csv ^
  --out data/processed/kml/DJI_0011_filtered_center_path.kml ^
  --name "DJI_0011 filtered predicted center path"
```

## Better command when rerunning the actual video localizer

The stronger improvement is to rerun the video localizer with temporal filtering enabled, because then the filter can choose between the top-k visual candidates before committing to one candidate.

```bash
python -m gps_ex1.pipeline.localize_video ^
  --video data/raw/DJI_0010.mp4 ^
  --reference-index data/processed/reference_index.npz ^
  --out data/processed/DJI_0010_predictions_temporal.csv ^
  --every-n-frames 30 ^
  --top-k 40 ^
  --prediction-target center ^
  --temporal-filter ^
  --min-inliers 6 ^
  --min-good-matches 8 ^
  --max-speed-mps 18 ^
  --base-gate-m 55 ^
  --hard-jump-m 180 ^
  --ema-alpha 0.35
```

Run the same command for `DJI_0011.mp4`.

## How to analyze path quality

```bash
python -m gps_ex1.tools.analyze_predictions ^
  --prediction-csv data/processed/DJI_0010_predictions.csv data/processed/DJI_0010_predictions_filtered.csv
```

The important diagnostics are median jump, max jump, and jumps over 100/200 meters. Since frames are sampled roughly once per second, hundreds-of-meters jumps are not physically plausible for this dataset.

## Current result from the provided CSVs

| File | Median jump | Max jump | Jumps >200m |
|---|---:|---:|---:|
| DJI_0010 raw center | 289.2 m | 1023.1 m | 485 |
| DJI_0010 filtered center | 0.0 m | 198.6 m | 0 |
| DJI_0011 raw center | 130.2 m | 869.4 m | 161 |
| DJI_0011 filtered center | 0.0 m | 171.2 m | 0 |

This is more realistic than the raw output, but it should be described honestly: it is a temporal-gated baseline, not proof of final metric accuracy, because the assignment test videos do not include `DJI_0010.SRT` / `DJI_0011.SRT` for ground-truth evaluation.
