# Debug timecodes, drone-position path, and YOLO/semantic next step

## Why this update exists

The temporal-filter KMLs are much more realistic than the raw KMLs, but the next debugging question is:

> Did the query frame really match the chosen reference frame?

To answer that, the prediction CSV now includes readable timecodes, and the project includes a tool that exports side-by-side debug images:

- left: the query/test video frame
- right: the matched reference frame

This makes it easy to open the exact time in the video and check whether the chosen match is visually logical.

## New CSV columns

New localizer outputs now include:

- `query_timecode` — readable time in the query/test video, e.g. `00:02:15.000`
- `matched_reference_timecode` — readable time in the matched reference video
- `raw_pred_latitude`, `raw_pred_longitude` — coordinate before temporal smoothing/holding
- `filter_reason` — why the filter accepted/held/rejected the match

Older CSVs can be upgraded without rerunning localization:

```bat
python -m gps_ex1.tools.add_timecodes ^
  --prediction-csv data/processed/DJI_0010_predictions_temporal.csv ^
  --out data/processed/DJI_0010_predictions_temporal_timecoded.csv
```

## Export match-debug images

After you have the actual `.mp4` videos and extracted reference frames locally:

```bat
python -m gps_ex1.tools.export_match_debug ^
  --prediction-csv data/processed/DJI_0010_predictions_temporal_timecoded.csv ^
  --query-video data/raw/DJI_0010.mp4 ^
  --out-dir data/processed/debug_matches/DJI_0010 ^
  --accepted-only ^
  --max-rows 50
```

Open the generated `.jpg` files. Each one shows the query frame beside the matched reference frame.

## Derive a drone-position path

The assignment target is the center point of the camera image, but a drone-position path is useful in Google Earth because it often looks more physically intuitive.

Use:

```bat
python -m gps_ex1.tools.derive_target_predictions ^
  --prediction-csv data/processed/DJI_0010_predictions_temporal_timecoded.csv ^
  --reference-index data/processed/reference_index.npz ^
  --target drone ^
  --out data/processed/DJI_0010_drone_predictions_timecoded.csv
```

Then export center and drone together:

```bat
python -m gps_ex1.tools.export_kml ^
  --prediction-csv data/processed/DJI_0010_predictions_temporal_timecoded.csv data/processed/DJI_0010_drone_predictions_timecoded.csv ^
  --out data/processed/kml/DJI_0010_timecoded_center_and_drone_paths.kml ^
  --name "DJI_0010 timecoded center and drone paths"
```

## YOLO / semantic masking recommendation

YOLO should not replace visual localization. It should be used as an auxiliary semantic mask.

Recommended use:

1. Detect dynamic objects such as people and vehicles.
2. Mask those pixels before global-descriptor extraction and ORB/feature matching.
3. Optionally record semantic labels for presentation/debugging.

Important limitation:

- Pretrained COCO-style YOLO models are useful for dynamic object masking.
- They are not enough for robust road/building/roof semantic localization, because roads/buildings are better handled by a semantic-segmentation model trained on aerial/urban classes.

Best next improvement after this update:

1. Use the new match-debug images to find whether failures are caused by wrong visual matches.
2. Add stronger reference-candidate continuity: prefer the same reference flight and nearby reference frame index.
3. Add optional YOLO/segmentation masking only after the above is measured.
