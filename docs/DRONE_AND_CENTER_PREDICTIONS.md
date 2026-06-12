# Drone-position and camera-center prediction outputs

The online localizer can output two different coordinates for each query frame:

1. `center` — the estimated coordinate of the center pixel / where the camera is looking.
2. `drone` — the estimated coordinate of the drone itself.

The assignment mainly asks for the center-point coordinate, but the drone-position path is useful as a bonus visualization layer in Google Earth.

## Predict center point from video

```bat
python -m gps_ex1.pipeline.localize_video --video data/raw/DJI_0010.mp4 --reference-index data/processed/reference_index.npz --out data/processed/DJI_0010_center_predictions.csv --every-n-frames 30 --top-k 10 --prediction-target center
```

## Predict drone position from video

```bat
python -m gps_ex1.pipeline.localize_video --video data/raw/DJI_0010.mp4 --reference-index data/processed/reference_index.npz --out data/processed/DJI_0010_drone_predictions.csv --every-n-frames 30 --top-k 10 --prediction-target drone
```

## Faster option: convert an existing prediction CSV

The visual match is the same for `center` and `drone`; only the coordinate taken from the matched reference frame changes.
If you already created a center prediction CSV, convert it without rerunning the video matcher:

```bat
python -m gps_ex1.tools.change_prediction_target --prediction-csv data/processed/DJI_0010_predictions.csv --reference-index data/processed/reference_index.npz --target drone --out data/processed/DJI_0010_drone_predictions.csv
```

## Export both paths to one KML

```bat
python -m gps_ex1.tools.export_kml --prediction-csv data/processed/DJI_0010_predictions.csv data/processed/DJI_0010_drone_predictions.csv --out data/processed/kml/DJI_0010_center_and_drone_paths.kml --name "DJI_0010 predicted center and drone paths"
```

Open the `.kml` in Google Earth.
