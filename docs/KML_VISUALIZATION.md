# KML Visualization

This project can export paths to KML so they can be opened in Google Earth or imported into Google My Maps.

## Predicted path from a GNSS-denied test video

```bat
python -m gps_ex1.tools.export_kml --prediction-csv data/processed/DJI_0010_predictions.csv --out data/processed/kml/DJI_0010_predicted_center_path.kml --name "DJI_0010 predicted center path"
```

A less noisy version can keep only stronger geometric matches:

```bat
python -m gps_ex1.tools.export_kml --prediction-csv data/processed/DJI_0010_predictions.csv --out data/processed/kml/DJI_0010_predicted_center_path_filtered.kml --name "DJI_0010 predicted center path filtered" --min-inliers 8
```

## GNSS paths from reference SRT files

```bat
python -m gps_ex1.tools.export_kml --srt data/raw/DJI_0006.SRT data/raw/DJI_0007.SRT data/raw/DJI_0008.SRT data/raw/DJI_0009.SRT --out data/processed/kml/reference_gnss_drone_paths.kml --name "Reference GNSS drone paths" --stride 30
```

## Estimated center-point paths stored in the reference index

```bat
python -m gps_ex1.tools.export_kml --reference-index data/processed/reference_index.npz --index-target center --out data/processed/kml/reference_estimated_center_paths.kml --name "Reference estimated center paths"
```

## Drone paths stored in the reference index

```bat
python -m gps_ex1.tools.export_kml --reference-index data/processed/reference_index.npz --index-target drone --out data/processed/kml/reference_index_drone_paths.kml --name "Reference indexed drone paths"
```

## Viewing

Use Google Earth for the simplest local workflow. For Google Maps, open Google My Maps, create a map, and import the KML file into a layer.
