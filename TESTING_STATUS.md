# Testing status

Generated package: GPS_EX1_ready

Checks run before zipping:

```text
python -m pip install -e .
python -m pytest -q
# result: 14 passed

python -m gps_ex1.tools.analyze_predictions \
  --prediction-csv data/processed/DJI_0010_predictions_filtered.csv data/processed/DJI_0011_predictions_filtered.csv
```

Existing filtered output diagnostics:

```text
DJI_0010_predictions_filtered.csv
rows: 831
points with coordinates: 824
accepted: 176 | held: 648 | rejected: 7
median jump: 0.00 m
max jump: 198.63 m
jumps >200m: 0

DJI_0011_predictions_filtered.csv
rows: 360
points with coordinates: 360
accepted: 157 | held: 203 | rejected: 0
median jump: 0.00 m
max jump: 171.23 m
jumps >200m: 0
```

Notes:

- The baseline works without DINOv2/LightGlue.
- DINOv2/LightGlue mode is hooked up behind optional CLI flags and requires `requirements-modern.txt`.
- Large video files are not included; only SRTs, URL files, existing CSV/KML/debug outputs, code, tests, and docs are included.
