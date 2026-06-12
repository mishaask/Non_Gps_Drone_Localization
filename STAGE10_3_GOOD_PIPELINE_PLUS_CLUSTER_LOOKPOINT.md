# Stage 10.3 good pipeline + cluster-lookpoint refinement

This overlay keeps the good pipeline logic only:

1. Stage 10.3 trusted path-guided pipeline:
   - dominant seed path
   - seed support validation
   - suffix / landing-tail rescue
   - path-guided local reruns
   - merge with expected-path fallback
2. A safe cluster-lookpoint refinement tool:
   - `gps_ex1.tools.stage10_cluster_lookpoint_refine`

The cluster tool is the safe version of the Stage 11 idea. It must be run **after** Stage 10.3 merge. It does not run the old weak VisDrone/basic first-pass pipeline and it does not pick new global reference frames. It only takes accepted visual matches from the good Stage 10.3 merged CSV and moves the output coordinate toward the RANSAC inlier cluster / visible landmark in the already-selected reference frame.

## Correct order

```bat
cd /d "E:\לימודים\GPS\GPS_EX1_stage9_fixed_grid_regions"
.venv\Scripts\activate
```

Run local reruns and merge as before. After you have:

```text
data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_3_merged_tail_rescue_top8.csv
```

run cluster-lookpoint refinement:

```bat
python -m gps_ex1.tools.stage10_cluster_lookpoint_refine ^
  --prediction-csv data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_3_merged_tail_rescue_top8.csv ^
  --query-video data/raw/DJI_0010.mp4 ^
  --reference-index data/processed/reference_index_anyloc_gem_masked.npz ^
  --out data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_3_cluster_lookpoint_refined.csv ^
  --project-root . ^
  --debug-dir data/processed/debug_matches/DJI_0010_stage10_3_cluster_lookpoint_refine_debug ^
  --accepted-only ^
  --match-backend lightglue ^
  --min-inliers 10 ^
  --replace-filtered
```

Then export KML from the cluster-refined CSV:

```bat
python -m gps_ex1.tools.export_kml ^
  --prediction-csv data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_3_cluster_lookpoint_refined.csv ^
  --out data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_3_CLUSTER_LOOKPOINT_FULL_PATH.kml ^
  --use-filtered ^
  --name "DJI_0010 Stage 10.3 cluster-lookpoint refined full path"
```

Open:

```bat
explorer data\processed\stage10_3_DJI_0010_every45_tail_rescue
explorer data\processed\debug_matches\DJI_0010_stage10_3_cluster_lookpoint_refine_debug
```

## Important warning

Do not run the old Stage 11 on:

```text
DJI_0010_predictions_visdrone_masked_fast.csv
```

That is the weak old pipeline and can produce a wrong clustered path. Use the Stage 10.3 merged CSV as input instead.

## Stage 10.4 addition

This overlay also includes `stage10_local_best_cluster_refine.py`, which tries to upgrade fallback rows using the best local candidate from Stage 10.3 rerun windows. Run:

- `run_stage10_4_dji0010_local_best_from_top8.bat` to test existing top8 rerun candidates.
- `run_stage10_4_dji0010_stronger_top25_search.bat` to run a stronger top25/more-scale local search first.

Use the debug folders to inspect the new local-best guesses before trusting the KML.
