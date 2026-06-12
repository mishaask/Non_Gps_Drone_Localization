# Stage 9.1 — Path Consistency Filter

Stage 9 fixed the search-space problem by rerunning bad frames only inside fixed map-grid regions.  Stage 9.1 fixes the next problem: a visually plausible single-frame match can still jump from the real area to a visually similar wrong area and then jump back.

The new tool is:

```bat
python -m gps_ex1.tools.stage9_path_consistency_filter
```

It reads a merged Stage 9 CSV, assigns every prediction to the nearest fixed grid cell, then rejects isolated accepted predictions that are far from nearby accepted predictions.

## Recommended command

```bat
python -m gps_ex1.tools.stage9_path_consistency_filter ^
  --prediction-csv data/processed/stage9_every45_grid15/stage9_merged_grid_008_013_top8.csv ^
  --grid-regions-csv data/processed/stage9_every45_grid15/stage9_grid_regions.csv ^
  --out data/processed/stage9_every45_grid15/stage9_merged_grid_008_013_top8_path_filtered.csv ^
  --teleport-threshold-m 160 ^
  --bridge-threshold-m 120 ^
  --context-window 10 ^
  --support-window 8 ^
  --min-same-grid-support 1 ^
  --filtered-fill interpolate ^
  --override-filter-accepted
```

Then export only path-trusted accepted debug images:

```bat
python -m gps_ex1.tools.export_feature_match_debug ^
  --prediction-csv data/processed/stage9_every45_grid15/stage9_merged_grid_008_013_top8_path_filtered.csv ^
  --query-video data/raw/DJI_0011.mp4 ^
  --out-dir data/processed/debug_matches/DJI_0011_every45_stage9_1_path_filtered_accepted ^
  --project-root . ^
  --max-rows 1000 ^
  --accepted-only ^
  --query-scale-mode selected ^
  --query-scale-fill blur ^
  --match-backend lightglue ^
  --header-alpha 0.15 ^
  --show-outliers
```

For Google Earth, export the smoothed/interpolated path:

```bat
python -m gps_ex1.tools.export_kml ^
  --prediction-csv data/processed/stage9_every45_grid15/stage9_merged_grid_008_013_top8_path_filtered.csv ^
  --out data/processed/stage9_every45_grid15/stage9_1_path_filtered_interpolated.kml ^
  --use-filtered ^
  --name "DJI_0011 Stage 9.1 path-filtered interpolated center path"
```

And export only trusted raw visual points:

```bat
python -m gps_ex1.tools.export_kml ^
  --prediction-csv data/processed/stage9_every45_grid15/stage9_merged_grid_008_013_top8_path_filtered.csv ^
  --out data/processed/stage9_every45_grid15/stage9_1_path_filtered_accepted_only.kml ^
  --accepted-only ^
  --name "DJI_0011 Stage 9.1 accepted-only trusted visual points"
```

## Optional manual region lock

If visual inspection shows the drone should stay in specific grid cells, use:

```bat
--trusted-grid-regions 7,8
```

or reject a known confusing cell:

```bat
--banned-grid-regions 13
```

Use manual locks only for debugging/evaluation, not as a general automatic solution.
