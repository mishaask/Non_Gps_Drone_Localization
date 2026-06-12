# Stage 10.1 — Dominant seed path filtering

Stage 10.1 fixes the problem where the Stage 10 seed KML could still contain several disconnected accepted-match islands.

The old Stage 10 filter removed isolated green -> red -> green teleports, but it could still keep a longer wrong cluster. Stage 10.1 adds a dominant-component filter before local rerun planning:

1. Build all first-pass accepted seed candidates.
2. Connect candidates only if the distance and speed between them are physically plausible.
3. Find connected seed components.
4. Keep only the largest / highest-confidence component as the seed path.
5. Reject the other seed islands before interpolation and local reruns.

Recommended planner command:

```bat
python -m gps_ex1.tools.stage10_path_guided_planner ^
  --prediction-csv data/processed/DJI_0011_every45_stage10_firstpass_top4_nomask.csv ^
  --reference-index data/processed/reference_index_anyloc_gem_masked.npz ^
  --out-dir data/processed/stage10_1_every45_dominant_path ^
  --seed-min-confidence 6.0 ^
  --seed-min-inliers 10 ^
  --seed-min-good-matches 20 ^
  --seed-max-ref-area-frac 0.12 ^
  --seed-max-ref-width-frac 0.45 ^
  --seed-max-ref-height-frac 0.45 ^
  --teleport-threshold-m 100 ^
  --bridge-threshold-m 70 ^
  --max-anchor-speed-mps 20 ^
  --seed-context-window 10 ^
  --dominant-edge-max-distance-m 160 ^
  --dominant-edge-max-speed-mps 25 ^
  --dominant-min-component-size 4 ^
  --rerun-mode all-non-seed ^
  --local-radius-m 120 ^
  --max-local-radius-m 260 ^
  --min-local-references 100 ^
  --window-spacing-m 120 ^
  --max-frames-per-window 45
```

To disable the new behavior for comparison, add:

```bat
--no-dominant-seed-component
```

The planner now writes extra columns in `stage10_seed_path.csv` and `stage10_path_rerun_plan.csv`:

- `dominant_component_id`
- `dominant_support_count`
- `seed_reject_reason`

A good run should show fewer trusted seed anchors than the raw first pass when Google Earth shows disconnected islands.
