# Stage 9 — fixed map-grid regions

Stage 9 fixes the Stage 8 problem where regions were built from first-pass predictions. Instead, it divides the reference flight footprint itself into a fixed grid, usually 5 columns × 3 rows = 15 regions.

Pipeline:

1. Run a fast first pass on DJI_0011.
2. Build fixed grid regions from the reference index coordinates.
3. Use only high-confidence accepted first-pass frames as temporal anchors.
4. Mark rejected / weak / suspicious jump frames.
5. Rerun those frames only against the grid cells supported by nearby anchors and the most popular strong cells.
6. Safely merge only accepted, score-improving, context-approved rerun results.
7. Export final debug images for every sampled frame.

Important outputs:

- `stage9_grid_regions.csv` — the fixed grid cells and reference counts.
- `stage9_reference_frame_regions.csv` — which reference frame belongs to which cell.
- `stage9_grid_context_rerun_plan.csv` — per-query-frame temporal context and candidate grid cells.
- `rerun_grid_region_XXX_frames.txt` — frames to rerun against a specific grid cell.
- `local_reference_index_grid_XXX.npz` — reference subset for each grid cell, with halo overlap.
- `stage9_merged_grid_top8.csv` — final merged predictions.

Recommended first experiment:

- DJI_0011 every 45 frames
- first pass `--top-k 4`
- rerun pass `--top-k 8`
- grid: `5 × 3`
- halo: `100m`

The grid uses local East/North meters internally, not raw latitude/longitude degrees, so cells are roughly meter-consistent over the Ariel University footprint.
