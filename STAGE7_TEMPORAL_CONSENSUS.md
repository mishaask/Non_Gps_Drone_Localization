# Stage 7 — Temporal Consensus + Local Rerun Preparation

Stage 7 is intentionally debug-friendly. It does **not** hide the raw Stage 6
predictions. It reads a Stage 6 CSV, finds high-confidence anchor predictions,
groups them into GPS regions, flags suspicious jumps/gaps, and prepares local
reference indexes + frame lists for targeted reruns.

## What Stage 7 adds

- `gps_ex1.tools.stage7_temporal_consensus`
  - Finds reliable anchors.
  - Clusters anchors into popular GPS regions.
  - Flags rejected frames inside a good region context.
  - Flags accepted frames that jump far away from neighboring anchors.
  - Writes `stage7_temporal_corrected.csv` and `stage7_regions.csv`.
  - Optionally writes local reference indexes per popular region.
  - Writes frame lists for suspicious/gap frames that should be rerun locally.

- `gps_ex1.tools.stage7_merge_reruns`
  - Merges one or more local rerun CSVs back into the Stage 7 CSV.
  - Replaces a frame only if the local rerun is accepted and improves the score.

## Step 7A — Temporal consensus analysis

Run this after your Stage 6 top-50 CSV exists:

```bat
python -m gps_ex1.tools.stage7_temporal_consensus ^
  --prediction-csv data/processed/DJI_0011_stage6_selected_frames_lightglue_cluster_top50.csv ^
  --reference-index data/processed/reference_index_anyloc_gem_masked.npz ^
  --out-dir data/processed/stage7_temporal_top50 ^
  --region-radius-m 120 ^
  --neighbor-window 8 ^
  --jump-threshold-m 180 ^
  --anchor-min-confidence 6.0 ^
  --anchor-min-inliers 10 ^
  --anchor-min-good-matches 20 ^
  --anchor-max-ref-area-frac 0.12 ^
  --anchor-max-ref-width-frac 0.45 ^
  --anchor-max-ref-height-frac 0.45 ^
  --fill-gaps ^
  --write-local-reference-indexes ^
  --local-reference-radius-m 180 ^
  --min-local-references 120 ^
  --max-regions 5
```

Outputs:

```text
data/processed/stage7_temporal_top50/stage7_temporal_corrected.csv
data/processed/stage7_temporal_top50/stage7_regions.csv
data/processed/stage7_temporal_top50/stage7_suspicious_or_gap_frames.txt
data/processed/stage7_temporal_top50/local_reference_index_region_001.npz
data/processed/stage7_temporal_top50/rerun_frames_region_001.txt
data/processed/stage7_temporal_top50/README_run_local_reruns.bat.txt
```

## Step 7B — Local rerun for suspicious/gap frames

Open this generated file:

```bat
notepad data\processed\stage7_temporal_top50\README_run_local_reruns.bat.txt
```

Copy the region command you want to test into CMD. The most important one is usually
`region_001`, because it has the most high-confidence anchors.

The generated rerun command uses only reference frames near the likely region and
only processes the suspicious/gap query frames for that region.

## Step 7C — Merge rerun results back into Stage 7

Example for region 001:

```bat
python -m gps_ex1.tools.stage7_merge_reruns ^
  --base-csv data/processed/stage7_temporal_top50/stage7_temporal_corrected.csv ^
  --rerun-csv data/processed/stage7_temporal_top50/stage7_rerun_region_001.csv ^
  --out data/processed/stage7_temporal_top50/stage7_merged_after_region001.csv
```

If you rerun several regions:

```bat
python -m gps_ex1.tools.stage7_merge_reruns ^
  --base-csv data/processed/stage7_temporal_top50/stage7_temporal_corrected.csv ^
  --rerun-csv ^
  data/processed/stage7_temporal_top50/stage7_rerun_region_001.csv ^
  data/processed/stage7_temporal_top50/stage7_rerun_region_002.csv ^
  --out data/processed/stage7_temporal_top50/stage7_merged_after_local_reruns.csv
```

## Debug notes

This stage is not final production smoothing. It is meant to show exactly what
happened:

- raw per-frame prediction stays in the original columns
- Stage 7 status is written in `stage7_temporal_status`
- gap/suspicious correction is written in `stage7_corrected_latitude` and `stage7_corrected_longitude`
- local rerun replacements are marked by `stage7_local_rerun_replaced`

Good statuses:

```text
anchor_consistent
accepted_near_consensus
```

Suspicious/debug statuses:

```text
rejected_inside_consensus_context
suspicious_jump_from_consensus
low_confidence_consensus_context
no_consensus_region
```
