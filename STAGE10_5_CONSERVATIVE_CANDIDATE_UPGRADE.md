# Stage 10.5 Conservative Candidate Upgrade

This overlay replaces the bad Stage 10.4 behavior with a conservative version of the same idea.

## Goal

For frames that are still `path_expected_fallback`, we search the already-created local rerun CSVs and try to upgrade the fallback point **only if** the local candidate is genuinely good.

This is different from Stage 10.4:

- Stage 10.4: used the best local candidate too easily.
- Stage 10.5: uses a local candidate only if it passes candidate-quality checks, cluster-quality checks, and expected-path distance checks.

## Important labels

- `cluster_refined_visual_match` means an existing strong Stage 10.3 visual match was refined by the cluster lookpoint.
- `candidate_upgrade_visual_match` means a fallback frame was upgraded using a good local rerun candidate.
- `path_expected_fallback` means no good visual candidate was found, so the smooth Stage 10.3 path fallback was kept.

## Recommended first test

Run Stage 10.5 on your existing Stage 10.3 top8 reruns:

```bat
run_stage10_5_dji0010_candidate_upgrade_top8.bat
```

Inspect:

```text
data\processed\stage10_3_DJI_0010_every45_tail_rescue\DJI_0010_stage10_5_CANDIDATE_UPGRADE_TOP8_FULL_PATH_numbered.kml
data\processed\debug_matches\DJI_0010_stage10_5_candidate_upgrade_top8_debug
```

## Stronger candidate search

If top8 does not add enough points, run:

```bat
run_stage10_5_dji0010_stronger_search_then_upgrade.bat
```

This makes top25/more-scale local reruns first, then upgrades only accepted/good candidates.

## How to judge results

Do not judge only by the number of upgraded points. Check the debug images.

Good result:

- more `candidate_upgrade_visual_match` points,
- KML remains close to the good Stage 10.3 path,
- debug images show visually correct matches.

Bad result:

- upgraded points jump away from the known path,
- debug images show wrong roofs/roads/trees,
- many weak candidates replace fallback points.

If bad, keep Stage 10.3 or tighten Stage 10.5 thresholds.
