# Stage 10.5B: Metric parser fix + rejection diagnostics

This overlay patches the conservative candidate-upgrade stage.

## Why this exists

Stage 10.5 top25 produced many candidate CSV rows, but the final upgrade step printed:

```text
fallback_with_candidate: 526
fallback_candidates_quality_ok: 0
fallback_upgraded: 0
candidate_rejected_quality: 526
```

The likely issue was that candidate quality metrics were read from only generic columns:

```text
good_matches
inlier_ratio
confidence
```

while our pipeline usually writes columns such as:

```text
orb_good_matches
verification_inlier_ratio
retrieval_similarity
homography_inliers
```

So good candidates could be parsed as `good_matches=0` and `inlier_ratio=0`, causing every fallback candidate to be rejected.

## What changed

`src/gps_ex1/tools/stage10_conservative_candidate_upgrade.py` now reads all known aliases:

```text
confidence: confidence, match_confidence, retrieval_similarity, descriptor_similarity, similarity, score
inliers: homography_inliers, inliers, verification_inliers, ransac_inliers, num_inliers
good matches: good_matches, matches, orb_good_matches, verification_good_matches, num_good_matches, total_good_matches
inlier ratio: inlier_ratio, verification_inlier_ratio, homography_inlier_ratio, ransac_inlier_ratio
```

If no ratio column exists, it computes:

```text
homography_inliers / good_matches
```

It also writes optional diagnostics using `--diagnostics-csv`.

## Run without rerunning the expensive top25 search

After extracting this overlay over the Stage 10.3 folder, run:

```bat
cd /d "E:\לימודים\GPS\GPS_EX1_stage9_fixed_grid_regions"
.venv\Scripts\activate
python -m pytest -q
run_stage10_5b_dji0010_recheck_existing_top25_metricfix.bat
```

This reuses existing files like:

```text
DJI_0010_stage10_5_candidate_search_window_001_top25.csv
...
DJI_0010_stage10_5_candidate_search_window_012_top25.csv
```

It produces:

```text
DJI_0010_stage10_5B_candidate_upgrade_top25_metricfix.csv
DJI_0010_stage10_5B_CANDIDATE_UPGRADE_TOP25_METRICFIX_FULL_PATH.kml
DJI_0010_stage10_5B_CANDIDATE_UPGRADE_TOP25_METRICFIX_FULL_PATH_numbered.kml
DJI_0010_stage10_5B_candidate_upgrade_top25_diagnostics.csv
```

## How to judge the result

Good result:

```text
fallback_upgraded: small/moderate number
candidate_rejected_quality: still high
path remains close to Stage 10.3
upgraded debug images look correct
```

Bad result:

```text
fallback_upgraded: huge number
path becomes messy
upgraded debug images look visually wrong
```

If top25 still upgrades 0 candidates after this fix, the diagnostics CSV will show whether the candidates truly have low inliers/matches or whether another field mismatch exists.
