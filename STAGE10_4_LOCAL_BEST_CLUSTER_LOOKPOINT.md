# Stage 10.4 Local-Best Cluster Lookpoint

This overlay keeps the good Stage 10.3 pipeline and adds the next idea:

> For frames that Stage 10.3 marked as `path_expected_fallback`, use the best local rerun candidate from the expected path area, then compute the lookpoint from the LightGlue/RANSAC inlier cluster.

This is intentionally not a global search. It only uses candidates from the local path windows produced by Stage 10.3, so it is much safer than the old Stage 11 global/VisDrone approach.

## Output labels

- `cluster_refined_visual_match`: already accepted Stage 10.3 visual match refined to the inlier cluster.
- `local_best_cluster_guess`: fallback point upgraded using the best local candidate cluster.
- `path_expected_fallback`: no usable local-best candidate; keep the Stage 10.3 expected path.

## Recommended use

Run after Stage 10.3 local reruns and merge. First test with existing top8 rerun CSVs. If it helps, try the stronger top25/more-scales rerun.
