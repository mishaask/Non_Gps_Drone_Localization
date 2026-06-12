# Stage 10.5C — confidence scale fix

## The bug (why 10.5 and 10.5B rejected all 526 candidates)

`stage10_conservative_candidate_upgrade` checked candidate confidence FIRST,
before inliers/good-matches. Its confidence alias list included
`retrieval_similarity`, which is a cosine similarity in [0, 1] — but the gate
`--min-candidate-confidence 5.0` is on the planner scale (row_confidence,
typically ~10-25 for good matches). localize_video CSVs have no explicit
`confidence` column, so EVERY candidate was read as confidence ≈ 0.2-0.9,
failed `< 5.0`, and was logged as `candidate_rejected_quality`
(reason `candidate_confidence_lt_5`). Even the ~40 genuinely accepted
top25 window matches died at this gate.

The 10.5B "metricfix" overlay fixed inliers/good-matches/ratio aliases —
the wrong metric. The confidence gate was the killer.

## The fix

- Similarity columns are no longer treated as confidence.
- When no explicit confidence column exists, the tool now computes the same
  composite `row_confidence()` score used by the Stage 10 planner, seed
  filtering, and merge tools — so `--min-candidate-confidence` finally means
  the same thing everywhere in the pipeline.
- Diagnostics CSV now shows `candidate_confidence_source = computed_row_confidence`.

## Changed files
- src/gps_ex1/tools/stage10_conservative_candidate_upgrade.py
- tests/test_stage10_candidate_metric_parser_fix.py (regression tests)

## How to run (reuses your existing top25 CSVs — no expensive re-search)

1. Extract this zip over:
   E:\לימודים\GPS\GPS_EX1_stage9_fixed_grid_regions
   (choose Replace)

2. cd /d "E:\לימודים\GPS\GPS_EX1_stage9_fixed_grid_regions"
   .venv\Scripts\activate
   python -m pytest -q
   (expect all tests passing, now ~39)

3. run_stage10_5b_dji0010_recheck_existing_top25_metricfix.bat

4. In the summary, fallback_candidates_quality_ok should now be > 0
   (roughly the number of accepted window matches, ~35-45) and
   fallback_upgraded should finally be non-zero. Some quality-ok candidates
   may still be rejected by the cluster refine or the 140 m distance gate —
   that is the conservative safety working as intended, and those counts
   appear separately as candidate_rejected_cluster / candidate_rejected_distance.
