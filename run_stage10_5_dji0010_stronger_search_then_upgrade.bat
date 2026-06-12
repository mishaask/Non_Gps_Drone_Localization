@echo off
REM Run this from E:\לימודים\GPS\GPS_EX1_stage9_fixed_grid_regions
REM This script intentionally does NOT cd into the Hebrew path, to avoid codepage/path corruption.
call .venv\Scripts\activate

for %%W in (001 002 003 004 005 006 007 008 009 010 011 012) do (
  if exist "data\processed\stage10_3_DJI_0010_every45_tail_rescue\rerun_path_window_%%W_frames.txt" (
    echo Running Stage 10.5 stronger candidate search window %%W
    python -m gps_ex1.pipeline.localize_video --video data/raw/DJI_0010.mp4 --reference-index data/processed/stage10_3_DJI_0010_every45_tail_rescue/local_reference_index_path_window_%%W.npz --out data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_5_candidate_search_window_%%W_top25.csv --query-frame-list data/processed/stage10_3_DJI_0010_every45_tail_rescue/rerun_path_window_%%W_frames.txt --top-k 25 --query-scales 0.15,0.2,0.3,0.4 --query-scale-fill blur --prediction-target center --descriptor-backend anyloc-gem --descriptor-image-size 322 --verification-backend lightglue --min-inliers 8 --min-good-matches 15 --min-inlier-ratio 0.18 --max-reference-inlier-area-frac 0.14 --max-reference-inlier-width-frac 0.45 --max-reference-inlier-height-frac 0.45
  )
)

python -m gps_ex1.tools.stage10_conservative_candidate_upgrade --merged-csv data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_3_merged_tail_rescue_top8.csv --rerun-dir data/processed/stage10_3_DJI_0010_every45_tail_rescue --rerun-glob "DJI_0010_stage10_5_candidate_search_window_*_top25.csv" --query-video data/raw/DJI_0010.mp4 --reference-index data/processed/reference_index_anyloc_gem_masked.npz --out data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_5_candidate_upgrade_top25.csv --project-root . --debug-dir data/processed/debug_matches/DJI_0010_stage10_5_candidate_upgrade_top25_debug --match-backend lightglue --min-cluster-inliers 8 --max-distance-from-expected-m 140 --min-candidate-confidence 5.0 --min-candidate-inliers 8 --min-candidate-good-matches 15 --min-candidate-inlier-ratio 0.18 --max-debug-images 250

python -m gps_ex1.tools.export_kml --prediction-csv data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_5_candidate_upgrade_top25.csv --out data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_5_CANDIDATE_UPGRADE_TOP25_FULL_PATH.kml --use-filtered --name "DJI_0010 Stage 10.5 candidate-upgrade top25 full path"

python make_numbered_kml.py --csv data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_5_candidate_upgrade_top25.csv --out data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_5_CANDIDATE_UPGRADE_TOP25_FULL_PATH_numbered.kml --name "DJI_0010 Stage 10.5 candidate-upgrade top25 numbered path"

explorer data\processed\stage10_3_DJI_0010_every45_tail_rescue
explorer data\processed\debug_matches\DJI_0010_stage10_5_candidate_upgrade_top25_debug
