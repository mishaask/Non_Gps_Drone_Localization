@echo off
REM Run this from E:\לימודים\GPS\GPS_EX1_stage9_fixed_grid_regions
REM This script intentionally does NOT cd into the Hebrew path, to avoid codepage/path corruption.
call .venv\Scripts\activate

python -m gps_ex1.tools.stage10_conservative_candidate_upgrade --merged-csv data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_3_merged_tail_rescue_top8.csv --rerun-dir data/processed/stage10_3_DJI_0010_every45_tail_rescue --rerun-glob "DJI_0010_stage10_3_rerun_path_window_*_top8.csv" --query-video data/raw/DJI_0010.mp4 --reference-index data/processed/reference_index_anyloc_gem_masked.npz --out data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_5_candidate_upgrade_top8.csv --project-root . --debug-dir data/processed/debug_matches/DJI_0010_stage10_5_candidate_upgrade_top8_debug --match-backend lightglue --min-cluster-inliers 8 --max-distance-from-expected-m 140 --min-candidate-confidence 5.0 --min-candidate-inliers 8 --min-candidate-good-matches 15 --min-candidate-inlier-ratio 0.18 --max-debug-images 150

python -m gps_ex1.tools.export_kml --prediction-csv data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_5_candidate_upgrade_top8.csv --out data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_5_CANDIDATE_UPGRADE_TOP8_FULL_PATH.kml --use-filtered --name "DJI_0010 Stage 10.5 candidate-upgrade top8 full path"

python make_numbered_kml.py --csv data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_5_candidate_upgrade_top8.csv --out data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_5_CANDIDATE_UPGRADE_TOP8_FULL_PATH_numbered.kml --name "DJI_0010 Stage 10.5 candidate-upgrade top8 numbered path"

explorer data\processed\stage10_3_DJI_0010_every45_tail_rescue
explorer data\processed\debug_matches\DJI_0010_stage10_5_candidate_upgrade_top8_debug
