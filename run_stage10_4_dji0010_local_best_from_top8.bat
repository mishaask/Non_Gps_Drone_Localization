@echo off
cd /d "E:\לימודים\GPS\GPS_EX1_stage9_fixed_grid_regions"
call .venv\Scripts\activate

python -m gps_ex1.tools.stage10_local_best_cluster_refine --merged-csv data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_3_merged_tail_rescue_top8.csv --rerun-dir data/processed/stage10_3_DJI_0010_every45_tail_rescue --rerun-glob "DJI_0010_stage10_3_rerun_path_window_*_top8.csv" --query-video data/raw/DJI_0010.mp4 --reference-index data/processed/reference_index_anyloc_gem_masked.npz --out data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_4_local_best_cluster_top8.csv --project-root . --debug-dir data/processed/debug_matches/DJI_0010_stage10_4_local_best_cluster_top8_debug --match-backend lightglue --min-inliers 4 --max-distance-from-expected-m 180 --max-debug-images 300

python -m gps_ex1.tools.export_kml --prediction-csv data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_4_local_best_cluster_top8.csv --out data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_4_LOCAL_BEST_CLUSTER_TOP8_FULL_PATH.kml --use-filtered --name "DJI_0010 Stage 10.4 local-best cluster top8 full path"

python make_numbered_kml.py --csv data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_4_local_best_cluster_top8.csv --out data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_4_LOCAL_BEST_CLUSTER_TOP8_FULL_PATH_numbered.kml --name "DJI_0010 Stage 10.4 local-best cluster top8 numbered path"

explorer data\processed\stage10_3_DJI_0010_every45_tail_rescue
explorer data\processed\debug_matches\DJI_0010_stage10_4_local_best_cluster_top8_debug
