@echo off
cd /d "E:\לימודים\GPS\GPS_EX1_stage9_fixed_grid_regions"
call .venv\Scripts\activate

for %%W in (001 002 003 004 005 006 007 008 009 010 011 012) do (
  if exist "data\processed\stage10_3_DJI_0010_every45_tail_rescue\rerun_path_window_%%W_frames.txt" (
    echo Running stronger local-best search window %%W
    python -m gps_ex1.pipeline.localize_video --video data/raw/DJI_0010.mp4 --reference-index data/processed/stage10_3_DJI_0010_every45_tail_rescue/local_reference_index_path_window_%%W.npz --out data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_4_rerun_path_window_%%W_top25_more_scales.csv --query-frame-list data/processed/stage10_3_DJI_0010_every45_tail_rescue/rerun_path_window_%%W_frames.txt --top-k 25 --query-scales 0.15,0.2,0.3,0.4 --query-scale-fill blur --prediction-target center --descriptor-backend anyloc-gem --descriptor-image-size 322 --verification-backend lightglue --min-inliers 8 --min-good-matches 15 --min-inlier-ratio 0.18 --max-reference-inlier-area-frac 0.14 --max-reference-inlier-width-frac 0.45 --max-reference-inlier-height-frac 0.45
  )
)

python -m gps_ex1.tools.stage10_merge_path_reruns --base-csv data/processed/DJI_0010_every45_stage10_firstpass_top4_nomask.csv --plan-csv data/processed/stage10_3_DJI_0010_every45_tail_rescue/stage10_path_rerun_plan.csv --rerun-dir data/processed/stage10_3_DJI_0010_every45_tail_rescue --rerun-glob "DJI_0010_stage10_4_rerun_path_window_*_top25_more_scales.csv" --out data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_4_merged_top25_more_scales.csv --max-distance-from-expected-m 180 --min-candidate-confidence 5.0 --clear-untrusted-matches

python -m gps_ex1.tools.stage10_local_best_cluster_refine --merged-csv data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_4_merged_top25_more_scales.csv --rerun-dir data/processed/stage10_3_DJI_0010_every45_tail_rescue --rerun-glob "DJI_0010_stage10_4_rerun_path_window_*_top25_more_scales.csv" --query-video data/raw/DJI_0010.mp4 --reference-index data/processed/reference_index_anyloc_gem_masked.npz --out data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_4_local_best_cluster_top25_more_scales.csv --project-root . --debug-dir data/processed/debug_matches/DJI_0010_stage10_4_local_best_cluster_top25_debug --match-backend lightglue --min-inliers 4 --max-distance-from-expected-m 180 --max-debug-images 400

python -m gps_ex1.tools.export_kml --prediction-csv data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_4_local_best_cluster_top25_more_scales.csv --out data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_4_LOCAL_BEST_CLUSTER_TOP25_FULL_PATH.kml --use-filtered --name "DJI_0010 Stage 10.4 local-best cluster top25 full path"

python make_numbered_kml.py --csv data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_4_local_best_cluster_top25_more_scales.csv --out data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_4_LOCAL_BEST_CLUSTER_TOP25_FULL_PATH_numbered.kml --name "DJI_0010 Stage 10.4 local-best cluster top25 numbered path"

explorer data\processed\stage10_3_DJI_0010_every45_tail_rescue
explorer data\processed\debug_matches\DJI_0010_stage10_4_local_best_cluster_top25_debug
