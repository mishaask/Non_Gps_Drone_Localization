@echo off
cd /d "E:\לימודים\GPS\GPS_EX1_stage9_fixed_grid_regions"
call .venv\Scripts\activate

python -m gps_ex1.tools.stage10_merge_path_reruns --base-csv data/processed/DJI_0010_every45_stage10_firstpass_top4_nomask.csv --plan-csv data/processed/stage10_3_DJI_0010_every45_tail_rescue/stage10_path_rerun_plan.csv --rerun-dir data/processed/stage10_3_DJI_0010_every45_tail_rescue --rerun-glob "DJI_0010_stage10_3_rerun_path_window_*_top8.csv" --out data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_3_merged_tail_rescue_top8.csv --max-distance-from-expected-m 140 --min-candidate-confidence 6.0 --clear-untrusted-matches

python -m gps_ex1.tools.stage10_cluster_lookpoint_refine --prediction-csv data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_3_merged_tail_rescue_top8.csv --query-video data/raw/DJI_0010.mp4 --reference-index data/processed/reference_index_anyloc_gem_masked.npz --out data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_3_cluster_lookpoint_refined.csv --project-root . --debug-dir data/processed/debug_matches/DJI_0010_stage10_3_cluster_lookpoint_refine_debug --accepted-only --match-backend lightglue --min-inliers 10 --replace-filtered

python -m gps_ex1.tools.export_kml --prediction-csv data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_3_cluster_lookpoint_refined.csv --out data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_3_CLUSTER_LOOKPOINT_FULL_PATH.kml --use-filtered --name "DJI_0010 Stage 10.3 cluster-lookpoint refined full path"

python -m gps_ex1.tools.export_feature_match_debug --prediction-csv data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_3_cluster_lookpoint_refined.csv --query-video data/raw/DJI_0010.mp4 --out-dir data/processed/debug_matches/DJI_0010_stage10_3_FINAL_cluster_refined_feature_debug --project-root . --max-rows 1000 --accepted-only --query-scale-mode selected --query-scale-fill blur --match-backend lightglue --header-alpha 0.15 --show-outliers

explorer data\processed\stage10_3_DJI_0010_every45_tail_rescue
explorer data\processed\debug_matches\DJI_0010_stage10_3_cluster_lookpoint_refine_debug
