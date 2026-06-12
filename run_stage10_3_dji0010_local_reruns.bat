@echo off
cd /d "E:\לימודים\GPS\GPS_EX1_stage9_fixed_grid_regions"
call .venv\Scripts\activate

for %%W in (001 002 003 004 005 006 007 008 009 010 011 012) do (
  if exist "data\processed\stage10_3_DJI_0010_every45_tail_rescue\rerun_path_window_%%W_frames.txt" (
    echo Running DJI_0010 Stage 10.3 local rerun window %%W
    python -m gps_ex1.pipeline.localize_video --video data/raw/DJI_0010.mp4 --reference-index data/processed/stage10_3_DJI_0010_every45_tail_rescue/local_reference_index_path_window_%%W.npz --out data/processed/stage10_3_DJI_0010_every45_tail_rescue/DJI_0010_stage10_3_rerun_path_window_%%W_top8.csv --query-frame-list data/processed/stage10_3_DJI_0010_every45_tail_rescue/rerun_path_window_%%W_frames.txt --top-k 8 --query-scales 0.2,0.3 --query-scale-fill blur --prediction-target center --descriptor-backend anyloc-gem --descriptor-image-size 322 --verification-backend lightglue --min-inliers 10 --min-good-matches 20 --min-inlier-ratio 0.25 --max-reference-inlier-area-frac 0.10 --max-reference-inlier-width-frac 0.35 --max-reference-inlier-height-frac 0.35
  )
)
