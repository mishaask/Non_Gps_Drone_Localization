# Match quality diagnosis and next improvement

The side-by-side debug images show that the current baseline is over-accepting weak matches.
The important fix is not to make the KML smoother. The important fix is to reject bad visual evidence before it becomes a coordinate.

## Why this happens

The current baseline uses a simple global descriptor and ORB/homography reranking. This can be fooled by repetitive trees, parking lots, bright roads, shadows, and roofs. A homography with 6-10 inliers is not enough proof that a low-flying query frame and a high-altitude aerial reference frame are the same place.

## New tool

`gps_ex1.tools.audit_match_quality` reads an existing prediction CSV and checks whether each query/reference pair is visually compatible using:

- stricter minimum inliers
- stricter minimum ORB good matches
- simple sky/horizon mismatch rejection
- lightweight scene-signature distance

It does not use GNSS.

Example:

```bat
python -m gps_ex1.tools.audit_match_quality ^
  --prediction-csv data/processed/DJI_0010_predictions_temporal.csv ^
  --query-video data/raw/DJI_0010.mp4 ^
  --out data/processed/DJI_0010_predictions_temporal_audited.csv ^
  --min-inliers 18 ^
  --min-good-matches 25
```

If this rejects most matches, that is useful: it means the data/viewpoint overlap is not strong enough for the simple baseline and the report should say so honestly.

## Recommended real improvement path

1. Stop accepting weak matches.
2. Use match debug images as qualitative evidence.
3. Add a stronger image retrieval method such as DINOv2/CosPlace/NetVLAD.
4. Replace ORB with SuperPoint+LightGlue or LoFTR.
5. Add optional YOLO/segmentation masking only as a support layer.
6. For a stronger final project, build a geo-referenced 2D/3D reference map with COLMAP/hloc/OpenDroneMap-style tooling and localize query frames against that map.
