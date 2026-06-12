"""Compare prediction CSVs from different descriptor backends.

This is useful for Stage 3 experiments where we run the same query video with
basic, DINOv2, and AnyLoc-style descriptors and want a quick quantitative view
before opening KML/debug images.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import mean, median

from gps_ex1.geometry.geo import haversine_m


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int_or_zero(value: str | None) -> int:
    parsed = _float_or_none(value)
    return 0 if parsed is None else int(parsed)


def summarize_csv(path: Path) -> dict[str, object]:
    rows = 0
    accepted = 0
    points: list[tuple[float, float]] = []
    inliers: list[int] = []
    good_matches: list[int] = []
    inlier_ratios: list[float] = []
    inside_count = 0
    geometry_ok_count = 0
    reasons: Counter[str] = Counter()
    flights: Counter[str] = Counter()
    query_scales: Counter[str] = Counter()
    reference_bbox_areas: list[float] = []

    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows += 1
            reason = (row.get("filter_reason") or "").strip() or "unknown"
            reasons[reason] += 1
            if (row.get("filter_accepted") or "").strip() == "1":
                accepted += 1
            flight = (row.get("matched_reference_flight") or "").strip()
            if flight:
                flights[flight] += 1
            scale = (row.get("selected_query_scale") or "").strip()
            if scale:
                query_scales[scale] += 1
            lat = _float_or_none(row.get("pred_latitude"))
            lon = _float_or_none(row.get("pred_longitude"))
            if lat is not None and lon is not None:
                points.append((lat, lon))
            inliers.append(_int_or_zero(row.get("homography_inliers")))
            good_matches.append(_int_or_zero(row.get("orb_good_matches")))
            ratio = _float_or_none(row.get("verification_inlier_ratio"))
            if ratio is not None:
                inlier_ratios.append(ratio)
            if (row.get("projected_center_inside") or "").strip() == "1":
                inside_count += 1
            if (row.get("homography_geometry_ok") or "").strip() == "1":
                geometry_ok_count += 1
            ref_bbox_area = _float_or_none(row.get("reference_inlier_bbox_area_frac"))
            if ref_bbox_area is not None:
                reference_bbox_areas.append(ref_bbox_area)

    jumps = [haversine_m(points[i-1][0], points[i-1][1], points[i][0], points[i][1]) for i in range(1, len(points))]
    return {
        "path": str(path),
        "rows": rows,
        "points": len(points),
        "accepted": accepted,
        "accepted_pct": 0.0 if rows == 0 else 100.0 * accepted / rows,
        "median_inliers": 0.0 if not inliers else float(median(inliers)),
        "mean_inliers": 0.0 if not inliers else float(mean(inliers)),
        "median_good_matches": 0.0 if not good_matches else float(median(good_matches)),
        "median_inlier_ratio": 0.0 if not inlier_ratios else float(median(inlier_ratios)),
        "center_inside_pct": 0.0 if rows == 0 else 100.0 * inside_count / rows,
        "geometry_ok_pct": 0.0 if rows == 0 else 100.0 * geometry_ok_count / rows,
        "median_jump_m": 0.0 if not jumps else float(median(jumps)),
        "max_jump_m": 0.0 if not jumps else float(max(jumps)),
        "top_reasons": reasons.most_common(6),
        "top_flights": flights.most_common(6),
        "median_ref_bbox_area": 0.0 if not reference_bbox_areas else float(median(reference_bbox_areas)),
        "top_query_scales": query_scales.most_common(8),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare localization CSVs from basic/DINOv2/AnyLoc descriptor runs.")
    parser.add_argument("--prediction-csv", nargs="+", required=True, type=Path)
    args = parser.parse_args()

    summaries = [summarize_csv(path) for path in args.prediction_csv]
    print("\nDescriptor run comparison")
    print("=" * 100)
    print(f"{'CSV':45} {'rows':>5} {'acc%':>7} {'med_inl':>8} {'med_ratio':>9} {'ref_bbox':>9} {'inside%':>8} {'geom%':>7} {'med_jump':>9} {'max_jump':>9}")
    for s in summaries:
        name = Path(str(s["path"])).name[:45]
        print(
            f"{name:45} {s['rows']:5d} {s['accepted_pct']:7.1f} {s['median_inliers']:8.1f} "
            f"{s['median_inlier_ratio']:9.2f} {s['median_ref_bbox_area']:9.3f} {s['center_inside_pct']:8.1f} {s['geometry_ok_pct']:7.1f} "
            f"{s['median_jump_m']:9.1f} {s['max_jump_m']:9.1f}"
        )

    print("\nDetails")
    print("=" * 100)
    for s in summaries:
        print(f"\n{s['path']}")
        print(f"  accepted: {s['accepted']} / {s['rows']} ({s['accepted_pct']:.1f}%)")
        print(f"  top reasons: {s['top_reasons']}")
        print(f"  top matched flights: {s['top_flights']}")
        print(f"  median reference inlier bbox area frac: {s['median_ref_bbox_area']:.4f}")
        if s.get("top_query_scales"):
            print(f"  selected query scales: {s['top_query_scales']}")


if __name__ == "__main__":
    main()
