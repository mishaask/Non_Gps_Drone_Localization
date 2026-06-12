"""Analyze prediction CSV path smoothness and filter diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median

from gps_ex1.geometry.geo import haversine_m


def _float_or_none(text: str | None) -> float | None:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    return float(text)


def analyze_prediction_csv(path: Path) -> dict[str, float | int]:
    points: list[tuple[float, float]] = []
    accepted = 0
    held = 0
    rejected = 0
    rows = 0

    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows += 1
            lat = _float_or_none(row.get("pred_latitude"))
            lon = _float_or_none(row.get("pred_longitude"))
            if lat is not None and lon is not None:
                points.append((lat, lon))

            if (row.get("filter_accepted") or "").strip() == "1":
                accepted += 1
            elif (row.get("filter_accepted") or "").strip() == "0":
                if (row.get("filter_reason") or "").endswith("_held_last"):
                    held += 1
                else:
                    rejected += 1

    jumps = [haversine_m(points[i - 1][0], points[i - 1][1], points[i][0], points[i][1]) for i in range(1, len(points))]
    if not jumps:
        return {
            "rows": rows,
            "points": len(points),
            "accepted": accepted,
            "held": held,
            "rejected": rejected,
            "median_jump_m": 0.0,
            "mean_jump_m": 0.0,
            "max_jump_m": 0.0,
            "jumps_over_100m": 0,
            "jumps_over_200m": 0,
        }

    return {
        "rows": rows,
        "points": len(points),
        "accepted": accepted,
        "held": held,
        "rejected": rejected,
        "median_jump_m": float(median(jumps)),
        "mean_jump_m": float(mean(jumps)),
        "max_jump_m": float(max(jumps)),
        "jumps_over_100m": sum(jump > 100.0 for jump in jumps),
        "jumps_over_200m": sum(jump > 200.0 for jump in jumps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Print path smoothness diagnostics for prediction CSV files.")
    parser.add_argument("--prediction-csv", nargs="+", required=True, type=Path)
    args = parser.parse_args()

    for path in args.prediction_csv:
        stats = analyze_prediction_csv(path)
        print(f"\n{path}")
        print(f"  rows: {stats['rows']}")
        print(f"  points with coordinates: {stats['points']}")
        print(f"  accepted: {stats['accepted']} | held: {stats['held']} | rejected: {stats['rejected']}")
        print(f"  median jump: {stats['median_jump_m']:.2f} m")
        print(f"  mean jump: {stats['mean_jump_m']:.2f} m")
        print(f"  max jump: {stats['max_jump_m']:.2f} m")
        print(f"  jumps >100m: {stats['jumps_over_100m']}")
        print(f"  jumps >200m: {stats['jumps_over_200m']}")


if __name__ == "__main__":
    main()
