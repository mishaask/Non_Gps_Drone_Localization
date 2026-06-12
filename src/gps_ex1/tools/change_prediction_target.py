"""Rewrite a prediction CSV from center-point coordinates to drone coordinates, or vice versa.

The visual matching decision is independent of whether we output the matched
reference frame's drone coordinate or its estimated camera-center coordinate.
This tool reuses an existing localize_video output CSV and swaps the coordinate
columns using reference_index.npz, so you do not have to rerun video matching.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gps_ex1.preprocess.reference_index import load_reference_index


def _build_coordinate_lookup(reference_index_path: Path, target: str) -> dict[tuple[str, int], tuple[float, float]]:
    index = load_reference_index(reference_index_path)
    lookup: dict[tuple[str, int], tuple[float, float]] = {}
    for i, flight_id in enumerate(index.flight_ids):
        key = (str(flight_id), int(index.frame_indices[i]))
        if target == "center":
            lookup[key] = (float(index.center_lats[i]), float(index.center_lons[i]))
        elif target == "drone":
            lookup[key] = (float(index.drone_lats[i]), float(index.drone_lons[i]))
        else:
            raise ValueError(f"Unsupported target: {target}")
    return lookup


def change_prediction_target(
    prediction_csv: Path,
    reference_index_path: Path,
    output_csv: Path,
    target: str,
) -> None:
    lookup = _build_coordinate_lookup(reference_index_path, target=target)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rewritten = 0
    missing = 0
    with prediction_csv.open("r", newline="", encoding="utf-8") as in_fp, output_csv.open("w", newline="", encoding="utf-8") as out_fp:
        reader = csv.DictReader(in_fp)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {prediction_csv}")
        required = {"matched_reference_flight", "matched_reference_frame_index", "pred_latitude", "pred_longitude", "prediction_target"}
        missing_columns = sorted(required - set(reader.fieldnames))
        if missing_columns:
            raise RuntimeError(f"CSV is missing required columns: {missing_columns}")

        writer = csv.DictWriter(out_fp, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            flight_id = (row.get("matched_reference_flight") or "").strip()
            frame_text = (row.get("matched_reference_frame_index") or "").strip()
            if flight_id and frame_text:
                key = (flight_id, int(float(frame_text)))
                coord = lookup.get(key)
                if coord is not None:
                    lat, lon = coord
                    row["prediction_target"] = target
                    row["pred_latitude"] = f"{lat:.8f}"
                    row["pred_longitude"] = f"{lon:.8f}"
                    rewritten += 1
                else:
                    missing += 1
            else:
                missing += 1
            writer.writerow(row)

    print(f"Wrote: {output_csv}")
    print(f"Rows rewritten to target={target}: {rewritten}")
    if missing:
        print(f"Rows without a reference-index match: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Change a localize_video predictions CSV between center/drone coordinate targets.")
    parser.add_argument("--prediction-csv", required=True, type=Path, help="Existing localize_video output CSV")
    parser.add_argument("--reference-index", required=True, type=Path, help="reference_index.npz used for localization")
    parser.add_argument("--target", choices=["center", "drone"], required=True, help="Coordinate target to write into pred_latitude/pred_longitude")
    parser.add_argument("--out", required=True, type=Path, help="Output rewritten prediction CSV")
    args = parser.parse_args()

    change_prediction_target(
        prediction_csv=args.prediction_csv,
        reference_index_path=args.reference_index,
        output_csv=args.out,
        target=args.target,
    )


if __name__ == "__main__":
    main()
