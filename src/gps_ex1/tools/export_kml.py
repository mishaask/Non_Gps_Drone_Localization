"""Export SRT GNSS paths, predicted localization CSVs, or reference-index paths to KML."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from gps_ex1.io.srt_parser import parse_srt_file, valid_gps_records
from gps_ex1.preprocess.reference_index import load_reference_index
from gps_ex1.visualization.kml_export import KmlPath, KmlPoint, write_kml


def _flight_name_from_path(path: Path) -> str:
    return path.stem.replace("_predictions", "").replace("_center", " center").replace("_drone", " drone")


def _points_from_srt(path: Path, stride: int) -> list[KmlPoint]:
    records = valid_gps_records(parse_srt_file(path))
    points: list[KmlPoint] = []
    for i, record in enumerate(records):
        if i % stride != 0:
            continue
        if record.latitude is None or record.longitude is None:
            continue
        points.append(
            KmlPoint(
                latitude=record.latitude,
                longitude=record.longitude,
                altitude_m=record.rel_alt,
                description=f"SRT index={record.index}, time={record.start_s:.3f}s",
            )
        )
    return points


def _paths_from_srts(paths: Iterable[Path], stride: int) -> list[KmlPath]:
    output: list[KmlPath] = []
    for path in paths:
        points = _points_from_srt(path, stride=stride)
        output.append(
            KmlPath(
                name=f"{path.stem} GNSS drone path",
                points=points,
                description=f"Ground-truth drone GNSS path parsed from {path.name}",
            )
        )
    return output


def _float_or_none(text: str | None) -> float | None:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    return float(text)


def _prediction_target_from_csv(path: Path) -> str:
    """Read the prediction target label from the first data row, if present."""

    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            target = (row.get("prediction_target") or "").strip().lower()
            if target in {"center", "drone"}:
                return target
    return "unknown"


def _points_from_prediction_csv(path: Path, stride: int, min_inliers: int | None, use_filtered: bool = False, accepted_only: bool = False) -> list[KmlPoint]:
    points: list[KmlPoint] = []
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row_number, row in enumerate(reader):
            if row_number % stride != 0:
                continue
            if accepted_only and (row.get("filter_accepted") or "").strip() not in {"1", "true", "True"}:
                continue

            if use_filtered:
                lat = _float_or_none(row.get("filtered_latitude"))
                lon = _float_or_none(row.get("filtered_longitude"))
            else:
                lat = _float_or_none(row.get("pred_latitude"))
                lon = _float_or_none(row.get("pred_longitude"))
            if lat is None or lon is None:
                continue
            inliers_text = (row.get("homography_inliers") or "").strip()
            inliers = int(float(inliers_text)) if inliers_text else 0
            if min_inliers is not None and inliers < min_inliers:
                continue
            query_frame = row.get("query_frame_index", "")
            query_time = row.get("query_time_s", "")
            target = row.get("prediction_target", "")
            matched_flight = row.get("matched_reference_flight", "")
            matched_frame = row.get("matched_reference_frame_index", "")
            similarity = row.get("retrieval_similarity", "")
            filter_reason = row.get("filter_reason", "")
            filter_accepted = row.get("filter_accepted", "")
            points.append(
                KmlPoint(
                    latitude=lat,
                    longitude=lon,
                    description=(
                        f"prediction_target={target}, "
                        f"query_frame={query_frame}, query_time_s={query_time}, "
                        f"matched_reference={matched_flight}:{matched_frame}, "
                        f"homography_inliers={inliers}, retrieval_similarity={similarity}, "
                        f"filter_accepted={filter_accepted}, filter_reason={filter_reason}"
                    ),
                )
            )
    return points


def _paths_from_prediction_csvs(paths: Iterable[Path], stride: int, min_inliers: int | None, use_filtered: bool = False, accepted_only: bool = False) -> list[KmlPath]:
    output: list[KmlPath] = []
    for path in paths:
        target = _prediction_target_from_csv(path)
        target_label = "center-point" if target == "center" else "drone-position" if target == "drone" else "localization"
        points = _points_from_prediction_csv(path, stride=stride, min_inliers=min_inliers, use_filtered=use_filtered, accepted_only=accepted_only)
        output.append(
            KmlPath(
                name=f"{_flight_name_from_path(path)} predicted {target_label} path",
                points=points,
                description=f"Predicted GNSS-denied {target_label} path from {path.name}",
            )
        )
    return output


def _paths_from_reference_index(path: Path, target: str, stride: int) -> list[KmlPath]:
    index = load_reference_index(path)
    grouped: dict[str, list[tuple[float, KmlPoint]]] = defaultdict(list)

    for i, flight_id in enumerate(index.flight_ids):
        if i % stride != 0:
            continue
        if target == "center":
            lat = float(index.center_lats[i])
            lon = float(index.center_lons[i])
            layer_suffix = "estimated center-point path"
        elif target == "drone":
            lat = float(index.drone_lats[i])
            lon = float(index.drone_lons[i])
            layer_suffix = "indexed drone path"
        else:
            raise ValueError(f"Unsupported target: {target}")

        grouped[str(flight_id)].append(
            (
                float(index.video_times_s[i]),
                KmlPoint(
                    latitude=lat,
                    longitude=lon,
                    description=f"frame={int(index.frame_indices[i])}, time={float(index.video_times_s[i]):.3f}s",
                ),
            )
        )

    output: list[KmlPath] = []
    for flight_id, timed_points in sorted(grouped.items()):
        timed_points.sort(key=lambda item: item[0])
        output.append(KmlPath(name=f"{flight_id} {layer_suffix}", points=[point for _, point in timed_points]))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GPS_EX1 paths to KML for Google Earth / Google My Maps.")
    parser.add_argument("--srt", nargs="*", type=Path, default=[], help="One or more DJI SRT files to export as GNSS drone paths")
    parser.add_argument("--prediction-csv", nargs="*", type=Path, default=[], help="One or more localize_video prediction CSVs")
    parser.add_argument("--reference-index", type=Path, default=None, help="Optional reference_index.npz to export")
    parser.add_argument("--index-target", choices=["center", "drone"], default="center", help="Which coordinate from the reference index to export")
    parser.add_argument("--out", required=True, type=Path, help="Output KML file")
    parser.add_argument("--name", default="GPS EX1 visualization", help="KML document name")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth point in exported paths")
    parser.add_argument("--min-inliers", type=int, default=None, help="For prediction CSVs, keep only rows with at least this many homography inliers")
    parser.add_argument("--use-filtered", action="store_true", help="For filtered CSVs, export filtered_latitude/filtered_longitude when available")
    parser.add_argument("--accepted-only", action="store_true", help="For filtered CSVs, export only rows with filter_accepted=1")
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    paths: list[KmlPath] = []
    paths.extend(_paths_from_srts(args.srt, stride=args.stride))
    paths.extend(_paths_from_prediction_csvs(args.prediction_csv, stride=args.stride, min_inliers=args.min_inliers, use_filtered=args.use_filtered, accepted_only=args.accepted_only))
    if args.reference_index is not None:
        paths.extend(_paths_from_reference_index(args.reference_index, target=args.index_target, stride=args.stride))

    if not paths:
        raise RuntimeError("No input paths were provided. Use --srt, --prediction-csv, or --reference-index.")

    non_empty = [path for path in paths if len(path.points) > 0]
    if not non_empty:
        raise RuntimeError("The provided inputs did not contain any valid coordinates.")

    write_kml(args.out, non_empty, document_name=args.name)
    for path in non_empty:
        print(f"Layer: {path.name} | points: {len(path.points)}")
    print(f"Wrote KML: {args.out}")


if __name__ == "__main__":
    main()
