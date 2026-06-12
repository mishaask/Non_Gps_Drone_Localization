"""CLI tool for inspecting DJI SRT telemetry files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gps_ex1.geometry.geo import haversine_m
from gps_ex1.io.srt_parser import parse_srt_file, valid_gps_records


def _write_csv(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "index",
                "start_s",
                "end_s",
                "timestamp_text",
                "latitude",
                "longitude",
                "rel_alt",
                "abs_alt",
                "focal_len",
                "iso",
                "shutter",
                "fnum",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.index,
                    f"{record.start_s:.3f}",
                    f"{record.end_s:.3f}",
                    record.timestamp_text or "",
                    "" if record.latitude is None else f"{record.latitude:.8f}",
                    "" if record.longitude is None else f"{record.longitude:.8f}",
                    "" if record.rel_alt is None else f"{record.rel_alt:.3f}",
                    "" if record.abs_alt is None else f"{record.abs_alt:.3f}",
                    "" if record.focal_len is None else f"{record.focal_len:.3f}",
                    "" if record.iso is None else record.iso,
                    record.shutter or "",
                    "" if record.fnum is None else record.fnum,
                ]
            )


def inspect_file(path: Path, csv_out: Path | None = None) -> None:
    records = parse_srt_file(path)
    valid = valid_gps_records(records)

    print(f"File: {path.name}")
    print(f"  total records: {len(records)}")
    print(f"  valid GNSS records: {len(valid)}")

    if records:
        print(f"  time range: {records[0].start_s:.3f}s -> {records[-1].end_s:.3f}s")

    if valid:
        first = valid[0]
        last = valid[-1]
        path_distance = 0.0
        previous = valid[0]
        for current in valid[1:]:
            path_distance += haversine_m(previous.latitude, previous.longitude, current.latitude, current.longitude)
            previous = current

        print(f"  first valid GNSS: index={first.index}, lat={first.latitude:.7f}, lon={first.longitude:.7f}")
        print(f"  last valid GNSS:  index={last.index}, lat={last.latitude:.7f}, lon={last.longitude:.7f}")
        print(f"  rel altitude range: {min(r.rel_alt for r in valid if r.rel_alt is not None):.2f}m -> {max(r.rel_alt for r in valid if r.rel_alt is not None):.2f}m")
        print(f"  approximate traveled distance from SRT samples: {path_distance:.1f}m")

    if csv_out is not None:
        _write_csv(csv_out, valid)
        print(f"  wrote valid GNSS CSV: {csv_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect DJI SRT telemetry files.")
    parser.add_argument("srt_files", nargs="+", type=Path, help="One or more .SRT files")
    parser.add_argument("--csv-dir", type=Path, default=None, help="Optional output directory for parsed CSV files")
    args = parser.parse_args()

    for srt_file in args.srt_files:
        csv_out = None
        if args.csv_dir is not None:
            csv_out = args.csv_dir / f"{srt_file.stem}_valid_gps.csv"
        inspect_file(srt_file, csv_out)


if __name__ == "__main__":
    main()
