"""Build the first offline artifact: per-reference-flight telemetry metadata.

This is Step 1 of the final solution. Later, extracted video keyframes will be
joined with this CSV by frame timestamp.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gps_ex1.io.srt_parser import parse_srt_file, valid_gps_records


def build_reference_metadata(srt_files: list[Path], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "flight_id",
                "record_index",
                "start_s",
                "end_s",
                "timestamp_text",
                "latitude",
                "longitude",
                "rel_alt",
                "abs_alt",
                "focal_len",
            ]
        )

        for srt_file in srt_files:
            flight_id = srt_file.stem
            records = valid_gps_records(parse_srt_file(srt_file))
            for record in records:
                writer.writerow(
                    [
                        flight_id,
                        record.index,
                        f"{record.start_s:.3f}",
                        f"{record.end_s:.3f}",
                        record.timestamp_text or "",
                        f"{record.latitude:.8f}",
                        f"{record.longitude:.8f}",
                        "" if record.rel_alt is None else f"{record.rel_alt:.3f}",
                        "" if record.abs_alt is None else f"{record.abs_alt:.3f}",
                        "" if record.focal_len is None else f"{record.focal_len:.3f}",
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reference telemetry metadata CSV from SRT files.")
    parser.add_argument("--srt", nargs="+", required=True, type=Path, help="Reference SRT files")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV path")
    args = parser.parse_args()

    build_reference_metadata(args.srt, args.out)
    print(f"Wrote reference telemetry metadata to {args.out}")


if __name__ == "__main__":
    main()
