"""Add human-readable video timecodes to an existing prediction CSV.

This is a lightweight debug helper. It does not rerun localization; it simply
adds query_timecode and matched_reference_timecode columns derived from the
existing numeric query_time_s and matched_reference_time_s columns.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gps_ex1.io.timecode import seconds_to_timecode


def add_timecodes_to_csv(prediction_csv: Path, output_csv: Path) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with prediction_csv.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not fieldnames:
        raise RuntimeError(f"CSV has no header: {prediction_csv}")

    if "query_timecode" not in fieldnames:
        insert_at = fieldnames.index("query_time_s") + 1 if "query_time_s" in fieldnames else len(fieldnames)
        fieldnames.insert(insert_at, "query_timecode")
    if "matched_reference_timecode" not in fieldnames:
        insert_at = fieldnames.index("matched_reference_time_s") + 1 if "matched_reference_time_s" in fieldnames else len(fieldnames)
        fieldnames.insert(insert_at, "matched_reference_timecode")

    for row in rows:
        row["query_timecode"] = seconds_to_timecode(row.get("query_time_s"))
        row["matched_reference_timecode"] = seconds_to_timecode(row.get("matched_reference_time_s"))

    with output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add HH:MM:SS.mmm timecode columns to a prediction CSV.")
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = add_timecodes_to_csv(args.prediction_csv, args.out)
    print(f"Wrote timecoded predictions: {args.out}")
    print(f"Rows: {rows}")


if __name__ == "__main__":
    main()
