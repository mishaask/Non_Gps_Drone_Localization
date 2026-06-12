"""Extract sampled keyframes from a video and sync them with SRT telemetry."""

from __future__ import annotations

import argparse
from pathlib import Path

from gps_ex1.io.srt_parser import parse_srt_file, valid_gps_records
from gps_ex1.io.video_frames import extract_sampled_frames, write_frame_metadata_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract sampled frames and join each frame with nearest SRT telemetry.")
    parser.add_argument("--video", required=True, type=Path, help="Input drone video file")
    parser.add_argument("--srt", required=True, type=Path, help="Matching SRT telemetry file")
    parser.add_argument("--flight-id", required=True, help="Flight identifier, e.g. DJI_0006")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for extracted images")
    parser.add_argument("--metadata-csv", required=True, type=Path, help="Output CSV for frame metadata")
    parser.add_argument("--every-n-frames", type=int, default=30, help="Frame sampling stride. 30 ~= 1 FPS for 30fps video")
    args = parser.parse_args()

    telemetry = valid_gps_records(parse_srt_file(args.srt))
    records = extract_sampled_frames(
        video_path=args.video,
        telemetry=telemetry,
        output_dir=args.out_dir,
        flight_id=args.flight_id,
        every_n_frames=args.every_n_frames,
    )
    write_frame_metadata_csv(records, args.metadata_csv)
    print(f"Extracted {len(records)} frames")
    print(f"Wrote metadata to {args.metadata_csv}")


if __name__ == "__main__":
    main()
