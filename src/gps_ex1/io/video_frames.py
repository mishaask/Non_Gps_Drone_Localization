"""Video frame extraction and timestamp synchronization utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from gps_ex1.io.models import TelemetryRecord
from gps_ex1.io.srt_parser import nearest_record_by_time


@dataclass(frozen=True)
class ExtractedFrameRecord:
    """Metadata for one extracted video frame."""

    flight_id: str
    frame_index: int
    video_time_s: float
    image_path: str
    telemetry_index: Optional[int]
    telemetry_time_s: Optional[float]
    time_error_s: Optional[float]
    latitude: Optional[float]
    longitude: Optional[float]
    rel_alt: Optional[float]
    abs_alt: Optional[float]


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for frame extraction. Install it with: pip install opencv-python"
        ) from exc
    return cv2


def extract_sampled_frames(
    video_path: str | Path,
    telemetry: list[TelemetryRecord],
    output_dir: str | Path,
    flight_id: str,
    every_n_frames: int = 30,
    image_extension: str = ".jpg",
) -> list[ExtractedFrameRecord]:
    """Extract sampled frames and link each image to nearest telemetry.

    Args:
        video_path: Path to the original drone video.
        telemetry: Parsed SRT records for this video.
        output_dir: Directory where images should be written.
        flight_id: Name used in output filenames and metadata.
        every_n_frames: Sampling stride. For 30 fps video, 30 means about 1 FPS.
        image_extension: Usually .jpg or .png.

    Returns:
        Metadata rows for all extracted frames.
    """

    if every_n_frames <= 0:
        raise ValueError("every_n_frames must be positive")

    cv2 = _require_cv2()
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    extracted: list[ExtractedFrameRecord] = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % every_n_frames == 0:
            video_time_s = frame_index / fps
            nearest = nearest_record_by_time(telemetry, video_time_s)
            image_name = f"{flight_id}_frame_{frame_index:06d}{image_extension}"
            image_path = output_dir / image_name

            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(f"Could not write extracted frame: {image_path}")

            if nearest is None:
                extracted.append(
                    ExtractedFrameRecord(
                        flight_id=flight_id,
                        frame_index=frame_index,
                        video_time_s=video_time_s,
                        image_path=str(image_path),
                        telemetry_index=None,
                        telemetry_time_s=None,
                        time_error_s=None,
                        latitude=None,
                        longitude=None,
                        rel_alt=None,
                        abs_alt=None,
                    )
                )
            else:
                extracted.append(
                    ExtractedFrameRecord(
                        flight_id=flight_id,
                        frame_index=frame_index,
                        video_time_s=video_time_s,
                        image_path=str(image_path),
                        telemetry_index=nearest.index,
                        telemetry_time_s=nearest.start_s,
                        time_error_s=abs(nearest.start_s - video_time_s),
                        latitude=nearest.latitude,
                        longitude=nearest.longitude,
                        rel_alt=nearest.rel_alt,
                        abs_alt=nearest.abs_alt,
                    )
                )

        frame_index += 1

    cap.release()
    return extracted


def write_frame_metadata_csv(records: Iterable[ExtractedFrameRecord], output_csv: str | Path) -> None:
    """Write extracted-frame metadata to CSV."""

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "flight_id",
                "frame_index",
                "video_time_s",
                "image_path",
                "telemetry_index",
                "telemetry_time_s",
                "time_error_s",
                "latitude",
                "longitude",
                "rel_alt",
                "abs_alt",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.flight_id,
                    record.frame_index,
                    f"{record.video_time_s:.3f}",
                    record.image_path,
                    "" if record.telemetry_index is None else record.telemetry_index,
                    "" if record.telemetry_time_s is None else f"{record.telemetry_time_s:.3f}",
                    "" if record.time_error_s is None else f"{record.time_error_s:.3f}",
                    "" if record.latitude is None else f"{record.latitude:.8f}",
                    "" if record.longitude is None else f"{record.longitude:.8f}",
                    "" if record.rel_alt is None else f"{record.rel_alt:.3f}",
                    "" if record.abs_alt is None else f"{record.abs_alt:.3f}",
                ]
            )
