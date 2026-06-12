"""Build and load a lightweight visual reference index.

This is a practical first baseline, not the final research-grade hloc/COLMAP
version. It creates one global descriptor per extracted reference frame, stores
its metadata, and later lets the online localizer retrieve visually similar
reference frames quickly.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from gps_ex1.features.global_descriptors import BasicDescriptorExtractor, create_descriptor_extractor
from gps_ex1.segmentation.dynamic_masks import DynamicObjectMasker
from gps_ex1.geometry.geo import haversine_m
from gps_ex1.geometry.projection import bearing_deg, camera_center_ground_point


@dataclass(frozen=True)
class FrameMetadata:
    flight_id: str
    frame_index: int
    video_time_s: float
    image_path: str
    drone_lat: float
    drone_lon: float
    rel_alt: Optional[float]
    heading_deg: Optional[float] = None
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    center_offset_m: Optional[float] = None


@dataclass(frozen=True)
class ReferenceIndex:
    descriptors: np.ndarray
    image_paths: np.ndarray
    flight_ids: np.ndarray
    frame_indices: np.ndarray
    video_times_s: np.ndarray
    drone_lats: np.ndarray
    drone_lons: np.ndarray
    center_lats: np.ndarray
    center_lons: np.ndarray
    headings_deg: np.ndarray
    center_offsets_m: np.ndarray


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install it with: pip install opencv-python") from exc
    return cv2


def resolve_image_path(image_path_text: str, base_dir: Path | None = None) -> Path:
    """Resolve image paths written by the frame-extraction CSV.

    Windows CSVs may contain backslashes. This helper keeps the code usable on
    both Windows and Linux/macOS.
    """

    raw = image_path_text.strip()
    path = Path(raw)
    if path.exists():
        return path

    normalized = Path(raw.replace("\\", "/"))
    if normalized.exists():
        return normalized

    if base_dir is not None:
        candidate = base_dir / path
        if candidate.exists():
            return candidate
        candidate = base_dir / normalized
        if candidate.exists():
            return candidate

    return path


def read_frame_metadata_csv(csv_path: str | Path) -> list[FrameMetadata]:
    csv_path = Path(csv_path)
    rows: list[FrameMetadata] = []
    with csv_path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            lat_text = (row.get("latitude") or "").strip()
            lon_text = (row.get("longitude") or "").strip()
            image_path = (row.get("image_path") or "").strip()
            if not lat_text or not lon_text or not image_path:
                continue
            lat = float(lat_text)
            lon = float(lon_text)
            if abs(lat) < 1e-12 and abs(lon) < 1e-12:
                continue
            rel_alt_text = (row.get("rel_alt") or "").strip()
            rows.append(
                FrameMetadata(
                    flight_id=(row.get("flight_id") or csv_path.stem).strip(),
                    frame_index=int(float(row.get("frame_index") or 0)),
                    video_time_s=float(row.get("video_time_s") or 0.0),
                    image_path=image_path,
                    drone_lat=lat,
                    drone_lon=lon,
                    rel_alt=None if not rel_alt_text else float(rel_alt_text),
                )
            )
    return rows


def _estimate_headings_for_one_flight(rows: list[FrameMetadata]) -> list[FrameMetadata]:
    if not rows:
        return []

    sorted_rows = sorted(rows, key=lambda row: row.video_time_s)
    headings: list[float] = []
    last_good_heading = 0.0

    for i, row in enumerate(sorted_rows):
        heading: Optional[float] = None

        if i + 1 < len(sorted_rows):
            nxt = sorted_rows[i + 1]
            if haversine_m(row.drone_lat, row.drone_lon, nxt.drone_lat, nxt.drone_lon) > 0.25:
                heading = bearing_deg(row.drone_lat, row.drone_lon, nxt.drone_lat, nxt.drone_lon)

        if heading is None and i > 0:
            prev = sorted_rows[i - 1]
            if haversine_m(prev.drone_lat, prev.drone_lon, row.drone_lat, row.drone_lon) > 0.25:
                heading = bearing_deg(prev.drone_lat, prev.drone_lon, row.drone_lat, row.drone_lon)

        if heading is None:
            heading = last_good_heading
        else:
            last_good_heading = heading

        headings.append(heading)

    return [
        FrameMetadata(
            flight_id=row.flight_id,
            frame_index=row.frame_index,
            video_time_s=row.video_time_s,
            image_path=row.image_path,
            drone_lat=row.drone_lat,
            drone_lon=row.drone_lon,
            rel_alt=row.rel_alt,
            heading_deg=headings[i],
        )
        for i, row in enumerate(sorted_rows)
    ]


def estimate_headings(rows: Iterable[FrameMetadata]) -> list[FrameMetadata]:
    by_flight: dict[str, list[FrameMetadata]] = {}
    for row in rows:
        by_flight.setdefault(row.flight_id, []).append(row)

    output: list[FrameMetadata] = []
    for flight_rows in by_flight.values():
        output.extend(_estimate_headings_for_one_flight(flight_rows))
    return sorted(output, key=lambda row: (row.flight_id, row.video_time_s))


def add_camera_center_coordinates(
    rows: Iterable[FrameMetadata],
    camera_angle_deg: float,
    angle_convention: str,
    default_altitude_m: float | None = None,
) -> list[FrameMetadata]:
    output: list[FrameMetadata] = []
    for row in rows:
        altitude = row.rel_alt
        if altitude is None or abs(altitude) < 1e-9:
            altitude = default_altitude_m
        if altitude is None:
            center_lat, center_lon, offset_m = row.drone_lat, row.drone_lon, 0.0
        else:
            center_lat, center_lon, offset_m = camera_center_ground_point(
                drone_lat=row.drone_lat,
                drone_lon=row.drone_lon,
                heading_deg=row.heading_deg or 0.0,
                altitude_m=altitude,
                camera_angle_deg=camera_angle_deg,
                convention=angle_convention,  # type: ignore[arg-type]
            )
        output.append(
            FrameMetadata(
                flight_id=row.flight_id,
                frame_index=row.frame_index,
                video_time_s=row.video_time_s,
                image_path=row.image_path,
                drone_lat=row.drone_lat,
                drone_lon=row.drone_lon,
                rel_alt=row.rel_alt,
                heading_deg=row.heading_deg,
                center_lat=center_lat,
                center_lon=center_lon,
                center_offset_m=offset_m,
            )
        )
    return output


def compute_global_descriptor(image_path: str | Path) -> np.ndarray:
    """Backward-compatible wrapper for the lightweight baseline descriptor."""

    return BasicDescriptorExtractor().describe_path(image_path)


def build_reference_index(
    metadata_rows: list[FrameMetadata],
    output_npz: str | Path,
    base_dir: str | Path | None = None,
    limit: int | None = None,
    descriptor_backend: str = "basic",
    descriptor_model: str = "dinov2_vits14",
    descriptor_device: str | None = None,
    descriptor_image_size: int = 518,
    masker: DynamicObjectMasker | None = None,
    masked_frame_dir: str | Path | None = None,
) -> ReferenceIndex:
    base_path = None if base_dir is None else Path(base_dir)
    rows = metadata_rows if limit is None else metadata_rows[:limit]

    # If masked reference frames are written to disk, describe those saved images
    # directly. Otherwise wrap the descriptor extractor so it masks on the fly.
    descriptor_extractor = create_descriptor_extractor(
        descriptor_backend,
        masker=None if masked_frame_dir is not None else masker,
        model_name=descriptor_model,
        device=descriptor_device,
        image_size=descriptor_image_size,
    )
    print(f"Descriptor backend: {descriptor_extractor.name}")

    descriptors: list[np.ndarray] = []
    kept_rows: list[FrameMetadata] = []

    masked_root = None if masked_frame_dir is None else Path(masked_frame_dir)

    for row in rows:
        image_path = resolve_image_path(row.image_path, base_path)
        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue

        stored_image_path = image_path
        if masker is not None and masked_root is not None:
            # Store masked reference frames so later verification/debug uses the
            # same car/person-suppressed images that the index was built from.
            rel_folder = row.flight_id or image_path.parent.name
            target_dir = masked_root / rel_folder
            target_dir.mkdir(parents=True, exist_ok=True)
            stored_image_path = target_dir / image_path.name
            if not stored_image_path.exists():
                cv2 = _require_cv2()
                masked_image = masker.mask_path(image_path)
                if not cv2.imwrite(str(stored_image_path), masked_image):
                    raise RuntimeError(f"Could not write masked reference frame: {stored_image_path}")

        descriptors.append(descriptor_extractor.describe_path(stored_image_path))
        kept_rows.append(
            FrameMetadata(
                flight_id=row.flight_id,
                frame_index=row.frame_index,
                video_time_s=row.video_time_s,
                image_path=str(stored_image_path),
                drone_lat=row.drone_lat,
                drone_lon=row.drone_lon,
                rel_alt=row.rel_alt,
                heading_deg=row.heading_deg,
                center_lat=row.center_lat,
                center_lon=row.center_lon,
                center_offset_m=row.center_offset_m,
            )
        )

    if not descriptors:
        raise RuntimeError("No reference descriptors were created. Check image paths and CSV files.")

    index = ReferenceIndex(
        descriptors=np.stack(descriptors, axis=0).astype(np.float32),
        image_paths=np.array([row.image_path for row in kept_rows]),
        flight_ids=np.array([row.flight_id for row in kept_rows]),
        frame_indices=np.array([row.frame_index for row in kept_rows], dtype=np.int64),
        video_times_s=np.array([row.video_time_s for row in kept_rows], dtype=np.float32),
        drone_lats=np.array([row.drone_lat for row in kept_rows], dtype=np.float64),
        drone_lons=np.array([row.drone_lon for row in kept_rows], dtype=np.float64),
        center_lats=np.array([row.center_lat if row.center_lat is not None else row.drone_lat for row in kept_rows], dtype=np.float64),
        center_lons=np.array([row.center_lon if row.center_lon is not None else row.drone_lon for row in kept_rows], dtype=np.float64),
        headings_deg=np.array([row.heading_deg if row.heading_deg is not None else np.nan for row in kept_rows], dtype=np.float32),
        center_offsets_m=np.array([row.center_offset_m if row.center_offset_m is not None else np.nan for row in kept_rows], dtype=np.float32),
    )
    save_reference_index(index, output_npz)
    return index


def save_reference_index(index: ReferenceIndex, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        descriptors=index.descriptors,
        image_paths=index.image_paths,
        flight_ids=index.flight_ids,
        frame_indices=index.frame_indices,
        video_times_s=index.video_times_s,
        drone_lats=index.drone_lats,
        drone_lons=index.drone_lons,
        center_lats=index.center_lats,
        center_lons=index.center_lons,
        headings_deg=index.headings_deg,
        center_offsets_m=index.center_offsets_m,
    )


def load_reference_index(path: str | Path) -> ReferenceIndex:
    data = np.load(Path(path), allow_pickle=False)
    return ReferenceIndex(
        descriptors=data["descriptors"].astype(np.float32),
        image_paths=data["image_paths"],
        flight_ids=data["flight_ids"],
        frame_indices=data["frame_indices"],
        video_times_s=data["video_times_s"],
        drone_lats=data["drone_lats"],
        drone_lons=data["drone_lons"],
        center_lats=data["center_lats"],
        center_lons=data["center_lons"],
        headings_deg=data["headings_deg"],
        center_offsets_m=data["center_offsets_m"],
    )
