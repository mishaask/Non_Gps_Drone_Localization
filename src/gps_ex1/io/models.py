"""Shared data models for the GPS_EX1 visual-navigation project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TelemetryRecord:
    """One telemetry sample parsed from a DJI SRT subtitle block.

    The test-video SRT may contain GNSS, but during the real-time localization
    pipeline those GNSS fields must be ignored and used only for evaluation.
    """

    index: int
    start_s: float
    end_s: float
    timestamp_text: Optional[str]
    srt_counter: Optional[int]
    diff_time_ms: Optional[int]
    latitude: Optional[float]
    longitude: Optional[float]
    rel_alt: Optional[float]
    abs_alt: Optional[float]
    focal_len: Optional[float]
    iso: Optional[int]
    shutter: Optional[str]
    fnum: Optional[int]
    raw_text: str

    @property
    def has_valid_gps(self) -> bool:
        """Return True when latitude/longitude look usable.

        DJI SRT files often begin with latitude=0 and longitude=0 before GNSS
        becomes available. Those values are not useful for our reference map.
        """

        if self.latitude is None or self.longitude is None:
            return False
        return not (abs(self.latitude) < 1e-12 and abs(self.longitude) < 1e-12)
