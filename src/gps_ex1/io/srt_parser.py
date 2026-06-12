"""DJI SRT telemetry parser.

This parser is intentionally dependency-free. It extracts the fields needed for
Ex1: frame timing, GNSS coordinates, relative/absolute altitude, focal length,
and a few camera settings useful for diagnostics.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional

from gps_ex1.io.models import TelemetryRecord

_TIME_RE = re.compile(
    r"(?P<sh>\d{2}):(?P<sm>\d{2}):(?P<ss>\d{2}),(?P<sms>\d{3})\s*-->\s*"
    r"(?P<eh>\d{2}):(?P<em>\d{2}):(?P<es>\d{2}),(?P<ems>\d{3})"
)

_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+\b")


def _to_seconds(hours: str, minutes: str, seconds: str, millis: str) -> float:
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0


def _parse_time_range(line: str) -> tuple[float, float]:
    match = _TIME_RE.search(line)
    if not match:
        raise ValueError(f"Invalid SRT time range: {line!r}")
    groups = match.groupdict()
    start = _to_seconds(groups["sh"], groups["sm"], groups["ss"], groups["sms"])
    end = _to_seconds(groups["eh"], groups["em"], groups["es"], groups["ems"])
    return start, end


def _find_float(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(1))


def _find_int(pattern: str, text: str) -> Optional[int]:
    match = re.search(pattern, text)
    if not match:
        return None
    return int(match.group(1))


def _find_str(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text)
    if not match:
        return None
    return match.group(1).strip()


def parse_srt_text(text: str) -> List[TelemetryRecord]:
    """Parse DJI SRT text into telemetry records.

    Bad or incomplete blocks are skipped rather than crashing the full run.
    """

    records: list[TelemetryRecord] = []
    blocks = re.split(r"\n\s*\n", text.strip())

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        try:
            index = int(lines[0])
            start_s, end_s = _parse_time_range(lines[1])
        except ValueError:
            continue

        body = "\n".join(lines[2:])
        timestamp_match = _DATE_RE.search(body)
        timestamp_text = timestamp_match.group(0) if timestamp_match else None

        record = TelemetryRecord(
            index=index,
            start_s=start_s,
            end_s=end_s,
            timestamp_text=timestamp_text,
            srt_counter=_find_int(r"SrtCnt\s*:\s*(\d+)", body),
            diff_time_ms=_find_int(r"DiffTime\s*:\s*(\d+)\s*ms", body),
            latitude=_find_float(r"\[latitude:\s*(-?\d+(?:\.\d+)?)\]", body),
            longitude=_find_float(r"\[longitude:\s*(-?\d+(?:\.\d+)?)\]", body),
            rel_alt=_find_float(r"\[rel_alt:\s*(-?\d+(?:\.\d+)?)", body),
            abs_alt=_find_float(r"abs_alt:\s*(-?\d+(?:\.\d+)?)\]", body),
            focal_len=_find_float(r"\[focal_len\s*:\s*(-?\d+(?:\.\d+)?)\]", body),
            iso=_find_int(r"\[iso\s*:\s*(\d+)\]", body),
            shutter=_find_str(r"\[shutter\s*:\s*([^\]]+)\]", body),
            fnum=_find_int(r"\[fnum\s*:\s*(\d+)\]", body),
            raw_text=body,
        )
        records.append(record)

    return records


def parse_srt_file(path: str | Path) -> List[TelemetryRecord]:
    """Read and parse a DJI SRT file."""

    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_srt_text(text)


def valid_gps_records(records: Iterable[TelemetryRecord]) -> list[TelemetryRecord]:
    """Return only records with non-zero GNSS coordinates."""

    return [record for record in records if record.has_valid_gps]


def nearest_record_by_time(records: list[TelemetryRecord], time_s: float) -> Optional[TelemetryRecord]:
    """Find the telemetry record nearest to a video timestamp in seconds.

    For large production runs this can be optimized with binary search. For the
    first implementation, this simple version is easier to read and test.
    """

    if not records:
        return None
    return min(records, key=lambda record: abs(record.start_s - time_s))
