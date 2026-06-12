"""Small helpers for human-readable video timecodes."""

from __future__ import annotations


def seconds_to_timecode(seconds: float | int | str | None) -> str:
    """Convert seconds to HH:MM:SS.mmm.

    The prediction CSV already stores numeric seconds. This helper adds a value
    that is easier to paste into VLC / video players and easier to read in a
    report/debug table.
    """

    if seconds is None:
        return ""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return ""
    if value < 0:
        return ""

    total_ms = int(round(value * 1000.0))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
