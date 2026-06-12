"""Path-level metrics for comparing predicted coordinates to SRT truth."""

from __future__ import annotations

import math
from statistics import median
from typing import Iterable, Sequence

from gps_ex1.geometry.geo import haversine_m


def distance_errors_m(
    predicted_latlon: Sequence[tuple[float, float]],
    truth_latlon: Sequence[tuple[float, float]],
) -> list[float]:
    """Return per-sample horizontal errors in meters."""

    if len(predicted_latlon) != len(truth_latlon):
        raise ValueError("predicted_latlon and truth_latlon must have the same length")

    return [
        haversine_m(pred_lat, pred_lon, truth_lat, truth_lon)
        for (pred_lat, pred_lon), (truth_lat, truth_lon) in zip(predicted_latlon, truth_latlon)
    ]


def summarize_errors(errors_m: Iterable[float]) -> dict[str, float]:
    """Compute common localization error statistics."""

    errors = sorted(float(error) for error in errors_m)
    if not errors:
        return {"count": 0.0, "mean_m": math.nan, "rmse_m": math.nan, "median_m": math.nan, "p95_m": math.nan}

    mean = sum(errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    p95_index = min(len(errors) - 1, math.ceil(0.95 * len(errors)) - 1)
    return {
        "count": float(len(errors)),
        "mean_m": mean,
        "rmse_m": rmse,
        "median_m": median(errors),
        "p95_m": errors[p95_index],
    }
