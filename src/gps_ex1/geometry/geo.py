"""Small geographic helper functions for local path calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class LocalPoint:
    """A point in a local East/North coordinate frame, in meters."""

    east_m: float
    north_m: float


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance between two lat/lon points in meters."""

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def latlon_to_local_xy(lat: float, lon: float, origin_lat: float, origin_lon: float) -> LocalPoint:
    """Approximate lat/lon as local East/North meters near an origin.

    This equirectangular approximation is accurate enough for the small areas in
    the assignment videos and keeps the first implementation simple.
    """

    lat_rad = math.radians(lat)
    origin_lat_rad = math.radians(origin_lat)
    d_lat = math.radians(lat - origin_lat)
    d_lon = math.radians(lon - origin_lon)

    east = EARTH_RADIUS_M * d_lon * math.cos((lat_rad + origin_lat_rad) / 2.0)
    north = EARTH_RADIUS_M * d_lat
    return LocalPoint(east_m=east, north_m=north)


def local_xy_to_latlon(east_m: float, north_m: float, origin_lat: float, origin_lon: float) -> tuple[float, float]:
    """Convert local East/North meters back to approximate lat/lon."""

    lat = origin_lat + math.degrees(north_m / EARTH_RADIUS_M)
    avg_lat_rad = math.radians((lat + origin_lat) / 2.0)
    lon = origin_lon + math.degrees(east_m / (EARTH_RADIUS_M * math.cos(avg_lat_rad)))
    return lat, lon
