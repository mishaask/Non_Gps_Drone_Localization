"""Ground-projection helpers for estimating the camera center point.

The DJI SRT files in the starter dataset contain drone latitude/longitude and
relative altitude, but not full camera pose. For the first assignment baseline we
estimate the viewing direction from the drone motion bearing and use the known
camera angle from the assignment statement.
"""

from __future__ import annotations

import math
from typing import Literal

EARTH_RADIUS_M = 6_371_000.0
AngleConvention = Literal["from-horizon", "from-nadir"]


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return initial bearing from point 1 to point 2 in degrees clockwise from north."""

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    theta = math.degrees(math.atan2(y, x))
    return (theta + 360.0) % 360.0


def destination_point(lat: float, lon: float, bearing_degrees: float, distance_m: float) -> tuple[float, float]:
    """Move from lat/lon by distance along bearing and return new lat/lon."""

    angular_distance = distance_m / EARTH_RADIUS_M
    bearing = math.radians(bearing_degrees)
    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)

    sin_phi2 = math.sin(phi1) * math.cos(angular_distance) + math.cos(phi1) * math.sin(angular_distance) * math.cos(bearing)
    phi2 = math.asin(max(-1.0, min(1.0, sin_phi2)))

    y = math.sin(bearing) * math.sin(angular_distance) * math.cos(phi1)
    x = math.cos(angular_distance) - math.sin(phi1) * math.sin(phi2)
    lambda2 = lambda1 + math.atan2(y, x)

    new_lat = math.degrees(phi2)
    new_lon = (math.degrees(lambda2) + 540.0) % 360.0 - 180.0
    return new_lat, new_lon


def ground_distance_from_camera_angle(
    altitude_m: float,
    camera_angle_deg: float,
    convention: AngleConvention = "from-horizon",
) -> float:
    """Estimate horizontal ground distance from drone to image center point.

    Args:
        altitude_m: Drone height above the local takeoff/ground plane.
        camera_angle_deg: Camera tilt angle in degrees.
        convention:
            "from-horizon": 60 means the camera points 60 degrees below the horizon.
            "from-nadir": 60 means the camera is 60 degrees away from straight down.

    Returns:
        Horizontal distance in meters from drone location to the ground point at
        the image center under a flat-ground approximation.
    """

    if altitude_m < 0:
        altitude_m = abs(altitude_m)
    angle = math.radians(camera_angle_deg)
    if angle <= 0.0 or angle >= math.pi / 2.0:
        raise ValueError("camera_angle_deg must be between 0 and 90 degrees for this simple baseline")

    if convention == "from-horizon":
        return altitude_m / math.tan(angle)
    if convention == "from-nadir":
        return altitude_m * math.tan(angle)
    raise ValueError(f"Unsupported camera angle convention: {convention}")


def camera_center_ground_point(
    drone_lat: float,
    drone_lon: float,
    heading_deg: float,
    altitude_m: float,
    camera_angle_deg: float,
    convention: AngleConvention = "from-horizon",
) -> tuple[float, float, float]:
    """Estimate the ground coordinate seen at the center of the image.

    Returns:
        (center_lat, center_lon, horizontal_distance_m)
    """

    distance_m = ground_distance_from_camera_angle(altitude_m, camera_angle_deg, convention)
    center_lat, center_lon = destination_point(drone_lat, drone_lon, heading_deg, distance_m)
    return center_lat, center_lon, distance_m
