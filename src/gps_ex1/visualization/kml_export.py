"""KML export helpers for visualizing GNSS and predicted paths.

The KML coordinate order is longitude, latitude, altitude, as required by KML.
These helpers intentionally use only the Python standard library so they work in
our lightweight assignment environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from xml.sax.saxutils import escape


@dataclass(frozen=True)
class KmlPoint:
    """One geographic sample that can be written to a KML path."""

    latitude: float
    longitude: float
    altitude_m: float | None = None
    name: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class KmlPath:
    """A named path layer inside a KML file."""

    name: str
    points: Sequence[KmlPoint]
    description: str | None = None


def _coordinate_text(points: Sequence[KmlPoint]) -> str:
    lines: list[str] = []
    for point in points:
        altitude = 0.0 if point.altitude_m is None else float(point.altitude_m)
        lines.append(f"{point.longitude:.8f},{point.latitude:.8f},{altitude:.3f}")
    return "\n              ".join(lines)


def _placemark_for_path(path: KmlPath, line_width: int = 4) -> str:
    description = "" if path.description is None else f"<description>{escape(path.description)}</description>"
    coords = _coordinate_text(path.points)
    return f"""
        <Placemark>
          <name>{escape(path.name)}</name>
          {description}
          <Style>
            <LineStyle><width>{line_width}</width></LineStyle>
          </Style>
          <LineString>
            <tessellate>1</tessellate>
            <altitudeMode>clampToGround</altitudeMode>
            <coordinates>
              {coords}
            </coordinates>
          </LineString>
        </Placemark>"""


def _placemark_for_point(point: KmlPoint, fallback_name: str) -> str:
    name = point.name or fallback_name
    description = "" if point.description is None else f"<description>{escape(point.description)}</description>"
    altitude = 0.0 if point.altitude_m is None else float(point.altitude_m)
    return f"""
        <Placemark>
          <name>{escape(name)}</name>
          {description}
          <Point><coordinates>{point.longitude:.8f},{point.latitude:.8f},{altitude:.3f}</coordinates></Point>
        </Placemark>"""


def build_kml_document(
    paths: Iterable[KmlPath],
    document_name: str = "GPS EX1 paths",
    include_start_end_points: bool = True,
) -> str:
    """Build a complete KML document string from one or more paths."""

    path_list = [path for path in paths if len(path.points) > 0]
    placemarks: list[str] = []
    for path in path_list:
        placemarks.append(_placemark_for_path(path))
        if include_start_end_points and len(path.points) >= 1:
            placemarks.append(
                _placemark_for_point(
                    KmlPoint(
                        latitude=path.points[0].latitude,
                        longitude=path.points[0].longitude,
                        altitude_m=path.points[0].altitude_m,
                        name=f"{path.name} start",
                    ),
                    "start",
                )
            )
            placemarks.append(
                _placemark_for_point(
                    KmlPoint(
                        latitude=path.points[-1].latitude,
                        longitude=path.points[-1].longitude,
                        altitude_m=path.points[-1].altitude_m,
                        name=f"{path.name} end",
                    ),
                    "end",
                )
            )

    body = "\n".join(placemarks)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{escape(document_name)}</name>
{body}
  </Document>
</kml>
"""


def write_kml(path: str | Path, paths: Iterable[KmlPath], document_name: str = "GPS EX1 paths") -> None:
    """Write paths to a KML file."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_kml_document(paths, document_name=document_name), encoding="utf-8")
