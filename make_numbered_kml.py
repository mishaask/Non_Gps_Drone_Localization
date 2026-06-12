import argparse
import csv
import html
from pathlib import Path

LAT_COLUMNS = [
    "filtered_latitude", "filtered_lat", "filtered_predicted_lat", "filtered_center_lat",
    "expected_latitude", "pred_latitude", "predicted_lat", "center_lat", "lat",
]
LON_COLUMNS = [
    "filtered_longitude", "filtered_lon", "filtered_predicted_lon", "filtered_center_lon",
    "expected_longitude", "pred_longitude", "predicted_lon", "center_lon", "lon",
]
ACCEPT_COLUMNS = [
    "stage10_filter_accepted", "stage9_path_filter_accepted", "filter_accepted",
    "filtered_accepted", "accepted", "is_accepted", "seed_trusted", "original_accepted",
]


def pick(row, names):
    for name in names:
        value = str(row.get(name, "")).strip()
        if value and value.lower() not in ("none", "nan"):
            return value
    return ""


def is_true(value):
    return str(value).strip().lower() in ("1", "true", "yes", "accepted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--accepted-only", action="store_true")
    parser.add_argument("--name", default="Numbered GPS path")
    args = parser.parse_args()

    points = []
    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if args.accepted_only and not is_true(pick(row, ACCEPT_COLUMNS)):
                continue
            lat = pick(row, LAT_COLUMNS)
            lon = pick(row, LON_COLUMNS)
            if not lat or not lon:
                continue
            try:
                latf = float(lat)
                lonf = float(lon)
            except ValueError:
                continue
            frame = pick(row, ["query_frame_index", "query_frame", "frame", "query_frame_idx"])
            time_s = pick(row, ["query_time_s", "time_s", "timestamp_s", "t"])
            reason = pick(row, ["seed_reject_reason", "stage10_merge_reason", "reason", "verification_reason", "filter_reason"])
            points.append((latf, lonf, frame, time_s, reason))

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    lines.append("<Document>")
    lines.append(f"<name>{html.escape(args.name)}</name>")
    lines.append('''
<Style id="numberedPoint">
  <IconStyle>
    <scale>0.8</scale>
    <Icon><href>http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png</href></Icon>
  </IconStyle>
  <LabelStyle><scale>0.9</scale></LabelStyle>
</Style>
<Style id="pathLine">
  <LineStyle><color>ffffffff</color><width>3</width></LineStyle>
</Style>
''')
    if points:
        lines.append("<Placemark>")
        lines.append(f"<name>{html.escape(args.name)} line</name>")
        lines.append("<styleUrl>#pathLine</styleUrl>")
        lines.append("<LineString><tessellate>1</tessellate><coordinates>")
        for lat, lon, *_ in points:
            lines.append(f"{lon},{lat},0")
        lines.append("</coordinates></LineString>")
        lines.append("</Placemark>")

    for idx, (lat, lon, frame, time_s, reason) in enumerate(points, start=1):
        label = f"{idx:03d}"
        if frame:
            label += f" | frame {frame}"
        desc = [f"Order: {idx}", f"Frame: {frame}", f"Time: {time_s}", f"Reason: {reason}", f"Lat/Lon: {lat:.8f}, {lon:.8f}"]
        lines.append("<Placemark>")
        lines.append(f"<name>{html.escape(label)}</name>")
        lines.append("<styleUrl>#numberedPoint</styleUrl>")
        lines.append(f"<description>{html.escape(chr(10).join(desc))}</description>")
        lines.append(f"<Point><coordinates>{lon},{lat},0</coordinates></Point>")
        lines.append("</Placemark>")

    lines.append("</Document></kml>")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(points)} numbered points to: {args.out}")


if __name__ == "__main__":
    main()
