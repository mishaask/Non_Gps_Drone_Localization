from gps_ex1.visualization.kml_export import KmlPath, KmlPoint, build_kml_document


def test_kml_uses_longitude_latitude_order():
    doc = build_kml_document([KmlPath(name="demo", points=[KmlPoint(latitude=32.1, longitude=35.2)])])
    assert "35.20000000,32.10000000" in doc


def test_kml_escapes_names():
    doc = build_kml_document([KmlPath(name="A&B", points=[KmlPoint(latitude=1.0, longitude=2.0)])])
    assert "A&amp;B" in doc
