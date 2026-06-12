from gps_ex1.geometry.projection import bearing_deg, camera_center_ground_point, ground_distance_from_camera_angle


def test_bearing_east_is_about_90():
    assert abs(bearing_deg(32.0, 35.0, 32.0, 35.001) - 90.0) < 1.0


def test_ground_distance_from_horizon():
    distance = ground_distance_from_camera_angle(100.0, 45.0, "from-horizon")
    assert 99.0 < distance < 101.0


def test_camera_center_returns_offset():
    lat, lon, offset = camera_center_ground_point(32.0, 35.0, 90.0, 100.0, 45.0, "from-horizon")
    assert offset > 99.0
    assert lon > 35.0
