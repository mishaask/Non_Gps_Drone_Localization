from pathlib import Path

import numpy as np

from gps_ex1.tools.stage10_cluster_lookpoint_refine import _cluster_metrics, _pixel_to_ground_offset, _accepted


def test_cluster_metrics_bbox_fraction():
    pts = np.array([[10, 10], [30, 20], [20, 40]], dtype=np.float32)
    area, width, height = _cluster_metrics(pts, (100, 200))
    assert round(width, 3) == 0.100
    assert round(height, 3) == 0.300
    assert round(area, 3) == 0.030


def test_pixel_to_ground_offset_center_is_near_zero():
    east, north = _pixel_to_ground_offset(
        cluster_x=500,
        cluster_y=250,
        image_w=1000,
        image_h=500,
        heading_deg=45,
        camera_angle_deg=60,
        angle_convention="from-horizon",
        horizontal_fov_deg=73,
        fallback_altitude_m=119,
    )
    assert abs(east) < 1e-9
    assert abs(north) < 1e-9


def test_accepted_helper():
    assert _accepted({"filter_accepted": "1"})
    assert _accepted({"filter_accepted": "true"})
    assert not _accepted({"filter_accepted": "0"})
