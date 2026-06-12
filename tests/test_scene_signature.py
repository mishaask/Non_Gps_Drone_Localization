import numpy as np

from gps_ex1.quality.scene_signature import compute_scene_signature_bgr, scene_compatibility_reason


def test_scene_signature_detects_sky_like_top_region():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :, :] = (255, 180, 80)  # BGR sky-ish blue
    sig = compute_scene_signature_bgr(img)
    assert sig.top_sky_ratio > 0.8
    assert sig.low_altitude_proxy > 0.8


def test_scene_compatibility_rejects_sky_mismatch():
    q = np.zeros((100, 100, 3), dtype=np.uint8)
    q[:50, :, :] = (255, 180, 80)
    r = np.zeros((100, 100, 3), dtype=np.uint8)
    r[:, :, :] = (70, 120, 70)
    ok, reason, _dist = scene_compatibility_reason(
        compute_scene_signature_bgr(q),
        compute_scene_signature_bgr(r),
        max_top_sky_delta=0.2,
    )
    assert not ok
    assert reason == "sky_horizon_mismatch"
