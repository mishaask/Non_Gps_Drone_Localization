"""Lightweight scene signatures for debugging false visual matches.

These helpers are intentionally classical/CPU-only.  They do not solve semantic
segmentation, but they catch many obvious bad matches, such as comparing a
low-altitude frame with sky/horizon to a top-down aerial reference frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SceneSignature:
    sky_ratio: float
    top_sky_ratio: float
    vegetation_ratio: float
    road_gray_ratio: float
    bright_ground_ratio: float
    dark_shadow_ratio: float
    edge_density: float
    low_altitude_proxy: float

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.sky_ratio,
                self.top_sky_ratio,
                self.vegetation_ratio,
                self.road_gray_ratio,
                self.bright_ground_ratio,
                self.dark_shadow_ratio,
                self.edge_density,
                self.low_altitude_proxy,
            ],
            dtype=np.float32,
        )


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install it with: pip install opencv-python") from exc
    return cv2


def compute_scene_signature_bgr(image_bgr: np.ndarray) -> SceneSignature:
    """Compute a simple scene signature from a BGR image.

    The values are rough fractions in [0, 1].  The goal is not perfect class
    segmentation; the goal is to reject visually impossible candidate pairs.
    """

    cv2 = _require_cv2()
    if image_bgr is None or image_bgr.size == 0:
        return SceneSignature(0, 0, 0, 0, 0, 0, 0, 0)

    small = cv2.resize(image_bgr, (320, 180), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    # Approximate classes using broad color/brightness rules.
    sky = ((h >= 85) & (h <= 120) & (s >= 25) & (v >= 110))
    vegetation = ((h >= 35) & (h <= 95) & (s >= 35) & (v >= 45))
    road_gray = ((s <= 45) & (v >= 45) & (v <= 210))
    bright_ground = ((h >= 10) & (h <= 35) & (s >= 20) & (v >= 120)) | ((s <= 55) & (v >= 190))
    dark_shadow = (v <= 55)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)

    top = sky[: max(1, sky.shape[0] // 3), :]
    sky_ratio = float(np.mean(sky))
    top_sky_ratio = float(np.mean(top))
    vegetation_ratio = float(np.mean(vegetation))
    road_gray_ratio = float(np.mean(road_gray))
    bright_ground_ratio = float(np.mean(bright_ground))
    dark_shadow_ratio = float(np.mean(dark_shadow))
    edge_density = float(np.mean(edges > 0))

    # A practical proxy for non-top-down/low-altitude frames: a large amount
    # of sky in the upper third or a strong sky+trees horizon appearance.
    low_altitude_proxy = min(1.0, max(top_sky_ratio, 0.65 * sky_ratio + 0.35 * vegetation_ratio))

    return SceneSignature(
        sky_ratio=sky_ratio,
        top_sky_ratio=top_sky_ratio,
        vegetation_ratio=vegetation_ratio,
        road_gray_ratio=road_gray_ratio,
        bright_ground_ratio=bright_ground_ratio,
        dark_shadow_ratio=dark_shadow_ratio,
        edge_density=edge_density,
        low_altitude_proxy=low_altitude_proxy,
    )


def compute_scene_signature_path(image_path: str | Path) -> SceneSignature:
    cv2 = _require_cv2()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    return compute_scene_signature_bgr(image)


def scene_distance(a: SceneSignature, b: SceneSignature) -> float:
    """Weighted L1 scene-signature distance; lower is more compatible."""

    weights = np.array([2.0, 2.5, 1.2, 1.0, 0.8, 0.6, 0.5, 2.0], dtype=np.float32)
    return float(np.sum(np.abs(a.as_vector() - b.as_vector()) * weights) / np.sum(weights))


def scene_compatibility_reason(
    query: SceneSignature,
    reference: SceneSignature,
    max_scene_distance: float = 0.22,
    max_top_sky_delta: float = 0.22,
) -> tuple[bool, str, float]:
    """Return whether two scenes are compatible, plus reason and distance."""

    dist = scene_distance(query, reference)
    top_sky_delta = abs(query.top_sky_ratio - reference.top_sky_ratio)

    if top_sky_delta > max_top_sky_delta:
        return False, "sky_horizon_mismatch", dist
    if dist > max_scene_distance:
        return False, "scene_signature_mismatch", dist
    return True, "scene_compatible", dist
