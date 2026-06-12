"""ORB matching and homography scoring for candidate reranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class OrbMatchScore:
    good_matches: int
    homography_inliers: int
    mean_match_distance: float
    homography_found: bool = False
    inlier_ratio: float = 0.0
    projected_center_x: float | None = None
    projected_center_y: float | None = None
    projected_center_inside: bool = True
    projected_quad_area_frac: float | None = None
    projected_quad_valid: bool = True
    geometry_reason: str = "not_checked"
    query_inlier_bbox_area_frac: float | None = None
    query_inlier_bbox_width_frac: float | None = None
    query_inlier_bbox_height_frac: float | None = None
    reference_inlier_bbox_area_frac: float | None = None
    reference_inlier_bbox_width_frac: float | None = None
    reference_inlier_bbox_height_frac: float | None = None

    @property
    def is_valid(self) -> bool:
        return (
            self.good_matches >= 8
            and self.homography_inliers >= 4
            and self.projected_center_inside
            and self.projected_quad_valid
        )



def _bbox_spread_metrics(points: np.ndarray, shape_hw: tuple[int, int]) -> tuple[float | None, float | None, float | None]:
    """Return bbox area/width/height fractions for a set of 2D points.

    The values are used only as debug/gating signals. A large reference bbox
    usually means matches are scattered over unrelated structures; a compact
    bbox means the match is concentrated on one landmark/object region.
    """

    if points is None or len(points) < 2:
        return None, None, None
    h, w = shape_hw
    if h <= 0 or w <= 0:
        return None, None, None
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if len(pts) < 2:
        return None, None, None
    x0 = float(np.min(pts[:, 0]))
    y0 = float(np.min(pts[:, 1]))
    x1 = float(np.max(pts[:, 0]))
    y1 = float(np.max(pts[:, 1]))
    width_frac = max(0.0, min(1.0, (x1 - x0) / max(float(w), 1.0)))
    height_frac = max(0.0, min(1.0, (y1 - y0) / max(float(h), 1.0)))
    area_frac = width_frac * height_frac
    return area_frac, width_frac, height_frac

def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install it with: pip install opencv-python") from exc
    return cv2


def _normalize_feature_mask(mask: np.ndarray | None, shape_hw: tuple[int, int]) -> np.ndarray | None:
    """Convert an ignore/dynamic mask into OpenCV's allowed-keypoint mask.

    OpenCV expects a single-channel uint8 mask where non-zero pixels are valid
    places to detect keypoints. In this project the incoming masks are dynamic
    object masks where 255 means "ignore this pixel", so we invert them here.
    """

    if mask is None:
        return None
    cv2 = _require_cv2()
    h, w = shape_hw
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask.astype(np.uint8)
    if not np.any(mask):
        return None
    return cv2.bitwise_not(mask)


def _homography_geometry_diagnostics(
    homography: np.ndarray | None,
    query_shape_hw: tuple[int, int],
    reference_shape_hw: tuple[int, int],
    min_quad_area_frac: float = 0.0005,
    max_quad_area_frac: float = 5.0,
) -> tuple[float | None, float | None, bool, float | None, bool, str]:
    """Measure whether the query center maps to a plausible reference location.

    For this assignment the final output is the coordinate of the camera-frame
    center. A homography that maps the center outside the matched reference frame
    is therefore not usable, even when RANSAC technically returned a model.
    """

    if homography is None:
        return None, None, False, None, False, "homography_not_found"

    cv2 = _require_cv2()
    qh, qw = query_shape_hw
    rh, rw = reference_shape_hw

    try:
        center_q = np.float32([[[qw / 2.0, qh / 2.0]]])
        projected_center = cv2.perspectiveTransform(center_q, homography).reshape(2)
        cx = float(projected_center[0])
        cy = float(projected_center[1])

        corners_q = np.float32([[0, 0], [qw - 1, 0], [qw - 1, qh - 1], [0, qh - 1]]).reshape(-1, 1, 2)
        projected_corners = cv2.perspectiveTransform(corners_q, homography).reshape(-1, 2)
    except Exception:
        return None, None, False, None, False, "homography_projection_failed"

    if not np.all(np.isfinite(projected_corners)) or not np.isfinite(cx) or not np.isfinite(cy):
        return cx if np.isfinite(cx) else None, cy if np.isfinite(cy) else None, False, None, False, "homography_non_finite_projection"

    center_inside = 0.0 <= cx < float(rw) and 0.0 <= cy < float(rh)

    area_px = abs(float(cv2.contourArea(projected_corners.astype(np.float32))))
    ref_area = max(float(rw * rh), 1.0)
    area_frac = area_px / ref_area

    quad_valid = min_quad_area_frac <= area_frac <= max_quad_area_frac
    if not center_inside:
        reason = "projected_center_outside_reference"
    elif not quad_valid:
        reason = "homography_quad_area_out_of_range"
    else:
        reason = "ok"

    return cx, cy, center_inside, area_frac, quad_valid, reason


def score_orb_homography(
    query_bgr: np.ndarray,
    reference_image_path: str | Path,
    nfeatures: int = 1500,
    ratio_test: float = 0.75,
    image_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    query_ignore_mask: np.ndarray | None = None,
    reference_ignore_mask: np.ndarray | None = None,
) -> OrbMatchScore:
    """Score visual alignment between query frame and reference image.

    If ``query_ignore_mask`` / ``reference_ignore_mask`` are supplied, ORB
    detects keypoints only outside those masked dynamic objects. This is better
    than drawing gray blobs into the image because it does not create artificial
    blob edges that ORB can latch onto.
    """

    cv2 = _require_cv2()
    reference_bgr = cv2.imread(str(reference_image_path), cv2.IMREAD_COLOR)
    if reference_bgr is None:
        return OrbMatchScore(good_matches=0, homography_inliers=0, mean_match_distance=float("inf"), geometry_reason="reference_image_missing")

    if image_transform is not None:
        query_bgr = image_transform(query_bgr)
        reference_bgr = image_transform(reference_bgr)
        query_ignore_mask = None
        reference_ignore_mask = None

    query_gray = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)

    allowed_q = _normalize_feature_mask(query_ignore_mask, query_gray.shape[:2])
    allowed_r = _normalize_feature_mask(reference_ignore_mask, reference_gray.shape[:2])

    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints_q, descriptors_q = orb.detectAndCompute(query_gray, allowed_q)
    keypoints_r, descriptors_r = orb.detectAndCompute(reference_gray, allowed_r)

    if descriptors_q is None or descriptors_r is None or len(keypoints_q) < 8 or len(keypoints_r) < 8:
        return OrbMatchScore(good_matches=0, homography_inliers=0, mean_match_distance=float("inf"), geometry_reason="not_enough_keypoints")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(descriptors_q, descriptors_r, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        best, second = pair
        if best.distance < ratio_test * second.distance:
            good.append(best)

    if not good:
        return OrbMatchScore(good_matches=0, homography_inliers=0, mean_match_distance=float("inf"), geometry_reason="no_good_matches")

    mean_distance = float(np.mean([match.distance for match in good]))
    inliers = 0
    homography = None

    query_bbox_area = query_bbox_width = query_bbox_height = None
    reference_bbox_area = reference_bbox_width = reference_bbox_height = None
    if len(good) >= 4:
        src_pts = np.float32([keypoints_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_r[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is not None:
            inlier_mask = mask.ravel().astype(bool)
            inliers = int(inlier_mask.sum())
            query_bbox_area, query_bbox_width, query_bbox_height = _bbox_spread_metrics(src_pts.reshape(-1, 2)[inlier_mask], query_gray.shape[:2])
            reference_bbox_area, reference_bbox_width, reference_bbox_height = _bbox_spread_metrics(dst_pts.reshape(-1, 2)[inlier_mask], reference_gray.shape[:2])

    inlier_ratio = float(inliers / max(len(good), 1))
    cx, cy, center_inside, area_frac, quad_valid, geometry_reason = _homography_geometry_diagnostics(
        homography,
        query_shape_hw=query_gray.shape[:2],
        reference_shape_hw=reference_gray.shape[:2],
    )

    return OrbMatchScore(
        good_matches=len(good),
        homography_inliers=inliers,
        mean_match_distance=mean_distance,
        homography_found=homography is not None,
        inlier_ratio=inlier_ratio,
        projected_center_x=cx,
        projected_center_y=cy,
        projected_center_inside=center_inside,
        projected_quad_area_frac=area_frac,
        projected_quad_valid=quad_valid,
        geometry_reason=geometry_reason,
        query_inlier_bbox_area_frac=query_bbox_area,
        query_inlier_bbox_width_frac=query_bbox_width,
        query_inlier_bbox_height_frac=query_bbox_height,
        reference_inlier_bbox_area_frac=reference_bbox_area,
        reference_inlier_bbox_width_frac=reference_bbox_width,
        reference_inlier_bbox_height_frac=reference_bbox_height,
    )
