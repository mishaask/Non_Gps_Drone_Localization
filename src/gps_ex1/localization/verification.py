"""Candidate verification backends.

The baseline ORB verifier is always available through OpenCV. LightGlue is wired
as an optional stronger verifier for machines that install the modern dependency
set. Both backends return the same score object so the rest of the pipeline can
remain unchanged.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

import numpy as np

from gps_ex1.localization.orb_matcher import (
    OrbMatchScore,
    _homography_geometry_diagnostics,
    score_orb_homography,
)


@dataclass(frozen=True)
class VerificationScore:
    good_matches: int
    homography_inliers: int
    mean_match_distance: float
    backend: str = "orb"
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


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("OpenCV is required. Install it with: pip install opencv-python") from exc
    return cv2


def score_candidate_alignment(
    query_bgr: np.ndarray,
    reference_image_path: str | Path,
    backend: str = "orb",
    image_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    dynamic_masker: Any | None = None,
) -> VerificationScore:
    """Score one query/reference image pair with the requested backend.

    When ``dynamic_masker`` is supplied for the ORB backend, ORB keypoints are
    ignored inside detected cars/objects instead of physically matching against
    gray/blurred blobs. This prevents artificial mask borders from becoming new
    false features.
    """

    normalized = backend.strip().lower()
    if normalized in {"orb", "opencv", "baseline"}:
        query_mask = None
        reference_mask = None
        if dynamic_masker is not None and image_transform is None:
            cv2 = _require_cv2()
            query_mask = dynamic_masker.dynamic_mask_bgr(query_bgr)
            reference_bgr = cv2.imread(str(reference_image_path), cv2.IMREAD_COLOR)
            if reference_bgr is not None:
                reference_mask = dynamic_masker.dynamic_mask_bgr(reference_bgr, cache_key=str(reference_image_path))
        orb_score: OrbMatchScore = score_orb_homography(
            query_bgr,
            reference_image_path,
            image_transform=image_transform,
            query_ignore_mask=query_mask,
            reference_ignore_mask=reference_mask,
        )
        return VerificationScore(
            good_matches=orb_score.good_matches,
            homography_inliers=orb_score.homography_inliers,
            mean_match_distance=orb_score.mean_match_distance,
            backend="orb",
            homography_found=orb_score.homography_found,
            inlier_ratio=orb_score.inlier_ratio,
            projected_center_x=orb_score.projected_center_x,
            projected_center_y=orb_score.projected_center_y,
            projected_center_inside=orb_score.projected_center_inside,
            projected_quad_area_frac=orb_score.projected_quad_area_frac,
            projected_quad_valid=orb_score.projected_quad_valid,
            geometry_reason=orb_score.geometry_reason,
            query_inlier_bbox_area_frac=orb_score.query_inlier_bbox_area_frac,
            query_inlier_bbox_width_frac=orb_score.query_inlier_bbox_width_frac,
            query_inlier_bbox_height_frac=orb_score.query_inlier_bbox_height_frac,
            reference_inlier_bbox_area_frac=orb_score.reference_inlier_bbox_area_frac,
            reference_inlier_bbox_width_frac=orb_score.reference_inlier_bbox_width_frac,
            reference_inlier_bbox_height_frac=orb_score.reference_inlier_bbox_height_frac,
        )
    if normalized in {"lightglue", "superpoint-lightglue", "sp-lightglue"}:
        if image_transform is not None:
            query_bgr = image_transform(query_bgr)
        elif dynamic_masker is not None:
            query_bgr = dynamic_masker.mask_bgr(query_bgr)
        return score_lightglue_homography(query_bgr, reference_image_path)
    raise ValueError(f"Unknown verification backend: {backend}. Use 'orb' or 'lightglue'.")



_LIGHTGLUE_RUNTIME = None


class _LightGlueRuntime:
    """Small cached wrapper around SuperPoint + LightGlue.

    Creating the neural models for every candidate pair is extremely slow. The
    verifier can score thousands of candidate pairs during one localization run,
    so we keep the extractor/matcher alive and cache reference features by path.
    """

    def __init__(self, max_num_keypoints: int = 2048):
        try:
            import torch  # type: ignore
            from lightglue import LightGlue, SuperPoint  # type: ignore
            from lightglue.utils import rbd  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "LightGlue backend requires torch and lightglue. Install optional dependencies with: "
                "python -m pip install -r requirements-stage5.txt"
            ) from exc

        self.torch = torch
        self.rbd = rbd
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        self.reference_feature_cache: dict[str, tuple[np.ndarray, Any]] = {}

    def _bgr_to_tensor(self, bgr: np.ndarray):
        rgb = bgr[:, :, ::-1].copy()
        return self.torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device)

    def extract(self, bgr: np.ndarray):
        with self.torch.inference_mode():
            return self.extractor.extract(self._bgr_to_tensor(bgr))

    def reference_features(self, reference_image_path: str | Path, reference_bgr: np.ndarray):
        key = str(Path(reference_image_path).resolve())
        cached = self.reference_feature_cache.get(key)
        if cached is not None and tuple(int(v) for v in cached[0]) == tuple(reference_bgr.shape[:2]):
            return cached[1]
        feats = self.extract(reference_bgr)
        # Store only the shape, not the full image, to keep the cache small.
        self.reference_feature_cache[key] = (np.array(reference_bgr.shape[:2], dtype=np.int32), feats)
        return feats

    def match_points(self, query_bgr: np.ndarray, reference_bgr: np.ndarray, reference_image_path: str | Path):
        with self.torch.inference_mode():
            feats0 = self.extract(query_bgr)
            feats1 = self.reference_features(reference_image_path, reference_bgr)
            matches01 = self.matcher({"image0": feats0, "image1": feats1})
            feats0_s, feats1_s, matches01_s = [self.rbd(x) for x in [feats0, feats1, matches01]]
            matches = matches01_s["matches"]
            if len(matches) == 0:
                return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
            points0 = feats0_s["keypoints"][matches[..., 0]].detach().cpu().numpy().astype(np.float32)
            points1 = feats1_s["keypoints"][matches[..., 1]].detach().cpu().numpy().astype(np.float32)
            return points0, points1


def _get_lightglue_runtime() -> _LightGlueRuntime:
    global _LIGHTGLUE_RUNTIME
    if _LIGHTGLUE_RUNTIME is None:
        _LIGHTGLUE_RUNTIME = _LightGlueRuntime()
    return _LIGHTGLUE_RUNTIME


def lightglue_match_points(query_bgr: np.ndarray, reference_bgr: np.ndarray, reference_image_path: str | Path):
    """Return matched keypoint coordinates for debug visualization."""
    return _get_lightglue_runtime().match_points(query_bgr, reference_bgr, reference_image_path)



def _bbox_spread_metrics(points: np.ndarray, shape_hw: tuple[int, int]) -> tuple[float | None, float | None, float | None]:
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
    return width_frac * height_frac, width_frac, height_frac

def score_lightglue_homography(query_bgr: np.ndarray, reference_image_path: str | Path) -> VerificationScore:
    """Optional LightGlue + SuperPoint verification."""

    cv2 = _require_cv2()
    reference_bgr = cv2.imread(str(reference_image_path), cv2.IMREAD_COLOR)
    if reference_bgr is None:
        return VerificationScore(0, 0, float("inf"), backend="lightglue", geometry_reason="reference_image_missing")

    points0, points1 = lightglue_match_points(query_bgr, reference_bgr, reference_image_path)
    if len(points0) == 0:
        return VerificationScore(0, 0, float("inf"), backend="lightglue", geometry_reason="no_matches")

    inliers = 0
    homography = None
    query_bbox_area = query_bbox_width = query_bbox_height = None
    reference_bbox_area = reference_bbox_width = reference_bbox_height = None
    if len(points0) >= 4:
        homography, mask = cv2.findHomography(points0.reshape(-1, 1, 2), points1.reshape(-1, 1, 2), cv2.RANSAC, 5.0)
        if mask is not None:
            inlier_mask = mask.ravel().astype(bool)
            inliers = int(inlier_mask.sum())
            query_bbox_area, query_bbox_width, query_bbox_height = _bbox_spread_metrics(points0[inlier_mask], query_bgr.shape[:2])
            reference_bbox_area, reference_bbox_width, reference_bbox_height = _bbox_spread_metrics(points1[inlier_mask], reference_bgr.shape[:2])

    displacement = np.linalg.norm(points0 - points1, axis=1)
    mean_distance = float(np.mean(displacement)) if len(displacement) else float("inf")
    inlier_ratio = float(inliers / max(len(points0), 1))
    cx, cy, center_inside, area_frac, quad_valid, geometry_reason = _homography_geometry_diagnostics(
        homography,
        query_shape_hw=query_bgr.shape[:2],
        reference_shape_hw=reference_bgr.shape[:2],
    )
    return VerificationScore(
        int(len(points0)),
        inliers,
        mean_distance,
        backend="lightglue",
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
