"""Temporal consistency helpers for GNSS-denied visual localization.

The first baseline chooses the visually best reference frame independently for
all query frames. Drone videos contain many repetitive structures, so this can
produce impossible jumps. This module adds lightweight online filtering: a
candidate must be visually plausible and physically plausible relative to the
previous accepted estimate.

Stage-2 adds strict geometric gates. A candidate is no longer accepted merely
because OpenCV produced a homography: it must have enough inliers, a reasonable
inlier ratio, and a projected query center that falls inside the reference image.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log1p
from typing import Sequence

from gps_ex1.geometry.geo import haversine_m


@dataclass(frozen=True)
class CandidateObservation:
    """One candidate localization result for a query frame."""

    index: int
    latitude: float
    longitude: float
    matched_flight: str
    matched_frame_index: int
    matched_time_s: float
    retrieval_distance: float
    retrieval_similarity: float
    good_matches: int
    homography_inliers: int
    mean_match_distance: float
    query_scale: float = 1.0
    homography_found: bool = True
    inlier_ratio: float = 1.0
    projected_center_x: float | None = None
    projected_center_y: float | None = None
    projected_center_inside: bool = True
    projected_quad_area_frac: float | None = None
    projected_quad_valid: bool = True
    geometry_reason: str = "ok"
    query_inlier_bbox_area_frac: float | None = None
    query_inlier_bbox_width_frac: float | None = None
    query_inlier_bbox_height_frac: float | None = None
    reference_inlier_bbox_area_frac: float | None = None
    reference_inlier_bbox_width_frac: float | None = None
    reference_inlier_bbox_height_frac: float | None = None


@dataclass(frozen=True)
class TemporalSelection:
    """Chosen candidate and diagnostic data."""

    candidate: CandidateObservation | None
    accepted: bool
    reason: str
    visual_score: float
    temporal_distance_m: float | None
    fused_score: float


@dataclass(frozen=True)
class TemporalFilterConfig:
    """Configuration for the online temporal filter."""

    min_inliers: int = 12
    min_good_matches: int = 12
    min_inlier_ratio: float = 0.25
    require_center_inside: bool = True
    require_valid_homography_quad: bool = True
    max_reference_inlier_bbox_area_frac: float = 1.0
    max_reference_inlier_bbox_width_frac: float = 1.0
    max_reference_inlier_bbox_height_frac: float = 1.0
    max_query_inlier_bbox_area_frac: float = 1.0
    max_speed_mps: float = 18.0
    base_gate_m: float = 55.0
    hard_jump_m: float = 180.0
    visual_weight: float = 1.0
    continuity_weight: float = 1.3
    same_flight_bonus: float = 0.20
    high_confidence_inliers: int = 24
    high_confidence_multiplier: float = 1.8
    ema_alpha: float = 0.35


@dataclass(frozen=True)
class OnlineTrackState:
    """State kept by the online localizer."""

    latitude: float
    longitude: float
    query_time_s: float
    matched_flight: str
    accepted_count: int = 1


def visual_quality_score(candidate: CandidateObservation) -> float:
    """Convert heterogeneous matching diagnostics into one monotonic score.

    This score is not meant to be a calibrated probability. It is a practical
    ranking value where larger is better. It favors geometric inliers, then
    good ORB matches, then retrieval similarity, and softly penalizes weak/large
    match distances and unstable homographies.
    """

    inlier_term = 1.35 * log1p(max(candidate.homography_inliers, 0))
    match_term = 0.45 * log1p(max(candidate.good_matches, 0))
    retrieval_term = 1.10 * float(candidate.retrieval_similarity)
    ratio_term = 1.25 * max(0.0, min(float(candidate.inlier_ratio), 1.0))

    distance_penalty = 0.0
    if isfinite(candidate.mean_match_distance):
        # ORB Hamming distances around 30-70 are common. Penalize very large
        # mean distances without letting this dominate the geometry terms.
        distance_penalty = max(0.0, (candidate.mean_match_distance - 35.0) / 80.0)

    geometry_penalty = 0.0
    if not candidate.projected_center_inside:
        geometry_penalty += 2.5
    if not candidate.projected_quad_valid:
        geometry_penalty += 1.5
    if candidate.reference_inlier_bbox_area_frac is not None:
        # Prefer candidates where the RANSAC inliers are concentrated on one
        # coherent landmark/structure instead of scattered over the full frame.
        geometry_penalty += 1.25 * max(0.0, candidate.reference_inlier_bbox_area_frac - 0.12)

    return inlier_term + match_term + retrieval_term + ratio_term - distance_penalty - geometry_penalty


def visual_gate_reason(candidate: CandidateObservation, cfg: TemporalFilterConfig) -> str:
    """Return 'accepted' if candidate passes strict visual/geometry checks."""

    if not candidate.homography_found:
        return "homography_not_found"
    if candidate.good_matches < cfg.min_good_matches:
        return "not_enough_good_matches"
    if candidate.homography_inliers < cfg.min_inliers:
        return "not_enough_inliers"
    if candidate.inlier_ratio < cfg.min_inlier_ratio:
        return "weak_inlier_ratio"
    if cfg.require_center_inside and not candidate.projected_center_inside:
        return "projected_center_outside"
    if cfg.require_valid_homography_quad and not candidate.projected_quad_valid:
        return "bad_homography_geometry"
    if (
        candidate.reference_inlier_bbox_area_frac is not None
        and candidate.reference_inlier_bbox_area_frac > cfg.max_reference_inlier_bbox_area_frac
    ):
        return "reference_inliers_scattered_area"
    if (
        candidate.reference_inlier_bbox_width_frac is not None
        and candidate.reference_inlier_bbox_width_frac > cfg.max_reference_inlier_bbox_width_frac
    ):
        return "reference_inliers_scattered_width"
    if (
        candidate.reference_inlier_bbox_height_frac is not None
        and candidate.reference_inlier_bbox_height_frac > cfg.max_reference_inlier_bbox_height_frac
    ):
        return "reference_inliers_scattered_height"
    if (
        candidate.query_inlier_bbox_area_frac is not None
        and candidate.query_inlier_bbox_area_frac > cfg.max_query_inlier_bbox_area_frac
    ):
        return "query_inliers_scattered_area"
    return "accepted"


def _allowed_distance_m(dt_s: float, cfg: TemporalFilterConfig, candidate: CandidateObservation) -> float:
    base = max(cfg.base_gate_m, cfg.max_speed_mps * max(dt_s, 0.1) * 2.0)
    if candidate.homography_inliers >= cfg.high_confidence_inliers:
        base *= cfg.high_confidence_multiplier
    return base


def select_visual_candidate(
    candidates: Sequence[CandidateObservation],
    cfg: TemporalFilterConfig,
) -> TemporalSelection:
    """Choose the visually best candidate without temporal continuity.

    This is used for raw/no-temporal experiments. It still applies the stage-2
    visual gates, so obviously bad homographies are marked as rejected.
    """

    if not candidates:
        return TemporalSelection(None, False, "no_candidates", 0.0, None, float("-inf"))

    best_selection: TemporalSelection | None = None
    for candidate in candidates:
        visual = visual_quality_score(candidate)
        reason = visual_gate_reason(candidate, cfg)
        selection = TemporalSelection(candidate, reason == "accepted", reason, visual, None, visual)
        if best_selection is None or selection.fused_score > best_selection.fused_score:
            best_selection = selection

    assert best_selection is not None
    return best_selection


def select_temporal_candidate(
    candidates: Sequence[CandidateObservation],
    previous: OnlineTrackState | None,
    query_time_s: float,
    cfg: TemporalFilterConfig,
) -> TemporalSelection:
    """Choose a candidate using visual evidence plus physical continuity."""

    if not candidates:
        return TemporalSelection(None, False, "no_candidates", 0.0, None, float("-inf"))

    best_selection: TemporalSelection | None = None

    for candidate in candidates:
        visual = visual_quality_score(candidate)
        visual_reason = visual_gate_reason(candidate, cfg)
        visually_valid = visual_reason == "accepted"

        temporal_distance: float | None = None
        continuity_penalty = 0.0
        same_flight_bonus = 0.0
        reason = visual_reason
        accepted = visually_valid

        if previous is not None:
            dt_s = max(query_time_s - previous.query_time_s, 0.0)
            allowed = _allowed_distance_m(dt_s, cfg, candidate)
            temporal_distance = haversine_m(previous.latitude, previous.longitude, candidate.latitude, candidate.longitude)

            if candidate.matched_flight == previous.matched_flight:
                same_flight_bonus = cfg.same_flight_bonus

            # Soft penalty inside the hard gate; hard reject for very unrealistic
            # jumps unless the candidate is unusually strong.
            continuity_penalty = cfg.continuity_weight * (temporal_distance / max(allowed, 1.0))
            hard_jump = max(cfg.hard_jump_m, allowed * 2.2)
            high_confidence = candidate.homography_inliers >= cfg.high_confidence_inliers and candidate.inlier_ratio >= max(cfg.min_inlier_ratio, 0.35)
            if temporal_distance > hard_jump and not high_confidence:
                accepted = False
                reason = "temporal_jump_rejected"
            elif not visually_valid:
                reason = visual_reason

        fused = cfg.visual_weight * visual - continuity_penalty + same_flight_bonus
        selection = TemporalSelection(
            candidate=candidate,
            accepted=accepted,
            reason=reason,
            visual_score=visual,
            temporal_distance_m=temporal_distance,
            fused_score=fused,
        )

        if best_selection is None or selection.fused_score > best_selection.fused_score:
            best_selection = selection

    assert best_selection is not None
    return best_selection


def hold_state_at_time(previous: OnlineTrackState, query_time_s: float) -> OnlineTrackState:
    """Keep the same coordinates but advance the online time stamp.

    This matters when a frame is rejected and the KML holds the previous
    position. Advancing the time prevents the next accepted candidate from
    getting an unrealistically large distance allowance merely because many
    weak frames were skipped.
    """

    return OnlineTrackState(
        latitude=previous.latitude,
        longitude=previous.longitude,
        query_time_s=query_time_s,
        matched_flight=previous.matched_flight,
        accepted_count=previous.accepted_count,
    )


def exponential_smooth_state(
    previous: OnlineTrackState | None,
    candidate: CandidateObservation,
    query_time_s: float,
    cfg: TemporalFilterConfig,
) -> OnlineTrackState:
    """Create the next online state, smoothing coordinates after the first point."""

    if previous is None:
        return OnlineTrackState(
            latitude=candidate.latitude,
            longitude=candidate.longitude,
            query_time_s=query_time_s,
            matched_flight=candidate.matched_flight,
            accepted_count=1,
        )

    alpha = min(max(cfg.ema_alpha, 0.0), 1.0)
    return OnlineTrackState(
        latitude=(1.0 - alpha) * previous.latitude + alpha * candidate.latitude,
        longitude=(1.0 - alpha) * previous.longitude + alpha * candidate.longitude,
        query_time_s=query_time_s,
        matched_flight=candidate.matched_flight,
        accepted_count=previous.accepted_count + 1,
    )
