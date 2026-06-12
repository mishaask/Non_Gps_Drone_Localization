from gps_ex1.localization.temporal_filter import (
    CandidateObservation,
    OnlineTrackState,
    TemporalFilterConfig,
    select_temporal_candidate,
)


def _candidate(lat, lon, inliers=10, good=12, flight="DJI_0006"):
    return CandidateObservation(
        index=0,
        latitude=lat,
        longitude=lon,
        matched_flight=flight,
        matched_frame_index=0,
        matched_time_s=0.0,
        retrieval_distance=0.2,
        retrieval_similarity=0.8,
        good_matches=good,
        homography_inliers=inliers,
        mean_match_distance=42.0,
    )


def test_temporal_filter_rejects_impossible_jump():
    cfg = TemporalFilterConfig(min_inliers=6, min_good_matches=8, max_speed_mps=15.0, base_gate_m=30.0, hard_jump_m=120.0)
    previous = OnlineTrackState(latitude=32.1000, longitude=35.2000, query_time_s=0.0, matched_flight="DJI_0006")
    far_candidate = _candidate(32.1100, 35.2100, inliers=8, good=12)

    selection = select_temporal_candidate([far_candidate], previous, query_time_s=1.0, cfg=cfg)

    assert selection.candidate is far_candidate
    assert not selection.accepted
    assert selection.reason == "temporal_jump_rejected"


def test_temporal_filter_accepts_near_candidate():
    cfg = TemporalFilterConfig(min_inliers=6, min_good_matches=8, max_speed_mps=15.0, base_gate_m=80.0, hard_jump_m=150.0)
    previous = OnlineTrackState(latitude=32.1000, longitude=35.2000, query_time_s=0.0, matched_flight="DJI_0006")
    near_candidate = _candidate(32.1002, 35.2002, inliers=8, good=12)

    selection = select_temporal_candidate([near_candidate], previous, query_time_s=1.0, cfg=cfg)

    assert selection.accepted
    assert selection.reason == "accepted"
