from gps_ex1.tools.stage10_conservative_candidate_upgrade import (
    _candidate_confidence,
    _candidate_confidence_with_source,
    _candidate_good_matches,
    _candidate_inlier_ratio,
    _candidate_inliers,
    _candidate_quality_ok,
)


class Args:
    require_candidate_accepted = True
    require_reason_accepted = False
    min_candidate_confidence = 5.0
    min_candidate_inliers = 8
    min_candidate_good_matches = 15
    min_candidate_inlier_ratio = 0.18


def test_candidate_parser_reads_localize_video_column_names():
    row = {
        "matched_reference_image": "ref.jpg",
        "filter_accepted": "1",
        "filter_reason": "accepted",
        "retrieval_similarity": "0.75",
        "homography_inliers": "10",
        "orb_good_matches": "32",
        "verification_inlier_ratio": "0.3125",
        "projected_center_inside": "1",
        "homography_geometry_ok": "1",
    }
    assert _candidate_inliers(row) == 10
    assert _candidate_good_matches(row) == 32
    assert _candidate_inlier_ratio(row) == 0.3125
    ok, reason = _candidate_quality_ok(row, Args())
    assert ok
    assert reason == "candidate_quality_ok"


def test_candidate_parser_computes_ratio_when_ratio_column_missing():
    row = {
        "matched_reference_image": "ref.jpg",
        "filter_accepted": "1",
        "confidence": "9",
        "homography_inliers": "9",
        "orb_good_matches": "18",
    }
    assert _candidate_inlier_ratio(row) == 0.5
    ok, reason = _candidate_quality_ok(row, Args())
    assert ok
    assert reason == "candidate_quality_ok"


def test_similarity_columns_are_not_misread_as_planner_scale_confidence():
    """Regression test for the Stage 10.5/10.5B 526x candidate_rejected_quality bug.

    localize_video CSVs have no explicit confidence column; the only matching
    alias used to be retrieval_similarity (cosine similarity in [0, 1]).
    Comparing that against --min-candidate-confidence 5.0 rejected every
    candidate, including genuinely accepted strong matches.
    """
    row = {
        "matched_reference_image": "ref.jpg",
        "matched_reference_flight": "DJI_0006",
        "matched_reference_frame_index": "19620",
        "filter_accepted": "1",
        "filter_reason": "accepted",
        "orb_good_matches": "34",
        "homography_inliers": "18",
        "verification_inlier_ratio": "0.42",
        "retrieval_similarity": "0.71",
        "projected_center_inside": "1",
        "homography_geometry_ok": "1",
    }
    conf, source = _candidate_confidence_with_source(row)
    assert source == "computed_row_confidence"
    assert conf > 5.0  # strong accepted match must clear the planner-scale gate
    ok, reason = _candidate_quality_ok(row, Args())
    assert ok, reason


def test_weak_rejected_candidate_still_fails_quality():
    row = {
        "matched_reference_image": "ref.jpg",
        "filter_accepted": "0",
        "filter_reason": "weak_visual_match",
        "orb_good_matches": "9",
        "homography_inliers": "4",
        "verification_inlier_ratio": "0.10",
        "retrieval_similarity": "0.55",
        "projected_center_inside": "0",
        "homography_geometry_ok": "0",
    }
    ok, reason = _candidate_quality_ok(row, Args())
    assert not ok
    assert reason == "candidate_not_accepted"


def test_explicit_confidence_column_still_takes_priority():
    row = {
        "matched_reference_image": "ref.jpg",
        "filter_accepted": "1",
        "confidence": "12.5",
        "homography_inliers": "20",
        "orb_good_matches": "40",
        "verification_inlier_ratio": "0.5",
    }
    conf, source = _candidate_confidence_with_source(row)
    assert conf == 12.5
    assert source == "confidence"
