from gps_ex1.tools.stage10_conservative_candidate_upgrade import (
    _candidate_confidence,
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
        "retrieval_similarity": "7.5",
        "homography_inliers": "10",
        "orb_good_matches": "32",
        "verification_inlier_ratio": "0.3125",
    }
    assert _candidate_confidence(row) == 7.5
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
