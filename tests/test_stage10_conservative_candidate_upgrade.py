from gps_ex1.tools.stage10_conservative_candidate_upgrade import _candidate_quality_ok, _is_fallback, _is_existing_visual


class Args:
    require_candidate_accepted = True
    require_reason_accepted = False
    min_candidate_confidence = 5.0
    min_candidate_inliers = 8
    min_candidate_good_matches = 15
    min_candidate_inlier_ratio = 0.18


def test_fallback_detection():
    assert _is_fallback({"stage10_final_source": "path_expected_fallback"})
    assert _is_fallback({"filtered_method": "path_expected_fallback"})
    assert not _is_fallback({"filtered_method": "cluster_refined_visual_match"})


def test_existing_visual_detection():
    assert _is_existing_visual({"filter_accepted": "1", "matched_reference_image": "ref.jpg"})
    assert not _is_existing_visual({"filtered_method": "path_expected_fallback", "matched_reference_image": "ref.jpg"})


def test_candidate_quality_requires_accepted_by_default():
    row = {"matched_reference_image": "ref.jpg", "filter_accepted": "0", "confidence": "100", "homography_inliers": "20", "good_matches": "40", "inlier_ratio": "0.5"}
    ok, reason = _candidate_quality_ok(row, Args())
    assert not ok
    assert reason == "candidate_not_accepted"


def test_candidate_quality_accepts_good_candidate():
    row = {"matched_reference_image": "ref.jpg", "filter_accepted": "1", "confidence": "6", "homography_inliers": "10", "good_matches": "20", "inlier_ratio": "0.3"}
    ok, reason = _candidate_quality_ok(row, Args())
    assert ok
    assert reason == "candidate_quality_ok"


def test_candidate_quality_accepts_pipeline_column_names():
    row = {
        "matched_reference_image": "ref.jpg",
        "filter_accepted": "1",
        "retrieval_similarity": "7.25",
        "homography_inliers": "10",
        "orb_good_matches": "28",
        "verification_inlier_ratio": "0.357",
    }
    ok, reason = _candidate_quality_ok(row, Args())
    assert ok
    assert reason == "candidate_quality_ok"


def test_candidate_quality_computes_ratio_from_pipeline_columns():
    row = {
        "matched_reference_image": "ref.jpg",
        "filter_accepted": "1",
        "confidence": "8",
        "homography_inliers": "9",
        "orb_good_matches": "18",
    }
    ok, reason = _candidate_quality_ok(row, Args())
    assert ok
    assert reason == "candidate_quality_ok"
