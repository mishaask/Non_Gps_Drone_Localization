from gps_ex1.tools.stage10_local_best_cluster_refine import _score_candidate, _haversine_m, _build_candidate_map


def test_score_candidate_prefers_accepted_and_inliers():
    weak = {"confidence": "5", "homography_inliers": "2", "good_matches": "10", "filter_accepted": "0"}
    strong = {"confidence": "5", "homography_inliers": "12", "good_matches": "20", "filter_accepted": "1"}
    assert _score_candidate(strong) > _score_candidate(weak)


def test_haversine_zero():
    assert _haversine_m(32.1, 35.2, 32.1, 35.2) < 1e-6
