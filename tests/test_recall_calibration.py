import numpy as np

from okagent.recall_calibration import (classification_metrics,
                                        clopper_pearson_lower,
                                        repeated_calibration,
                                        select_recall_threshold)


def test_threshold_meets_recall_and_keeps_score_ties_together():
    threshold, result = select_recall_threshold(
        [1, 0, 1, 0, 1], [0.9, 0.8, 0.8, 0.2, 0.1], 2 / 3,
    )
    assert threshold == 0.8
    assert result["tp"] == 2
    assert result["fp"] == 1
    assert classification_metrics([1, 0, 1, 0, 1],
                                  [0.9, 0.8, 0.8, 0.2, 0.1], threshold)["selected"] == 3


def test_clopper_pearson_lower_bound_is_conservative():
    assert clopper_pearson_lower(0, 10) == 0
    assert 0.7 < clopper_pearson_lower(10, 10) < 1


def test_repeated_calibration_reports_all_seeds():
    labels = np.array([0] * 160 + [1] * 40)
    scores = np.r_[np.linspace(0, .7, 160), np.linspace(.4, 1, 40)]
    result = repeated_calibration(labels, scores, target_recall=.8,
                                  seeds=[1, 2, 3], calibration_fraction=.25)
    assert result["valid_runs"] == 3
    assert len(result["runs"]) == 3
    assert 0 <= result["recall_achievement_rate"] <= 1
