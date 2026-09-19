import json

import pytest

from okagent.qwen_online_trainer import merge_labels, select_threshold


def test_merge_labels_deduplicates_and_rejects_conflicts(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps([
        {"candidate_id": "b", "label": 0},
        {"candidate_id": "a", "label": 1},
    ]))
    second.write_text(json.dumps([
        {"candidate_id": "a", "label": 1},
        {"candidate_id": "c", "label": 0},
    ]))
    assert merge_labels([first, second]) == [
        {"candidate_id": "a", "label": 1},
        {"candidate_id": "b", "label": 0},
        {"candidate_id": "c", "label": 0},
    ]
    second.write_text(json.dumps([{"candidate_id": "a", "label": 0}]))
    with pytest.raises(ValueError, match="conflicting labels"):
        merge_labels([first, second])


def test_threshold_prefers_precision_while_meeting_recall():
    threshold, metrics = select_threshold(
        [1, 1, 0, 0], [0.9, 0.8, 0.7, 0.1], target_recall=0.9,
        min_validation_positive=2,
    )
    assert threshold == pytest.approx(0.8)
    assert metrics["proxy_valid"] is True
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_threshold_requires_both_validation_classes():
    threshold, metrics = select_threshold([0, 0], [0.2, 0.1], target_recall=0.9)
    assert threshold is None
    assert metrics["proxy_valid"] is False


def test_threshold_rejects_too_few_validation_positives():
    threshold, metrics = select_threshold([1, 0], [0.9, 0.1], target_recall=0.9)
    assert threshold is None
    assert metrics["proxy_valid"] is False
    assert metrics["validation_positive"] == 1
