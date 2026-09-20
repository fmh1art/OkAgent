import json

import pytest

from okagent.qwen_online_trainer import (load_resume_texts, merge_labels,
                                         resume_chunks, select_threshold)


def test_full_unstructured_text_joins_all_rows(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    with duckdb.connect(str(tmp_path / "data.duckdb")) as con:
        con.execute("CREATE TABLE candidate_segments(candidate_id VARCHAR, segment VARCHAR, text VARCHAR)")
        con.executemany("INSERT INTO candidate_segments VALUES (?,?,?)", [
            ("a", "unstructured", "later"), ("a", "unstructured", "earlier"),
            ("a", "structured", "exclude me"), ("b", "unstructured", "other")])
    texts = load_resume_texts(tmp_path, ["a"])
    assert texts == {"a": "earlier\n\nlater"}


def test_resume_chunks_cover_every_token_without_truncation():
    class FakeTokenizer:
        def __call__(self, value, **kwargs):
            assert kwargs == {"add_special_tokens": False, "truncation": False}
            return {"input_ids": [ord(char) + 3 for char in value]}

        def build_inputs_with_special_tokens(self, ids):
            return [1, *ids, 2]

    tokenizer = FakeTokenizer()
    context, resume = "岗位要求", "ABCDEFGHIJKLMNO"
    from okagent.qwen_online_trainer import _prompt_parts
    prefix, suffix = _prompt_parts(context)
    prefix_size, suffix_size = len(prefix), len(suffix)
    maximum = prefix_size + suffix_size + 2 + 4
    chunks = resume_chunks(tokenizer, context, resume, maximum)
    assert len(chunks) == 4
    assert all(len(chunk) <= maximum for chunk in chunks)
    recovered = [token - 3 for chunk in chunks
                 for token in chunk[1 + prefix_size:-(suffix_size + 1)]]
    assert recovered == [ord(char) for char in resume]
    with pytest.raises(ValueError, match="cannot fit"):
        resume_chunks(tokenizer, context, resume, prefix_size + suffix_size + 2)

    class NoSpecialBuilder:
        def __call__(self, value, **kwargs):
            return tokenizer(value, **kwargs)

    plain = resume_chunks(NoSpecialBuilder(), context, resume,
                          prefix_size + suffix_size + 4)
    assert len(plain) == 4
    assert all(len(chunk) <= prefix_size + suffix_size + 4 for chunk in plain)


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
