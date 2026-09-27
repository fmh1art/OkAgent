import json

import pytest

from okagent.batch_teacher import (BatchResult, compact_resume, export_labels,
                                   parse_batch_response, save_failure, save_result, usage)


def test_parse_batch_response_validates_ids_and_normalizes_scores():
    content = json.dumps({"candidates": [
        {"candidate_id": "a", "overall": 80, "education": 70,
         "experience": 60, "technical_skills": 90, "projects": 75,
         "research": 40, "evidence_quality": 85,
         "missing_requirements": ["实习时长"]},
    ]}, ensure_ascii=False)
    rows = parse_batch_response(content, ["a"])
    assert rows[0]["overall"] == 0.8
    with pytest.raises(ValueError, match="IDs"):
        parse_batch_response(content, ["b"])


def test_compact_resume_preserves_sections_and_limits_text():
    result = compact_resume([
        ("unstructured", "duplicate"), ("work", "abcdefgh"),
        ("projects", "ijklmnop"),
    ], max_chars=7, per_section_chars=4)
    assert result == {"projects": "ijkl", "work": "abc"}


def test_batch_ledger_accounts_request_tokens_once(tmp_path):
    ledger = tmp_path / "labels.sqlite"
    labels = [{"candidate_id": cid, **{field: 0.5 for field in (
        "overall", "education", "experience", "technical_skills", "projects",
        "research", "evidence_quality")}, "missing_requirements": []}
              for cid in ("a", "b")]
    save_result(ledger, BatchResult("r1", labels, 100, 20, "model", 1))
    assert [row["candidate_id"] for row in export_labels(ledger)] == ["a", "b"]
    assert usage(ledger) == {
        "completed_requests": 1, "failed_requests": 0,
        "candidate_slots": 2, "unique_candidates": 2,
        "input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "attempts": 1,
    }


def test_failed_billed_response_is_included_in_usage(tmp_path):
    ledger = tmp_path / "labels.sqlite"
    save_failure(ledger, "bad", 4, 2, "ValueError",
                 input_tokens=80, output_tokens=30, model="model")
    save_failure(ledger, "bad", 4, 1, "ValueError",
                 input_tokens=20, output_tokens=10, model="model")
    assert usage(ledger) == {
        "completed_requests": 0, "failed_requests": 1,
        "candidate_slots": 0, "unique_candidates": 0,
        "input_tokens": 100, "output_tokens": 40, "total_tokens": 140, "attempts": 3,
    }
