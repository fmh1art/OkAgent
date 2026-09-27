import json
import threading
import time
from types import SimpleNamespace

from benchmarks import batch_label_job01
from okagent.batch_teacher import BatchResult, LABEL_DIMENSIONS


def test_batch_requests_run_concurrently_but_persist_complete_ledger(tmp_path, monkeypatch):
    ids = [f"c{index}" for index in range(8)]
    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps(ids), encoding="utf-8")
    llm_config = tmp_path / "llm.json"
    llm_config.write_text(json.dumps({
        "openai_base_url": "https://example.test/v3",
        "llm_name": "test-model", "key": "not-a-real-key",
    }), encoding="utf-8")
    monkeypatch.setattr(batch_label_job01, "resolve_job01",
                        lambda path: (tmp_path / "data.db", tmp_path / "truth.db",
                                      {"job_title": "test"}, "2026-07-22"))
    monkeypatch.setattr(batch_label_job01, "load_candidates",
                        lambda data, wanted, **kwargs: {
                            cid: {"candidate_id": cid, "sections": {"work": "x"}}
                            for cid in wanted})
    threads = set()

    def fake_call_batch(**kwargs):
        threads.add(threading.get_ident())
        time.sleep(.02)
        candidates = kwargs["candidates"]
        labels = [{"candidate_id": row["candidate_id"],
                   **{field: .5 for field in LABEL_DIMENSIONS},
                   "missing_requirements": []} for row in candidates]
        return BatchResult("-".join(row["candidate_id"] for row in candidates),
                           labels, 100, 20, "test-model", 1), None

    monkeypatch.setattr(batch_label_job01, "call_batch", fake_call_batch)
    output = tmp_path / "output"
    report = batch_label_job01.run(SimpleNamespace(
        config=tmp_path / "hiring.json", llm_config=llm_config, ids=ids_path,
        output=output, batch_size=2, workers=3, max_chars=100,
        per_section_chars=50, attempts=1,
    ))
    assert report["complete"] is True
    assert report["usage"]["completed_requests"] == 4
    assert report["mean_candidates_per_completed_request"] == 2
    assert report["completed_request_reduction_vs_single"] == .5
    assert len(threads) >= 2
