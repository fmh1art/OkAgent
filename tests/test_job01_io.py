import json

from okagent.job01_io import resolve_job01


def test_resolve_job01_without_loading_resume_rows(tmp_path):
    raw = tmp_path / "raw"
    descriptions = raw / "job-description-20260722"
    embeddings = raw / "job-candidate-embedding-20260722"
    descriptions.mkdir(parents=True)
    embeddings.mkdir()
    (descriptions / "01_test.json").write_text(
        json.dumps({"job_title": "test"}), encoding="utf-8")
    (embeddings / "hiring_job01_full_segvec.db").touch()
    (embeddings / "hiring_job01_llm_pass.db").touch()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "raw_root": "raw", "job": "job01", "as_of": "2026-07-22",
    }), encoding="utf-8")
    data, labels, job, as_of = resolve_job01(config)
    assert data.name == "hiring_job01_full_segvec.db"
    assert labels.name == "hiring_job01_llm_pass.db"
    assert job["job_title"] == "test"
    assert as_of == "2026-07-22"
