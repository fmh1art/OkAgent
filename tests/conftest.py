import json

import duckdb
import pytest

from okagent.data import prepare


@pytest.fixture(autouse=True)
def isolated_llm_config(tmp_path, monkeypatch):
    monkeypatch.setenv("OKAGENT_LLM_CONFIG", str(tmp_path / "no-llm.json"))


@pytest.fixture
def source(tmp_path):
    raw = tmp_path / "raw"
    descriptions = raw / "job-description-20260722"
    databases = raw / "job-candidate-embedding-20260722"
    descriptions.mkdir(parents=True)
    databases.mkdir()
    job = dict(job_id="123", job_title="测试岗位", must_have_qualifications=["有相关经历"],
               nice_to_have_qualifications=["可选技能"], jd="详细职位描述")
    (descriptions / "01_123.json").write_text(json.dumps(job), encoding="utf-8")
    (descriptions / "06_456.json").write_text(json.dumps(job), encoding="utf-8")
    (raw / "llm_prompt.txt").write_text(
        '时间 {{ current_time }}\n测试匹配规则\n# 岗位画像\n{{ job_info.job_title }}\n'
        '{% for qual in job_info.must_have_qualifications %}{{ qual }}\n{% endfor %}'
        '# 候选人画像\n{{ candidate.education_experience }}\n# 输出要求\n'
        '{"candidate_id": "{{ candidate.candidate_id }}", "is_match": {"result": true}}', encoding="utf-8")
    candidates = databases / "hiring_job01_full_segvec.db"
    with duckdb.connect(str(candidates)) as con:
        con.execute("CREATE TABLE candidate_segments(candidate_id VARCHAR, segment VARCHAR, text VARCHAR, vec FLOAT[3])")
        con.executemany("INSERT INTO candidate_segments VALUES (?,?,?,?)", [
            (cid, "education", "测试文本 " + cid, [1, 2, 3]) for cid in "abcdef"])
        con.execute("INSERT INTO candidate_segments VALUES ('a','notes',NULL,NULL)")
        con.execute("CREATE TABLE _meta(key VARCHAR,value VARCHAR)")
        con.execute("INSERT INTO _meta VALUES ('secret','HIDDEN_METADATA')")
    with duckdb.connect(str(databases / "hiring_job01_llm_pass.db")) as con:
        con.execute("CREATE TABLE llm_pass(candidate_id VARCHAR,llm_pass TINYINT)")
        con.executemany("INSERT INTO llm_pass VALUES (?,?)", [(cid, int(cid in "ace")) for cid in "abcdef"])
        con.execute("INSERT INTO llm_pass VALUES ('outside-library',1)")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(raw_root=str(raw), job="job01", as_of="2026-07-22", max_calls=3)))
    return config


@pytest.fixture
def run(source, tmp_path):
    run_dir = tmp_path / "run"
    prepare(source, run_dir)
    return run_dir
