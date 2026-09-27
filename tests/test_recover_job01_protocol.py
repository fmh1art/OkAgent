import json
import sqlite3

from benchmarks.recover_job01_protocol import export_source, inventory


def test_inventory_finds_seed11_ledger_without_exposing_ids(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "job": "job01", "seed": 11, "budget": 2000,
    }), encoding="utf-8")
    ledger = run / "teacher.sqlite"
    with sqlite3.connect(ledger) as db:
        db.execute("CREATE TABLE labels(candidate_id TEXT,status TEXT)")
        db.executemany("INSERT INTO labels VALUES (?,?)", [
            ("a", "complete"), ("b", "complete"), ("c", "failed")])
    rows = inventory(tmp_path)
    assert len(rows) == 1
    assert rows[0]["count"] == 2
    assert "ids" not in rows[0]
    assert len(rows[0]["sha256"]) == 64
    assert rows[0]["metadata"]["seed"] == 11


def test_export_source_freezes_sorted_unique_ids(tmp_path):
    source = tmp_path / "sampled_ids.json"
    source.write_text(json.dumps(["b", "a", "b"]), encoding="utf-8")
    output = tmp_path / "frozen.json"
    result = export_source(source, output)
    assert json.loads(output.read_text(encoding="utf-8")) == ["a", "b"]
    assert result["count"] == 2
