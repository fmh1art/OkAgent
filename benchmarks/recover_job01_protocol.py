"""Inventory and freeze candidate-ID protocols from historical job01 runs.

The inventory never writes raw candidate IDs.  After reviewing its hashes and
manifest metadata, use ``--export-source`` to freeze one exact source explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


JSON_ID_NAMES = {
    "sampled_ids.json", "train_ids.json", "active_ids.json", "universe_ids.json",
    "pool_ids.json", "validation_ids.json", "unsampled_ids.json",
}


def read_json_ids(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        # Some split artifacts keep several ID arrays in one object. Their union
        # is the actual reserved teacher set only when explicitly exported later.
        arrays = [part for key, part in value.items()
                  if key.endswith("_ids") and isinstance(part, list)]
        if not arrays:
            raise ValueError("JSON object contains no *_ids arrays")
        value = [item for part in arrays for item in part]
    if not isinstance(value, list):
        raise ValueError("candidate ID artifact is not a list")
    ids = [row.get("candidate_id") if isinstance(row, dict) else row for row in value]
    if any(not isinstance(cid, str) or not cid for cid in ids):
        raise ValueError("candidate IDs must be non-empty strings")
    return sorted(set(ids))


def read_sqlite_ids(path: Path) -> list[str]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "labels" not in tables:
            raise ValueError("no labels table")
        columns = {row[1] for row in db.execute("PRAGMA table_info(labels)")}
        if "candidate_id" not in columns:
            raise ValueError("labels table has no candidate_id")
        where = " WHERE status='complete'" if "status" in columns else ""
        ids = [row[0] for row in db.execute(
            "SELECT DISTINCT candidate_id FROM labels" + where + " ORDER BY candidate_id")]
    if any(not isinstance(cid, str) or not cid for cid in ids):
        raise ValueError("ledger contains invalid candidate IDs")
    return ids


def id_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def nearby_manifest(path: Path, root: Path) -> tuple[str | None, dict]:
    current = path.parent
    while current == root or root in current.parents:
        for name in ("manifest.json", "experiment.json", "report.json"):
            candidate = current / name
            if candidate.is_file() and candidate != path:
                try:
                    value = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                manifest = value.get("manifest", value) if isinstance(value, dict) else {}
                keep = {key: manifest.get(key) for key in (
                    "job", "seed", "budget", "max_calls", "teacher", "method",
                    "validation_count", "cold_count", "active_count",
                ) if manifest.get(key) is not None}
                return str(candidate.relative_to(root)), keep
        if current == root:
            break
        current = current.parent
    return None, {}


def inspect_source(path: Path, root: Path) -> dict | None:
    try:
        ids = read_sqlite_ids(path) if path.suffix in (".sqlite", ".db") else read_json_ids(path)
    except (OSError, ValueError, sqlite3.Error, json.JSONDecodeError):
        return None
    if not ids:
        return None
    manifest_path, metadata = nearby_manifest(path, root)
    return {
        "source": str(path.relative_to(root)), "kind": "sqlite_labels" if path.suffix in (
            ".sqlite", ".db") else "json_ids",
        "count": len(ids), "sha256": id_hash(ids),
        "likely_seed11_budget2000": len(ids) == 2_000 and metadata.get("seed") == 11,
        "manifest": manifest_path, "metadata": metadata,
    }


def inventory(root: Path) -> list[dict]:
    root = root.resolve()
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in JSON_ID_NAMES or path.name == "split.json" or path.suffix == ".sqlite":
            item = inspect_source(path, root)
            if item is not None:
                candidates.append(item)
    return sorted(candidates, key=lambda row: (
        not row["likely_seed11_budget2000"], abs(row["count"] - 2_000), row["source"]))


def export_source(source: Path, output: Path) -> dict:
    ids = read_sqlite_ids(source) if source.suffix in (".sqlite", ".db") else read_json_ids(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"source": str(source.resolve()), "output": str(output.resolve()),
            "count": len(ids), "sha256": id_hash(ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports/protocol/job01_id_source_inventory.json"))
    parser.add_argument("--export-source", type=Path)
    parser.add_argument("--export-output", type=Path,
                        default=Path("reports/protocol/job01_seed11_sampled_ids.json"))
    args = parser.parse_args()
    if args.export_source:
        result = export_source(args.export_source, args.export_output)
    else:
        rows = inventory(args.root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        result = {"sources": len(rows), "likely_seed11_budget2000": sum(
            row["likely_seed11_budget2000"] for row in rows), "output": str(args.output)}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
