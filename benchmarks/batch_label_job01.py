"""Resumable job01 multi-candidate batch prompting with multi-dimensional labels.

One request shares the job description across several structured resumes.  This
is real prompt batching (not merely concurrent single-candidate requests), and
the SQLite ledger accounts token usage once per HTTP request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import requests

from okagent.batch_teacher import (LABEL_DIMENSIONS, BatchResult, SYSTEM_PROMPT,
                                   build_batch_payload, compact_resume, completed_ids,
                                   export_labels, parse_batch_response, request_id,
                                   save_failure, save_result, setup_ledger, usage)
from okagent.job01_io import resolve_job01


def read_ids(path: Path) -> list[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    ids = [row["candidate_id"] if isinstance(row, dict) else row for row in rows]
    if any(not isinstance(cid, str) for cid in ids) or len(ids) != len(set(ids)):
        raise ValueError("ID file must contain unique strings")
    return ids


def load_candidates(data_path: Path, ids: list[str], *, max_chars: int,
                    per_section_chars: int) -> dict[str, dict]:
    wanted = set(ids)
    parts = {cid: [] for cid in ids}
    with duckdb.connect(str(data_path), read_only=True) as db:
        for cid, segment, text in db.execute(
            "SELECT candidate_id,segment,text FROM candidate_segments "
            "WHERE segment<>'unstructured' ORDER BY candidate_id,segment"
        ).fetchall():
            if cid in wanted:
                parts[cid].append((segment, text or ""))
    missing = [cid for cid in ids if not parts[cid]]
    if missing:
        raise ValueError(f"{len(missing)} candidates have no structured sections")
    return {cid: {"candidate_id": cid, "sections": compact_resume(
        parts[cid], max_chars=max_chars, per_section_chars=per_section_chars)} for cid in ids}


def call_batch(*, candidates: list[dict], job: dict, as_of: str, endpoint: str,
               model: str, key: str, protocol_hash: str, attempts: int = 4):
    ids = [row["candidate_id"] for row in candidates]
    batch_id = request_id(ids, protocol_hash)
    payload = {
        "model": model, "temperature": 0, "max_tokens": max(512, 220 * len(ids)),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(
                build_batch_payload(job, as_of, candidates), ensure_ascii=False)},
        ],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    error = "unknown"
    fatal = False
    cumulative_input_tokens = 0
    cumulative_output_tokens = 0
    response_model = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=(10, 180))
            if response.status_code != 200:
                error = f"HTTP {response.status_code}"
                fatal = response.status_code in (400, 401, 402, 403, 404)
                if response.status_code not in (408, 413, 429) and not 500 <= response.status_code < 600:
                    break
            else:
                body = response.json()
                token_usage = body.get("usage", {})
                prompt_tokens = int(token_usage.get("prompt_tokens", 0))
                completion_tokens = int(token_usage.get("completion_tokens", 0))
                if prompt_tokens < 0 or completion_tokens < 0:
                    raise ValueError("negative token usage")
                cumulative_input_tokens += prompt_tokens
                cumulative_output_tokens += completion_tokens
                response_model = body.get("model") or response_model
                rows = parse_batch_response(body["choices"][0]["message"]["content"], ids)
                return BatchResult(
                    batch_id, rows, cumulative_input_tokens, cumulative_output_tokens,
                    response_model, attempt,
                ), None
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            error = type(exc).__name__
        if attempt < attempts:
            time.sleep(min(8, 2 ** (attempt - 1)))
    return None, (batch_id, attempts, error, fatal, cumulative_input_tokens,
                  cumulative_output_tokens, response_model)


def run(args) -> dict:
    data_path, _, job, as_of = resolve_job01(args.config)
    settings = json.loads(args.llm_config.read_text(encoding="utf-8"))
    endpoint = settings["openai_base_url"].rstrip("/") + "/chat/completions"
    model, key = settings["llm_name"], settings["key"]
    ids = read_ids(args.ids)
    candidates = load_candidates(data_path, ids, max_chars=args.max_chars,
                                 per_section_chars=args.per_section_chars)
    protocol = {
        "job": job, "as_of": as_of, "system": SYSTEM_PROMPT, "model": model,
        "max_chars": args.max_chars, "per_section_chars": args.per_section_chars,
        "label_dimensions": list(LABEL_DIMENSIONS),
    }
    protocol_hash = hashlib.sha256(json.dumps(
        protocol, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {**protocol, "protocol_sha256": protocol_hash,
                "requested_candidates": len(ids), "batch_size": args.batch_size}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("saved run uses a different batch-label protocol")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    ledger = args.output / "batch_teacher.sqlite"
    setup_ledger(ledger)
    done = completed_ids(ledger)
    queue = [[candidates[cid] for cid in ids[start:start + args.batch_size] if cid not in done]
             for start in range(0, len(ids), args.batch_size)]
    queue = [batch for batch in queue if batch]
    failed_singletons = []
    while queue:
        wave = [queue.pop(0) for _ in range(min(args.workers, len(queue)))]
        retry_batches = []
        fatal_errors = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(
                call_batch, candidates=batch, job=job, as_of=as_of,
                endpoint=endpoint, model=model, key=key,
                protocol_hash=protocol_hash, attempts=args.attempts): batch
                for batch in wave}
            for future in as_completed(futures):
                batch = futures[future]
                result, failure = future.result()
                # SQLite writes stay on this orchestration thread even though HTTP
                # requests run concurrently.
                if result is not None:
                    save_result(ledger, result)
                    continue
                (batch_id, attempts, error, fatal, input_tokens,
                 output_tokens, response_model) = failure
                save_failure(ledger, batch_id, len(batch), attempts, error,
                             input_tokens=input_tokens, output_tokens=output_tokens,
                             model=response_model)
                if fatal:
                    fatal_errors.append(error)
                elif len(batch) > 1:
                    middle = len(batch) // 2
                    retry_batches.extend([batch[:middle], batch[middle:]])
                else:
                    failed_singletons.append(batch[0]["candidate_id"])
        if fatal_errors:
            raise RuntimeError(f"fatal batch-teacher error: {fatal_errors[0]}")
        queue[0:0] = retry_batches
        print(json.dumps({"labeled": len(completed_ids(ledger)), "requested": len(ids),
                          "queued_batches": len(queue), "failed_singletons": len(failed_singletons)}),
              flush=True)
    labels = export_labels(ledger)
    (args.output / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    token_usage = usage(ledger)
    unique = max(token_usage["unique_candidates"], 1)
    report = {"complete": len(labels) == len(ids), "usage": token_usage,
              "failed_candidate_ids": failed_singletons,
              "mean_candidates_per_completed_request": (
                  len(labels) / max(token_usage["completed_requests"], 1)),
              "input_tokens_per_labeled_candidate": token_usage["input_tokens"] / unique,
              "output_tokens_per_labeled_candidate": token_usage["output_tokens"] / unique,
              "completed_request_reduction_vs_single": 1 - (
                  token_usage["completed_requests"] / unique)}
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--ids", type=Path, required=True,
                   help="Frozen candidate IDs; use a training-only set, never test IDs")
    p.add_argument("--config", type=Path, default=Path("_config/hiring.json"))
    p.add_argument("--llm-config", type=Path, default=Path("_config/llm.json"))
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--workers", type=int, default=4,
                   help="Concurrent multi-candidate requests; SQLite writes remain serialized")
    p.add_argument("--max-chars", type=int, default=8_000)
    p.add_argument("--per-section-chars", type=int, default=2_000)
    p.add_argument("--attempts", type=int, default=4)
    return p


def main() -> None:
    args = parser().parse_args()
    if min(args.batch_size, args.workers, args.max_chars,
           args.per_section_chars, args.attempts) <= 0:
        raise ValueError("batch and retry settings must be positive")
    print(json.dumps(run(args), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
