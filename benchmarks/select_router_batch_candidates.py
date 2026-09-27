"""Select additional training candidates from multi-proxy/router disagreement.

Selection consumes only frozen model scores and an exclusion set. It never opens
historical labels, so the resulting IDs are safe to send to the batch teacher.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


STRATEGIES = ("top", "boundary", "expert_disagreement", "router_disagreement", "coverage")
DEFAULT_FRACTIONS = {
    "top": .30,
    "boundary": .25,
    "expert_disagreement": .25,
    "router_disagreement": .10,
    "coverage": .10,
}


def read_ids(path: Path) -> list[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    ids = [row.get("candidate_id") if isinstance(row, dict) else row for row in rows]
    if any(not isinstance(cid, str) or not cid for cid in ids):
        raise ValueError(f"invalid candidate IDs in {path}")
    return ids


def allocate_counts(budget: int, fractions: dict[str, float]) -> dict[str, int]:
    if budget <= 0 or set(fractions) != set(STRATEGIES) or any(v < 0 for v in fractions.values()):
        raise ValueError("invalid acquisition budget or fractions")
    total = sum(fractions.values())
    if total <= 0:
        raise ValueError("acquisition fractions sum to zero")
    raw = {name: budget * value / total for name, value in fractions.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = budget - sum(counts.values())
    order = sorted(STRATEGIES, key=lambda name: (-(raw[name] - counts[name]), name))
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def select_candidates(candidate_ids: list[str], scores: dict[str, np.ndarray], *,
                      excluded: set[str], budget: int, seed: int = 11,
                      fractions: dict[str, float] | None = None) -> tuple[list[str], list[dict]]:
    required = {"global", "experience", "credentials", "stacking_router"}
    if not required.issubset(scores):
        raise ValueError(f"scores are missing {sorted(required - set(scores))}")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be unique")
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in scores.items()}
    if any(value.shape != (len(candidate_ids),) for value in arrays.values()):
        raise ValueError("score arrays must align with candidate IDs")
    available = np.asarray([cid not in excluded for cid in candidate_ids], dtype=bool)
    if budget > int(available.sum()):
        raise ValueError("acquisition budget exceeds available candidates")
    counts = allocate_counts(budget, fractions or DEFAULT_FRACTIONS)
    experts = np.column_stack([arrays[name] for name in ("global", "experience", "credentials")])
    stacking = arrays["stacking_router"]
    if "embedding_router" in arrays:
        embedding = arrays["embedding_router"]
    elif "expert_mean" in arrays:
        embedding = arrays["expert_mean"]
    else:
        raise ValueError("scores need embedding_router or expert_mean")
    rng = np.random.default_rng(seed)
    jitter = rng.random(len(candidate_ids)) * 1e-12
    criteria = {
        "top": -stacking + jitter,
        "boundary": np.abs(stacking - .5) + jitter,
        "expert_disagreement": -np.std(experts, axis=1) + jitter,
        "router_disagreement": -np.abs(stacking - embedding) + jitter,
        "coverage": rng.random(len(candidate_ids)),
    }
    selected = []
    trace = []
    chosen = set(excluded)
    for strategy in STRATEGIES:
        if counts[strategy] == 0:
            continue
        order = np.argsort(criteria[strategy], kind="stable")
        taken = 0
        for index in order:
            cid = candidate_ids[int(index)]
            if not available[index] or cid in chosen:
                continue
            selected.append(cid)
            chosen.add(cid)
            trace.append({
                "candidate_id": cid, "strategy": strategy,
                "stacking_score": float(stacking[index]),
                "embedding_router_score": float(embedding[index]),
                "expert_scores": {name: float(arrays[name][index]) for name in (
                    "global", "experience", "credentials")},
                "expert_std": float(np.std(experts[index])),
            })
            taken += 1
            if taken == counts[strategy]:
                break
    # Heavy overlap between strategies can leave a quota short. Fill strictly by
    # model uncertainty and record the fallback rather than silently under-spend.
    if len(selected) < budget:
        for index in np.argsort(criteria["boundary"], kind="stable"):
            cid = candidate_ids[int(index)]
            if available[index] and cid not in chosen:
                selected.append(cid)
                chosen.add(cid)
                trace.append({"candidate_id": cid, "strategy": "boundary_fill",
                              "stacking_score": float(stacking[index])})
                if len(selected) == budget:
                    break
    if len(selected) != budget or len(selected) != len(set(selected)):
        raise AssertionError("acquisition did not produce the exact unique budget")
    return selected, trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("router_output", type=Path,
                        help="Directory containing candidate_ids.json and scores.npz")
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate_ids = read_ids(args.router_output / "candidate_ids.json")
    with np.load(args.router_output / "scores.npz") as archive:
        scores = {name: archive[name] for name in archive.files}
    excluded = {cid for path in args.exclude for cid in read_ids(path)}
    selected, trace = select_candidates(candidate_ids, scores, excluded=excluded,
                                        budget=args.budget, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "candidate_ids.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "selection_trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"budget": args.budget, "selected": len(selected), "excluded": len(excluded),
               "seed": args.seed, "strategy_counts": {
                   name: sum(row["strategy"] == name for row in trace)
                   for name in set(row["strategy"] for row in trace)},
               "historical_labels_read": False}
    (args.output / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
