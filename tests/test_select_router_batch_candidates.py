import numpy as np

from benchmarks.select_router_batch_candidates import (allocate_counts,
                                                        select_candidates)


def test_acquisition_is_exact_unique_and_excludes_existing_labels():
    count = 120
    ids = [f"c{index}" for index in range(count)]
    base = np.linspace(0, 1, count)
    scores = {
        "global": base,
        "experience": base[::-1],
        "credentials": np.abs(base - .5),
        "expert_mean": np.full(count, .4),
        "stacking_router": base,
        "embedding_router": np.clip(base + np.sin(np.arange(count)) * .2, 0, 1),
    }
    excluded = set(ids[:20])
    selected, trace = select_candidates(ids, scores, excluded=excluded, budget=50, seed=3)
    assert len(selected) == len(set(selected)) == len(trace) == 50
    assert not set(selected) & excluded
    assert {row["strategy"] for row in trace}.issuperset({
        "top", "boundary", "expert_disagreement", "router_disagreement", "coverage"})
    assert selected == select_candidates(ids, scores, excluded=excluded,
                                         budget=50, seed=3)[0]


def test_fraction_allocation_spends_exact_budget():
    counts = allocate_counts(17, {
        "top": .3, "boundary": .25, "expert_disagreement": .25,
        "router_disagreement": .1, "coverage": .1,
    })
    assert sum(counts.values()) == 17
