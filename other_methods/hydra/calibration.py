"""Hydra's target_recall_threshold calibration, including oracle-cache corrections."""
import numpy as np


def uniform_scores(scores):
    """Global empirical CDF with mid-ranks; equal probabilities remain tied."""
    values, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((2 * ends - counts) / (2 * len(scores)))[inverse]


def calibrate(scores, cache, acquire, rng, sample_size=512, target_recall=0.7,
              include_training=True, failure_probability=-1):
    """Mutates cache only through acquire(); never trains on calibration labels.

    cache[i] is -1 until labeled. acquire(indices) fills it. Sampling is WITH
    replacement; duplicate draws retain multiplicity but purchase one label.
    Returns (probability threshold or None, diagnostics).
    """
    n = len(scores)
    uniform = uniform_scores(scores)
    training = np.flatnonzero(cache >= 0)
    population = np.flatnonzero(cache < 0) if include_training else np.arange(n)
    size = min(sample_size, len(population))
    census = size == len(population)
    info = dict(success=False, draws=size, census=census)
    if not census and size < 2:
        return None, dict(info, reason="fewer than two calibration draws")
    if census:
        draws = population
        correction = np.full(size, 1 / n)
    else:
        roots = np.sqrt(uniform[population])
        distribution = 0.9 * roots / roots.sum() + 0.1 / len(population)
        offsets = rng.choice(len(population), size=size, replace=True, p=distribution)
        draws = population[offsets]
        correction = 1 / (n * distribution[offsets])
    # Preserve first occurrence order, as Hydra does when scheduling oracle rows.
    uncached = np.array(list(dict.fromkeys(int(i) for i in draws if cache[i] < 0)), dtype=int)
    acquire(uncached)
    info.update(indices=draws.tolist(), correction_factors=correction.tolist(),
                unique_new_labels=len(uncached))
    overrides = uncached if include_training else np.concatenate((training, uncached))
    positive_overrides = overrides[cache[overrides] == 1]
    exact = np.count_nonzero(cache[training] == 1) / n if include_training else 0.0
    weights = correction * (cache[draws] == 1)
    # Prefix sums implement the original threshold sweep without an N x draws loop.
    order = np.argsort(uniform[draws], kind="stable")
    draw_scores = uniform[draws][order]
    prefix = np.r_[0.0, np.cumsum(weights[order])]
    divisor = 1 if census else size
    total_mass = prefix[-1] / divisor
    override_scores = np.sort(uniform[positive_overrides])
    thresholds = np.unique(uniform)[::-1]

    def masses(threshold):
        below = prefix[np.searchsorted(draw_scores, threshold, side="left")] / divisor
        restored = np.searchsorted(override_scores, threshold, side="left") / n
        return exact + total_mass - below + restored, np.maximum(0, below - restored)

    def choose(target):
        retained, missed = masses(thresholds)
        total = retained + missed
        recall = np.divide(retained, total, out=np.zeros_like(total), where=total > 0)
        valid = np.flatnonzero((total > 0) & (recall >= target))
        return float(thresholds[valid[0]]) if len(valid) else None

    threshold = choose(target_recall)
    if threshold is None:
        return None, dict(info, reason="no estimated positive mass or qualifying threshold")
    retained_upper, missed_lower = masses(threshold)
    if not census:
        scale = 0 if failure_probability == -1 else np.sqrt(2 * np.log(2 / failure_probability))
        retained_draws = weights * (uniform[draws] >= threshold)
        missed_draws = weights * (uniform[draws] < threshold)
        retained_upper += retained_draws.std(ddof=1) / np.sqrt(size) * scale
        missed_lower -= missed_draws.std(ddof=1) / np.sqrt(size) * scale
    retained_upper = np.clip(retained_upper, 0, 1)
    missed_lower = np.clip(missed_lower, 0, 1)
    if retained_upper + missed_lower <= 0:
        return None, dict(info, reason="nonpositive confidence-bound mass")
    inflated = float(np.clip(retained_upper / (retained_upper + missed_lower), target_recall, 1))
    threshold = choose(inflated)
    if threshold is None:
        return None, dict(info, reason="no qualifying confidence-adjusted threshold")
    original = float(np.min(scores[uniform >= threshold]))
    retained, missed = masses(threshold)
    return original, dict(info, success=True, uniform_threshold=threshold,
                          inflated_target=inflated, estimated_recall=float(retained / (retained + missed)))
