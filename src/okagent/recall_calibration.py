"""Recall-constrained ranking metrics and repeated threshold calibration."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def classification_metrics(labels: Sequence[int], scores: Sequence[float],
                           threshold: float) -> dict:
    y = np.asarray(labels, dtype=np.int8)
    probability = np.asarray(scores, dtype=np.float64)
    predicted = probability >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "count": len(y), "positive": int(y.sum()), "selected": int(predicted.sum()),
        "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def clopper_pearson_lower(successes: int, trials: int, *, alpha: float = 0.05) -> float:
    if not 0 < alpha < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial confidence interval inputs")
    if successes == 0:
        return 0.0
    from scipy.stats import beta
    return float(beta.ppf(alpha, successes, trials - successes + 1))


def select_recall_threshold(labels: Sequence[int], scores: Sequence[float],
                            target_recall: float, *, require_lower_bound: bool = False,
                            alpha: float = 0.05) -> tuple[float, dict]:
    """Choose maximum precision among tied-score thresholds satisfying recall."""
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0,1]")
    y = np.asarray(labels, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    if y.shape != values.shape or y.ndim != 1 or y.sum() == 0:
        raise ValueError("labels/scores must align and contain a positive")
    order = np.lexsort((np.arange(len(values)), -values))
    y, values = y[order], values[order]
    positives = int(y.sum())
    tp = fp = 0
    candidates = []
    index = 0
    while index < len(values):
        threshold = float(values[index])
        end = index
        while end < len(values) and values[end] == threshold:
            if y[end]:
                tp += 1
            else:
                fp += 1
            end += 1
        recall = tp / positives
        lower = clopper_pearson_lower(tp, positives, alpha=alpha)
        constraint = lower if require_lower_bound else recall
        if constraint >= target_recall:
            precision = tp / (tp + fp)
            candidates.append((precision, threshold, recall, lower, tp, fp))
        index = end
    if not candidates:
        raise ValueError("no threshold satisfies the requested recall constraint")
    precision, threshold, recall, lower, tp, fp = max(candidates)
    return threshold, {
        "threshold": threshold, "precision": precision, "recall": recall,
        "recall_lower_bound": lower, "tp": tp, "fp": fp,
        "target_recall": target_recall, "lower_bound_required": require_lower_bound,
        "alpha": alpha,
    }


def repeated_calibration(labels: Sequence[int], scores: Sequence[float], *,
                         target_recall: float, seeds: Sequence[int],
                         calibration_fraction: float = 0.2,
                         require_lower_bound: bool = False,
                         alpha: float = 0.05) -> dict:
    from sklearn.model_selection import train_test_split

    y = np.asarray(labels, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    indices = np.arange(len(y))
    runs = []
    for seed in seeds:
        test_idx, calibration_idx = train_test_split(
            indices, test_size=calibration_fraction, stratify=y, random_state=seed,
        )
        try:
            threshold, calibration = select_recall_threshold(
                y[calibration_idx], values[calibration_idx], target_recall,
                require_lower_bound=require_lower_bound, alpha=alpha,
            )
            test = classification_metrics(y[test_idx], values[test_idx], threshold)
            test["recall_lower_bound"] = clopper_pearson_lower(
                test["tp"], test["tp"] + test["fn"], alpha=alpha)
            runs.append({"seed": int(seed), "valid": True, "calibration": calibration,
                         "test": test, "threshold": threshold})
        except ValueError as exc:
            runs.append({"seed": int(seed), "valid": False, "reason": str(exc)})
    valid = [row for row in runs if row["valid"]]
    if not valid:
        return {"runs": runs, "valid_runs": 0, "target_recall": target_recall}
    summary = {
        "valid_runs": len(valid), "run_count": len(runs), "target_recall": target_recall,
        "recall_achievement_rate": sum(
            row["test"]["recall"] >= target_recall for row in valid) / len(valid),
        "mean_precision": float(np.mean([row["test"]["precision"] for row in valid])),
        "std_precision": float(np.std([row["test"]["precision"] for row in valid])),
        "mean_recall": float(np.mean([row["test"]["recall"] for row in valid])),
        "std_recall": float(np.std([row["test"]["recall"] for row in valid])),
        "mean_f1": float(np.mean([row["test"]["f1"] for row in valid])),
        "std_f1": float(np.std([row["test"]["f1"] for row in valid])),
        "worst_recall": min(row["test"]["recall"] for row in valid),
        "mean_test_recall_lower_bound": float(np.mean([
            row["test"]["recall_lower_bound"] for row in valid])),
        "mean_threshold": float(np.mean([row["threshold"] for row in valid])),
        "std_threshold": float(np.std([row["threshold"] for row in valid])),
        "lower_bound_required_on_calibration": require_lower_bound,
        "alpha": alpha,
    }
    return {**summary, "runs": runs}
