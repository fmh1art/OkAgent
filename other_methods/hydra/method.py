"""Standalone reproduction of the archive's configured ML/active-learning path."""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .calibration import calibrate


@dataclass(frozen=True)
class Config:
    sample_size: int = 1024
    calibration_sample_size: int = 512
    target_recall: float = 0.7
    calibration_include_training: bool = True
    failure_probability: float = -1
    lr_lambda: float = 0.0
    lr_threshold: float = 0.5
    step: int = 32
    adaptive: bool = True
    stop_fraction: float = 1.0
    stop_threshold: float = 0.995
    patience: int = 3
    max_calls: int = 2000
    batch_size: int = 1
    seed: int = 42

    def validate(self):
        for name in ("sample_size", "step", "patience", "max_calls", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.calibration_sample_size) is not int or self.calibration_sample_size < 0:
            raise ValueError("calibration_sample_size must be a nonnegative integer")
        if not 0 < self.target_recall < 1 or not 0 <= self.stop_fraction <= 1:
            raise ValueError("target_recall must be in (0,1); stop_fraction in [0,1]")
        if self.failure_probability != -1 and not 0 < self.failure_probability < 1:
            raise ValueError("failure_probability must be -1 or in (0,1)")
        if not np.isfinite(self.lr_lambda) or self.lr_lambda < 0 or not np.isfinite(self.lr_threshold):
            raise ValueError("Invalid logistic-regression parameters")
        if not -1 <= self.stop_threshold <= 1:
            raise ValueError("stop_threshold must be in [-1,1]")


def fit_logistic(features, labels, regularization=0):
    """mlpack objective: summed log loss + lambda/2 * ||w||², intercept unpenalized."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)

    def objective(parameters):
        w, intercept = parameters[:-1], parameters[-1]
        logits = x @ w + intercept
        loss = np.logaddexp(0, (1 - 2 * y) * logits).sum() + regularization / 2 * (w @ w)
        residual = expit(logits) - y
        gradient = np.r_[x.T @ residual + regularization * w, residual.sum()]
        return loss, gradient

    result = minimize(objective, np.zeros(x.shape[1] + 1), jac=True, method="L-BFGS-B",
                      options={"maxiter": 1000, "gtol": 1e-6, "ftol": 1e-12})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Logistic-regression optimization failed: {result.message}")
    return result.x


def similarity_extremes(available, scores, size):
    order = available[np.lexsort((available, scores[available]))]
    low = size // 2
    return np.sort(np.r_[order[:low], order[::-1][:size - low]])


def active_batch(available, labels, probabilities, size, rng):
    """10% random exploration, minority enrichment, then boundary uncertainty."""
    random_n = min(int(np.floor(0.1 * size + 0.5)), size)  # C++ round, not Python's bankers' rounding.
    selected = list(rng.choice(available, random_n, replace=False))
    available = available[~np.isin(available, selected)]
    boundary = 1 - 2 * np.abs(probabilities - 0.5)
    positive = np.count_nonzero(labels == 1)
    minority = min(positive, len(labels) - positive)

    def append_ranked(score, count):
        nonlocal available
        ranked = available[np.lexsort((available, -score[available]))][:count]
        selected.extend(ranked)
        available = available[~np.isin(available, ranked)]

    if minority / len(labels) < 0.35:
        needed = max(0, int(np.ceil(0.35 * (len(labels) + size) - minority)))
        quota = min(needed, int(np.floor(0.8 * size)), size - len(selected))
        minority_prob = probabilities if positive < len(labels) - positive else 1 - probabilities
        append_ranked(boundary + 0.75 * minority_prob, quota)
    append_ranked(boundary, size - len(selected))
    return np.sort(selected).astype(int)


def kappa(previous, current):
    a, b = previous.mean(), current.mean()
    expected = a * b + (1 - a) * (1 - b)
    return None if expected == 1 else float(((previous == current).mean() - expected) / (1 - expected))


def run(candidate_ids, vectors, label, query_vector=None, *, config=None, initialization=None):
    """label(ids) returns one 0/1 per ID in order, doing at most one request.

    With a query vector, similarity bootstrap follows Hydra. Without it,
    bootstrap is random, as requested for this local adaptation.
    All model inputs exclude historical labels; oracle labels enter via label().
    """
    config = config or Config()
    config.validate()
    ids = list(candidate_ids)
    if any(not isinstance(cid, str) or not cid for cid in ids) or len(set(ids)) != len(ids):
        raise ValueError("candidate_ids must be unique nonempty strings")
    x = np.array(vectors, dtype=np.float32, copy=True)
    if x.ndim != 2 or x.shape[0] != len(ids) or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("vectors must be a finite N x D matrix aligned with candidate_ids")
    initialization = initialization or ("similarity" if query_vector is not None else "random")
    if initialization not in ("similarity", "random"):
        raise ValueError("initialization must be similarity or random")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x /= np.where(norms > 0, norms, 1)  # Hydra normalizes precomputed feature vectors too.
    if initialization == "similarity":
        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape != (x.shape[1],) or not np.isfinite(query).all():
            raise ValueError("Provide a query_vector from the resume embedding model, or explicitly use initialization='random'")
        norm = np.linalg.norm(query)
        similarities = x @ (query / norm if norm > 0 else query)
    else:
        similarities = None
    rng = np.random.default_rng(config.seed)
    n = len(ids)
    cache = np.full(n, -1, dtype=np.int8)
    usage = dict(llm_calls=0, labeled_candidates=0)

    def acquire(indices):
        pending = list(dict.fromkeys(int(i) for i in indices if cache[i] < 0))
        for start in range(0, len(pending), config.batch_size):
            batch = pending[start:start + config.batch_size]
            if usage["llm_calls"] >= config.max_calls:
                raise RuntimeError("LLM call budget exhausted")
            usage["llm_calls"] += 1  # Failed requests are charged; no automatic retries.
            try:
                answer = np.asarray(label([ids[i] for i in batch]))
                if answer.shape != (len(batch),) or answer.dtype.kind not in "biu" or not np.isin(answer, [0, 1]).all():
                    raise ValueError("label callback must return one boolean/integer 0 or 1 per requested ID")
            except Exception as error:
                raise RuntimeError(f"Labeling failed after {usage['llm_calls']} charged requests") from error
            cache[batch] = answer
            usage["labeled_candidates"] += len(batch)

    limit = min(config.sample_size, n)
    stop_size = min(n, int(np.ceil(config.stop_fraction * n))) if config.adaptive else 0
    stop_ids = np.sort(rng.choice(n, stop_size, replace=False))
    training, history, kappa_history = [], [], []
    previous = parameters = None
    probabilities = np.zeros(n)
    stopped = False
    while len(training) < limit and usage["llm_calls"] < config.max_calls:
        available = np.flatnonzero(cache < 0)
        size = min(config.step, limit - len(training), (config.max_calls - usage["llm_calls"]) * config.batch_size)
        if parameters is None:
            batch = (similarity_extremes(available, similarities, size) if similarities is not None
                     else np.sort(rng.choice(available, size, replace=False)))
        else:
            batch = active_batch(available, cache[training], probabilities, size, rng)
        acquire(batch)
        training.extend(int(i) for i in batch)
        labels = cache[training]
        if np.unique(labels).size == 1:
            probabilities = np.full(n, float(labels[0]))
        else:
            parameters = fit_logistic(x[training], labels, config.lr_lambda)
            probabilities = expit(x @ parameters[:-1] + parameters[-1])
            if stop_size:
                current = probabilities[stop_ids] >= config.lr_threshold
                if previous is not None:
                    agreement = kappa(previous, current)
                    if agreement is not None:
                        kappa_history.append(agreement)
                        stopped = bool(len(kappa_history) >= config.patience and
                                       np.mean(kappa_history[-config.patience:]) > config.stop_threshold)
                previous = current
        history.append(dict(training_size=len(training), positives=int(np.count_nonzero(labels == 1)),
                            last_kappa=kappa_history[-1] if kappa_history else None))
        if stopped:
            break
    threshold = config.lr_threshold
    calibration = dict(success=False, reason="single-class training or calibration disabled")
    if parameters is not None and config.calibration_sample_size:
        budget = min(config.calibration_sample_size, (config.max_calls - usage["llm_calls"]) * config.batch_size)
        learned, calibration = calibrate(probabilities, cache, acquire, rng, budget, config.target_recall,
                                         config.calibration_include_training, config.failure_probability)
        if learned is not None:
            threshold = learned
    selected = probabilities >= threshold
    # With one-class labels, Hydra returns the constant class irrespective of LR threshold.
    if n and parameters is None:
        selected[:] = cache[training[0]] == 1
    selected[cache >= 0] = cache[cache >= 0].astype(bool)
    return dict(candidate_ids=[ids[i] for i in np.flatnonzero(selected)], scores=probabilities,
                parameters=parameters, usage=usage,
                summary=dict(method="hydra_active_learning_lr_target_recall", config=asdict(config),
                             initialization=initialization, threshold=float(threshold), training_indices=training,
                             labeled_indices=np.flatnonzero(cache >= 0).tolist(), calibration=calibration,
                             stopped_by_stabilization=stopped, rounds=history))
