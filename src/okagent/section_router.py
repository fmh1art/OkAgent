"""Candidate-level multi-proxy models over resume section embeddings.

The module deliberately keeps historical/teacher labels at candidate level.  A
candidate with many resume sections therefore contributes exactly one training
example to each expert instead of receiving a larger loss weight.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


DEFAULT_EXPERT_GROUPS = {
    "global": ("unstructured",),
    "experience": ("projects", "work"),
    "credentials": ("education", "awards", "languages", "applications"),
}


def aggregate_section_rows(
    candidate_ids: Sequence[str],
    rows: Sequence[tuple[str, str, Sequence[float]]],
    segments: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-pool selected section vectors and return a presence mask.

    Pooling happens before model training, so every candidate has one vector and
    one loss contribution regardless of resume length or section count.
    """
    index = {cid: pos for pos, cid in enumerate(candidate_ids)}
    if len(index) != len(candidate_ids):
        raise ValueError("candidate_ids must be unique")
    allowed = set(segments)
    dimension = None
    sums = None
    counts = np.zeros(len(candidate_ids), dtype=np.int32)
    for cid, segment, vector in rows:
        if cid not in index or segment not in allowed or vector is None:
            continue
        value = np.asarray(vector, dtype=np.float32)
        if value.ndim != 1:
            raise ValueError("section vectors must be one-dimensional")
        if dimension is None:
            dimension = len(value)
            sums = np.zeros((len(candidate_ids), dimension), dtype=np.float32)
        if len(value) != dimension:
            raise ValueError("section vector dimensions do not match")
        sums[index[cid]] += value
        counts[index[cid]] += 1
    if dimension is None:
        raise ValueError(f"no vectors found for segments {sorted(allowed)}")
    present = counts > 0
    sums[present] /= counts[present, None]
    return sums, present


def _binary_proxy(random_state: int, class_weight="balanced"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            C=0.25, class_weight=class_weight, max_iter=2_000,
            solver="liblinear", random_state=random_state,
        ),
    )


def _multiclass_router(random_state: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=2_000,
            solver="lbfgs", random_state=random_state,
        ),
    )


@dataclass
class ExpertFit:
    name: str
    model: object
    missing_probability: float


class SectionRouterEnsemble:
    """Three section experts with leakage-safe stacking and embedding routing.

    Router targets and stacker inputs are built from out-of-fold (OOF) expert
    predictions.  The final experts are then refit on all training candidates.
    """

    def __init__(self, *, folds: int = 5, seed: int = 11):
        if folds < 2:
            raise ValueError("folds must be at least 2")
        self.folds = folds
        self.seed = seed
        self.expert_names: list[str] = []
        self.experts: dict[str, ExpertFit] = {}
        self.stacker = None
        self.router = None
        self.router_classes_: np.ndarray | None = None
        self.training_summary_: dict | None = None

    @staticmethod
    def _check_inputs(features: Mapping[str, np.ndarray],
                      presence: Mapping[str, np.ndarray], labels: Sequence[int]):
        names = list(features)
        if len(names) < 2 or set(names) != set(presence):
            raise ValueError("features and presence must contain at least two identical experts")
        y = np.asarray(labels, dtype=np.int8)
        if y.ndim != 1 or set(np.unique(y)) != {0, 1}:
            raise ValueError("labels must be a one-dimensional array containing both classes")
        n = len(y)
        dimension = None
        clean_features = {}
        clean_presence = {}
        for name in names:
            matrix = np.asarray(features[name], dtype=np.float32)
            mask = np.asarray(presence[name], dtype=bool)
            if matrix.ndim != 2 or len(matrix) != n or mask.shape != (n,):
                raise ValueError(f"invalid matrix or presence mask for expert {name}")
            if dimension is None:
                dimension = matrix.shape[1]
            elif matrix.shape[1] != dimension:
                raise ValueError("all expert embeddings must have the same dimension")
            clean_features[name] = matrix
            clean_presence[name] = mask
        return names, clean_features, clean_presence, y

    def fit(self, features: Mapping[str, np.ndarray],
            presence: Mapping[str, np.ndarray], labels: Sequence[int], *,
            expert_labels: Mapping[str, Sequence[int]] | None = None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold

        names, x, masks, y = self._check_inputs(features, presence, labels)
        task_labels = {name: y for name in names}
        if expert_labels is not None:
            if set(expert_labels) != set(names):
                raise ValueError("expert_labels must contain every expert")
            for name in names:
                values = np.asarray(expert_labels[name], dtype=np.int8)
                if values.shape != y.shape or not set(np.unique(values)).issubset({0, 1}):
                    raise ValueError(f"invalid expert labels for {name}")
                task_labels[name] = values
        minimum_class = int(np.bincount(y).min())
        folds = min(self.folds, minimum_class)
        if folds < 2:
            raise ValueError("each class needs at least two candidates for OOF routing")
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
        oof = np.zeros((len(y), len(names)), dtype=np.float64)
        expert_counts = {}
        priors = {}

        for expert_index, name in enumerate(names):
            mask = masks[name]
            expert_counts[name] = int(mask.sum())
            expert_y = task_labels[name]
            if len(np.unique(expert_y[mask])) < 2:
                raise ValueError(f"expert {name} does not observe both classes")
            prior = float(expert_y[mask].mean())
            priors[name] = prior
            oof[:, expert_index] = prior
            for train_idx, valid_idx in splitter.split(x[name], y):
                train_present = train_idx[mask[train_idx]]
                valid_present = valid_idx[mask[valid_idx]]
                if len(valid_present) == 0 or len(np.unique(expert_y[train_present])) < 2:
                    continue
                proxy = _binary_proxy(self.seed + expert_index)
                proxy.fit(x[name][train_present], expert_y[train_present])
                oof[valid_present, expert_index] = proxy.predict_proba(
                    x[name][valid_present])[:, 1]

        presence_matrix = np.column_stack([masks[name] for name in names]).astype(np.float64)
        stack_features = np.column_stack([oof, presence_matrix])
        self.stacker = LogisticRegression(
            C=0.25, class_weight="balanced", max_iter=2_000,
            solver="liblinear", random_state=self.seed,
        ).fit(stack_features, y)

        # The best OOF expert per candidate is a supervised routing target.  The
        # router only sees the global resume embedding, never the evaluation label.
        losses = -(y[:, None] * np.log(np.clip(oof, 1e-6, 1)) +
                   (1 - y[:, None]) * np.log(np.clip(1 - oof, 1e-6, 1)))
        losses = np.where(presence_matrix > 0, losses, np.inf)
        routing_target = np.argmin(losses, axis=1)
        global_name = "global" if "global" in names else names[0]
        global_present = masks[global_name]
        if len(np.unique(routing_target[global_present])) >= 2:
            self.router = _multiclass_router(self.seed).fit(
                x[global_name][global_present], routing_target[global_present])
            self.router_classes_ = np.asarray(self.router.classes_, dtype=int)

        self.experts = {}
        for expert_index, name in enumerate(names):
            proxy = _binary_proxy(self.seed + expert_index)
            proxy.fit(x[name][masks[name]], task_labels[name][masks[name]])
            self.experts[name] = ExpertFit(name, proxy, priors[name])
        self.expert_names = names
        self.training_summary_ = {
            "candidate_count": len(y),
            "positive_count": int(y.sum()),
            "folds": folds,
            "experts": expert_counts,
            "expert_priors": priors,
            "expert_positive_counts": {
                name: int(task_labels[name][masks[name]].sum()) for name in names
            },
            "router_target_counts": {
                names[index]: int((routing_target == index).sum())
                for index in range(len(names))
            },
            "embedding_router_fitted": self.router is not None,
        }
        return self

    def expert_probabilities(self, features: Mapping[str, np.ndarray],
                             presence: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if not self.experts:
            raise ValueError("ensemble is not fitted")
        count = len(np.asarray(features[self.expert_names[0]]))
        probabilities = np.zeros((count, len(self.expert_names)), dtype=np.float64)
        present = np.zeros_like(probabilities)
        for index, name in enumerate(self.expert_names):
            matrix = np.asarray(features[name], dtype=np.float32)
            mask = np.asarray(presence[name], dtype=bool)
            if len(matrix) != count or mask.shape != (count,):
                raise ValueError(f"invalid prediction inputs for {name}")
            probabilities[:, index] = self.experts[name].missing_probability
            selected = matrix if mask.all() else matrix[mask]
            probabilities[mask, index] = self.experts[name].model.predict_proba(
                selected)[:, 1]
            present[:, index] = mask
        return probabilities, present

    def predict_all(self, features: Mapping[str, np.ndarray],
                    presence: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        probabilities, present = self.expert_probabilities(features, presence)
        available = np.maximum(present.sum(axis=1), 1)
        mean = (probabilities * present).sum(axis=1) / available
        results = {name: probabilities[:, index]
                   for index, name in enumerate(self.expert_names)}
        results["expert_mean"] = mean
        results["stacking_router"] = self.stacker.predict_proba(
            np.column_stack([probabilities, present]))[:, 1]
        if self.router is not None:
            global_name = "global" if "global" in self.expert_names else self.expert_names[0]
            raw_weights = self.router.predict_proba(
                np.asarray(features[global_name], dtype=np.float32))
            weights = np.zeros_like(probabilities)
            weights[:, self.router_classes_] = raw_weights
            weights *= present
            row_sum = weights.sum(axis=1, keepdims=True)
            zero = row_sum[:, 0] == 0
            weights[zero] = present[zero]
            row_sum = np.maximum(weights.sum(axis=1, keepdims=True), 1)
            weights /= row_sum
            results["embedding_router"] = (probabilities * weights).sum(axis=1)
        return results
