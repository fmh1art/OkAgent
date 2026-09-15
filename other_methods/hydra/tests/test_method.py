import json

import duckdb
import numpy as np
import pytest
from scipy.special import expit

from other_methods.hydra import Config, run, run_hiring
from other_methods.hydra.calibration import calibrate, uniform_scores
from other_methods.hydra.method import active_batch, fit_logistic, kappa, similarity_extremes
from okagent.data import prepare
from okagent.evaluation import evaluate


def test_logistic_objective_and_regularization():
    x = np.array([[-3.0], [-2], [-1], [1], [2], [3]])
    labels = np.array([0, 0, 0, 1, 1, 1])
    parameters = fit_logistic(x, labels)
    scores = expit(x @ parameters[:-1] + parameters[-1])
    assert np.array_equal(scores >= 0.5, labels)
    regularized = fit_logistic(x, labels, regularization=10)
    assert 0.5 < expit(3 * regularized[0] + regularized[1]) < scores[-1]
    assert abs(parameters[-1]) < 1e-5


def test_source_sampling_rules_and_degenerate_kappa():
    assert similarity_extremes(np.arange(8), np.zeros(8), 5).tolist() == [0, 1, 5, 6, 7]
    probabilities = np.array([0.01, 0.2, 0.45, 0.5, 0.55, 0.8, 0.99])
    batch = active_batch(np.arange(7), np.array([0] * 9 + [1]), probabilities, 4, np.random.default_rng(0))
    assert batch.tolist() == [1, 2, 3, 4]
    assert kappa(np.zeros(4), np.zeros(4)) is None
    assert kappa(np.array([0, 1]), np.array([0, 1])) == 1
    assert uniform_scores(np.array([0.2, 0.2, 0.1, 0.9])).tolist() == [0.5, 0.5, 0.125, 0.875]


@pytest.mark.parametrize("include_training", [True, False])
@pytest.mark.parametrize("failure_probability", [-1, 0.2])
def test_calibration_matches_independent_source_equations(include_training, failure_probability):
    scores = np.array([0.02, 0.02, 0.1, 0.15, 0.3, 0.4, 0.7, 0.8, 0.9, 0.99])
    truth = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    cache = np.array([1, -1, -1, -1, -1, -1, -1, -1, -1, 0], dtype=np.int8)
    training = np.flatnonzero(cache >= 0)
    requested = []

    def acquire(indices):
        requested.extend(indices.tolist())
        cache[indices] = truth[indices]

    target = 0.7
    threshold, info = calibrate(scores, cache, acquire, np.random.default_rng(9), 6, target,
                                include_training, failure_probability)
    assert len(requested) == len(set(requested))
    assert set(training).isdisjoint(requested)
    draws = info["indices"]
    weights = info["correction_factors"]
    n = len(scores)
    # Direct, slow source equations in original score space, independent of CDF/prefix sums.
    def masses(t):
        retained = sum(truth[i] for i in training) / n if include_training else 0
        missed = 0
        for i, w in zip(draws, weights):
            if truth[i]:
                if scores[i] >= t:
                    retained += w / len(draws)
                else:
                    missed += w / len(draws)
        overrides = requested if include_training else list(training) + requested
        for i in overrides:
            if truth[i] and scores[i] < t:
                retained += 1 / n
                missed -= 1 / n
        return retained, max(0, missed)

    def choose(r):
        for t in sorted(set(scores), reverse=True):
            kept, missed = masses(t)
            if kept + missed > 0 and kept / (kept + missed) >= r:
                return t

    initial = choose(target)
    kept, missed = masses(initial)
    if failure_probability != -1:
        z_kept = [w * truth[i] * (scores[i] >= initial) for i, w in zip(draws, weights)]
        z_missed = [w * truth[i] * (scores[i] < initial) for i, w in zip(draws, weights)]
        scale = np.sqrt(2 * np.log(2 / failure_probability) / len(draws))
        kept += np.std(z_kept, ddof=1) * scale
        missed -= np.std(z_missed, ddof=1) * scale
    kept, missed = np.clip([kept, missed], 0, 1)
    inflated = np.clip(kept / (kept + missed), target, 1)
    assert threshold == choose(inflated)
    assert info["success"]


def test_duplicate_draws_keep_multiplicity_but_one_label():
    class FixedDraws:
        def choice(self, n, size, replace, p):
            return np.array([0, 0, 2, 2])

    scores = np.linspace(0.1, 0.9, 10)
    cache = np.full(10, -1, dtype=np.int8)
    requested = []

    def acquire(indices):
        requested.extend(indices)
        cache[indices] = 1

    _, info = calibrate(scores, cache, acquire, FixedDraws(), 4)
    assert requested == [0, 2]
    assert info["indices"] == [0, 0, 2, 2]
    assert info["draws"] == 4
    assert info["unique_new_labels"] == 2


def test_census_and_all_cached_labels():
    scores = np.linspace(0.05, 0.95, 6)
    truth = np.array([1, 0, 1, 0, 1, 0])
    cache = np.array([1, 0, -1, -1, -1, -1], dtype=np.int8)
    requested = []

    def acquire(indices):
        requested.extend(indices)
        cache[indices] = truth[indices]

    threshold, info = calibrate(scores, cache, acquire, np.random.default_rng(3), 10)
    assert requested == [2, 3, 4, 5]
    assert threshold == max(scores)
    assert info["estimated_recall"] == 1
    requested.clear()
    _, info = calibrate(scores, cache, acquire, np.random.default_rng(3), 10)
    assert requested == []
    assert info["census"]


def test_budget_and_oracle_overrides():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(80, 5))
    truth = (x[:, 0] > 0).astype(int)
    requested = []

    def label(ids):
        requested.append(ids)
        return truth[list(map(int, ids))]

    result = run(list(map(str, range(80))), x, label,
                 config=Config(sample_size=32, calibration_sample_size=16, step=4, adaptive=False,
                               batch_size=3, max_calls=4))
    assert len(requested) == 4
    assert result["usage"]["llm_calls"] == 4
    assert result["summary"]["initialization"] == "random"
    selected = set(result["candidate_ids"])
    for ids in requested:
        for cid in ids:
            assert (cid in selected) == bool(truth[int(cid)])


@pytest.mark.parametrize("constant", [0, 1])
def test_single_class_preserves_source_behavior(constant):
    x = np.eye(12)
    result = run(list(map(str, range(12))), x, lambda ids: [constant] * len(ids),
                 config=Config(sample_size=8, step=2, lr_threshold=1.1))
    assert len(result["candidate_ids"]) == (12 if constant else 0)
    assert result["usage"]["llm_calls"] == 8
    assert not result["summary"]["stopped_by_stabilization"]
    assert result["parameters"] is None


def test_callback_failures_and_missing_query_are_explicit():
    calls = []

    def bad_label(ids):
        calls.append(ids)
        return [2] * len(ids)

    with pytest.raises(ValueError, match="query_vector"):
        run(["a", "b"], np.eye(2), bad_label, initialization="similarity")
    assert not calls
    with pytest.raises(RuntimeError, match="after 1 charged"):
        run(["a", "b"], np.eye(2), bad_label)
    assert len(calls) == 1


def test_adaptive_rounds_are_json_serializable():
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(60, 4))
    truth = (vectors[:, 0] > 0).astype(int)
    result = run(list(map(str, range(60))), vectors, lambda ids: truth[list(map(int, ids))],
                 config=Config(sample_size=24, step=4, calibration_sample_size=4, patience=1, stop_threshold=1.0))
    assert len(result["summary"]["rounds"]) == 6
    assert result["summary"]["stopped_by_stabilization"] is False
    json.dumps(result["summary"], allow_nan=False)


def test_hiring_adapter_writes_existing_evaluation_contract(tmp_path):
    raw = tmp_path / "raw"
    descriptions = raw / "job-description-20260722"
    databases = raw / "job-candidate-embedding-20260722"
    descriptions.mkdir(parents=True)
    databases.mkdir()
    (descriptions / "01_123.json").write_text(json.dumps(dict(job_title="test", must_have_qualifications=["test"])))
    (raw / "llm_prompt.txt").write_text("test prompt")
    with duckdb.connect(str(databases / "hiring_job01_full_segvec.db")) as con:
        con.execute("CREATE TABLE candidate_segments(candidate_id VARCHAR,segment VARCHAR,text VARCHAR,vec FLOAT[2])")
        con.executemany("INSERT INTO candidate_segments VALUES (?,'unstructured','text',?)",
                        [(str(i), [float(i - 4), 1.0]) for i in range(8)])
    with duckdb.connect(str(databases / "hiring_job01_llm_pass.db")) as con:
        con.execute("CREATE TABLE llm_pass(candidate_id VARCHAR,llm_pass TINYINT)")
        con.executemany("INSERT INTO llm_pass VALUES (?,?)", [(str(i), int(i > 3)) for i in range(8)])
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(raw_root=str(raw), job="job01", as_of="2026-07-22", max_calls=20)))
    run_dir = tmp_path / "run"
    prepare(config, run_dir)
    result = run_hiring(run_dir, query_vector=[1, 0], config=Config(sample_size=4, step=2,
                        calibration_sample_size=8, adaptive=False, max_calls=20))
    assert result["usage"]["mode"] == "replay"
    report = evaluate(run_dir)
    assert report["recall"] == report["precision"] == 1
    assert report["llm_calls"] == 8
    assert (run_dir / "workspace/output/hydra_model.npz").is_file()


def test_concurrent_callbacks_preserve_algorithm():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(80, 5))
    truth = (x[:, 0] > 0).astype(int)
    options = dict(sample_size=32, calibration_sample_size=16, step=8, adaptive=False)
    def label(ids):
        return truth[list(map(int, ids))]
    a = run(list(map(str, range(80))), x, label, config=Config(**options))
    b = run(list(map(str, range(80))), x, label, config=Config(**options, label_workers=4))
    assert a['candidate_ids'] == b['candidate_ids']
    assert a['usage'] == b['usage']
    assert a['summary']['training_indices'] == b['summary']['training_indices']
    assert np.array_equal(a['scores'], b['scores'])
