import numpy as np

from okagent.section_router import SectionRouterEnsemble, aggregate_section_rows


def test_aggregate_section_rows_is_candidate_level_and_tracks_missing():
    ids = ["a", "b", "c"]
    rows = [
        ("a", "work", [1.0, 3.0]),
        ("a", "projects", [3.0, 5.0]),
        ("b", "education", [8.0, 8.0]),
    ]
    matrix, present = aggregate_section_rows(ids, rows, ("work", "projects"))
    np.testing.assert_allclose(matrix[0], [2.0, 4.0])
    np.testing.assert_allclose(matrix[1:], 0)
    assert present.tolist() == [True, False, False]


def test_router_returns_one_candidate_score_per_method():
    rng = np.random.default_rng(7)
    n, dimension = 90, 12
    labels = np.array([0, 1] * (n // 2), dtype=np.int8)
    global_x = rng.normal(size=(n, dimension)).astype(np.float32)
    global_x[:, 0] += labels * 1.5
    experience_x = rng.normal(size=(n, dimension)).astype(np.float32)
    experience_x[:, 1] += labels * 1.2
    credentials_x = rng.normal(size=(n, dimension)).astype(np.float32)
    credentials_x[:, 2] += labels
    features = {
        "global": global_x,
        "experience": experience_x,
        "credentials": credentials_x,
    }
    presence = {
        "global": np.ones(n, dtype=bool),
        "experience": np.arange(n) % 4 != 0,
        "credentials": np.ones(n, dtype=bool),
    }
    model = SectionRouterEnsemble(folds=3, seed=11).fit(features, presence, labels)
    scores = model.predict_all(features, presence)
    assert {"global", "experience", "credentials", "expert_mean",
            "stacking_router"}.issubset(scores)
    for values in scores.values():
        assert values.shape == (n,)
        assert np.all((0 <= values) & (values <= 1))
    assert model.training_summary_["candidate_count"] == n
