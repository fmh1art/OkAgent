import json
import tempfile
import unittest
from pathlib import Path

import duckdb
import joblib
import numpy as np

from job01_full_eval import source_profile, prepare, train_folds, evaluate, DEFAULT_MODEL
from job01_full_agent import run_full_pipeline


def fixture(root, n=40):
    folder = root / 'job-candidate-embedding-20260722'
    folder.mkdir()
    rng = np.random.default_rng(12)
    with duckdb.connect(str(folder / 'hiring_job01_full_segvec.db')) as con:
        con.execute('CREATE TABLE candidate_segments(candidate_id VARCHAR,segment VARCHAR,text VARCHAR,vec FLOAT[4])')
        rows = [(str(i), 'unstructured', 'fixture', (rng.normal(size=4) + (i % 2)).tolist()) for i in range(n)]
        con.executemany('INSERT INTO candidate_segments VALUES (?,?,?,?)', rows)
    with duckdb.connect(str(folder / 'hiring_job01_llm_pass.db')) as con:
        con.execute('CREATE TABLE llm_pass(candidate_id VARCHAR,llm_pass TINYINT)')
        con.executemany('INSERT INTO llm_pass VALUES (?,?)', [(str(i), i % 2) for i in range(n)])
    return folder


class FullEvaluationTest(unittest.TestCase):
    def test_custom_model_and_threshold_independent_validation(self):
        from adaptive_eval import prepare_search, validate_candidate
        from sklearn.naive_bayes import GaussianNB
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            run = root / 'run'
            run.mkdir()
            prepare_search({'data_dir': str(root), 'expected_count': 40, 'dimension': 4,
                            'test_size': 0.2, 'validation_size': 0.2, 'seed': 42}, run)
            directory = run / 'rounds/000001'
            directory.mkdir(parents=True)
            with np.load(run / 'data/train.npz') as data:
                model = GaussianNB().fit(data['x'], data['y'])
            joblib.dump(model, directory / 'model.joblib')
            report = validate_candidate({'run_dir': str(run), 'round_dir': str(directory),
                'observation': {'model_file': 'model.joblib', 'threshold': 0.7, 'f1': 999}})
            self.assertEqual(report['model_class'], 'GaussianNB')
            self.assertEqual(report['threshold'], 0.7)
            self.assertLessEqual(report['validation']['f1'], 1)
            self.assertNotIn('test', report)

    def test_full_chain_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            run = root / 'run'
            config = {'data_dir': str(root), 'folds': 2, 'seed': 10, 'expected_count': 40, 'dimension': 4}
            result = run_full_pipeline(config, run, timeout=90)
            self.assertEqual(result['full_rows'], 40)
            self.assertEqual(sum(result['test'][k] for k in ['tp', 'fp', 'fn', 'tn']), 8)
            import csv
            with (run / 'test_predictions.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), len({r['candidate_id'] for r in rows}))
            history_len = len((run / 'trajectory.jsonl').read_text().splitlines())
            resumed = run_full_pipeline(config, run, timeout=90)
            self.assertEqual(resumed['test'], result['test'])
            self.assertEqual(len((run / 'trajectory.jsonl').read_text().splitlines()), history_len)
            with self.assertRaisesRegex(ValueError, 'Configuration'):
                run_full_pipeline({**config, 'seed': 11}, run, timeout=90)

    def test_missing_vector_and_duplicate_label_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = fixture(root)
            with duckdb.connect(str(folder / 'hiring_job01_full_segvec.db')) as con:
                con.execute("DELETE FROM candidate_segments WHERE candidate_id='0'")
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                source_profile(root, 40, 4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = fixture(root)
            with duckdb.connect(str(folder / 'hiring_job01_llm_pass.db')) as con:
                con.execute("UPDATE llm_pass SET candidate_id='1' WHERE candidate_id='0'")
            with self.assertRaisesRegex(ValueError, 'unique'):
                source_profile(root, 40, 4)

    def test_rejects_leaked_model_and_changed_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            run = root / 'run'
            run.mkdir()
            prepare(root, run, 2, 10, 40, 4)
            train_folds(run, DEFAULT_MODEL)
            with self.assertRaisesRegex(ValueError, 'locked'):
                train_folds(run, {**DEFAULT_MODEL, 'C': 2.0})
            path = run / 'model_fold_0.joblib'
            saved = joblib.load(path)
            saved['train_indices'] = np.arange(40)
            joblib.dump(saved, path)
            with self.assertRaisesRegex(ValueError, 'leakage'):
                evaluate(run)

if __name__ == '__main__':
    unittest.main()
