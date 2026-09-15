"""Export aggregate evidence only; keep resumes, candidate IDs and full trajectories local."""
import json
import sqlite3
from pathlib import Path
from statistics import mean, median

import duckdb

ROOT = Path(__file__).resolve().parents[1]


def summarize(run):
    result = json.loads((run / 'experiment.json').read_text())
    path = run / 'evaluation.json'
    if path.exists():
        result['evaluation'] = json.loads(path.read_text())
    responses, labeled = [], []
    path = run / 'workspace/output/semantic.sqlite'
    if path.exists():
        with sqlite3.connect(path) as con:
            rows = con.execute('SELECT candidate_id,label,response FROM queries').fetchall()
        labeled = [(cid, label) for cid, label, _ in rows if label is not None]
        responses = [json.loads(response) for _, _, response in rows if response]
        result.update(label_attempts=len(rows), successful_labels=len(labeled),
                      positive_labels=sum(label for _, label in labeled),
                      unfinished_or_failed_labels=len(rows)-len(labeled))
        task = json.loads((run / 'task.json').read_text())
        with duckdb.connect(task['labels'], read_only=True) as con:
            gold = dict(con.execute('SELECT candidate_id,llm_pass FROM llm_pass').fetchall())
        pairs = [(label, gold[cid]) for cid, label in labeled if gold.get(cid) in (0, 1)]
        result['label_agreement_on_queried_sample'] = dict(
            n=len(pairs), agree=sum(a == b for a, b in pairs),
            live_positive_gold_positive=sum(a == b == 1 for a, b in pairs),
            live_positive_gold_negative=sum(a == 1 and b == 0 for a, b in pairs),
            live_negative_gold_positive=sum(a == 0 and b == 1 for a, b in pairs))
    latencies = [r['_latency_seconds'] for r in responses if '_latency_seconds' in r]
    result['label_tokens'] = tokens([r.get('_usage', {}) for r in responses])
    if latencies:
        result['label_latency_seconds'] = dict(mean=mean(latencies), median=median(latencies), maximum=max(latencies))
    # Each trajectory contains its own agent calls; workers are separate from their planner.
    trajectories = list(run.glob('*.trajectory.json')) + list(run.glob('workspace/operators/*/trajectory.json'))
    usages, models = [], set()
    for path in trajectories:
        trajectory = json.loads(path.read_text())
        for message in trajectory['messages']:
            response = message.get('extra', {}).get('response', {})
            if message['role'] == 'assistant' and response:
                usages.append(response.get('usage', {}))
                models.add(response.get('model', 'unknown'))
    result['agent_calls'] = len(usages)
    result['agent_tokens'] = tokens(usages)
    result['agent_models'] = sorted(models)
    return result


def tokens(usages):
    result = {key: sum(u.get(key, 0) or 0 for u in usages)
              for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
    for group, key in [('prompt_tokens_details', 'cached_tokens'), ('completion_tokens_details', 'reasoning_tokens')]:
        result[key] = sum((u.get(group) or {}).get(key, 0) or 0 for u in usages)
    return result


def export():
    rows = [summarize(p.parent) for p in sorted((ROOT / 'results/comparison').glob('*/*/experiment.json'))]
    (ROOT / 'benchmarks/results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    return rows
