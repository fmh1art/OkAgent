# job01 multi-proxy router experiment

This experiment implements two related ideas without adding a GPU dependency:

1. split each resume into three candidate-level embedding views;
2. train one proxy for each view and route/stack their scores.

The views are frozen as:

| Proxy | Resume sections |
|---|---|
| `global` | `unstructured` |
| `experience` | `projects`, `work` |
| `credentials` | `education`, `awards`, `languages`, `applications` |

All vectors within a view are mean-pooled before fitting. A candidate therefore
contributes one row and one loss term even if their resume has more sections.

## Methods reported

- each standalone proxy;
- unweighted mean of available proxies;
- `stacking_router`: logistic stacking over out-of-fold expert probabilities and
  section-presence flags;
- `embedding_router`: a multinomial gate that reads the global resume embedding
  and predicts which out-of-fold expert is most reliable.

The stacker and gate are trained only on out-of-fold expert predictions. Final
expert models are then refit on the complete frozen training set.

## Existing-label run

Use exactly one frozen 2,000-ID source. For the completed Qwen-Doubao run:

First inventory historical ID sources without exporting candidate IDs:

```bash
python -m benchmarks.recover_job01_protocol \
  --root results/comparison \
  --output reports/protocol/job01_id_source_inventory.json
```

The inventory records counts, SHA-256 ID-set hashes, nearby seed/budget metadata,
and flags an exact 2,000-ID seed-11 source when found. After reviewing the source,
freeze it explicitly with `--export-source`; the scanner never guesses among
ambiguous runs.

```bash
python -m benchmarks.multi_proxy_router_job01 \
  --train-ledger results/comparison/qwen_doubao_segment_soft_al_full34761_mc2000_r1/workspace/output/qwen_active_full/teacher.sqlite \
  --expected-train-count 2000 \
  --seed 11 \
  --output results/comparison/multi_proxy_router_job01_seed11
```

This run trains the proxies on historical binary labels for those IDs and excludes
all 2,000 IDs from strict-unsampled evaluation. It is a controlled architecture
test, but it is only a strict USA comparison if the ledger contains the exact
USA seed-11 IDs.

## Batch-prompted multi-task labels

First freeze a training-only ID JSON file. Never pass calibration or test IDs.
One API request contains six candidates by default and returns seven continuous
labels per candidate:

```bash
python -m benchmarks.batch_label_job01 \
  results/comparison/job01_batch_multilabel_seed11 \
  --ids reports/protocol/job01_seed11_sampled_ids.json \
  --batch-size 6 \
  --workers 4
```

Each worker sends a true multi-candidate request; all SQLite writes remain serialized
on the orchestration thread. The run is resumable. `batch_teacher.sqlite` stores request-level token usage once,
while `labels.json` stores candidate-level `overall`, `education`, `experience`,
`technical_skills`, `projects`, `research`, and `evidence_quality` scores. The API
key is read only from `_config/llm.json` and is never written to outputs.
Token accounting includes every billed HTTP-200 attempt, including malformed JSON,
failed parent batches that are later bisected, and attempts from earlier resumptions.
The report includes input/output tokens per labeled candidate and completed-request
reduction versus one request per candidate.
Stable internal candidate IDs are never sent to the external model: each request
uses `candidate_000`, `candidate_001`, and so on, then maps validated responses back
to real IDs locally before writing the ledger.

Then train section-specific experts from the additional labels:

```bash
python -m benchmarks.multi_proxy_router_job01 \
  --train-ids reports/protocol/job01_seed11_sampled_ids.json \
  --batch-labels results/comparison/job01_batch_multilabel_seed11/labels.json \
  --expected-train-count 2000 \
  --seed 11 \
  --output results/comparison/multi_proxy_router_multilabel_job01_seed11
```

`global` uses `overall`; `experience` uses the mean of experience/project scores;
`credentials` uses the mean of education/technical-skill/research scores. The
proxy losses use the continuous scores through class-balanced soft BCE (implemented
as an exactly equivalent weighted two-row expansion), rather than rounding away
teacher confidence. The router still learns against thresholded `overall`, so it
can correct systematic expert bias while the final task remains binary ranking.

## Selecting additional batch-label candidates

After an initial router run, select a new training-only batch without reading any
historical evaluation labels:

```bash
python -m benchmarks.select_router_batch_candidates \
  results/comparison/multi_proxy_router_job01_seed11 \
  --exclude reports/protocol/job01_seed11_sampled_ids.json \
  --budget 1000 \
  --seed 12 \
  --output results/comparison/job01_router_acquisition_round2
```

The fixed acquisition mixture contains high-score candidates, decision-boundary
candidates, high expert disagreement, high router disagreement, and a random
coverage slice. It produces an exact unique budget plus a per-candidate reason
trace. Pass its `candidate_ids.json` directly to `batch_label_job01`, then combine
the frozen round-one and round-two data for the next proxy run by repeating the
CLI flags (conflicting duplicate labels are rejected):

```bash
python -m benchmarks.multi_proxy_router_job01 \
  --train-ids reports/protocol/job01_seed11_sampled_ids.json \
  --train-ids results/comparison/job01_router_acquisition_round2/candidate_ids.json \
  --batch-labels results/comparison/job01_batch_multilabel_seed11/labels.json \
  --batch-labels results/comparison/job01_batch_multilabel_round2/labels.json \
  --expected-train-count 3000 \
  --output results/comparison/multi_proxy_router_multilabel_job01_round2
```

## Outputs and evaluation

- `model.pkl`: fitted experts and routers;
- `candidate_ids.json` + `scores.npz`: full-population scores;
- `report.json`: strict-unsampled AP, P@R80, P@R90, and 20-seed calibration;
- `report.md`: automatically rendered metric tables and protocol caveats, written
  when the background experiment finishes without requiring active monitoring;
- both empirical-recall and 95% Clopper-Pearson lower-bound calibration;
- Recall achievement rate, worst-seed Recall, mean/std Precision, Recall, F1,
  threshold, and Recall lower bound.

Historical unsampled labels are not used for fitting. They are used only after all
scores have been produced, for the benchmark diagnostics in `report.json`.
