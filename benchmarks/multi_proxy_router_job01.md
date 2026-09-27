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
  --batch-size 6
```

The run is resumable. `batch_teacher.sqlite` stores request-level token usage once,
while `labels.json` stores candidate-level `overall`, `education`, `experience`,
`technical_skills`, `projects`, `research`, and `evidence_quality` scores. The API
key is read only from `_config/llm.json` and is never written to outputs.

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
router still learns against `overall`, so it can correct systematic expert bias.

## Outputs and evaluation

- `model.pkl`: fitted experts and routers;
- `candidate_ids.json` + `scores.npz`: full-population scores;
- `report.json`: strict-unsampled AP, P@R80, P@R90, and 20-seed calibration;
- both empirical-recall and 95% Clopper-Pearson lower-bound calibration;
- Recall achievement rate, worst-seed Recall, mean/std Precision, Recall, F1,
  threshold, and Recall lower bound.

Historical unsampled labels are not used for fitting. They are used only after all
scores have been produced, for the benchmark diagnostics in `report.json`.
