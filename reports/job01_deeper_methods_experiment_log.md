# job01 deeper methods experiment log

**Implementation date:** 2026-09-28

## Implemented methods

| Method | Status | Key property |
|---|---:|---|
| Candidate-level resume splitting | Implemented | Three fixed section groups; one row/loss per candidate |
| LLM batch prompting | Implemented | Multiple candidates share one JD/system prompt per HTTP request |
| Multi-dimensional teacher labels | Implemented | Seven continuous labels plus missing hard requirements |
| Three section proxies | Implemented | Global, experience, credentials |
| OOF stacking router | Implemented | No in-sample expert predictions are used to train the stacker |
| Embedding router | Implemented | Global resume embedding selects/weights proxy experts |
| Recall-constrained calibration | Implemented | 20 seeds, empirical and Clopper-Pearson lower-bound variants |
| job01 architecture run | Complete | Existing frozen 2,000-ID ledger; architecture-only protocol |
| job01 batch multi-label run | Pending | Teacher labels have not yet been generated on job01 |

## Leakage and fairness controls

- Training IDs are frozen by JSON or an existing teacher SQLite ledger.
- Training IDs are removed from every strict-unsampled metric.
- Before scoring, historical-label queries are restricted to the frozen training
  IDs (or omitted entirely for batch-teacher runs); full unsampled truth is opened
  only after every candidate score has been frozen.
- Each expert produces out-of-fold training scores for both routers.
- Full expert models are fitted only after router training features are frozen.
- Batch-label input must cover training candidates; calibration/test IDs must not
  be passed to the labeler.
- Historical labels outside the training set are opened only for final metrics.
- The report records whether the ID source is the exact USA seed-11 protocol; if
  that ID set cannot be recovered, results must not claim a strict USA comparison.

## Test status

Local test command (UTF-8 mode):

```text
pytest -q
47 passed, 2 skipped
```

The two skipped tests are pre-existing optional integration paths. A non-UTF-8
Windows locale initially caused one pre-existing test to decode a UTF-8 prompt as
GBK; rerunning with `PYTHONUTF8=1` produced the result above.

The test suite includes an end-to-end synthetic DuckDB run of the complete
multi-proxy pipeline: section loading, candidate-level pooling, three experts,
OOF routers, strict-unsampled exclusion, repeated calibration, model persistence,
compressed score output, and JSON report generation. Synthetic metrics are not
copied into the formal job01 results table.

## Server experiment results

The first real job01 server experiment completed on 2026-09-28. It reused the
existing Qwen-Doubao 2,000-candidate ledger and therefore evaluates the proxy and
router architecture only; it is not an exact USA seed-11 comparison.

| Training labels | Method | Unsampled AP | P@R80 | P@R90 | 20-seed R80 achievement | 20-seed R90 achievement |
|---|---|---:|---:|---:|---:|---:|
| Existing frozen 2,000 IDs | global | 0.3595 | 11.07% | 4.74% | 35% | 25% |
| Existing frozen 2,000 IDs | experience | 0.3341 | 4.45% | 2.95% | 60% | 45% |
| Existing frozen 2,000 IDs | credentials | 0.0767 | 2.13% | 1.59% | 65% | 35% |
| Existing frozen 2,000 IDs | expert mean | 0.3472 | 7.57% | 5.20% | 55% | 40% |
| Existing frozen 2,000 IDs | stacking router | **0.3928** | 8.67% | **5.76%** | 60% | 40% |
| Existing frozen 2,000 IDs | embedding router | 0.2255 | 6.17% | 3.44% | 50% | 45% |
| Batch multi-label 2,000 IDs | Pending | — | — | — | — | — |

Supplemental metrics for the completed architecture run:

| Method | 20-seed job01 R80 calibration F1 | F1 std. | Mean test recall | Oracle-Overwrite AP |
|---|---:|---:|---:|---:|
| global | **19.25%** | 2.11pp | 77.47% | 0.5790 |
| experience | 8.33% | 0.18pp | 80.17% | 0.5614 |
| credentials | 4.15% | 0.28pp | 79.89% | 0.3542 |
| expert mean | 14.41% | 1.93pp | 79.34% | 0.5770 |
| stacking router | 15.77% | 2.05pp | 79.34% | **0.6063** |
| embedding router | 11.35% | 1.10pp | 79.40% | 0.4797 |

The calibration values above use job01-only 20-seed stratified calibration with
an R80 target. They are not the cross-job Shared-Mixture F1 used by LR-U/US/US3/USA,
and they are not the old single-split maximum-F1 values. Oracle-Overwrite replaces
the frozen 2,000 sampled candidates' scores with historical binary truth and then
computes AP on the full 34,761-candidate pool. US and US3 cannot be backfilled for
P@R80 or Oracle-Overwrite from the current artifacts because their original
candidate-level score arrays were not saved.

Protocol details:

- Comparison claim: `architecture_only`.
- Training ID SHA-256: `ad8b48f29482d2dfc2af1cf32ef2f8d17ae738e4fabf16e9b5c8ef440b6067fd`.
- Training candidates / positives: 2,000 / 88.
- Strict-unsampled candidates / positives: 32,761 / 294.
- Model seed / OOF folds: 11 / 5.
- Training IDs were excluded from evaluation and full unsampled truth was opened
  only after scores were frozen.
- This run reused an existing ledger, so it made no new teacher-LLM requests;
  request and token usage are not applicable. Runtime and peak memory were not
  instrumented in this run.

The stacking router achieved the best AP and P@R90 among the implemented proxy
methods. The global proxy achieved the best P@R80. Relative to the historical
USA figures, stacking has lower AP (0.3928 versus 0.4798), but higher point
precision at R80 and R90 (8.67% versus 5.80%, and 5.76% versus 2.98%). This is
descriptive only because the training ID protocol has not been proven identical.

The empirical 20-seed recall thresholds were unstable: no method reached either
target on every seed. The Clopper-Pearson lower-bound-constrained variant was much
safer (stacking achieved R80 and R90 on 100% of seeds), at lower mean precision
of 5.75% and 2.51% respectively.

Raw artifacts are preserved in `reports/job01_multi_proxy_router_results/`.
