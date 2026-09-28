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
| job01 batch multi-label run | Complete | 2,000 initial labels plus 1,000 disagreement-selected labels |
| job01 multi-label proxy runs | Complete | 2,000-label baseline and cumulative 3,000-label model |

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

## Batch multi-label and disagreement-acquisition results

Both batch-label jobs and both downstream proxy/router runs completed on
2026-09-28. The first round labeled the same frozen 2,000 IDs as the binary
architecture run. The second round labeled 1,000 new candidates selected from
the existing **binary-label router** scores; it was not selected by the new
multi-label model. No random-1,000 control run exists.

### LLM request and token efficiency

| Label round | Candidates | Completed requests | Request reduction vs. one/request | Input tokens | Output tokens | Total tokens | Protocol SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| Initial multi-label | 2,000 | 336 | 83.20% | 3,320,689 | 1,147,872 | 4,468,561 | `da18c291c8a7fca42cea06901af262d25ff333b49af3f56551878956a3ff7faa` |
| Disagreement round | 1,000 | 169 | 83.10% | 1,572,959 | 593,251 | 2,166,210 | `da18c291c8a7fca42cea06901af262d25ff333b49af3f56551878956a3ff7faa` |
| Combined | 3,000 | 505 | 83.17% | 4,893,648 | 1,741,123 | 6,634,771 | same protocol |

The reduction figure counts completed HTTP requests relative to one request per
candidate; it is not a measured token reduction because no single-candidate
token-control run was made. All billed HTTP-200 attempts, including malformed
responses later retried, remain included in token totals. Proxy interruptions
caused many zero-token retries, but the SQLite ledgers preserved completed rows
and allowed exact resumption.

### Native strict-unsampled reports

The initial multi-label run uses the same 2,000-ID hash as the binary run:
`ad8b48f29482d2dfc2af1cf32ef2f8d17ae738e4fabf16e9b5c8ef440b6067fd`.
The cumulative 3,000-ID hash is
`613ec27a1aef6e023410733ccde8fda653df505d82c9f1f5784fdeb6186c528c`.

| Training labels | Best method | Unsampled candidates / positives | AP | P@R80 | P@R90 | R80 calibration F1 | R80 achieved | R90 calibration F1 | R90 achieved |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Multi-label 2,000 | stacking router | 32,761 / 294 | 0.2025 | 5.68% | 3.74% | 10.36% | 60% | 7.17% | 50% |
| Multi-label cumulative 3,000 | stacking router | 31,761 / 123 | 0.0507 | 1.88% | 0.85% | 3.65% | 50% | 2.13% | 50% |

These two native reports must not be compared directly: the 3,000-label run
removes the acquired 1,000 candidates from evaluation, and those candidates
contain 171 of the 294 historical positives that were previously unsampled.
Consequently, the second test set is much harder and has only 123 positives.

### Common-holdout comparison

For a fair model comparison, all three saved score arrays were re-evaluated on
the same 31,761 candidates after excluding the union of all 3,000 training IDs.
This common holdout contains 123 historical positives.

| Training supervision | Method | Common-holdout AP | P@R80 | P@R90 |
|---|---|---:|---:|---:|
| Historical binary 2,000 | stacking router | **0.0571** | **1.95%** | **1.09%** |
| Batch multi-label 2,000 | stacking router | 0.0372 | 1.70% | 0.95% |
| Batch multi-label + disagreement 3,000 | stacking router | 0.0507 | 1.88% | 0.85% |

Adding the disagreement-selected 1,000 labels improved multi-label stacking AP
from 0.0372 to 0.0507, a 36.4% relative increase, and P@R80 from 1.70% to
1.88%. P@R90 decreased from 0.95% to 0.85%. The 3,000-label model still did not
recover the binary-label stacker's AP or precision. Therefore the experiment
supports the usefulness of the additional selected data within the multi-label
setup, but it does **not** show that the current multi-label teacher is better
than the historical binary teacher.

The selected 1,000 candidates contain 171 historical positives (17.1%), compared
with 294/32,761 (0.90%) in the original unsampled pool, approximately 19 times
the base rate. This demonstrates strong positive enrichment by the binary-router
selection. It does not isolate the causal benefit of disagreement acquisition,
because there is no random-1,000 control and the acquisition mixture also contains
300 top-score candidates.

Formal raw artifacts are preserved under
`reports/job01_parallel_multilabel_results/`. The common-holdout recomputation is
stored in `common_holdout_comparison.json`.
