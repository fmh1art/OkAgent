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
| job01 server runs | Pending | Waiting for authenticated non-interactive server access |

## Leakage and fairness controls

- Training IDs are frozen by JSON or an existing teacher SQLite ledger.
- Training IDs are removed from every strict-unsampled metric.
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
37 passed, 2 skipped
```

The two skipped tests are pre-existing optional integration paths. A non-UTF-8
Windows locale initially caused one pre-existing test to decode a UTF-8 prompt as
GBK; rerunning with `PYTHONUTF8=1` produced the result above.

## Server experiment results

No metrics are recorded yet. Do not fill this table from local synthetic data.

| Training labels | Method | Unsampled AP | P@R80 | P@R90 | R80 achievement | R90 achievement |
|---|---|---:|---:|---:|---:|---:|
| Existing frozen 2,000 IDs | Pending | — | — | — | — | — |
| Batch multi-label 2,000 IDs | Pending | — | — | — | — | — |

When the server run completes, copy the exact protocol hashes, ID-set source,
runtime, memory, request/token usage, all method metrics, and comparison against
LR-U/US/US3/USA into this file.
