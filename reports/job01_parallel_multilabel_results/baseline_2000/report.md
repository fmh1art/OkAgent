# job01 multi-proxy router results

## Protocol

- Comparison claim: `architecture_only`
- Training candidates: 2,000 (positive target: 239)
- Strict-unsampled candidates: 32,761 (historical positives: 294)
- Training ID SHA-256: `ad8b48f29482d2dfc2af1cf32ef2f8d17ae738e4fabf16e9b5c8ef440b6067fd`
- Pre-scoring truth scope: `none`
- Model seed / OOF folds: 11 / 5

## Ranking metrics

| Method | Unsampled AP | P@R80 | Actual R80 | P@R90 | Actual R90 |
|---|---:|---:|---:|---:|---:|
| global | 0.1923 | 4.74% | 80.27% | 2.91% | 90.14% |
| experience | 0.1187 | 4.10% | 80.95% | 2.79% | 90.14% |
| credentials | 0.0575 | 2.04% | 80.27% | 1.40% | 90.14% |
| expert_mean | 0.1962 | 5.58% | 80.61% | 3.69% | 90.14% |
| stacking_router | 0.2025 | 5.68% | 80.27% | 3.74% | 90.14% |
| embedding_router | 0.1471 | 4.08% | 80.27% | 2.50% | 90.14% |

## 20-seed recall calibration

| Method | Target | Precision mean±sd | Recall mean±sd | F1 mean±sd | Recall achieved | Worst recall | Mean recall LCB |
|---|---:|---:|---:|---:|---:|---:|---:|
| global | R80 | 4.72%±0.81% | 79.49%±4.65% | 8.89%±1.41% | 45% | 70.64% | 74.72% |
| global | R90 | 2.89%±0.48% | 88.83%±4.13% | 5.58%±0.89% | 40% | 79.15% | 84.93% |
| experience | R80 | 4.29%±0.68% | 77.26%±7.30% | 8.11%±1.14% | 55% | 59.57% | 72.39% |
| experience | R90 | 2.92%±0.51% | 88.19%±4.31% | 5.64%±0.96% | 35% | 78.30% | 84.22% |
| credentials | R80 | 1.96%±0.23% | 80.51%±4.52% | 3.83%±0.43% | 55% | 73.62% | 75.81% |
| credentials | R90 | 1.39%±0.17% | 89.81%±3.63% | 2.75%±0.33% | 55% | 83.40% | 86.03% |
| expert_mean | R80 | 5.60%±0.69% | 79.43%±5.38% | 10.43%±1.14% | 60% | 66.38% | 74.66% |
| expert_mean | R90 | 3.81%±0.65% | 88.64%±3.83% | 7.28%±1.17% | 50% | 77.45% | 84.70% |
| stacking_router | R80 | 5.55%±0.58% | 80.09%±4.87% | 10.36%±0.97% | 60% | 69.79% | 75.36% |
| stacking_router | R90 | 3.74%±0.65% | 88.74%±3.57% | 7.17%±1.18% | 50% | 80.43% | 84.81% |
| embedding_router | R80 | 4.20%±0.85% | 79.02%±6.15% | 7.94%±1.47% | 55% | 61.28% | 74.24% |
| embedding_router | R90 | 2.62%±0.50% | 89.09%±4.73% | 5.09%±0.93% | 65% | 79.57% | 85.24% |

## Lower-bound-constrained calibration

Thresholds in this table are eligible only when the calibration-set Clopper-Pearson 95% Recall lower bound reaches the target.

| Method | Target | Valid seeds | Test precision | Test recall | Recall achieved | Mean test Recall LCB |
|---|---:|---:|---:|---:|---:|---:|
| global | R80 | 20/20 | 3.10% | 87.87% | 95% | 83.84% |
| global | R90 | 20/20 | 1.61% | 96.19% | 100% | 93.60% |
| experience | R80 | 20/20 | 3.18% | 86.74% | 90% | 82.61% |
| experience | R90 | 20/20 | 1.78% | 95.43% | 90% | 92.62% |
| credentials | R80 | 20/20 | 1.49% | 88.11% | 100% | 84.13% |
| credentials | R90 | 20/20 | 1.05% | 96.40% | 100% | 93.80% |
| expert_mean | R80 | 20/20 | 4.23% | 86.53% | 90% | 82.38% |
| expert_mean | R90 | 20/20 | 2.04% | 95.45% | 95% | 92.65% |
| stacking_router | R80 | 20/20 | 4.11% | 87.40% | 95% | 83.32% |
| stacking_router | R90 | 20/20 | 1.93% | 95.66% | 95% | 92.92% |
| embedding_router | R80 | 20/20 | 2.85% | 87.49% | 85% | 83.45% |
| embedding_router | R90 | 20/20 | 1.83% | 96.15% | 90% | 93.59% |

## Router behavior

| Expert | Primary assignments | Assignment share | Mean gate weight | Stacker probability coefficient |
|---|---:|---:|---:|---:|
| global | 9,625 | 27.69% | 0.2833 | 2.6905 |
| experience | 20,383 | 58.64% | 0.5684 | 1.9227 |
| credentials | 4,753 | 13.67% | 0.1483 | 3.1490 |

## Summary

Best strict-unsampled AP: **stacking_router** (0.2025).

This run is an architecture comparison only. It must not be described as exceeding USA until the training ID hash is verified against the exact USA seed-11, 2,000-ID protocol.
