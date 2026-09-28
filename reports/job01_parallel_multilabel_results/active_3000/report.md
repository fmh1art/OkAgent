# job01 multi-proxy router results

## Protocol

- Comparison claim: `architecture_only`
- Training candidates: 3,000 (positive target: 502)
- Strict-unsampled candidates: 31,761 (historical positives: 123)
- Training ID SHA-256: `613ec27a1aef6e023410733ccde8fda653df505d82c9f1f5784fdeb6186c528c`
- Pre-scoring truth scope: `none`
- Model seed / OOF folds: 11 / 5

## Ranking metrics

| Method | Unsampled AP | P@R80 | Actual R80 | P@R90 | Actual R90 |
|---|---:|---:|---:|---:|---:|
| global | 0.0464 | 1.51% | 80.49% | 0.77% | 90.24% |
| experience | 0.0298 | 0.97% | 80.49% | 0.66% | 90.24% |
| credentials | 0.0140 | 0.63% | 82.11% | 0.57% | 90.24% |
| expert_mean | 0.0433 | 1.64% | 80.49% | 0.85% | 90.24% |
| stacking_router | 0.0507 | 1.88% | 80.49% | 0.85% | 90.24% |
| embedding_router | 0.0281 | 1.18% | 80.49% | 0.93% | 90.24% |

## 20-seed recall calibration

| Method | Target | Precision mean±sd | Recall mean±sd | F1 mean±sd | Recall achieved | Worst recall | Mean recall LCB |
|---|---:|---:|---:|---:|---:|---:|---:|
| global | R80 | 1.59%±0.44% | 76.48%±10.42% | 3.10%±0.84% | 40% | 58.16% | 68.66% |
| global | R90 | 0.88%±0.35% | 88.47%±8.08% | 1.74%±0.68% | 50% | 72.45% | 82.13% |
| experience | R80 | 1.08%±0.31% | 77.55%±9.42% | 2.13%±0.60% | 45% | 60.20% | 69.77% |
| experience | R90 | 0.68%±0.23% | 89.69%±8.42% | 1.35%±0.44% | 55% | 71.43% | 83.67% |
| credentials | R80 | 0.71%±0.11% | 76.63%±8.65% | 1.40%±0.21% | 30% | 59.18% | 68.73% |
| credentials | R90 | 0.55%±0.03% | 89.59%±4.16% | 1.10%±0.06% | 30% | 80.61% | 83.21% |
| expert_mean | R80 | 1.63%±0.30% | 76.33%±10.56% | 3.18%±0.56% | 50% | 46.94% | 68.47% |
| expert_mean | R90 | 1.02%±0.35% | 87.81%±7.08% | 2.02%±0.68% | 40% | 71.43% | 81.27% |
| stacking_router | R80 | 1.87%±0.30% | 76.17%±9.20% | 3.65%±0.56% | 50% | 61.22% | 68.26% |
| stacking_router | R90 | 1.08%±0.46% | 88.16%±7.65% | 2.13%±0.89% | 50% | 74.49% | 81.75% |
| embedding_router | R80 | 1.25%±0.22% | 77.40%±10.52% | 2.45%±0.42% | 55% | 53.06% | 69.65% |
| embedding_router | R90 | 0.91%±0.19% | 89.03%±7.73% | 1.81%±0.37% | 55% | 67.35% | 82.74% |

## Lower-bound-constrained calibration

Thresholds in this table are eligible only when the calibration-set Clopper-Pearson 95% Recall lower bound reaches the target.

| Method | Target | Valid seeds | Test precision | Test recall | Recall achieved | Mean test Recall LCB |
|---|---:|---:|---:|---:|---:|---:|
| global | R80 | 20/20 | 0.66% | 93.32% | 95% | 87.89% |
| global | R90 | 0/20 | — | — | — | — |
| experience | R80 | 20/20 | 0.58% | 92.14% | 90% | 86.54% |
| experience | R90 | 0/20 | — | — | — | — |
| credentials | R80 | 20/20 | 0.51% | 93.52% | 100% | 87.96% |
| credentials | R90 | 0/20 | — | — | — | — |
| expert_mean | R80 | 20/20 | 0.77% | 92.55% | 100% | 86.98% |
| expert_mean | R90 | 0/20 | — | — | — | — |
| stacking_router | R80 | 20/20 | 0.78% | 92.70% | 95% | 87.17% |
| stacking_router | R90 | 0/20 | — | — | — | — |
| embedding_router | R80 | 20/20 | 0.75% | 92.55% | 90% | 87.05% |
| embedding_router | R90 | 0/20 | — | — | — | — |

## Router behavior

| Expert | Primary assignments | Assignment share | Mean gate weight | Stacker probability coefficient |
|---|---:|---:|---:|---:|
| global | 9,860 | 28.37% | 0.2917 | 3.7556 |
| experience | 20,386 | 58.65% | 0.5683 | 2.1063 |
| credentials | 4,515 | 12.99% | 0.1400 | 2.1249 |

## Summary

Best strict-unsampled AP: **stacking_router** (0.0507).

This run is an architecture comparison only. It must not be described as exceeding USA until the training ID hash is verified against the exact USA seed-11, 2,000-ID protocol.
