# job01 multi-proxy router results

## Protocol

- Comparison claim: `architecture_only`
- Training candidates: 2,000 (positive target: 88)
- Strict-unsampled candidates: 32,761 (historical positives: 294)
- Training ID SHA-256: `ad8b48f29482d2dfc2af1cf32ef2f8d17ae738e4fabf16e9b5c8ef440b6067fd`
- Pre-scoring truth scope: `frozen_training_ids_only`
- Model seed / OOF folds: 11 / 5

## Ranking metrics

| Method | Unsampled AP | P@R80 | Actual R80 | P@R90 | Actual R90 |
|---|---:|---:|---:|---:|---:|
| global | 0.3595 | 11.07% | 80.27% | 4.74% | 90.14% |
| experience | 0.3341 | 4.45% | 81.29% | 2.95% | 90.14% |
| credentials | 0.0767 | 2.13% | 80.27% | 1.59% | 90.48% |
| expert_mean | 0.3472 | 7.57% | 80.27% | 5.20% | 90.14% |
| stacking_router | 0.3928 | 8.67% | 80.27% | 5.76% | 90.14% |
| embedding_router | 0.2255 | 6.17% | 80.27% | 3.44% | 90.14% |

## 20-seed recall calibration

| Method | Target | Precision mean±sd | Recall mean±sd | F1 mean±sd | Recall achieved | Worst recall | Mean recall LCB |
|---|---:|---:|---:|---:|---:|---:|---:|
| global | R80 | 11.04%±1.43% | 77.47%±5.02% | 19.25%±2.11% | 35% | 66.81% | 72.57% |
| global | R90 | 5.14%±1.18% | 88.34%±2.96% | 9.68%±2.05% | 25% | 81.70% | 84.35% |
| experience | R80 | 4.39%±0.11% | 80.17%±4.20% | 8.33%±0.18% | 60% | 69.36% | 75.44% |
| experience | R90 | 3.17%±0.42% | 89.34%±2.79% | 6.12%±0.78% | 45% | 84.26% | 85.47% |
| credentials | R80 | 2.13%±0.15% | 79.89%±5.54% | 4.15%±0.28% | 65% | 69.79% | 75.17% |
| credentials | R90 | 1.65%±0.18% | 88.57%±3.66% | 3.24%±0.35% | 35% | 81.70% | 84.63% |
| expert_mean | R80 | 7.96%±1.24% | 79.34%±4.47% | 14.41%±1.93% | 55% | 69.79% | 74.55% |
| expert_mean | R90 | 4.92%±1.08% | 89.72%±3.42% | 9.30%±1.94% | 40% | 81.28% | 85.92% |
| stacking_router | R80 | 8.79%±1.34% | 79.34%±4.47% | 15.77%±2.05% | 60% | 70.21% | 74.56% |
| stacking_router | R90 | 5.33%±1.15% | 89.51%±3.18% | 10.03%±2.05% | 40% | 82.13% | 85.67% |
| embedding_router | R80 | 6.12%±0.66% | 79.40%±3.07% | 11.35%±1.10% | 50% | 73.19% | 74.60% |
| embedding_router | R90 | 3.36%±0.83% | 89.04%±3.94% | 6.45%±1.54% | 45% | 82.13% | 85.17% |

## Lower-bound-constrained calibration

Thresholds in this table are eligible only when the calibration-set Clopper-Pearson 95% Recall lower bound reaches the target.

| Method | Target | Valid seeds | Test precision | Test recall | Recall achieved | Mean test Recall LCB |
|---|---:|---:|---:|---:|---:|---:|
| global | R80 | 20/20 | 5.86% | 86.96% | 100% | 82.81% |
| global | R90 | 20/20 | 2.08% | 95.91% | 100% | 93.20% |
| experience | R80 | 20/20 | 3.40% | 88.09% | 100% | 84.05% |
| experience | R90 | 20/20 | 1.89% | 95.79% | 90% | 93.07% |
| credentials | R80 | 20/20 | 1.75% | 87.32% | 95% | 83.24% |
| credentials | R90 | 20/20 | 1.24% | 95.74% | 95% | 93.01% |
| expert_mean | R80 | 20/20 | 5.32% | 88.28% | 100% | 84.30% |
| expert_mean | R90 | 20/20 | 2.33% | 95.70% | 100% | 92.93% |
| stacking_router | R80 | 20/20 | 5.75% | 88.51% | 100% | 84.56% |
| stacking_router | R90 | 20/20 | 2.51% | 95.64% | 100% | 92.85% |
| embedding_router | R80 | 20/20 | 4.00% | 86.70% | 100% | 82.54% |
| embedding_router | R90 | 20/20 | 1.65% | 95.72% | 100% | 92.98% |

## Router behavior

| Expert | Primary assignments | Assignment share | Mean gate weight | Stacker probability coefficient |
|---|---:|---:|---:|---:|
| global | 11,600 | 33.37% | 0.3353 | 2.4279 |
| experience | 8,618 | 24.79% | 0.2501 | 3.2822 |
| credentials | 14,543 | 41.84% | 0.4145 | 1.6413 |

## Summary

Best strict-unsampled AP: **stacking_router** (0.3928).

This run is an architecture comparison only. It must not be described as exceeding USA until the training ID hash is verified against the exact USA seed-11, 2,000-ID protocol.
