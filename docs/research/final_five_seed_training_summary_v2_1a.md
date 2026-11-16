# Final Five-Seed Training Summary v2.1A

Date: 2026-09-20

Status: **FIVE FIXED SEEDS TRAINED; STOPPED BEFORE HELD-OUT EVALUATION**

The frozen `[256, 256, 128]` SiLU policy was independently trained with seeds
17001 through 17005. Every run used AdamW, learning rate 0.001, batch size 512,
weight decay 0.0001, at most 50 epochs, patience 8, improvement tolerance
`1e-10`, gradient-norm clipping at 1.0, and minimum development MSE for
checkpoint selection. TRAIN alone entered optimization. DEVELOPMENT alone
entered checkpoint selection and early stopping.

| Seed | Epochs | Best epoch | Development MSE | MAE | RMSE | Runtime (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17001 | 45 | 37 | 0.000628643627 | 0.008481379643 | 0.025072766636 | 97.981 |
| 17002 | 42 | 34 | 0.000647738182 | 0.009133453726 | 0.025450701015 | 90.872 |
| 17003 | 36 | 28 | 0.000651709198 | 0.008814042986 | 0.025528595700 | 77.980 |
| 17004 | 48 | 40 | 0.000631657225 | 0.009688443133 | 0.025132791823 | 104.976 |
| 17005 | 47 | 39 | 0.000622424843 | 0.008480906208 | 0.024948443697 | 101.755 |

Across seeds, mean development MSE was `0.000636434615` with sample standard
deviation `0.000012657806`; mean MAE was `0.008919645139` with sample standard
deviation `0.000508056290`; and mean RMSE was `0.025226659774` with sample
standard deviation `0.000250628339`. The complete mean, sample standard
deviation, median, minimum, and maximum summaries—including every output—are
stored in the machine-readable provenance.

All five best checkpoints independently passed SHA-256 verification, fresh
model reload, the 42-input/5-output/110,341-parameter contract, and development
metric reproduction. Challenge, OOD-stress, and final sealed test records were
not accessed.

The frozen protocol requires all five neural seeds to be reported and prohibits
a best-seed-only result, but it contains no unambiguous representative-model,
checkpoint-aggregation, or ensemble rule. No seed is selected here. A
prospective amendment is required before any workflow that needs one
representative checkpoint. Challenge evaluation remains unopened.

Machine-readable provenance:
`artifacts/research_stage/final_five_seed_training_v2_1a.json`.
