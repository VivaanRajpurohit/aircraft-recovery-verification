# Final Split and Hold-Out Design

Status: preregistered before large-dataset generation.

| Split | Flights | Per tier | Seed range | Permitted use |
|---|---:|---:|---|---|
| Train | 576 | 96 | 20000–20575 | Optimization and training-only normalization |
| Development | 144 | 24 | 21000–21143 | Early stopping and bounded hyperparameter selection |
| Challenge validation | 144 | 24 | 22000–22143 | Hard known-family validation; no optimization |
| OOD-style simulator stress | 144 | 24 | 23000–23143 | Supported distribution-shift analysis |
| Final sealed test | 144 | 24 | 24000–24143 | One post-freeze evaluation only, with explicit authorization |

Splits are made at flight level and isolate flight ID, scenario instance, seed, record hash, and trajectory signature. No transition-row random split is permitted.

Challenge holds out severity values 0.40/0.65/0.82 and activation times 2.5/5.0/7.5 s. OOD-style stress holds out 0.22/0.52/0.85 and 2.25/5.5/7.25 s, plus supported compositions and boundary initial-condition bins. The sealed test uses unique seeds, severity values 0.39/0.64/0.84, and activation times 2.75/5.25/7.75 s. Those exact conditions cannot be used for tuning, debugging, threshold selection, or failure-case retraining.

The sealed split manifest may be generated and hashed by a non-evaluating packaging step, but its records and labels remain inaccessible until dataset generation, architecture, hyperparameters, preprocessing, monitor/fallback settings, recovery rules, metrics, and analysis code are frozen. Existing sealed data is not used as design feedback.

Challenge, OOD, and sealed labels describe simulator-domain distribution relationships—not real-world generalization.
