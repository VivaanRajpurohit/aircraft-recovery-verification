# Historical Neural Checkpoint Audit

Status terms are `VERIFIED`, `INFERRED`, and `UNKNOWN`. Filenames are not treated as provenance evidence.

## `checkpoints/phase3_quick/best.pt`

| Fact | Finding | Status |
|---|---|---|
| SHA-256 | `efbcc999dd43e7f9e75dc9738e67bb19f2b7fe994ecee626de10b296b8b62c25` | VERIFIED |
| First Git commit containing checkpoint | No commit on any local/remote branch contains it; only `checkpoints/.gitkeep` is tracked | UNKNOWN |
| Training dataset | `datasets/phase3_quick`, dataset ID `6d73dd...e349d6` | VERIFIED from adjacent summary; immutable linkage is incomplete |
| Flights/transitions | Manifest-backed values available locally; checkpoint directory itself does not embed complete source files | PARTIALLY VERIFIED |
| Feature order | `phase1-42-v1`; canonical 42 names in dataset metadata | VERIFIED locally |
| Five control targets | elevator, aileron, rudder, left throttle, right throttle | VERIFIED in generator/model code |
| Architecture | MLP 42 → 256 → 128, ReLU; bounded 5-control head, 5-mode head, 4 stabilization-target head; 45,710 parameters | VERIFIED |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 | VERIFIED |
| Batch/epochs | 512; 4 completed; best epoch 4 | VERIFIED |
| Early stopping | patience 3; minimum improvement `1e-10` | VERIFIED |
| Seed | 3400 | VERIFIED |
| Normalization | training split only, five-standard-deviation clipping | VERIFIED by current code/config; original immutable manifest linkage incomplete |
| Split | episode-level seeded shuffle | VERIFIED from local manifest/code |
| Scenario coverage | Quick Phase 2/3 simulator scenarios | VERIFIED from local metadata; not exhaustive |
| Label source | deterministic fallback proposed actions | VERIFIED by data generator code |
| Training entry point | `scripts/train_policy.py` using quick configuration | VERIFIED locally |

Classification: **provenance-incomplete and unsuitable as a primary scientific comparison**. It can remain a historical engineering baseline with this limitation.

## `checkpoints/policy_smoke_v1/best.pt`

| Fact | Finding | Status |
|---|---|---|
| SHA-256 | `daecaeb37eedd943804da7cf13ed90e4dccca4a80268f8003596f63d6dfbf4e4` | VERIFIED |
| First Git commit containing checkpoint | None; checkpoint is untracked | UNKNOWN |
| Dataset | `datasets/policy_smoke_v1`, ID `341681...c933f` | VERIFIED locally |
| Flights/transitions | 20 flights, 4,651 transitions | VERIFIED from metadata only; sealed test records were not opened |
| Feature contract | Exact 42-feature `phase1-42-v1` ordering preserved in preprocessing manifest | VERIFIED |
| Control order | elevator, aileron, rudder, throttle_left, throttle_right | VERIFIED |
| Architecture | MLP 42 → 256 → 128, ReLU; 45,710 parameters | VERIFIED |
| Optimizer | AdamW, LR 0.001, weight decay 0.0001 | VERIFIED |
| Batch/epochs | 256; 8 completed; best epoch 8 | VERIFIED |
| Early stopping | patience 5 | VERIFIED |
| Seed | 6300 | VERIFIED |
| Normalization | 3,151 training transitions only; 5σ clipping; test fitting explicitly false | VERIFIED |
| Split | flight/episode-level; 14 train, 3 validation, 3 sealed test | VERIFIED from manifest only |
| Scenario coverage | nominal, engine, actuator, sensor, disturbance, and boundary smoke cases | VERIFIED |
| Labels | deterministic fallback **proposed** controls; executed actions not labels | VERIFIED |
| Training script/config | `scripts/train_policy.py`, `configs/training/policy_smoke_imitation.yaml` | VERIFIED |

Classification: **partially reproducible historical pipeline-validation artifact**. The source/config/data remain local and internally linked, but the checkpoint and producing dirty worktree were never committed. It must not be called the final research model.

## Scientific use

Neither checkpoint is eligible for cherry-picked reporting. Both may be evaluated as named historical baselines on development/challenge scenarios, provided compatibility is checked and their provenance limitations accompany every result. Their current sealed test partitions remain unopened.

