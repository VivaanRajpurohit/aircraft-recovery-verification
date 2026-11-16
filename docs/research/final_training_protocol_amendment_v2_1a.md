# Final Training Protocol Amendment v2.1A

Date: 2026-09-20

Status: **ACTIVE IMMUTABLE PRE-TRAINING AMENDMENT**

This amendment resolves the neural-output, checkpoint-selection, and split-role
inconsistencies discovered before any v2.1 architecture optimization or
development, challenge, OOD-stress, or sealed evaluation occurred. It adds a
new provenance layer and does not rewrite any historical frozen file.

## Authoritative neural output contract

The v2.1A neural policy accepts the frozen ordered 42-element observation vector
and produces exactly five bounded control outputs in this fixed order:

1. `elevator`
2. `aileron`
3. `rudder`
4. `throttle_left`
5. `throttle_right`

The first three outputs use `tanh` bounds of [-1, 1]. The two throttle outputs
use sigmoid bounds of [0, 1]. The previously specified five mode logits and four
stabilization-target outputs are removed from the final learning objective.
The immutable approved v2.1 export contains no labels for those auxiliary
outputs, and the manuscript defines a five-control-output neural policy.
Auxiliary labels must not be synthesized, inferred, backfilled, or regenerated.

The frozen architecture candidates remain:

| Architecture ID | Hidden layers | Activation | Independently verified parameters |
| --- | --- | --- | ---: |
| `small_reference` | [128, 64] | ReLU | 14,085 |
| `current_reference` | [256, 128] | ReLU | 44,549 |
| `moderate` | [256, 256, 128] | SiLU | 110,341 |

Each count includes all weights and biases in the shared feed-forward trunk and
the single five-control output head.

## Optimization and checkpoint selection

The frozen v2.1 training protocol specifies mean-squared error for normalized
control commands. The v2.1-specific preflight implementation also defined
control MSE. The generic historical multi-head trainer's Huber objective is a
legacy code path and is not the v2.1A final-policy objective.

Training minimizes mean-squared error over the five ordered control targets.
After every epoch, the identical mean-squared-error objective is evaluated on
the development split. `best.pt` is the epoch with the minimum development MSE,
subject to the frozen strict improvement tolerance of `1e-10`. Early stopping
uses the same development MSE, patience 8, and the same improvement tolerance.
Development MAE, RMSE, and per-output MAE/RMSE are diagnostics only and cannot
replace MSE as the checkpoint criterion.

## Split roles and selection authority

| Split | Authorized v2.1A use |
| --- | --- |
| Train | Optimization only |
| Development | Architecture selection, hyperparameter selection, early stopping, and checkpoint selection |
| Challenge | Post-selection evaluation only |
| OOD-style simulator stress | Post-selection stress evaluation only |
| Final sealed test | Final confirmatory evaluation only |

Challenge, OOD-stress, and sealed outcomes cannot cause architecture reselection
or hyperparameter tuning. The prior challenge/closed-loop veto language is
superseded because it conflicts with the frozen split-role and access contract.
Architecture selection is minimum development MSE, with lower parameter count
as the deterministic exact-tie breaker.

## Preserved frozen state

This amendment does not alter:

- `final_policy_candidate_v2_1`;
- `final_policy_approved_v2_1`;
- the 576/144/144/144/144 split geometry;
- Tier 0 through Tier 5 definitions;
- final seeds 17001, 17002, 17003, 17004, and 17005;
- train-only normalization;
- the final metric contract;
- the final monitor or fallback;
- holdout semantics or leakage protections; or
- the sealed-access boundary.

The architecture screen retains seed 17001, AdamW, learning rate 0.001, batch
size 256, weight decay 0.0001, at most 50 epochs, patience 8, improvement
tolerance `1e-10`, and gradient-norm clipping at 1.0. The later optimizer grid
remains learning rates {0.0003, 0.001} by batch sizes {256, 512}. Five-seed final
training remains reserved for the later authorized stage.

## Provenance

The machine-readable companion is
`artifacts/research_stage/final_training_protocol_amendment_v2_1a.json`.
Its `amendment_logical_hash` is calculated over the canonical JSON object with
that field excluded. Amendment logical hash:
`4ecb3942f366ac862990b11b353a06a37b1f0e261004981e9ab5f1359cc92f1f`.
