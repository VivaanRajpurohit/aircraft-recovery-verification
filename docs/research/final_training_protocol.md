# Final Training Protocol

Status: **DRAFT FROZEN BEFORE LARGE-DATASET GENERATION**. Pilot approval passed, but final training remains blocked until the revised 1,152-flight design is generated, independently approved, split-audited, and preprocessing-frozen.

## Fixed architecture-study budget

Exactly three candidates may be compared:

1. Small reference: `[128, 64]`, ReLU.
2. Current reference: `[256, 128]`, ReLU.
3. Moderate: `[256, 256, 128]`, SiLU.

All retain 42 inputs, five bounded controls, five mode logits, and four stabilization targets. No unlimited search is permitted.

The total search budget is fixed at 12 runs: three architecture-screening runs using seed 17001, four optimizer-grid runs for the selected architecture using seed 17001, and five final independent seeds. Architecture screening uses LR 0.001 and batch 256. The optimizer grid is LR `{0.0003, 0.001}` × batch `{256, 512}`. No other learning rates, batch sizes, or architectures may be tried without a dated pre-training amendment.

Selection uses development and challenge validation only: first reject any candidate with worse safety violations or >10% oscillation increase versus the current reference; among remaining candidates select the lowest macro normalized control MAE, breaking ties with closed-loop recovery and then parameter count.

## Training controls

- Final seeds fixed before training: `17001, 17002, 17003, 17004, 17005`.
- AdamW; LR selected from `{0.0003, 0.001}`; weight decay 0.0001.
- Batch size selected from `{256, 512}`; epoch limit 50; patience 8; improvement tolerance `1e-10`.
- Gradient norm clipping 1.0.
- Loss weights initially controls 1.0, mode 0.25, stabilization targets 0.1.
- Any output reweighting requires a separate development-only ablation and dated decision.
- Normalization is fit only on approved training flights; exact 42-feature order is enforced; standard deviation floor is `1e-6`; clipping is ±5 training standard deviations; missing/reordered/unexpected fields are fatal.

Every seed writes to a unique checkpoint directory and records dataset, split, preprocessing, configuration, environment, architecture, initialization, optimizer, epoch history, duration, best epoch, and checkpoint SHA-256. All seeds are reported; no best-seed-only result is allowed.

## Loss, metrics, and naming

The training objective is weighted mean-squared error for five normalized controls plus cross-entropy for controller mode and normalized target loss using the fixed weights above. Development reporting includes per-output MAE/RMSE for elevator, aileron, rudder, left throttle, and right throttle; macro normalized MAE; mode metrics; saturation; and smoothness. Closed-loop recovery and safety metrics decide ties and may veto an offline winner.

Checkpoint directories use `checkpoints/final_policy_v1/<architecture_id>/seed_<seed>/`; `best.pt` and `last.pt` are never shared between seeds. Each checkpoint is tied to approved-dataset, split, preprocessing, configuration, and source hashes. SHA-256 is computed after writing and verified on load.

Training-only normalization uses the exact ordered 42-feature contract, standard-deviation floor `1e-6`, and ±5 training-standard-deviation clipping. Missing, unexpected, or reordered features are fatal. Development, challenge, OOD, and sealed records cannot contribute statistics.
