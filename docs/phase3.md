# Phase 3: Demonstrations and Neural Imitation Baseline

> Phase 3 is empirical simulation research. The neural policy is not formally
> verified and has no Phase 4 runtime safety monitor.

## Data boundary and leakage prevention

The expert is the Phase 2 deterministic fallback controller. Demonstrations are
collected in closed loop: each expert action advances the simulator and the next
expert input is the resulting observation. `transitions.npz` is compressed,
columnar numerical data. `analysis_ground_truth.jsonl` separately stores true
state and hidden failures. The training loader admits only an explicit array
allowlist and rejects unexpected or truth-named arrays.

Training inputs contain only the unchanged 42-feature controller vector.
Recoverability labels, hidden failures, true state, future failure schedules,
future states, termination outcomes, PID internals, and diversion scores never
enter the training tensor. Episode IDs, categories, seeds, known failures,
termination records, and navigation labels are metadata rather than inputs.

## Splits and validation

Splits operate on complete episode IDs, never timesteps. A seeded shuffle forms
train, validation, and test sets; configured categories are held out as OOD.
The manifest records every episode and is checked for overlap and full coverage.
Dataset validation checks 42-value finite observations, control bounds, throttle
bounds, mask agreement, feature/unit names, leakage fields, category counts, and
split integrity. Array-content, configuration, and manifest hashes identify the
artifacts reproducibly.

## Preprocessing

Statistics are fitted only on training episodes. Continuous values use mean/std
normalization with a configurable standard-deviation clip. Non-finite inputs are
deterministically replaced with zero before normalization. Binary health flags
and explicit validity/availability masks remain 0/1 and are not normalized.
Statistics and the feature names/version are embedded in every checkpoint.

Heading, bearing, runway heading, and wind direction remain direct degree
features to preserve the Phase 1 vector. This creates a discontinuity at
0/360 degrees; a sine/cosine encoding would require a future versioned vector.

## Policy and objectives

The default MLP is `42 -> 256 -> 128`, with ReLU after each hidden layer and two
heads. Primary controls use `tanh`; throttles use `sigmoid`; stabilization
targets use documented fixed scales; the emergency head predicts the five
existing public modes. The default model has 45,710 trainable parameters. ReLU
does not merge layers or imply safety; it supplies the nonlinear transformation
between learned affine layers. ReLU and SiLU are compared under controlled,
identical training settings and only measured differences are reported.

Training combines Huber control loss, emergency-mode cross entropy, and optional
scaled stabilization-target Huber loss. An inverse-frequency weighted sampler
balances modes with replacement; original and effective distributions and that
replacement behavior are recorded. AdamW, gradient clipping, early stopping,
periodic checkpoints, best-checkpoint selection, resume support, CPU/CUDA
selection, deterministic seeds, and optional CUDA AMP are implemented.

## Checkpoints and inference

Checkpoints include model and optimizer states, architecture, parameter count,
feature names and vector version, normalization, output rules, training config,
dataset/split identifiers, versions, validation metrics, epoch, and format
version. Load rejects incompatible versions, feature lists, parameter counts,
and malformed files.

The neural controller implements the unchanged controller protocol. Navigation
fields not directly predicted by the model are constructed deterministically
from the current validated observation. Neural output is labeled `not_checked`.
If inference raises, a deterministic software execution fallback is labeled
`neural_runtime_failure`; this is not a safety-monitor intervention and is
excluded from neural-performance interpretation.

## Evaluation

Offline evaluation reports action MAE/RMSE, mode accuracy and per-mode
precision/recall, disagreement, smoothness, saturation, latency, and peak GPU
memory. Paired closed-loop evaluation gives expert and neural controllers the
same scenario seed but separate simulators, so each controller receives its own
resulting next state. Recovery, stabilization, diversion, envelope, termination,
latency, smoothness, and saturation metrics are reported separately.

Low offline error does not imply stable or safe closed-loop behavior.

## Limitations

The expert is understandable but not proven optimal; imitation can reproduce
its weaknesses. Weighted sampling changes the effective distribution.
Distribution shift, accumulated closed-loop error, sparse approach/landing
labels, and direct degree encodings can degrade behavior. Dynamics, terrain, and
airports remain simplified abstractions. Phase 3 results are empirical. Phase 4
will add bounded formal analysis and a runtime monitor only for explicitly
specified properties, assumptions, models, and bounds.
