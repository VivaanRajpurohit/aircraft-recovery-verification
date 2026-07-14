# Change record 003: neural output bounds

## Objective

Attempt a sound output-bounding backend on the exact existing neural-policy architecture before considering a compact policy.

## Implementation and scientific rationale

Added float64 interval bound propagation through checkpoint normalization, each exact Linear/ReLU layer, and the control, mode, and target heads. Monotone tanh and sigmoid transformations retain bounded output semantics. Endpoint computations are rounded outward by one representable float64 value.

## Tests and verification results

Unit tests compare affine and small-network intervals against sampled executions. Sampling is used only to test the implementation and is not the proof method. The exact checkpoint feasibility command is `python scripts/check_exact_policy_bounds.py checkpoints/phase3_quick/best.pt`.

## Known limitations

The backend is sound but conservative for a represented input box and supports ReLU checkpoints only. It does not by itself verify the plant, monitor, or recovery property. A closed-loop classification is withheld until Gate C plant residuals exist.

## Effect on earlier artifacts

None. The checkpoint is loaded read-only; no simulator, controller, dataset, result, or replay artifact is modified.
