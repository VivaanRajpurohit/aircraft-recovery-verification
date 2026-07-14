# Bounded recovery-envelope specification

Protocol `v1.1.0-formal-smoke` analyzes repeated closed-loop transitions of the existing neural proposal, runtime monitor, projection/rejection, deterministic fallback, actuator delay, and a conservative interval plant under complete left-engine failure plus bounded aileron degradation, wind, and sensor error.

At each 0.1-second step the analyzer constructs the bounded 42-feature observation, propagates it through the exact checkpoint, applies all feasible assurance branches, advances the degraded plant, inflates the set by validated residuals, checks safety, and updates recovery dwell state. Branches may be split or soundly joined according to configuration.

The initial region, formal validity region, horizon, dwell, properties, solver limits, and uncertainty bounds are machine-readable in `configs/verification/`. The smoke horizon is 0.5 seconds, pilot horizon 3 seconds, and full target horizon 10 seconds. Only the full target uses the scientific 5-second recovery dwell.

The planned plant is an interval/LPV discretization of the simulator equations. Calibration uses deterministic factorial trajectories spanning controls and selected failures. Held-out validation uses separate seeds and interior/boundary points. For each state coordinate, residual inflation is the outward-rounded maximum absolute held-out error plus a declared numerical tolerance. No cell may be classified safe until the validation envelope covers its inputs.

The exact existing checkpoint is attempted first with sound IBP. Exact ReLU enumeration is not selected for the initial Windows workflow because the 384 ReLU units make exhaustive branching impractical. A compact policy is not currently recommended; it becomes an option only after measured exact-policy bound width or runtime prevents useful horizons, and it must remain separately named and evaluated.

This document is a software protocol, not a universal safety claim.
