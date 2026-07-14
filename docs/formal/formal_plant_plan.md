# Formal plant derivation and validation plan

## Proposed transition

The first plant is an interval/linear-parameter-varying discretization of `SimpleAircraftSimulator`, not a separately invented aircraft. With `dt = 0.1 s`, effective delayed commands, left/right availability `a_L/a_R`, and aileron effectiveness `eta_a`, the source equations give:

```text
T_L = throttle_left * a_L
T_R = throttle_right * a_R
v+ = v + dt * (8 * ((T_L + T_R)/2 - 0.5) - 0.02 * (v - 140)) + r_v
pitch+ = pitch + dt * 12 * elevator + r_pitch
bank+ = bank + dt * 30 * (eta_a * aileron + 0.12 * (T_R - T_L)) + r_bank
vertical_speed+ = v+ * 101.269 * sin(pitch+ in radians) + r_vs
clearance+ = clearance + dt * vertical_speed+ / 60 + r_clearance
```

Interval multiplication handles throttle availability and aileron effectiveness. Sine is bounded over the complete pitch interval. Actuator delay is a finite command queue of `ceil(delay_seconds / dt)` steps. Crosswind has no direct roll term in the current simulator, so any claimed crosswind-to-bank effect must be represented by a calibrated residual or excluded.

## Calibration data

Generate a new versioned calibration set only after approval: 2,500 stratified one-step state/action/failure samples plus 80 deterministic 100-step traces. Cover interior and boundary values of the declared validity region, command rates, asymmetric thrust, aileron effectiveness, and delays. Use dedicated calibration seeds and save the configuration, split manifest, hashes, and source commit.

## Held-out validation

Generate 1,500 separately seeded one-step samples plus 40 held-out 100-step traces. No state/action tuple or random seed may overlap calibration. Include validity-region boundaries and interpolate between calibration grid points. Report per-state maximum absolute error, quantiles, coverage, and multi-step drift.

## Residual bounds

For each state coordinate, use the outward-rounded maximum absolute held-out prediction error plus an explicit floating-point tolerance. When subregions have enough coverage, use piecewise residuals; otherwise retain the global maximum. A cell outside validation coverage is `outside_model_validity`. A bound widened beyond the configured approximation tolerance is `unknown_overapproximation`.

Gate C remains incomplete until these datasets exist, held-out errors are measured, and residual bounds are committed. Provisional residuals in smoke configuration cannot support a verified-safe classification.
