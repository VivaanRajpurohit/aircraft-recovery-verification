# Phase 4: Bounded Formal Abstraction and Runtime Monitor

> “Verified” in this phase means only that named logic obligations were checked
> by Z3 under the stated variables, bounds, assumptions, and one-step model. It
> does not mean that the neural network, simulator, or any real aircraft is
> universally safe.

## Mathematical abstraction

For time step `dt`, the monitor encodes affine next-state approximations:

```text
V'   = V + dt[T((tl+tr)/2 - 0.5) - D(V-Vcruise)] + dV
q'   = q + dt Ke elevator_effectiveness elevator + dq
phi' = phi + dt Ka aileron_effectiveness aileron + dphi
AoA' = AoA + dt KAoA elevator + dAoA
n'   = n + Kn |aileron| + dn
h'   = hclearance + vertical_speed dt / 60 - terrain_loss
```

Every disturbance is independently bounded by `configs/safety/default.yaml`.
Z3 searches for a disturbance assignment that violates any encoded one-step
property. `unsat` means no counterexample exists in this abstraction; `sat`
returns an example; `unknown` or timeout causes fallback.

## Encoded properties

- stall and overspeed margins;
- absolute bank, pitch, angle of attack, and load-factor bounds;
- positive modeled terrain clearance outside landing handling;
- primary and throttle command bounds;
- per-step command-rate bounds;
- invalid input cannot be accepted;
- an unsafe accepted action is inconsistent;
- rejection and unknown results require fallback;
- `UNRECOVERABLE -> NORMAL` is forbidden;
- `TERMINATED` has no outgoing operational transition.

Thresholds, checked state bounds, disturbances, horizon, timeout, assumptions,
and limitations are configuration values rather than code constants.

## Runtime decisions

The monitor returns accept, modify, reject-and-fallback, or
unknown-and-fallback. Projection enumerates rate-limited blends toward a
deterministic recovery command and selects the nearest candidate for which Z3
finds no bounded counterexample. If none exists, the Phase 2 deterministic
fallback executes. Original neural and final actions are both retained in logs,
along with solver status, counterexample, latency, modification magnitude, and
violated property names.

The projection search is finite and not an optimal controller. The fallback is
not itself proven to recover all states. The monitor uses observed data only and
receives no hidden simulator truth.

## Claims and limitations

Formal results cover the written Boolean invariants and one-step affine model
only. They exclude nonlinear simulator dynamics, longer horizons, correlated or
out-of-bound disturbances, unmodeled sensor behavior, terrain geometry,
floating-point implementation equivalence, hardware, and the complete neural
network. Runtime monitoring can reduce acceptance of actions with modeled
counterexamples; it cannot guarantee recovery, certification, or airworthiness.
