# Phase 2: Failures, Fallback Recovery, and Diversion

> Simulation-only research software. None of the behavior in this phase is
> formally verified, suitable for operational flight planning, or evidence of
> airworthiness.

## True state and controller observation

The simulator maintains true state and applies every active ground-truth
failure to its dynamics. It separately constructs a sensor-observed state and
then the validated controller packet. Only failures marked both `observable`
and `detected` appear in `active_failures` and health packets. Hidden failures
remain in `ground_truth_before`/`ground_truth_after` JSONL audit records but are
not disclosed to the controller.

Sensor bias and noise modify observations without modifying true state. Frozen
sensors retain their activation value. Because the Phase 1 schema deliberately
keeps aircraft fields required, a temporarily missing scalar sensor retains its
last valid bounded value; navigation-related loss also sets `data_valid=false`.
The ground-truth failure audit is the authoritative missing-data indicator.

## Failure model

Each failure has an identifier, type, subsystem, severity, activation mode,
optional activation time or state predicate, duration/permanence, ramp time,
observability/detection flags, and type-specific parameters. Linear ramp
intensity is `severity * min(1, elapsed / ramp_duration)`. The deterministic
engine supports more than three concurrent failures and keeps hidden truth
separate from controller knowledge.

The dynamics remain low-order. Effectiveness multipliers, stuck commands,
delayed command queues, command-rate bounds, engine authority, deterministic
seeded noise/turbulence, and simplified wind terms are abstractions rather than
validated component models.

## Deterministic fallback

The fallback is independent of any neural policy. Bounded PID loops control
pitch, roll, heading, and airspeed. Saturation-aware conditional integration and
integral clamps prevent windup. Rules lower pitch and raise the speed target
near stall, level excessive attitudes, compensate known asymmetric thrust with
rudder, scale commands for known remaining authority, select a diversion, and
command a wind-corrected heading. All gains and primary thresholds are in
`configs/controllers/fallback.yaml`.

## Emergency modes

Allowed transitions are:

```text
NORMAL -> FAILURE_DETECTED -> STABILIZE
STABILIZE -> GLIDE | DIVERT | UNRECOVERABLE
GLIDE -> DIVERT | APPROACH | UNRECOVERABLE
DIVERT -> APPROACH | STABILIZE | UNRECOVERABLE
APPROACH -> EMERGENCY_LANDING | STABILIZE | UNRECOVERABLE
EMERGENCY_LANDING -> TERMINATED | UNRECOVERABLE
UNRECOVERABLE -> TERMINATED
```

A configurable minimum dwell prevents rapid assessment/stabilization
oscillation. Each transition records old/new mode, time, trigger, state, known
failures, and selected airport in `summary.json`.

## Diversion and range abstractions

Airports are synthetic. The planner evaluates every candidate using distance,
constant-ratio glide range, constant-endurance powered range, runway length,
wind/crosswind, terrain penalty, control authority, engine availability,
turn-altitude loss, alignment, surface, and availability. Hard rejection reasons
are retained. The highest-scoring feasible airport is selected, which need not
be the nearest.

Glide range uses usable altitude times a configurable glide ratio. Powered
range assumes constant speed, endurance, reserve, and an engine-authority
factor. Turn loss assumes constant bank and descent rate. Wind correction uses
a bounded planar wind triangle. These are simulator abstractions and must not be
used as real-world flight-planning guidance.

## Recoverability and termination

Scenarios carry `expected_recoverability` as `recoverable`, `marginal`,
`unrecoverable`, or `unknown`. This experimental label is written to metadata
but never supplied to a controller. Termination categories distinguish duration
completion, stable recovery, diversion arrival, approach completion, ground
impact, sustained stall/overspeed, loss of control, terrain violation, invalid
state, explicitly unrecoverable conditions, and numerical instability.

## Claims and limitations

Phase 2 results demonstrate deterministic simulator behavior and automated test
coverage only. The model omits coupled six-degree-of-freedom aerodynamics,
validated propulsion and actuator models, terrain geometry, certification-grade
sensors, structural dynamics, and real aviation data. No formal-methods solver
is used in Phase 2, and no formal-verification claim has been established.
