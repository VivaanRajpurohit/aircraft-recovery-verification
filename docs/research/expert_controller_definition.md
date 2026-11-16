# Expert Controller Definition

## Identity

Training labels are produced by `DeterministicFallbackController` (`phase2_deterministic_fallback`), not by a human pilot, an optimal-control solver, or the runtime safety monitor. It is a deterministic PID/rule controller with a hysteretic emergency-mode state machine and diversion planner.

The term **expert** means only “the simulator controller selected to generate imitation labels.” It does not imply optimality, aerodynamic validation, certification, or real-aircraft expertise.

## Inputs observed

At each 100 ms control step the controller receives a structured `ControllerInput` containing:

- aircraft state: altitude, airspeed, pitch, roll, yaw, vertical speed, heading, position;
- derived envelope: angle of attack, load factor, stall and overspeed margins;
- control health: elevator/aileron/rudder effectiveness, stuck flags, actuator delay;
- engine health: left/right available thrust, failed flags, asymmetric thrust;
- controller-known active failures and emergency declaration;
- measured environment and navigation status;
- the full current list of candidate airports available to the diversion planner.

The controller does not read simulator ground-truth audit snapshots. However, it has **privileged structured context relative to the 42-feature neural policy**: its diversion planner can inspect the full airport list and it retains PID integrals, previous errors, emergency-mode state, dwell time, and selected-plan state. The neural observation contains a fixed summary rather than all planner inputs/history.

## Algorithm

1. The diversion planner ranks airports using configured runway, crosswind, terrain, availability, reachability, and control-authority rules.
2. A deterministic state machine selects NORMAL, FAILURE_DETECTED, STABILIZE, GLIDE, DIVERT, APPROACH, EMERGENCY_LANDING, UNRECOVERABLE, or TERMINATED, with a configured 2 s minimum dwell.
3. Pitch, roll, heading, and airspeed PID controllers produce bounded commands.
4. Stall-margin logic commands nose-down recovery and a higher target speed. Large pitch returns toward zero. Total thrust loss selects a −3° glide target.
5. Navigation heading error becomes a bank target limited to ±25°; rudder also compensates modeled asymmetric thrust.
6. Elevator, aileron, and rudder proposals divide by known effectiveness down to a 0.25 denominator and are then clamped.
7. Failed engines receive zero proposed throttle; otherwise both throttles use the bounded airspeed loop.

## Outputs and bounds

The label order is fixed:

1. `elevator` ∈ [−1, 1]
2. `aileron` ∈ [−1, 1]
3. `rudder` ∈ [−1, 1]
4. `throttle_left` ∈ [0, 1]
5. `throttle_right` ∈ [0, 1]

The controller also produces stabilization targets, mode, navigation selection, confidence, and a fallback safety-decision record. These auxiliary values are not part of the five-control target vector.

There is no explicit command slew-rate limiter in the expert. PID dynamics, the fixed step, output clipping, and simulator actuator delay constrain temporal response. This absence must not be described as a physical actuator-rate model.

## Proposed versus executed actions

Imitation labels are the expert's **proposed controls before actuator degradation/delay**. The candidate record separately stores:

- `expert_proposed_controls`;
- `monitor_modified_controls` when a monitor is part of a specifically declared runtime-assurance dataset (otherwise null);
- `executed_controls` from the simulator audit snapshot.

Reduced effectiveness, stuck surfaces, engine availability, and actuator delay can therefore make executed controls differ from proposals. Post-monitor or post-actuator values must never silently replace expert-proposal labels.

## Determinism and configuration

For a fixed scenario, simulator seed, airport list, configuration, initial controller reset, and software revision, the controller is deterministic. It contains no random-number generator. Reproducibility still depends on the simulator trajectory and exact serialized configuration.

Canonical metadata: `artifacts/research_stage/expert_controller_manifest.json`.

## Limitations

- Rule/PID gains were selected for this low-order simulator, not a physical aircraft.
- Exact health variables and planner context may make imitation easier than deployment with incomplete observations.
- It is not proven optimal and can generate valid data showing failed recovery.
- It does not model pilot behavior, actuator bandwidth, structural loading, or high-fidelity aerodynamics.
- Its fallback status is a software role, not a safety certification.

