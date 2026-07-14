# Formal state and action variables

The recovery-envelope abstraction uses SI-free aviation units matching the numerical simulator. One transition is 0.1 seconds.

## Continuous state

The ordered formal state vector has ten components:

1. `airspeed_kts` (kt)
2. `pitch_deg` (deg; positive nose-up)
3. `bank_deg` (deg; positive right-wing-down)
4. `vertical_speed_fpm` (ft/min; positive climb)
5. `terrain_clearance_ft` (ft)
6. `previous_elevator` (normalized command)
7. `previous_aileron` (normalized command)
8. `previous_rudder` (normalized command)
9. `previous_throttle_left` (normalized command)
10. `previous_throttle_right` (normalized command)

Heading is not presently a recovery property. It is retained by the numerical simulator and may be added to a later formal-model version if directional recovery becomes a claim.

## Discrete controller state

The transition also carries `emergency_mode`, `fallback_active`, `terminated`, and an actuator command queue of `ceil(delay_seconds / 0.1)` actions. These values are not packed into the continuous vector.

## Action

The ordered five-component action is `[elevator, aileron, rudder, throttle_left, throttle_right]`. Surface commands lie in `[-1, 1]`; throttles lie in `[0, 1]`. The configured maximum command change is 0.25 per 0.1-second step.

## Observation

The analyzed policy receives the unchanged 42-component FP32 observation contract. Sensor uncertainty is applied before observation construction; normalization and validity masks are part of the neural-bound computation. The formal state is not a replacement for the simulator state or policy observation.
