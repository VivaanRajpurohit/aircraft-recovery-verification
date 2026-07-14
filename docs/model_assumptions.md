# Simplified Model Assumptions and Numerical Contract

The Phase 1 simulator is a deterministic, low-order integration model for
software validation. It assumes a rigid intact aircraft, flat local Earth,
constant gravity, short time steps, synthetic weather, ideal navigation,
decoupled first-order attitude response, simplified thrust/drag, and no ground
effect. Lift, sideslip, inertia coupling, control saturation dynamics, fuel,
compressibility, terrain geometry, and failures are not physically modeled yet.
Catastrophic structural breakup is outside scope.

External API values use feet, knots, feet per minute, degrees, nautical miles,
seconds, milliseconds, and g as named. Conversion helpers isolate feet/meters
and knots/meters-per-second. Validated initial ranges include altitude
`[-1,000, 60,000] ft`, airspeed `[0, 600] kt`, pitch `[-90, 90] deg`, roll and
yaw `[-180, 180] deg`, load factor `[-3, 9] g`, angle of attack `[-20, 40] deg`,
normalized primary control `[-1, 1]`, throttle `[0, 1]`, latitude `[-90, 90]`,
and longitude `[-180, 180]`.

These validation ranges are representational bounds, not certified safety
limits. Safety thresholds live separately in `configs/safety/default.yaml`.

