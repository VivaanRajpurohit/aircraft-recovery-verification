# Model Limitations

- Evidence is limited to a low-order deterministic/stochastic simulator with explicitly configured disturbances and failures.
- The expert is a hand-designed PID/rule controller, not an optimal or human-pilot oracle.
- The neural policy may inherit expert weaknesses and has less planner/history context than the expert.
- Simulator actuator, sensor, engine, terrain, and aerodynamic behavior are simplified.
- The runtime monitor checks a declared abstraction; it does not certify the neural controller or actual aircraft safety.
- Recovery definitions and safety envelopes are experiment constructs and may omit relevant real-world hazards.
- OOD-style tests remain shifts inside the simulator domain and are not evidence of real-world generalization.
- Latency results apply only to recorded hardware/software and do not establish hard-real-time certification.
- Failed recoveries, systematic family weaknesses, fallback domination, and solver unknown/timeouts must remain visible.
- No claim of foolproof behavior, production readiness, certification, aerodynamic validation, or real-aircraft safety is made.

