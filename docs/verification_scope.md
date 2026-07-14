# Verification Scope and Claims

Phase 1 contains runtime data validation and empirical automated tests only. It
does **not** deliver or claim formal verification.

Phase 4 will encode bounded abstractions in Z3 for command bounds and rate
limits, monitor decision completeness, unsafe-action disposition, invalid-data
fallback activation, and selected discrete safety-envelope transitions.
Satisfiable/unsatisfiable examples, counterexamples, solver bounds, assumptions,
duration, and limitations will be recorded.

Dynamics fidelity, learned-controller quality, recovery success, diversion
feasibility, numerical implementation behavior, latency, and performance under
sampled failures remain empirical. A solver result will cover only the written
formula, finite bounds, discretization, and model assumptions. It will not prove
the full neural network, continuous real dynamics, sensors, actuators, hardware,
weather, pilot interaction, or any real aircraft universally safe.

