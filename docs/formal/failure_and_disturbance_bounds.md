# Failure, disturbance, and uncertainty bounds

The first formal family fixes complete left-engine failure and varies partial aileron authority, crosswind, delay, and optional sensor error. Bounds are closed intervals and configurable in `configs/verification/`.

| Quantity | Unit | Smoke | Pilot | Full target |
|---|---:|---:|---:|---:|
| Left thrust availability | fraction | 0 | 0 | 0 |
| Right thrust availability | fraction | 1 | 0.95–1.00 | 0.90–1.00 |
| Aileron effectiveness | fraction | 0.55–0.65 | 0.40–0.80 | 0.30–1.00 |
| Actuator delay | s | 0–0.10 | 0–0.20 | 0–0.30 |
| Crosswind | kt | −15–15 | −25–25 | −35–35 |
| Airspeed sensor error | kt | −1–1 | −3–3 | −5–5 |
| Bank sensor error | deg | −0.5–0.5 | −1.5–1.5 | −3–3 |
| Clearance sensor error | ft | −10–10 | −25–25 | −50–50 |

Crosswind is signed. The current simulator injects it into yaw rather than a resolved aerodynamic lateral-force model, so a separately calibrated roll residual is required before a bank-envelope result can pass Gate C. Gaussian simulator turbulence has unbounded mathematical support; formal experiments replace it with a declared deterministic interval, never with an unstated Gaussian guarantee.

All bounds are hypotheses for calibration and validation, not assertions about a real Cessna. Conditions outside the configured formal validity region must be classified `outside_model_validity`.
