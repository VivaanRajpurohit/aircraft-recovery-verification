# Change record 004: formal plant calibration

## Objective and scientific classification

Gate C derives and validates a conservative transition abstraction for the focused
complete-left-engine/partial-aileron/delay/wind family. The result is **an
intervalized formal representation of the existing low-order simulator dynamics**.
It is a hybrid: the equation structure is an algebraic extraction of the simulator's
low-order update, while coefficients are fitted solely from calibration records. The
fit recovers the simulator constants to floating-point precision.

This is software/model containment validation, not independent aircraft-model or
real-aircraft fidelity validation. It does not establish a safe recovery region.
Closed-loop neural-network and safety-monitor verification has not begun, and
higher-fidelity aircraft validation remains separate.

## Independent execution paths

Numerical records are produced by `generate_dataset` -> `_make_record` ->
`SimpleAircraftSimulator.step`. `_formal_state` then reads the simulator's mutated
true state through `audit_snapshot` and stores it as `next_state` in the JSONL record.

Fitting uses `fit_formal_plant` -> `_features` -> `numpy.linalg.lstsq` for airspeed,
pitch, bank, and clearance. The vertical-speed scale is the median calibration ratio.
`predict_point` computes development residuals, and only the final 20 percent of
one-step calibration records determine residual bounds.

Formal prediction uses `predict_point` for point errors and `formal_plant.transition`
for interval reachability. These functions perform their own NumPy/interval algebra.
They do not import or call the simulator transition, simulator coefficient tables, or
a shared state-transition helper. The validator loads the stored source state,
executed action, failure parameters, realized disturbances, sensor errors, and stored
`next_state`; it never instantiates or steps the simulator. Expected values are the
stored simulator outputs, while predictions are independently evaluated formal
expressions.

Dataset loading verifies the manifest, transition hash, record count, sample-ID hash,
and scenario-ID hash before fitting or validation. No normalization or output
regeneration is applied to stored `next_state` values.

## Data generation and isolation

The ordered state is airspeed, pitch, bank, vertical speed, terrain clearance, and
five previous executed commands. Terrain clearance is altitude above a fixed terrain
datum. Records include true and observed state, proposed and executed actions,
failures, realized pitch/roll disturbance, sensor error, provenance, and the
0.1-second timestep. Outputs are atomic JSONL with deterministic identifiers and
SHA-256 manifests.

Calibration uses seed 7101 with 160 independent one-step samples and six 20-step
traces. Validation uses seed 8101 with 100 independent one-step samples and four
20-step traces. Their seeds, sample IDs, scenario IDs, transition hashes, and manifest
hashes are disjoint. The first 128 calibration samples fit coefficients; the remaining
32 calibration samples define development residuals. No validation path is accepted
or opened by `fit_formal_plant`, no global record cache exists, and validation records
are not used for structure selection, fitting, or residual inflation. Model structure
was selected from the known simulator equations before held-out validation.

## Equation equivalence and coefficient recovery

For the configured `dt=0.1`, thrust acceleration 8 kt/s, drag coefficient 0.02/s,
cruise speed 140 kt, pitch rate 12 deg/s, and roll rate 30 deg/s:

- Airspeed is the exact unclamped simulator affine update in current airspeed and
  availability-weighted throttles. The expected coefficients are
  `[-0.12, 0.998, 0.4, 0.4]`. Complete left-engine failure fixes left availability at
  zero, so the left coefficient is unidentifiable and the minimum-norm fit returns
  zero without supporting extrapolation.
- Pitch is the exact unclamped update with coefficients `[0, 1, 1.2, 1]` for
  intercept, current pitch, executed elevator, and recorded realized pitch increment.
- Bank is the exact unclamped update with coefficients `[0, 1, 3, 0.36, 1]` for
  intercept, current bank, effective aileron, thrust asymmetry, and recorded realized
  roll increment.
- Vertical speed exactly uses `101.269 * airspeed * sin(pitch)`.
- Clearance exactly uses the simulator's updated vertical speed and coefficient
  `dt/60 = 1/600`.
- Previous-action states exactly become the recorded executed action.

The compact machine-readable comparison is in
`artifacts/formal_plant_coefficient_comparison.json`. Differences from known constants
are floating-point least-squares effects. Development residual extrema are likewise
floating-point effects and are expanded by an explicit `1e-8` numerical tolerance.
Outward rounding is still required because formal interval soundness cannot assume
real-number operations: floating-point addition, multiplication, sine, fitted
coefficients, and repeated propagation can otherwise round inward.

Simulator clamps are not represented because the modeled operating domain stays far
from the simulator's broad clamps. Yaw, heading, geographic motion, airport geometry,
termination logic, headwind effects, controller decisions, sensor-to-controller
propagation, and branching delay queues are excluded.

## Noise, disturbances, delay, and residual meaning

The simulator is deterministic for a fixed seed, but nonzero turbulence invokes
seeded Gaussian process perturbations to pitch and roll. Gate C records the realized
increments and supplies them as explicit fixed inputs to the formal transition;
therefore no stochastic process noise remains unmodeled *conditional on that recorded
trace*. Crosswind affects simulator yaw, which is outside the formal state. Sensor
errors affect recorded observations, not true plant state, and are not used in the
true-state transition.

The simulator uses discrete low-order updates. The formal equations reproduce the
same update order, so integration discrepancy is limited to floating-point evaluation
at this gate. Actuator delay is represented exactly for each stored transition through
the executed action. Queue resolution has a unit test, but branching over possible
delayed controller actions is not part of this validation.

No artificial noise was added to enlarge residuals. The nonzero residual envelopes
come from least-squares/evaluation roundoff plus the explicit `1e-8` safety tolerance.

## Formal model validity range

These limits are computational/modeling limits, not a physically validated flight
envelope:

| Quantity | Formal limit | Basis |
|---|---:|---|
| Airspeed | 100 to 180 kt | Chosen modeled operating domain padding calibration sampling at 105 to 175 kt |
| Pitch | -20 to 20 deg | Chosen modeled operating domain padding calibration sampling at -15 to 15 deg |
| Bank | -50 to 50 deg | Chosen modeled operating domain padding calibration sampling at -45 to 45 deg |
| Vertical speed | -20,000 to 20,000 ft/min | Conservative computational guard for the derived state; not independently sampled or physically validated |
| Clearance | 100 to 12,000 ft | Chosen verification scope padding calibration sampling at 500 to 10,000 ft |
| Actuator delay | 0 to 0.1 s | Exact calibration sampling range and recorded-executed-action scope |

Failure validity additionally fixes left availability at zero, right availability at
0.95 to 1.0, and aileron effectiveness at 0.4 to 0.8. Disturbance validity is
crosswind +/-25 kt, realized pitch increment +/-1 deg/step, and realized roll increment
+/-2 deg/step. These are formal model and calibration-domain choices, not airworthiness
or physical-fidelity claims.

## Held-out smoke results and interpretation

All 100 held-out one-step records were inside validity and contained for all five
modeled states. Maximum absolute errors were `2.8422e-13` kt airspeed, `5.3291e-15`
degrees pitch, `1.4211e-14` degrees bank, and `7.2760e-12` for vertical speed and
clearance in their respective units. These machine-precision errors mean that fitted
coefficients recovered an equation-equivalent representation of the same low-order
software dynamics under stored realized inputs. They are not evidence of real-aircraft
accuracy or independent physical-model agreement.

Rollout validation is **interval propagation under fixed recorded executed actions and
realized disturbances** (open-loop replay validation), not closed-loop controller
validation. Each trace begins from a narrow numerical box of radius `1e-9`, not a
meaningful operational uncertainty set. Four held-out traces were contained through
horizons 5, 10, and 20; maximum vertical-speed width was `1.1757e-4` ft/min. The small
20-step widths do not predict interval growth once observation uncertainty,
neural-network branching, monitor branching, action uncertainty, and larger initial
sets are introduced.

Perturbation tests independently alter bank response, throttle response,
vertical-speed scale, clearance integration, and residual inflation in temporary model
copies. Every corrupted model produces nonzero error and containment failures. No
corrupted model is saved as research evidence.

The corrected raw smoke artifacts occupy 1,162,439 bytes under
`results/formal_plant_smoke_v1` and remain ignored. The tracked summary is
`artifacts/formal_plant_smoke_summary.json`; smoke datasets are not publication
evidence.

## Reproduction

```powershell
python scripts/generate_plant_calibration_data.py configs/formal_plant/calibration_smoke.yaml
python scripts/generate_plant_calibration_data.py configs/formal_plant/validation_smoke.yaml
python scripts/fit_formal_plant.py results/formal_plant_smoke_v1/calibration configs/formal_plant/calibration_smoke.yaml results/formal_plant_smoke_v1/formal_plant_model.json
python scripts/validate_formal_plant.py results/formal_plant_smoke_v1/formal_plant_model.json results/formal_plant_smoke_v1/calibration results/formal_plant_smoke_v1/validation results/formal_plant_smoke_v1/validation_report.json
```
