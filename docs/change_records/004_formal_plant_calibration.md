# Change record 004: formal plant calibration

## Objective

Derive and validate a conservative transition abstraction from the existing internal numerical simulator for the focused complete-left-engine/partial-aileron/delay/wind family. This work does not verify a recovery region.

## Simulator variables and generation

The ordered state is airspeed, pitch, bank, vertical speed, terrain clearance, and five previous executed commands. Terrain clearance is altitude above a fixed terrain datum. The generator uses `SimpleAircraftSimulator` through reset, step, and audit interfaces. It records true and observed state, proposed and executed actions, failures, realized pitch/roll disturbance, sensor error, provenance, and the 0.1-second timestep. Outputs are atomic JSONL with deterministic identifiers and SHA-256 manifests.

Calibration uses seed 7101 with 160 independent one-step samples and six 20-step traces. Validation uses seed 8101 with 100 independent one-step samples and four 20-step traces. Sample IDs, scenario IDs, and generation seeds are disjoint. The first 128 calibration samples fit coefficients; the remaining 32 define development residuals. Held-out validation is never used to fit coefficients or residual margins.

## Model and coefficients

The model is a hybrid interval/LPV transition. Airspeed is affine in current airspeed and availability-weighted throttles. Pitch is affine in current pitch, executed elevator, and bounded realized pitch disturbance. Bank is affine in current bank, effectiveness-weighted aileron, engine-availability asymmetry, and bounded realized roll disturbance. Vertical speed uses `101.269 * airspeed * sin(pitch)`. Clearance integrates vertical speed. Previous commands update to executed commands. Interval products, sine extrema, affine evaluation, and residual addition are outward rounded.

The fitted coefficient dimensions are 4 for airspeed, 4 for pitch, 5 for bank, 1 for vertical-speed scale, and 3 for clearance. Complete left-engine failure is fixed in the validity region, so its throttle coefficient is intentionally unidentifiable and fitted as zero; no claim is made outside zero left availability.

## Residuals and validity

Signed development residual extrema are expanded by an explicit `1e-8` numerical tolerance and outward rounding. Validity is limited to 100–180 kt airspeed, ±20 degrees pitch, ±50 degrees bank, ±20,000 ft/min vertical speed, 100–12,000 ft clearance, left availability zero, right availability 0.95–1.0, aileron effectiveness 0.4–0.8, delay 0–0.1 seconds, crosswind ±25 kt, pitch disturbance ±1 degree/step, roll disturbance ±2 degrees/step, and the configured sensor-error ranges.

Sensor errors are observation uncertainty and do not alter true plant dynamics. Actuator delay is recorded through executed commands and has a tested queue resolver; the smoke rollout replays recorded executed commands rather than analyzing controller/queue branching.

## Smoke results

All 100 held-out one-step records were inside validity and contained for all five modeled states. Maximum absolute errors were `2.8422e-13` kt airspeed, `5.3291e-15` degrees pitch, `1.4211e-14` degrees bank, and `7.2760e-12` for both vertical speed and clearance in their respective units. All four held-out traces were contained at every step through horizons 5, 10, and 20. The maximum vertical-speed interval width was `1.1757e-4` ft/min. These near-machine-precision errors are expected because the abstraction preserves the low-order simulator equations and receives recorded realized disturbances; they must not be generalized to real aircraft dynamics or unmodeled disturbances.

The corrected smoke artifacts occupy 1,162,439 bytes under `results/formal_plant_smoke_v1`. The pre-fix exploratory output is preserved separately under `results/formal_plant_smoke_v1_pre_manifest_fix`. A lightweight tracked summary is in `artifacts/formal_plant_smoke_summary.json`.

## Tests and limitations

Tests cover deterministic generation, schema/order, split isolation, degradation/delay, atomic writes, deterministic hashes and fitting, finite coefficients, interval ordering, residual inflation, validity rejection, known unit equations, held-out containment, rollout propagation, and explicit containment-failure reporting. The simulator remains a low-order research abstraction. Crosswind affects yaw, which is outside this formal state. Bounded sensor uncertainty has not yet been propagated through the neural observation. No policy, monitor, or recovery property is analyzed here.

## Reproduction

```powershell
python scripts/generate_plant_calibration_data.py configs/formal_plant/calibration_smoke.yaml
python scripts/generate_plant_calibration_data.py configs/formal_plant/validation_smoke.yaml
python scripts/fit_formal_plant.py results/formal_plant_smoke_v1/calibration configs/formal_plant/calibration_smoke.yaml results/formal_plant_smoke_v1/formal_plant_model.json
python scripts/validate_formal_plant.py results/formal_plant_smoke_v1/formal_plant_model.json results/formal_plant_smoke_v1/calibration results/formal_plant_smoke_v1/validation results/formal_plant_smoke_v1/validation_report.json
```
