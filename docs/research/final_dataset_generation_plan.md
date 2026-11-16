# Final Dataset Generation Plan — Design Only

Status: **DESIGN AUDIT REQUIRED; GENERATION NOT AUTHORIZED**  
Configuration: `configs/training/final_policy_dataset_design_v1.yaml`

## Sample-size rationale

The 36-flight pilot has only six flights per tier and cannot support stable family-level estimates or five-seed training comparisons. Prospective analysis rejected the intermediate 720-flight plan. The accepted design has 192 flights per tier and 1,152 total: 576 train and 144 in each of development, challenge, OOD-style stress, and final sealed test. Using the pilot mean gives approximately 299,000 transitions; the planned 20–30 second durations imply a 230,400–345,600 range at 10 Hz.

The trajectory remains the independent experimental unit. Transition count is not treated as 187,000 independent recovery trials.

## Explicit scenario matrix

Each tier contains 120 flights. Tier 0 contains 120 nominal flights. Tier 1 contains 40 each of mild aileron degradation, partial thrust loss, and complete engine loss under mild initial conditions. Tiers 2–5 each contain four families of 30 flights, covering isolated moderate impairments, concurrent failures, severe supported combinations, and boundary in-domain conditions.

Aggregate factor exposure (overlapping for concurrent failures) is: no failure 120; reduced aileron 220; complete engine loss 190, balanced 95 left/95 right; partial thrust loss 100; actuator delay 210; deterministic sensor bias 120; turbulence/crosswind 120; reduced rudder 30.

Every family of 30 uses 18/3/3/3/3 flights across train/development/challenge/OOD/sealed. Every family of 40 uses 24/4/4/4/4. Nominal uses 72/12/12/12/12. Thus no non-nominal family is hidden in one split.

## Supported variation

- Airspeed 95–180 kt; bank −40° to +40°; pitch −15° to +15°.
- Altitude 1,200–10,000 ft and declared terrain clearance 500–3,000 ft.
- Failure activation 2–8 s.
- Supported left/right engine loss, partial thrust loss, aileron degradation 0.20–0.85, delays 0.10–0.80 s, deterministic airspeed/altitude/heading bias, turbulence/crosswind, and supported combinations.
- Multiple predetermined environment and disturbance seeds; no seed hunting.

`InitialStateConfig` does not support direct vertical-speed initialization: reset sets it to zero. Rather than inventing physics, the dataset will stratify measured vertical speed at failure activation after deterministic pre-failure warmup. The design audit must confirm adequate counts in the five specified vertical-speed bins.

Gaussian sensor noise is excluded until the realized draw is serialized for independent validation. Frozen/missing sensor cases require a specialized review contract before inclusion. Difficult or failed expert trajectories remain valid data evidence when structurally correct.

## Generation gates

Before generation, the configuration must pass count arithmetic, supported-parameter validation, seed/split isolation, held-out value isolation, expected compute/storage review, and schema-v3 contract coverage. This document does not authorize generation, preprocessing fitting, or training.
