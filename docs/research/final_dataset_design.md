# Final Dataset Design

## Scope

The dataset samples only failure types and parameters already implemented by `SimpleAircraftSimulator`. It does not invent new physics. Raw candidate files are immutable; validation reads serialized JSONL from disk. Final sealed-test content is excluded from generation, analysis, tuning, and pilot work.

## Difficulty tiers

| Tier | Exact development design bounds |
|---|---|
| 0 nominal | No failure; airspeed 125–155 kt, altitude 5,000–10,000 ft, pitch −5° to +7°, bank −20° to +20° |
| 1 isolated mild | One supported impairment; severity 0.20–0.40; activation 3–8 s; turbulence ≤0.10 |
| 2 isolated moderate | One impairment; severity 0.41–0.65; activation 2–7 s; delay ≤0.5 s |
| 3 concurrent moderate | Two supported failures, each severity 0.30–0.65; turbulence ≤0.20; delay ≤0.6 s |
| 4 severe in-domain | Supported engine/control combinations; severity 0.66–0.85 for degradations or declared complete engine failure; delay ≤0.8 s; turbulence ≤0.30 |
| 5 boundary/stress | Initial airspeed 95–125 or 155–180 kt, altitude/clearance 1,200–4,000 ft, pitch ±15°, bank ±40°, supported concurrent failures; no deliberately impossible state |

These tiers are experiment strata, not physical hazard categories.

## Planned inventory

- Pilot: 36 development flights, six per tier, using seeds 15000–15035.
- Audited full candidate design: 1,152 flights, 192 per tier; see `final_dataset_generation_plan.md` and `final_policy_dataset_design_v1.yaml`.
- Scenario-family balance target within each applicable tier: nominal, engine-only, control-only, engine+control, delay, sensor-error, disturbance, and boundary.
- Train 432; development 72; challenge 72; OOD-style 72; final sealed 72. All seed ranges are disjoint. No final-test record is opened for design feedback.

The pilot is permitted to show failed recoveries. Valid controller failures remain evidence and are never reclassified as invalid data.

## Record contract

Every transition records dataset/record/flight/scenario IDs, seed, tier/family/severity, step/timestamps, exact 42-feature vector and names, proposed/monitor-modified/executed controls, known and ground-truth failure state, activation parameters, actuator health/delay, sensor/environment values, before/after state, outcome/termination, configuration hashes, controller hash, Git revision, generator version, and canonical record hash.

## Splits

- Training: optimization only.
- Development: early stopping and bounded architecture selection.
- Challenge: harder known combinations/severity bands not identical to training.
- OOD-style simulator stress: unseen seeds/activation bands and compositions of supported factors.
- Final sealed test: inaccessible until the final freeze and explicit authorization.

Splits operate on complete flights and must also isolate seeds, scenario-instance IDs, record hashes, and near-duplicate trajectory signatures.
