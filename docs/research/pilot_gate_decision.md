# Pilot Gate Decision

## Decision

The initial decision was **STOP before final training** because the candidate pilot was structurally valid but had 6,234 unresolved review records. The independent contract audit has now resolved those findings without changing raw data. The current recommendation is **READY_FOR_APPROVAL_REVIEW**, while export remains prohibited until explicit authorization.

## Candidate evidence

- Immutable dataset: `final_policy_pilot_v2`
- Flights/transitions: 36 / 9,350
- Split flights: train 18, development 6, challenge 6, OOD-style stress 6
- Difficulty: six flights in every tier 0–5
- Validator before: 3,116 PASS, 0 FAIL, 6,234 REVIEW_REQUIRED
- Validator v3 after independent contract checks: 9,350 PASS, 0 FAIL, 0 REVIEW_REQUIRED
- Leakage/integrity errors: none
- Review triggers: declared actuator degradation and observed/true-state differences during declared sensor faults
- Approved export: not created; explicit authorization still required
- Sealed held-out test: not opened

The old review flags were traced to two overly broad validator rules. Validator v3 now reconstructs delayed executed commands from serialized history and deterministic sensor bias from separate true/observed fields. Adversarial examples remain rejected. Full evidence is in `candidate_review_disposition_audit.md` and `artifacts/research_stage/pilot_review_audit/`.

## Development-only baseline difficulty

Recovery requires roll ≤10°, pitch ≤7.5°, absolute vertical speed ≤750 ft/min, stall margin ≥10 kt, and nonnegative terrain clearance continuously for five seconds after the latest observed failure activation (or simulation start for nominal flights).

| Controller | Strict recoveries | Rate | Flights with safety violation | Violation duration |
|---|---:|---:|---:|---:|
| Constant idle | 14/36 | 38.9% | 36 | 254.4 s |
| Neutral cruise | 14/36 | 38.9% | 15 | 150.6 s |
| Historical Phase 3 checkpoint | 26/36 | 72.2% | 9 | 95.1 s |
| Smoke pipeline checkpoint | 31/36 | 86.1% | 5 | 50.2 s |
| Deterministic expert | 36/36 | 100.0% | 0 | 0.0 s |

The trivial baselines do not recover almost every flight, so the aggregate scenario set can differentiate controller behavior. Tier 1 remains too easy in isolation (6/6 for both trivial baselines), while Tier 5 is strongly discriminative (0/6 trivial, 1/6 smoke, 2/6 historical, 6/6 expert). This finding must remain visible in later dataset refinement.

## Consequences

Large final-data generation, preprocessing fitting, bounded architecture selection, multi-seed training, closed-loop final-policy evaluation, ablations, robustness analysis, and the final freeze remain not executed. The current final-experiment freeze remains explicitly **NOT FROZEN**. The next gate is explicit authorization to export the independently dispositioned pilot; this audit did not export it.

This is low-order simulator-domain evidence only. It is not evidence of real-aircraft safety, certification, aerodynamic validation, or universal recovery.
