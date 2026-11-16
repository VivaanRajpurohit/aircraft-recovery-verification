# Candidate Review-Disposition Audit

## Scope and preservation

This audit analyzes `final_policy_pilot_v2` only. It reads immutable serialized records from disk and does not call the simulator's live `step()` method. Before and after validation, `raw/transitions.jsonl` remained SHA-256 `40f3b590e3e585a6b8ab0d067cfe9fec768dbe999873bf0be9374e804ded3446` and `raw/flights.jsonl` remained `93dc3e3149a5ae6ab8e1dd056b8a04a761d1977190d28420a2bcb01606adcb66`.

No final model was trained, no approved dataset was exported, and no sealed-test record was opened.

## Root causes

| Class | Old validator reason | Occurrences | Classification | Scientific determination |
|---|---|---:|---|---|
| RC-ACT-001 | `actuator_degradation_requires_human_review_of_serialized_history` | 6,198 | VALIDATOR_BUG | Underlying records are EXPECTED_BY_MODEL and independently deterministic from serialized command history. |
| RC-SEN-001 | `observed_true_state_difference_under_declared_sensor_fault` | 1,116 | VALIDATOR_BUG | Underlying differences are EXPECTED_BY_MODEL deterministic airspeed bias; ground truth is preserved separately. |

The counts overlap: some records have both actuator and sensor findings. They represent 6,234 unique records across 25 flights, not 7,314 independent data problems.

Scenario-level old actuator findings were: partial aileron 722; engine+aileron 2,156; actuator/turbulence/thrust 1,120; sensor+actuator 1,080; rudder+crosswind 1,120. All 1,116 sensor findings came from the four `phase2_sensor_actuator` flights. Exact flight IDs, timestep ranges, failure schedules, severities, effectiveness ranges, representative records, neighboring context, and source hashes are retained in the 25 flight packets.

## Actuator contract

The stored fields have distinct semantics:

1. `expert_proposed_controls` is the expert command for the current observation.
2. Actuator delay selects the latest serialized proposed command whose timestamp is no later than `current_time - delay`. Flight-control delay affects elevator, aileron, and rudder; throttle delay affects throttles.
3. Stuck/rate-limit transformations would then affect the applied command. Those mechanisms do not occur in the 25 affected pilot flights; the new validator conservatively retains `REVIEW_REQUIRED` for them until a specialized serialized-history rule is available.
4. `executed_controls` stores the command after delay/stuck/rate-limit handling.
5. Surface effectiveness does **not** scale `executed_controls`. It scales that command inside the subsequent attitude dynamics. Therefore `executed ≈ proposed × effectiveness` is not the repository's contract.

Validator v3 independently reconstructs visible effectiveness as the product of `(1 - active intensity)` for applicable visible/detected degradation failures. It independently reconstructs delayed commands from prior serialized proposed controls. Every affected pilot command matched within absolute tolerance `1e-7`, including the exact binary-float timestamp comparison that determines delay-queue eligibility.

The v3 rule fails incorrect effectiveness, wrong-sign execution, wrong-actuator execution, out-of-range/mismatched effectiveness, missing degradation metadata, invalid delay metadata, and execution inconsistent with serialized history.

## Sensor contract

`aircraft_state_before` is ground-truth simulator state. `ordered_observation_42` is the controller-observed feature vector. They must not be made equal under sensor failure.

The pilot sensor fault is hidden `airspeed_bias` with severity 0.8, a two-second ramp, and `maximum_bias = 22 kt`. For each active record, the independently expected observation is:

`observed airspeed = clamp(true airspeed + active intensity × 22 kt, 0, 600 kt)`

Observed bias magnitudes ramp from 0.88 kt through 16.72 kt, then remain 17.6 kt. All 1,116 differences match that rule. Other unaffected state features match ground truth within the documented tolerance. `sensor_error_failures` identifies the same active sensor-fault IDs, and true state remains separate.

Validator v3 fails incorrect bias magnitude, corruption of the wrong feature, overwritten ground truth, missing/invalid fault metadata, metadata-set mismatch, and unexplained state discrepancies. Gaussian noise remains `REVIEW_REQUIRED` because the current schema stores its distribution but not the realized random draw; no such unresolved record occurs in this pilot.

## Disposition rules

- `ACT-SERIALIZED-HISTORY-001`: approve only when effectiveness, delay metadata, and all five executed controls match the independent serialized-history reconstruction.
- `SENSOR-DETERMINISTIC-BIAS-001`: approve only when metadata identifies the sensor and the observed value matches true state plus the deterministic active bias, including clamp/wrap behavior.

All 6,234 formerly flagged records satisfy every applicable deterministic rule. Each disposition records its rule IDs, evidence, validator version, record and flight IDs, UTC timestamp, source record hash, and explanation. No flight is approved merely because most of its transitions pass, and recovery outcome is not used to determine data validity.

## Before and after

| Level | Before PASS | Before REVIEW | Before FAIL | After PASS | After REVIEW | After FAIL |
|---|---:|---:|---:|---:|---:|---:|
| Records | 3,116 | 6,234 | 0 | 9,350 | 0 | 0 |
| Flights | 11 | 25 | 0 | 36 | 0 | 0 |

The only status changes are the 6,234 records for which the two broad warnings were replaced by independently computed contract checks. No raw record changed. Validator v3 report hash is `78a9ae1c2861970ea0ac4622e13826337444195fa5db575c470b5bdaedd90680`; integrity and leakage audits remain clean.

## Gate recommendation

**READY_FOR_APPROVAL_REVIEW.** Evidence supports approving 36 flights and 9,350 transitions, rejecting/excluding zero, subject to explicit authorization. This audit does not perform that export.

This is a low-order simulator data-validity determination. It is not evidence of real-aircraft safety, certification, aerodynamic validation, or universal recovery.
