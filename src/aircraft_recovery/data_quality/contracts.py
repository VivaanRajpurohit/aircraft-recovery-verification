"""Independent serialized-record checks for actuator and sensor contracts.

This module deliberately does not import or call the simulator. It reconstructs
documented transformations from immutable JSON records.
"""

from __future__ import annotations

from typing import Any

from aircraft_recovery.common.observations import OBSERVATION_FEATURES
from aircraft_recovery.data_quality.common import CONTROL_NAMES


TOLERANCE = 1e-7
SURFACE_FAILURES = {
    "reduced_elevator_effectiveness": "elevator",
    "reduced_aileron_effectiveness": "aileron",
    "reduced_rudder_effectiveness": "rudder",
}
SENSOR_TYPES = {
    "airspeed_bias", "altitude_bias", "heading_bias", "frozen_sensor_value",
    "gaussian_measurement_noise", "temporary_missing_sensor_data",
}
STATE_FIELDS = (
    "altitude_ft", "airspeed_kts", "pitch_deg", "roll_deg", "yaw_deg",
    "vertical_speed_fpm", "heading_deg", "latitude_deg", "longitude_deg",
)


def _check(name: str, status: str, reason: str | None = None, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"check": name, "status": status, "reason": reason, "evidence": evidence or {}}


def _active(record: dict[str, Any]) -> list[dict[str, Any]]:
    return list(record.get("ground_truth_failures", record.get("active_failures", [])))


def _scaled(failure: dict[str, Any], parameter: str) -> float | None:
    parameters = failure.get("parameters")
    if not isinstance(parameters, dict) or parameter not in parameters:
        return None
    return float(failure.get("intensity", failure.get("severity", 0.0))) * float(parameters[parameter])


def _effectiveness(failures: list[dict[str, Any]], visible_only: bool) -> dict[str, float]:
    expected = {name: 1.0 for name in ("elevator", "aileron", "rudder")}
    for failure in failures:
        if visible_only and not (failure.get("observable", False) and failure.get("detected", False)):
            continue
        surface = SURFACE_FAILURES.get(str(failure.get("failure_type")))
        if surface:
            intensity = float(failure.get("intensity", failure.get("severity", -1.0)))
            expected[surface] *= 1.0 - intensity
    return expected


def _delay_seconds(failures: list[dict[str, Any]], failure_type: str) -> tuple[float | None, str | None]:
    maximum = 0.0
    for failure in failures:
        if failure.get("failure_type") != failure_type:
            continue
        scaled = _scaled(failure, "maximum_delay_seconds")
        if scaled is None:
            return None, f"missing_maximum_delay_seconds:{failure.get('failure_id')}"
        maximum = max(maximum, scaled)
    return maximum, None


def actuator_contract_checks(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Reconstruct executed commands, visible health, and delays from history."""
    output: dict[str, list[dict[str, Any]]] = {}
    history: list[tuple[float, dict[str, float]]] = []
    for record in records:
        record_id = str(record["record_id"])
        failures = _active(record)
        proposed = dict(zip(CONTROL_NAMES, map(float, record["expert_proposed_controls"]), strict=True))
        executed = dict(zip(CONTROL_NAMES, map(float, record["executed_controls"]), strict=True))
        history.append((float(record["simulation_timestamp_seconds"]), proposed))
        checks: list[dict[str, Any]] = []

        visible_expected = _effectiveness(failures, visible_only=True)
        recorded_effectiveness = record.get("actuator_effectiveness")
        if not isinstance(recorded_effectiveness, dict):
            checks.append(_check("actuator_effectiveness_contract", "FAIL", "missing_degradation_metadata"))
        else:
            valid = all(
                name in recorded_effectiveness
                and abs(float(recorded_effectiveness[name]) - visible_expected[name]) <= TOLERANCE
                for name in visible_expected
            )
            checks.append(_check(
                "actuator_effectiveness_contract", "PASS" if valid else "FAIL",
                None if valid else "actuator_effectiveness_disagrees_with_active_failures",
                {"expected": visible_expected, "actual": recorded_effectiveness},
            ))

        actuator_delay, actuator_error = _delay_seconds(failures, "actuator_delay")
        throttle_delay, throttle_error = _delay_seconds(failures, "delayed_throttle_response")
        if actuator_error or throttle_error:
            checks.append(_check("actuator_delay_contract", "FAIL", actuator_error or throttle_error))
            output[record_id] = checks
            continue
        assert actuator_delay is not None and throttle_delay is not None
        visible = [item for item in failures if item.get("observable", False) and item.get("detected", False)]
        visible_delay, visible_error = _delay_seconds(visible, "actuator_delay")
        recorded_delay = record.get("actuator_delay_seconds")
        delay_valid = visible_error is None and recorded_delay is not None and abs(float(recorded_delay) - float(visible_delay)) <= TOLERANCE
        checks.append(_check(
            "actuator_delay_contract", "PASS" if delay_valid else "FAIL",
            None if delay_valid else "actuator_delay_metadata_mismatch",
            {"expected_visible_seconds": visible_delay, "actual_seconds": recorded_delay},
        ))

        unsupported = [item for item in failures if item.get("failure_type") in {
            "stuck_elevator", "stuck_aileron", "stuck_rudder", "control_command_rate_limitation",
        }]
        if unsupported:
            checks.append(_check(
                "executed_action_actuator_consistency", "REVIEW_REQUIRED",
                "serialized_stuck_or_rate_limit_history_requires_specialized_contract",
                {"failure_ids": [item.get("failure_id") for item in unsupported]},
            ))
            output[record_id] = checks
            continue

        expected = dict(proposed)
        for name in CONTROL_NAMES:
            delay = throttle_delay if name.startswith("throttle") else actuator_delay
            if delay <= 0.0:
                continue
            # Preserve the serialized binary-float comparison used by the delay
            # queue. Adding a tolerance changes which 100 ms sample is eligible.
            eligible = [commands for timestamp, commands in history
                        if timestamp <= float(record["simulation_timestamp_seconds"]) - delay]
            expected[name] = eligible[-1][name] if eligible else 0.0
        consistent = all(abs(executed[name] - expected[name]) <= TOLERANCE for name in CONTROL_NAMES)
        checks.append(_check(
            "executed_action_actuator_consistency", "PASS" if consistent else "FAIL",
            None if consistent else "executed_action_disagrees_with_serialized_actuator_contract",
            {
                "expected_executed": [expected[name] for name in CONTROL_NAMES],
                "actual_executed": [executed[name] for name in CONTROL_NAMES],
                "actuator_delay_seconds": actuator_delay,
                "throttle_delay_seconds": throttle_delay,
                "surface_effectiveness_applies_in_dynamics_not_to_executed_command": True,
            },
        ))
        output[record_id] = checks
    return output


def sensor_contract_checks(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Reconstruct observed state from true state and serialized fault metadata."""
    output: dict[str, list[dict[str, Any]]] = {}
    frozen_values: dict[str, float] = {}
    previous_observed: dict[str, float] | None = None
    for record in records:
        record_id = str(record["record_id"])
        true_state = {name: float(record["aircraft_state_before"][name]) for name in STATE_FIELDS}
        expected = dict(true_state)
        failures = _active(record)
        sensor_failures = [item for item in failures if str(item.get("failure_type")) in SENSOR_TYPES]
        metadata = record.get("sensor_error_failures")
        checks: list[dict[str, Any]] = []
        if not isinstance(metadata, list):
            checks.append(_check("sensor_fault_metadata", "FAIL", "missing_sensor_fault_metadata"))
        else:
            expected_ids = {str(item.get("failure_id")) for item in sensor_failures}
            actual_ids = {str(item.get("failure_id")) for item in metadata}
            checks.append(_check(
                "sensor_fault_metadata", "PASS" if actual_ids == expected_ids else "FAIL",
                None if actual_ids == expected_ids else "sensor_fault_metadata_set_mismatch",
                {"expected_ids": sorted(expected_ids), "actual_ids": sorted(actual_ids)},
            ))
        unverifiable: list[str] = []
        invalid_metadata: list[str] = []
        for failure in sensor_failures:
            kind = str(failure.get("failure_type"))
            failure_id = str(failure.get("failure_id"))
            parameters = failure.get("parameters")
            if not isinstance(parameters, dict):
                invalid_metadata.append(failure_id)
                continue
            if kind in {"airspeed_bias", "altitude_bias", "heading_bias"}:
                field = {"airspeed_bias": "airspeed_kts", "altitude_bias": "altitude_ft", "heading_bias": "heading_deg"}[kind]
                scaled = _scaled(failure, "maximum_bias")
                if scaled is None:
                    invalid_metadata.append(failure_id)
                    continue
                expected[field] += scaled
                if field == "heading_deg":
                    expected[field] %= 360.0
            elif kind == "frozen_sensor_value":
                field = str(parameters.get("field", ""))
                if field not in expected:
                    invalid_metadata.append(failure_id)
                    continue
                frozen_values.setdefault(failure_id, expected[field])
                expected[field] = frozen_values[failure_id]
            elif kind == "temporary_missing_sensor_data":
                field = str(parameters.get("field", ""))
                if field not in expected:
                    invalid_metadata.append(failure_id)
                elif previous_observed is not None:
                    expected[field] = previous_observed[field]
            elif kind == "gaussian_measurement_noise":
                unverifiable.append(failure_id)
        expected["altitude_ft"] = min(60000.0, max(-1000.0, expected["altitude_ft"]))
        expected["airspeed_kts"] = min(600.0, max(0.0, expected["airspeed_kts"]))
        expected["heading_deg"] %= 360.0
        observed = {name: float(record["ordered_observation_42"][OBSERVATION_FEATURES.index(name)]) for name in STATE_FIELDS}
        if invalid_metadata:
            checks.append(_check(
                "observed_state_sensor_contract", "FAIL", "sensor_fault_parameters_missing_or_invalid",
                {"failure_ids": invalid_metadata},
            ))
        elif unverifiable:
            checks.append(_check(
                "observed_state_sensor_contract", "REVIEW_REQUIRED", "stochastic_sensor_draw_not_serialized",
                {"failure_ids": unverifiable},
            ))
        else:
            mismatches = {
                name: {"expected": expected[name], "actual": observed[name]}
                for name in STATE_FIELDS if abs(observed[name] - expected[name]) > max(1e-5, abs(expected[name]) * 1e-6)
            }
            checks.append(_check(
                "observed_state_sensor_contract", "PASS" if not mismatches else "FAIL",
                None if not mismatches else "observed_state_disagrees_with_serialized_sensor_contract",
                {"mismatches": mismatches, "true_state_preserved": true_state},
            ))
        previous_observed = observed
        output[record_id] = checks
    return output


def flight_contract_checks(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    actuator = actuator_contract_checks(records)
    sensor = sensor_contract_checks(records)
    return {record["record_id"]: actuator[record["record_id"]] + sensor[record["record_id"]] for record in records}
