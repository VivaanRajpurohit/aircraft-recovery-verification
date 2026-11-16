from __future__ import annotations

import copy

import pytest

from aircraft_recovery.common.observations import OBSERVATION_FEATURES
from aircraft_recovery.data_quality.contracts import actuator_contract_checks, sensor_contract_checks


def _record() -> dict:
    state = {
        "altitude_ft": 5000.0, "airspeed_kts": 100.0, "pitch_deg": 2.0,
        "roll_deg": -3.0, "yaw_deg": 1.0, "vertical_speed_fpm": 100.0,
        "heading_deg": 250.0, "latitude_deg": 37.0, "longitude_deg": -120.0,
    }
    observation = [0.0] * 42
    for name, value in state.items():
        observation[OBSERVATION_FEATURES.index(name)] = value
    return {
        "record_id": "flight:step:000000",
        "simulation_timestamp_seconds": 0.0,
        "aircraft_state_before": state,
        "ordered_observation_42": observation,
        "expert_proposed_controls": [0.2, -0.3, 0.1, 0.6, 0.7],
        "executed_controls": [0.2, -0.3, 0.1, 0.6, 0.7],
        "actuator_effectiveness": {"elevator": 1.0, "aileron": 0.5, "rudder": 1.0},
        "actuator_delay_seconds": 0.0,
        "ground_truth_failures": [{
            "failure_id": "aileron-loss", "failure_type": "reduced_aileron_effectiveness",
            "affected_subsystem": "aileron", "severity": 0.5, "intensity": 0.5,
            "observable": True, "detected": True, "activation_time_seconds": 0.0,
            "parameters": {},
        }],
        "sensor_error_failures": [],
    }


def _statuses(checks: list[dict]) -> set[str]:
    return {item["status"] for item in checks}


def _actuator_checks(record: dict) -> list[dict]:
    return actuator_contract_checks([record])[record["record_id"]]


def _sensor_checks(record: dict) -> list[dict]:
    return sensor_contract_checks([record])[record["record_id"]]


def test_correct_degraded_actuator_command_is_verified() -> None:
    # Effectiveness changes dynamics, not the stored applied command.
    assert _statuses(_actuator_checks(_record())) == {"PASS"}


@pytest.mark.parametrize("mutation", [
    lambda row: row["executed_controls"].__setitem__(1, 0.3),  # wrong sign
    lambda row: row["executed_controls"].__setitem__(0, 0.1),  # wrong actuator
    lambda row: row["actuator_effectiveness"].__setitem__("aileron", 0.6),
    lambda row: row["actuator_effectiveness"].__setitem__("aileron", 1.2),
    lambda row: row.pop("actuator_effectiveness"),
])
def test_bad_degradation_examples_fail(mutation) -> None:
    row = _record()
    mutation(row)
    assert "FAIL" in _statuses(_actuator_checks(row))


def test_serialized_delay_history_is_reconstructed() -> None:
    first = _record()
    first["ground_truth_failures"] = []
    first["actuator_effectiveness"]["aileron"] = 1.0
    second = copy.deepcopy(first)
    second["record_id"] = "flight:step:000001"
    second["simulation_timestamp_seconds"] = 0.1
    second["expert_proposed_controls"] = [-0.2, 0.3, -0.1, 0.4, 0.3]
    second["executed_controls"] = first["expert_proposed_controls"][:3] + second["expert_proposed_controls"][3:]
    second["actuator_delay_seconds"] = 0.1
    second["ground_truth_failures"] = [{
        "failure_id": "delay", "failure_type": "actuator_delay",
        "severity": 1.0, "intensity": 1.0, "observable": True, "detected": True,
        "parameters": {"maximum_delay_seconds": 0.1},
    }]
    checks = actuator_contract_checks([first, second])
    assert _statuses(checks[second["record_id"]]) == {"PASS"}


def _biased_record() -> dict:
    row = _record()
    failure = {
        "failure_id": "airspeed-bias", "failure_type": "airspeed_bias",
        "affected_subsystem": "airspeed_sensor", "severity": 0.5, "intensity": 0.5,
        "observable": False, "detected": False, "activation_time_seconds": 0.0,
        "parameters": {"maximum_bias": 20.0},
    }
    row["ground_truth_failures"] = [failure]
    row["sensor_error_failures"] = [copy.deepcopy(failure)]
    row["ordered_observation_42"][OBSERVATION_FEATURES.index("airspeed_kts")] = 110.0
    row["actuator_effectiveness"]["aileron"] = 1.0
    return row


def test_correct_sensor_bias_preserves_true_state_and_passes() -> None:
    row = _biased_record()
    assert row["aircraft_state_before"]["airspeed_kts"] == 100.0
    assert _statuses(_sensor_checks(row)) == {"PASS"}


@pytest.mark.parametrize("mutation", [
    lambda row: row["ordered_observation_42"].__setitem__(OBSERVATION_FEATURES.index("airspeed_kts"), 109.0),
    lambda row: row["ordered_observation_42"].__setitem__(OBSERVATION_FEATURES.index("pitch_deg"), 12.0),
    lambda row: row["aircraft_state_before"].__setitem__("airspeed_kts", 110.0),
    lambda row: row["ground_truth_failures"][0].__setitem__("parameters", {}),
    lambda row: row.pop("sensor_error_failures"),
])
def test_bad_sensor_examples_fail(mutation) -> None:
    row = _biased_record()
    mutation(row)
    assert "FAIL" in _statuses(_sensor_checks(row))


def test_unexplained_state_discrepancy_fails() -> None:
    row = _record()
    row["ground_truth_failures"] = []
    row["actuator_effectiveness"]["aileron"] = 1.0
    row["ordered_observation_42"][OBSERVATION_FEATURES.index("airspeed_kts")] = 101.0
    assert "FAIL" in _statuses(_sensor_checks(row))
