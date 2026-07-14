from collections.abc import Callable

import pytest

from aircraft_recovery.config import AircraftConfig, ScenarioConfig
from aircraft_recovery.controllers import StabilizationController
from aircraft_recovery.failures import FailureDefinition, FailureType
from aircraft_recovery.simulator import SimpleAircraftSimulator


def failure(failure_type: FailureType, **updates: object) -> FailureDefinition:
    data: dict[str, object] = {
        "failure_id": failure_type.value,
        "failure_type": failure_type,
        "affected_subsystem": "test",
        "severity": 0.5,
        "activation": "start",
        "permanent": True,
        "observable": True,
        "detected": True,
    }
    data.update(updates)
    return FailureDefinition.model_validate(data)


@pytest.mark.parametrize("failure_type", list(FailureType))
def test_each_failure_type_executes_in_simulator(
    aircraft_config: AircraftConfig,
    scenario_factory: Callable[..., ScenarioConfig],
    failure_type: FailureType,
) -> None:
    parameters = {
        "field": "airspeed_kts",
        "maximum_bias": 10.0,
        "maximum_stddev": 2.0,
        "maximum_delay_seconds": 0.2,
        "maximum_rate_per_second": 0.5,
        "maximum_wind_kts": 20.0,
    }
    scenario = scenario_factory([failure(failure_type, parameters=parameters)])
    simulator = SimpleAircraftSimulator(aircraft_config, scenario)
    observation = simulator.reset(123)
    next_observation = simulator.step(StabilizationController().act(observation))
    assert next_observation.timestamp_seconds == pytest.approx(0.1)
    assert simulator.audit_snapshot()["ground_truth_failures"][0]["failure_type"] == failure_type.value


def test_three_concurrent_failures_are_active(
    aircraft_config: AircraftConfig,
    scenario_factory: Callable[..., ScenarioConfig],
) -> None:
    items = [
        failure(FailureType.LEFT_ENGINE_FAILURE, severity=1.0),
        failure(FailureType.REDUCED_AILERON, failure_id="aileron"),
        failure(FailureType.CROSSWIND_INCREASE, failure_id="wind"),
    ]
    observation = SimpleAircraftSimulator(aircraft_config, scenario_factory(items)).reset(5)
    assert len(observation.active_failures) == 3


def test_hidden_failure_is_not_disclosed(
    aircraft_config: AircraftConfig,
    scenario_factory: Callable[..., ScenarioConfig],
) -> None:
    item = failure(
        FailureType.LEFT_ENGINE_FAILURE, severity=1.0, observable=False, detected=False
    )
    simulator = SimpleAircraftSimulator(aircraft_config, scenario_factory([item]))
    observation = simulator.reset(5)
    assert observation.active_failures == []
    assert observation.engine_health.left_engine_thrust_available == 1.0
    assert len(simulator.audit_snapshot()["ground_truth_failures"]) == 1


def test_sensor_bias_does_not_change_true_state(
    aircraft_config: AircraftConfig,
    scenario_factory: Callable[..., ScenarioConfig],
) -> None:
    item = failure(
        FailureType.AIRSPEED_BIAS,
        severity=1.0,
        observable=False,
        detected=False,
        parameters={"maximum_bias": 20.0},
    )
    simulator = SimpleAircraftSimulator(aircraft_config, scenario_factory([item]))
    observation = simulator.reset(5)
    audit = simulator.audit_snapshot()
    assert observation.aircraft_state.airspeed_kts == pytest.approx(
        audit["true_state"]["airspeed_kts"] + 20.0
    )
    assert observation.active_failures == []


def test_stuck_surface_ignores_new_commands(
    aircraft_config: AircraftConfig,
    scenario_factory: Callable[..., ScenarioConfig],
) -> None:
    item = failure(
        FailureType.STUCK_ELEVATOR,
        severity=1.0,
        parameters={"stuck_value": 0.35},
    )
    simulator = SimpleAircraftSimulator(aircraft_config, scenario_factory([item]))
    observation = simulator.reset(5)
    action = StabilizationController().act(observation)
    action.control_commands.elevator = -1.0
    simulator.step(action)
    assert simulator.audit_snapshot()["applied_control_commands"]["elevator"] == pytest.approx(0.35)


def test_actuator_delay_holds_initial_neutral_command(
    aircraft_config: AircraftConfig,
    scenario_factory: Callable[..., ScenarioConfig],
) -> None:
    item = failure(
        FailureType.ACTUATOR_DELAY,
        severity=1.0,
        parameters={"maximum_delay_seconds": 0.5},
    )
    simulator = SimpleAircraftSimulator(aircraft_config, scenario_factory([item]))
    observation = simulator.reset(5)
    action = StabilizationController().act(observation)
    action.control_commands.elevator = 1.0
    simulator.step(action)
    assert simulator.audit_snapshot()["applied_control_commands"]["elevator"] == 0.0


def test_temporary_navigation_loss_recovers(
    aircraft_config: AircraftConfig,
    scenario_factory: Callable[..., ScenarioConfig],
) -> None:
    item = failure(
        FailureType.NAVIGATION_INVALID,
        severity=1.0,
        permanent=False,
        duration_seconds=0.2,
    )
    simulator = SimpleAircraftSimulator(aircraft_config, scenario_factory([item], duration_seconds=1.0))
    observation = simulator.reset(5)
    assert not observation.navigation.data_valid
    controller = StabilizationController()
    observation = simulator.step(controller.act(observation))
    observation = simulator.step(controller.act(observation))
    assert observation.navigation.data_valid

