from aircraft_recovery.controllers.emergency import EmergencyMode, EmergencyStateMachine
from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config
from aircraft_recovery.simulator import SimpleAircraftSimulator


def observation_at(timestamp: float, failures: list[str]):
    aircraft = load_config("configs/aircraft/simple_twin.yaml", AircraftConfig)
    scenario = load_config("configs/scenarios/phase1_nominal.yaml", ScenarioConfig)
    observation = SimpleAircraftSimulator(aircraft, scenario).reset(1)
    return observation.model_copy(update={"timestamp_seconds": timestamp, "active_failures": failures})


def test_allowed_recovery_mode_transitions() -> None:
    machine = EmergencyStateMachine(minimum_mode_duration_seconds=2.0)
    assert machine.update(observation_at(0.0, ["engine"]), "TEST1", 10.0) == EmergencyMode.FAILURE_DETECTED
    assert machine.update(observation_at(2.0, ["engine"]), "TEST1", 10.0) == EmergencyMode.STABILIZE
    assert machine.update(observation_at(4.0, ["engine"]), "TEST1", 10.0) == EmergencyMode.DIVERT
    assert [item.new_mode for item in machine.transitions] == ["FAILURE_DETECTED", "STABILIZE", "DIVERT"]


def test_mode_hysteresis_prevents_rapid_oscillation() -> None:
    machine = EmergencyStateMachine(minimum_mode_duration_seconds=2.0)
    machine.update(observation_at(0.0, ["engine"]), None, None)
    assert machine.update(observation_at(1.99, ["engine"]), None, None) == EmergencyMode.FAILURE_DETECTED
    assert len(machine.transitions) == 1

