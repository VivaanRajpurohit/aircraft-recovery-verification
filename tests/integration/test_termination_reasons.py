from collections.abc import Callable
import math

from aircraft_recovery.config import AircraftConfig, ScenarioConfig
from aircraft_recovery.simulator import SimpleAircraftSimulator


def simulator_for(
    aircraft: AircraftConfig,
    factory: Callable[..., ScenarioConfig],
    termination: dict[str, object] | None = None,
) -> SimpleAircraftSimulator:
    simulator = SimpleAircraftSimulator(aircraft, factory(termination=termination))
    simulator.reset(1)
    return simulator


def test_precise_ground_impact_classification(aircraft_config, scenario_factory) -> None:
    simulator = simulator_for(aircraft_config, scenario_factory)
    simulator._state["altitude_ft"] = -1.0
    simulator._update_termination()
    assert simulator.termination_reason() == "ground_impact"


def test_precise_loss_of_control_classification(aircraft_config, scenario_factory) -> None:
    simulator = simulator_for(aircraft_config, scenario_factory)
    simulator._state["roll_deg"] = 130.0
    simulator._update_termination()
    assert simulator.termination_reason() == "loss_of_control"


def test_precise_stall_timeout_classification(aircraft_config, scenario_factory) -> None:
    simulator = simulator_for(
        aircraft_config, scenario_factory, {"stall_timeout_seconds": 0.1}
    )
    simulator._state["airspeed_kts"] = 50.0
    simulator._update_termination()
    assert simulator.termination_reason() == "stall_not_recovered"


def test_precise_overspeed_timeout_classification(aircraft_config, scenario_factory) -> None:
    simulator = simulator_for(
        aircraft_config, scenario_factory, {"overspeed_timeout_seconds": 0.1}
    )
    simulator._state["airspeed_kts"] = 250.0
    simulator._update_termination()
    assert simulator.termination_reason() == "overspeed_not_recovered"


def test_precise_unrecoverable_classification(aircraft_config, scenario_factory) -> None:
    simulator = simulator_for(
        aircraft_config, scenario_factory, {"force_unrecoverable_at_seconds": 0.0}
    )
    simulator._update_termination()
    assert simulator.termination_reason() == "unrecoverable_condition"


def test_precise_terrain_classification(aircraft_config, scenario_factory) -> None:
    simulator = simulator_for(aircraft_config, scenario_factory)
    simulator.scenario.environment["terrain_clearance_ft"] = 0.0
    simulator._update_termination()
    assert simulator.termination_reason() == "terrain_clearance_violation"


def test_precise_invalid_state_classification(aircraft_config, scenario_factory) -> None:
    simulator = simulator_for(aircraft_config, scenario_factory)
    simulator._state["airspeed_kts"] = -1.0
    simulator._update_termination()
    assert simulator.termination_reason() == "invalid_state"


def test_precise_numerical_instability_classification(aircraft_config, scenario_factory) -> None:
    simulator = simulator_for(aircraft_config, scenario_factory)
    simulator._state["roll_deg"] = math.nan
    simulator._update_termination()
    assert simulator.termination_reason() == "numerical_instability"
