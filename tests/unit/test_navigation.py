import pytest

from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config
from aircraft_recovery.controllers import StabilizationController
from aircraft_recovery.navigation import DiversionPlanner
from aircraft_recovery.navigation.range_estimation import (
    estimate_glide_range_nm,
    turn_altitude_loss_ft,
    wind_components_kts,
)
from aircraft_recovery.simulator import SimpleAircraftSimulator


def test_glide_range_and_turn_loss_are_monotonic() -> None:
    assert estimate_glide_range_nm(10000.0) > estimate_glide_range_nm(5000.0) > 0.0
    assert turn_altitude_loss_ft(180.0, 130.0) > turn_altitude_loss_ft(30.0, 130.0)


def test_wind_components() -> None:
    headwind, crosswind = wind_components_kts(20.0, 0.0, 0.0)
    assert headwind == pytest.approx(20.0)
    assert crosswind == pytest.approx(0.0)
    _, crosswind = wind_components_kts(20.0, 90.0, 0.0)
    assert crosswind == pytest.approx(20.0)


def test_airport_ranking_is_not_nearest_only() -> None:
    aircraft = load_config("configs/aircraft/simple_twin.yaml", AircraftConfig)
    scenario = load_config("configs/scenarios/phase2_nominal.yaml", ScenarioConfig)
    observation = SimpleAircraftSimulator(aircraft, scenario).reset(scenario.seed)
    plan = DiversionPlanner().plan(observation, scenario.airports)
    assert scenario.airports[0].distance_nm < scenario.airports[1].distance_nm
    assert plan.selected_airport == "TEST2"


def test_airport_rejection_reasons_are_explained() -> None:
    aircraft = load_config("configs/aircraft/simple_twin.yaml", AircraftConfig)
    scenario = load_config("configs/scenarios/phase2_nominal.yaml", ScenarioConfig)
    scenario.airports[0].available = False
    scenario.airports[0].runway_length_ft = 1000.0
    observation = SimpleAircraftSimulator(aircraft, scenario).reset(scenario.seed)
    plan = DiversionPlanner().plan(observation, scenario.airports)
    candidate = next(item for item in plan.candidates if item.airport_id == "TEST1")
    assert "airport_unavailable" in candidate.rejection_reasons
    assert "runway_too_short" in candidate.rejection_reasons


def test_invalid_navigation_rejects_all_airports() -> None:
    aircraft = load_config("configs/aircraft/simple_twin.yaml", AircraftConfig)
    scenario = load_config("configs/scenarios/phase2_navigation_engine.yaml", ScenarioConfig)
    simulator = SimpleAircraftSimulator(aircraft, scenario)
    observation = simulator.reset(scenario.seed)
    controller = StabilizationController()
    while observation.timestamp_seconds < 2.0:
        observation = simulator.step(controller.act(observation))
    plan = DiversionPlanner().plan(observation, scenario.airports)
    assert not plan.estimated_reachable
    assert all("navigation_data_invalid" in item.rejection_reasons for item in plan.candidates)
