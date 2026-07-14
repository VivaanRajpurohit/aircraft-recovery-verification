from aircraft_recovery.common.observations import OBSERVATION_FEATURES, observation_to_vector
from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config
from aircraft_recovery.simulator import SimpleAircraftSimulator


def test_vector_order_and_shape_are_stable() -> None:
    aircraft = load_config("configs/aircraft/simple_twin.yaml", AircraftConfig)
    scenario = load_config("configs/scenarios/phase1_nominal.yaml", ScenarioConfig)
    observation = SimpleAircraftSimulator(aircraft, scenario).reset(scenario.seed)
    vector = observation_to_vector(observation)
    assert vector.shape == (len(OBSERVATION_FEATURES),)
    assert len(OBSERVATION_FEATURES) == 42
    assert vector[OBSERVATION_FEATURES.index("altitude_ft")] == 8500.0
    assert vector[OBSERVATION_FEATURES.index("airport_available")] == 1.0

