import numpy as np
import pytest

from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config
from aircraft_recovery.models import ControlCommands
from aircraft_recovery.safety import RuntimeSafetyMonitor, SafetyConfig
from aircraft_recovery.simulator import SimpleAircraftSimulator
from aircraft_recovery.verification import SolverCheck, Z3SafetySolver


@pytest.fixture
def safety_config() -> SafetyConfig:
    return load_config("configs/safety/default.yaml", SafetyConfig)


@pytest.fixture
def observation():
    aircraft = load_config("configs/aircraft/simple_twin.yaml", AircraftConfig)
    scenario = load_config("configs/scenarios/phase1_nominal.yaml", ScenarioConfig)
    return SimpleAircraftSimulator(aircraft, scenario).reset(scenario.seed)


def neutral() -> ControlCommands:
    return ControlCommands(
        elevator=0.0, aileron=0.0, rudder=0.0,
        throttle_left=0.5, throttle_right=0.5,
    )


def test_safe_action_is_accepted(safety_config, observation) -> None:
    decision = RuntimeSafetyMonitor(safety_config).decide(observation, neutral(), neutral())
    assert decision.status == "accept"
    assert not decision.fallback_activated


def test_rate_unsafe_action_is_projected(safety_config, observation) -> None:
    proposed = ControlCommands(
        elevator=1.0, aileron=1.0, rudder=1.0,
        throttle_left=1.0, throttle_right=1.0,
    )
    decision = RuntimeSafetyMonitor(safety_config).decide(observation, proposed, neutral())
    assert decision.status == "modify"
    assert decision.modification_magnitude > 0.0
    assert all(abs(getattr(decision.final_action, name)) <= 0.25 for name in ("elevator", "aileron", "rudder"))


def test_no_safe_action_activates_fallback(safety_config, observation) -> None:
    observation.aircraft_state.airspeed_kts = 98.0
    observation.flight_envelope.stall_margin_kts = 13.0
    decision = RuntimeSafetyMonitor(safety_config).decide(observation, neutral(), neutral())
    assert decision.status == "reject_fallback"
    assert decision.fallback_activated
    assert "stall_margin" in decision.violated_properties


def test_invalid_navigation_activates_fallback_without_solver(safety_config, observation) -> None:
    observation.navigation.data_valid = False
    decision = RuntimeSafetyMonitor(safety_config).decide(observation, neutral(), neutral())
    assert decision.status == "reject_fallback"
    assert decision.solver_result == "invalid"


class UnknownBackend:
    def __init__(self, timeout: bool) -> None:
        self.timeout = timeout

    def check_action(self, observation, commands, previous_commands):
        return SolverCheck(
            "unknown", 20.0, (), {"reason": "timeout" if self.timeout else "incomplete"},
            timeout=self.timeout, unknown=True,
        )


@pytest.mark.parametrize("timeout", [False, True])
def test_unknown_or_timeout_never_accepts(safety_config, observation, timeout) -> None:
    monitor = RuntimeSafetyMonitor(safety_config, UnknownBackend(timeout))
    decision = monitor.decide(observation, neutral(), neutral())
    assert decision.status == "unknown_fallback"
    assert decision.fallback_activated
    assert decision.unknown
    assert decision.timeout is timeout


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("airspeed_low", 98.0, "stall_margin"),
        ("airspeed_high", 208.0, "overspeed_margin"),
        ("roll", 61.0, "bank_angle"),
        ("pitch", 26.0, "pitch_range"),
        ("terrain", 10.0, "terrain_clearance"),
    ],
)
def test_encoded_envelope_properties_find_counterexamples(
    safety_config, observation, field, value, expected
) -> None:
    if field == "airspeed_low" or field == "airspeed_high":
        observation.aircraft_state.airspeed_kts = value
    elif field == "roll":
        observation.aircraft_state.roll_deg = value
    elif field == "pitch":
        observation.aircraft_state.pitch_deg = value
    else:
        observation.environment.terrain_clearance_ft = value
    check = Z3SafetySolver(safety_config).check_action(observation, neutral(), neutral())
    assert check.status == "unsafe"
    assert expected in check.violated_properties
    assert check.counterexample is not None


def test_formal_checked_bounds_are_ordered(safety_config) -> None:
    bounds = safety_config.checked_state_bounds
    assert bounds.minimum_airspeed_kts < bounds.maximum_airspeed_kts
    assert bounds.minimum_pitch_deg < bounds.maximum_pitch_deg
    assert bounds.minimum_load_factor_g < bounds.maximum_load_factor_g

