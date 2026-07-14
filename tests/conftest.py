from __future__ import annotations

from collections.abc import Callable

import pytest

from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config
from aircraft_recovery.failures import FailureDefinition


@pytest.fixture
def aircraft_config() -> AircraftConfig:
    return load_config("configs/aircraft/simple_twin.yaml", AircraftConfig)


@pytest.fixture
def scenario_factory() -> Callable[..., ScenarioConfig]:
    base = load_config("configs/scenarios/phase1_nominal.yaml", ScenarioConfig)

    def make(
        failures: list[FailureDefinition] | None = None,
        duration_seconds: float = 2.0,
        termination: dict[str, object] | None = None,
    ) -> ScenarioConfig:
        data = base.model_dump()
        data.update({
            "scenario_id": "test_scenario",
            "duration_seconds": duration_seconds,
            "failures": [failure.model_dump() for failure in failures or []],
            "termination": termination or {},
        })
        return ScenarioConfig.model_validate(data)

    return make

