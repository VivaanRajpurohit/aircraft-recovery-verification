from pathlib import Path

import pytest

from aircraft_recovery.config import AircraftConfig, load_config


def test_load_aircraft_config() -> None:
    config = load_config(Path("configs/aircraft/simple_twin.yaml"), AircraftConfig)
    assert config.time_step_seconds == 0.1
    assert config.controller_frequency_hz == 10.0


def test_missing_config_has_clear_error() -> None:
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config("does-not-exist.yaml", AircraftConfig)

