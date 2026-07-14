import pytest
from pydantic import ValidationError

from aircraft_recovery.models import ControlCommands, EngineHealth


def test_control_commands_enforce_ranges() -> None:
    with pytest.raises(ValidationError):
        ControlCommands(elevator=1.01, aileron=0, rudder=0, throttle_left=0.5, throttle_right=0.5)
    with pytest.raises(ValidationError):
        ControlCommands(elevator=0, aileron=0, rudder=0, throttle_left=-0.1, throttle_right=0.5)


def test_failed_engine_must_have_zero_available_thrust() -> None:
    with pytest.raises(ValidationError, match="zero left thrust"):
        EngineHealth(left_engine_failed=True, left_engine_thrust_available=0.2)

