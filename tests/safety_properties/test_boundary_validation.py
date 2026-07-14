"""Phase 1 boundary invariants; these are tests, not formal proofs."""

from aircraft_recovery.models import ControlCommands


def test_exact_command_boundaries_are_valid() -> None:
    command = ControlCommands(
        elevator=-1.0, aileron=1.0, rudder=0.0,
        throttle_left=0.0, throttle_right=1.0,
    )
    assert command.elevator == -1.0
    assert command.throttle_right == 1.0

