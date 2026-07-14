"""Controller interface shared by baseline and constrained controllers."""

from typing import Protocol

from aircraft_recovery.models import ControllerInput, ControllerOutput


class Controller(Protocol):
    """Maps one validated observation to one validated command packet."""

    name: str

    def act(self, observation: ControllerInput) -> ControllerOutput: ...

