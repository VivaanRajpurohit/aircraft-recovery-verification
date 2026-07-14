"""Replaceable simulator interface."""

from __future__ import annotations

from typing import Protocol

from aircraft_recovery.models import ControllerInput, ControllerOutput


class AircraftSimulator(Protocol):
    """Interface implemented by simplified and future JSBSim adapters."""

    @property
    def time_seconds(self) -> float: ...

    def reset(self, seed: int) -> ControllerInput: ...

    def step(self, action: ControllerOutput) -> ControllerInput: ...

    def is_terminal(self) -> bool: ...

    def termination_reason(self) -> str | None: ...

