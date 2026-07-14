"""Neural policy wrapped by the bounded runtime monitor and expert fallback."""

from __future__ import annotations

from pathlib import Path

from aircraft_recovery.config import SyntheticAirportConfig
from aircraft_recovery.controllers.fallback import DeterministicFallbackController, FallbackConfig
from aircraft_recovery.controllers.neural import NeuralBaselineController
from aircraft_recovery.models import ControllerInput, ControllerOutput, SafetyDecision
from aircraft_recovery.safety import RuntimeSafetyMonitor, SafetyConfig


class MonitoredNeuralController:
    """Compose neural proposal, bounded monitor, projection, and fallback."""

    name = "phase4_monitored_neural"

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str,
        fallback_config: FallbackConfig,
        airports: list[SyntheticAirportConfig],
        safety_config: SafetyConfig,
        monitor: RuntimeSafetyMonitor | None = None,
    ) -> None:
        self.neural = NeuralBaselineController(checkpoint_path, device)
        self.fallback = DeterministicFallbackController(fallback_config, airports)
        self.monitor = monitor or RuntimeSafetyMonitor(safety_config)
        self.model_checkpoint_identifier = self.neural.model_checkpoint_identifier
        self.checkpoint_metadata = self.neural.checkpoint_metadata
        self.last_proposed_action: ControllerOutput | None = None
        self.last_monitor_decision = None
        self.selected_airport: str | None = None
        self.counts = {"accept": 0, "modify": 0, "reject_fallback": 0, "unknown_fallback": 0}

    def reset(self) -> None:
        self.neural.reset()
        self.fallback.reset()
        self.monitor.reset()
        self.last_proposed_action = None
        self.last_monitor_decision = None
        self.selected_airport = None
        self.counts = {name: 0 for name in self.counts}

    def act(self, observation: ControllerInput) -> ControllerOutput:
        proposed = self.neural.act(observation)
        fallback = self.fallback.act(observation)
        decision = self.monitor.decide(
            observation, proposed.control_commands, fallback.control_commands
        )
        self.last_proposed_action = proposed
        self.last_monitor_decision = decision
        self.counts[decision.status] += 1
        if decision.fallback_activated:
            final = fallback.model_copy(deep=True)
            final.safety_decision = SafetyDecision(
                status="fallback", ai_action_accepted=False, fallback_activated=True,
                violated_constraints=list(decision.violated_properties),
                explanation=decision.reason,
            )
        else:
            final = proposed.model_copy(deep=True)
            final.control_commands = decision.final_action
            final.safety_decision = SafetyDecision(
                status="accepted" if decision.status == "accept" else "modified",
                ai_action_accepted=decision.status == "accept",
                fallback_activated=False,
                violated_constraints=list(decision.violated_properties),
                explanation=decision.reason,
            )
        self.selected_airport = final.navigation_commands.selected_airport
        return final

    def monitor_statistics(self) -> dict[str, int]:
        return dict(self.counts)

