"""Unverified Phase 3 neural baseline controller."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from aircraft_recovery.common.observations import observation_to_vector
from aircraft_recovery.controllers.stabilization import StabilizationController
from aircraft_recovery.data.dataset import MODE_NAMES
from aircraft_recovery.data.hashing import file_hash
from aircraft_recovery.data.preprocessing import ObservationPreprocessor, PreprocessingStatistics
from aircraft_recovery.models import (
    ControlCommands,
    ControllerInput,
    ControllerOutput,
    NavigationCommands,
    SafetyDecision,
    StabilizationTargets,
)
from aircraft_recovery.training.checkpointing import load_checkpoint
from aircraft_recovery.training.models import TARGET_SCALES, ImitationPolicyNetwork, PolicyArchitectureConfig
from aircraft_recovery.training.trainer import resolve_device


class NeuralBaselineController:
    """Checkpoint-backed baseline with no Phase 4 runtime safety filtering."""

    name = "phase3_neural_baseline"

    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        self.device = resolve_device(device)
        self.checkpoint_path = Path(checkpoint_path)
        checkpoint = load_checkpoint(self.checkpoint_path, map_location=self.device)
        architecture = PolicyArchitectureConfig.model_validate(checkpoint["architecture_configuration"])
        self.model = ImitationPolicyNetwork(architecture).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.preprocessor = ObservationPreprocessor(
            PreprocessingStatistics.from_dict(checkpoint["normalization_statistics"])
        )
        self.model_checkpoint_identifier = file_hash(self.checkpoint_path)
        self.checkpoint_metadata = {
            "path": str(self.checkpoint_path),
            "identifier": self.model_checkpoint_identifier,
            "dataset_identifier": checkpoint["dataset_identifier"],
            "split_manifest_identifier": checkpoint["split_manifest_identifier"],
            "parameter_count": checkpoint["parameter_count"],
            "epoch": checkpoint["epoch"],
        }
        self.execution_fallback = StabilizationController()
        self.runtime_execution_fallback_count = 0
        self.last_inference_latency_ms = 0.0

    def reset(self) -> None:
        self.runtime_execution_fallback_count = 0
        self.last_inference_latency_ms = 0.0

    def act(self, observation: ControllerInput) -> ControllerOutput:
        started = perf_counter()
        try:
            vector = observation_to_vector(observation)
            processed = self.preprocessor.transform(vector)
            tensor = torch.from_numpy(processed.astype(np.float32)).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                prediction = self.model(tensor)
            controls = prediction["controls"][0].detach().cpu().numpy()
            probabilities = torch.softmax(prediction["mode_logits"][0], dim=-1)
            mode_index = int(torch.argmax(probabilities).cpu())
            mode = MODE_NAMES[mode_index]
            confidence = float(probabilities[mode_index].cpu())
            if "targets_normalized" in prediction:
                normalized_targets = prediction["targets_normalized"][0].detach().cpu().numpy()
                target_values = normalized_targets * np.asarray(TARGET_SCALES, dtype=np.float32)
            else:
                state = observation.aircraft_state
                target_values = np.asarray([0.0, 0.0, 135.0, state.altitude_ft], dtype=np.float32)
            navigation_active = mode in {"divert", "approach", "land"} and observation.navigation.data_valid
            airport = observation.navigation.nearest_airports[0] if navigation_active and observation.navigation.nearest_airports else None
            approach_phase = "landing" if mode == "land" else "approach" if mode == "approach" else "diversion" if mode == "divert" else "none"
            self.last_inference_latency_ms = (perf_counter() - started) * 1000.0
            return ControllerOutput(
                timestamp_seconds=observation.timestamp_seconds,
                control_commands=ControlCommands(
                    elevator=float(controls[0]), aileron=float(controls[1]),
                    rudder=float(controls[2]), throttle_left=float(controls[3]),
                    throttle_right=float(controls[4]),
                ),
                stabilization_targets=StabilizationTargets(
                    target_pitch_deg=float(target_values[0]),
                    target_roll_deg=float(target_values[1]),
                    target_airspeed_kts=float(target_values[2]),
                    target_altitude_ft=float(target_values[3]),
                ),
                navigation_commands=NavigationCommands(
                    target_heading_deg=airport.bearing_deg if airport else observation.aircraft_state.heading_deg,
                    selected_airport=airport.airport_id if airport else None,
                    selected_runway_heading_deg=airport.runway_heading_deg if airport else None,
                    approach_phase=approach_phase,
                ),
                emergency_mode=mode,
                safety_decision=SafetyDecision(
                    status="not_checked",
                    ai_action_accepted=True,
                    fallback_activated=False,
                    violated_constraints=[],
                    explanation="Unverified Phase 3 neural baseline action; no Phase 4 safety monitor was applied.",
                ),
                confidence=confidence,
            )
        except Exception as error:
            self.last_inference_latency_ms = (perf_counter() - started) * 1000.0
            self.runtime_execution_fallback_count += 1
            output = self.execution_fallback.act(observation)
            output.safety_decision = SafetyDecision(
                status="fallback", ai_action_accepted=False, fallback_activated=True,
                violated_constraints=["neural_runtime_failure"],
                explanation=(
                    "Deterministic execution fallback used because neural inference failed; "
                    f"this is not a Phase 4 safety intervention ({type(error).__name__})."
                ),
            )
            output.confidence = 0.0
            return output

