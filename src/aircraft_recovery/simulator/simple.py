"""Deterministic low-order simulator with Phase 2 failure injection."""

from __future__ import annotations

from collections import deque
import math
from typing import Any

import numpy as np

from aircraft_recovery.config import AircraftConfig, ScenarioConfig
from aircraft_recovery.failures import ActiveFailure, FailureEngine, FailureType
from aircraft_recovery.models import (
    AircraftState,
    Airport,
    ControlHealth,
    ControllerInput,
    ControllerOutput,
    EngineHealth,
    Environment,
    FlightEnvelope,
    Mission,
    MissionPhase,
    Navigation,
)
from aircraft_recovery.navigation.range_estimation import estimate_glide_range_nm


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _scaled_parameter(failure: ActiveFailure, name: str, default: float) -> float:
    return float(failure.parameters.get(name, default)) * failure.intensity


class SimpleAircraftSimulator:
    """Low-order simulator separating true state from sensor observations.

    This implementation is a reproducible research abstraction, not a flight
    dynamics model suitable for operational use or airworthiness conclusions.
    """

    def __init__(self, aircraft: AircraftConfig, scenario: ScenarioConfig) -> None:
        self.aircraft = aircraft
        self.scenario = scenario
        self.failure_engine = FailureEngine(scenario.failures)
        self._rng = np.random.default_rng(scenario.seed)
        self._time_seconds = 0.0
        self._state: dict[str, float] = {}
        self._reason: str | None = None
        self._current_failures: list[ActiveFailure] = []
        self._last_observation: ControllerInput | None = None
        self._frozen_values: dict[str, float] = {}
        self._stuck_commands: dict[str, float] = {}
        self._last_applied = {name: 0.0 for name in ("elevator", "aileron", "rudder")}
        self._last_applied_commands: dict[str, float] = {
            "elevator": 0.0, "aileron": 0.0, "rudder": 0.0,
            "throttle_left": 0.0, "throttle_right": 0.0,
        }
        self._control_history: deque[tuple[float, dict[str, float]]] = deque()
        self._stall_seconds = 0.0
        self._overspeed_seconds = 0.0
        self._stable_seconds = 0.0
        self._airport_distances: dict[str, float] = {}

    @property
    def time_seconds(self) -> float:
        return self._time_seconds

    def reset(self, seed: int) -> ControllerInput:
        self._rng = np.random.default_rng(seed)
        initial = self.scenario.initial_state
        self._time_seconds = 0.0
        self._reason = None
        self.failure_engine.reset()
        self._frozen_values.clear()
        self._stuck_commands.clear()
        self._control_history.clear()
        self._stall_seconds = self._overspeed_seconds = self._stable_seconds = 0.0
        self._last_applied = {name: 0.0 for name in ("elevator", "aileron", "rudder")}
        self._last_applied_commands = {name: 0.0 for name in self._last_applied_commands}
        self._airport_distances = {airport.airport_id: airport.distance_nm for airport in self.scenario.airports}
        self._state = {
            "altitude_ft": initial.altitude_ft,
            "airspeed_kts": initial.airspeed_kts,
            "pitch_deg": initial.pitch_deg,
            "roll_deg": initial.roll_deg,
            "yaw_deg": initial.yaw_deg,
            "vertical_speed_fpm": 0.0,
            "heading_deg": initial.heading_deg,
            "latitude_deg": initial.latitude_deg,
            "longitude_deg": initial.longitude_deg,
        }
        self._current_failures = self.failure_engine.evaluate(0.0, self._state)
        self._last_observation = self._observation()
        return self._last_observation

    def step(self, action: ControllerOutput) -> ControllerInput:
        if self._reason is not None:
            raise RuntimeError(f"Cannot step a terminal simulator: {self._reason}")
        dt = self.aircraft.time_step_seconds
        self._current_failures = self.failure_engine.evaluate(self._time_seconds, self._state)
        effects = self._true_effects()
        proposed = action.control_commands.model_dump()
        commands = self._apply_command_failures(proposed, effects)
        state = self._state

        state["pitch_deg"] = _clamp(
            state["pitch_deg"] + commands["elevator"] * effects["elevator_effectiveness"]
            * self.aircraft.pitch_rate_deg_s * dt,
            -89.0, 89.0,
        )
        state["roll_deg"] = _clamp(
            state["roll_deg"] + (
                commands["aileron"] * effects["aileron_effectiveness"]
                + 0.12 * effects["thrust_asymmetry"]
            ) * self.aircraft.roll_rate_deg_s * dt,
            -179.0, 179.0,
        )
        state["yaw_deg"] = _clamp(
            state["yaw_deg"] + (
                commands["rudder"] * effects["rudder_effectiveness"]
                + 0.2 * effects["thrust_asymmetry"]
                + 0.003 * effects["crosswind_kts"]
            ) * self.aircraft.yaw_rate_deg_s * dt,
            -180.0, 180.0,
        )
        left_thrust = commands["throttle_left"] * effects["left_thrust_available"]
        right_thrust = commands["throttle_right"] * effects["right_thrust_available"]
        average_throttle = (left_thrust + right_thrust) / 2.0
        drag = self.aircraft.drag_coefficient_per_s * (
            state["airspeed_kts"] - self.aircraft.cruise_speed_kts
        )
        acceleration = self.aircraft.max_thrust_accel_kts_s * (average_throttle - 0.5) - drag
        state["airspeed_kts"] = _clamp(state["airspeed_kts"] + acceleration * dt, 0.0, 599.0)

        turbulence = effects["turbulence"]
        if turbulence > 0.0:
            state["pitch_deg"] = _clamp(state["pitch_deg"] + self._rng.normal(0.0, turbulence * 0.2), -89.0, 89.0)
            state["roll_deg"] = _clamp(state["roll_deg"] + self._rng.normal(0.0, turbulence * 0.5), -179.0, 179.0)
        pitch_rad = math.radians(state["pitch_deg"])
        state["vertical_speed_fpm"] = state["airspeed_kts"] * 101.269 * math.sin(pitch_rad)
        state["altitude_ft"] = _clamp(
            state["altitude_ft"] + state["vertical_speed_fpm"] * dt / 60.0,
            -999.0, 59999.0,
        )
        turn_rate = 9.81 * math.tan(math.radians(state["roll_deg"])) / max(
            state["airspeed_kts"] * 0.514444, 10.0
        ) * 180.0 / math.pi
        state["heading_deg"] = (state["heading_deg"] + turn_rate * dt) % 360.0
        ground_speed = max(0.0, state["airspeed_kts"] - effects["headwind_kts"])
        distance_nm = ground_speed * dt / 3600.0
        heading_rad = math.radians(state["heading_deg"])
        state["latitude_deg"] = _clamp(
            state["latitude_deg"] + distance_nm * math.cos(heading_rad) / 60.0, -90.0, 90.0
        )
        lon_scale = max(math.cos(math.radians(state["latitude_deg"])), 0.1)
        state["longitude_deg"] = _clamp(
            state["longitude_deg"] + distance_nm * math.sin(heading_rad) / (60.0 * lon_scale),
            -180.0, 180.0,
        )
        for airport in self.scenario.airports:
            alignment = math.cos(math.radians((airport.bearing_deg - state["heading_deg"] + 180.0) % 360.0 - 180.0))
            self._airport_distances[airport.airport_id] = max(
                0.0, self._airport_distances[airport.airport_id] - distance_nm * alignment
            )
        self._time_seconds += dt
        self._current_failures = self.failure_engine.evaluate(self._time_seconds, self._state)
        self._update_termination()
        self._last_observation = self._observation()
        return self._last_observation

    def is_terminal(self) -> bool:
        return self._reason is not None

    def termination_reason(self) -> str | None:
        return self._reason

    def audit_snapshot(self) -> dict[str, Any]:
        """Return ground truth for logging; never passed to the controller."""
        return {
            "true_state": dict(self._state),
            "observed_state": self._last_observation.aircraft_state.model_dump() if self._last_observation else None,
            "ground_truth_failures": [failure.model_dump(mode="json") for failure in self._current_failures],
            "controller_known_failures": list(self._last_observation.active_failures) if self._last_observation else [],
            "expected_recoverability": self.scenario.expected_recoverability,
            "airport_distances_nm": dict(self._airport_distances),
            "applied_control_commands": dict(self._last_applied_commands),
        }

    def _true_effects(self) -> dict[str, float]:
        effects = {
            "left_thrust_available": 1.0, "right_thrust_available": 1.0,
            "elevator_effectiveness": 1.0, "aileron_effectiveness": 1.0,
            "rudder_effectiveness": 1.0, "actuator_delay_seconds": 0.0,
            "throttle_delay_seconds": 0.0, "rate_limit_per_second": 100.0,
            "turbulence": float(self.scenario.environment.get("turbulence_intensity", 0.0)),
            "crosswind_kts": 0.0, "headwind_kts": 0.0,
            "thrust_asymmetry": 0.0,
        }
        for failure in self._current_failures:
            intensity, kind = failure.intensity, failure.failure_type
            if kind == FailureType.LEFT_ENGINE_FAILURE:
                effects["left_thrust_available"] *= 1.0 - intensity
            elif kind == FailureType.RIGHT_ENGINE_FAILURE:
                effects["right_thrust_available"] *= 1.0 - intensity
            elif kind == FailureType.PARTIAL_THRUST_LOSS:
                side = str(failure.parameters.get("engine", "both"))
                if side in {"left", "both"}:
                    effects["left_thrust_available"] *= 1.0 - intensity
                if side in {"right", "both"}:
                    effects["right_thrust_available"] *= 1.0 - intensity
            elif kind == FailureType.ASYMMETRIC_THRUST:
                effects["right_thrust_available"] *= 1.0 - intensity
            elif kind == FailureType.DELAYED_THROTTLE_RESPONSE:
                effects["throttle_delay_seconds"] = max(effects["throttle_delay_seconds"], _scaled_parameter(failure, "maximum_delay_seconds", 1.0))
            elif kind == FailureType.REDUCED_ELEVATOR:
                effects["elevator_effectiveness"] *= 1.0 - intensity
            elif kind == FailureType.REDUCED_AILERON:
                effects["aileron_effectiveness"] *= 1.0 - intensity
            elif kind == FailureType.REDUCED_RUDDER:
                effects["rudder_effectiveness"] *= 1.0 - intensity
            elif kind == FailureType.ACTUATOR_DELAY:
                effects["actuator_delay_seconds"] = max(effects["actuator_delay_seconds"], _scaled_parameter(failure, "maximum_delay_seconds", 1.0))
            elif kind == FailureType.CONTROL_RATE_LIMIT:
                effects["rate_limit_per_second"] = min(effects["rate_limit_per_second"], float(failure.parameters.get("maximum_rate_per_second", 0.5)))
            elif kind == FailureType.TURBULENCE_INCREASE:
                effects["turbulence"] = _clamp(effects["turbulence"] + intensity, 0.0, 1.0)
            elif kind == FailureType.CROSSWIND_INCREASE:
                effects["crosswind_kts"] += _scaled_parameter(failure, "maximum_wind_kts", 40.0)
            elif kind == FailureType.HEADWIND_CHANGE:
                effects["headwind_kts"] += _scaled_parameter(failure, "maximum_wind_kts", 40.0)
        effects["thrust_asymmetry"] = effects["left_thrust_available"] - effects["right_thrust_available"]
        return effects

    def _apply_command_failures(self, proposed: dict[str, float], effects: dict[str, float]) -> dict[str, float]:
        commands = dict(proposed)
        delay = max(effects["actuator_delay_seconds"], effects["throttle_delay_seconds"])
        self._control_history.append((self._time_seconds, dict(commands)))
        if delay > 0.0:
            eligible = [item for time, item in self._control_history if time <= self._time_seconds - delay]
            delayed = eligible[-1] if eligible else {name: 0.0 for name in commands}
            for name in commands:
                if name.startswith("throttle") and effects["throttle_delay_seconds"] <= 0.0:
                    continue
                if not name.startswith("throttle") and effects["actuator_delay_seconds"] <= 0.0:
                    continue
                commands[name] = delayed[name]
        for failure in self._current_failures:
            stuck_field = {
                FailureType.STUCK_ELEVATOR: "elevator",
                FailureType.STUCK_AILERON: "aileron",
                FailureType.STUCK_RUDDER: "rudder",
            }.get(failure.failure_type)
            if stuck_field:
                self._stuck_commands.setdefault(failure.failure_id, float(failure.parameters.get("stuck_value", commands[stuck_field])))
                commands[stuck_field] = self._stuck_commands[failure.failure_id]
        max_delta = effects["rate_limit_per_second"] * self.aircraft.time_step_seconds
        for name in ("elevator", "aileron", "rudder"):
            commands[name] = _clamp(commands[name], self._last_applied[name] - max_delta, self._last_applied[name] + max_delta)
            self._last_applied[name] = commands[name]
        self._last_applied_commands = dict(commands)
        return commands

    def _visible_health(self) -> tuple[ControlHealth, EngineHealth]:
        visible = [failure for failure in self._current_failures if failure.observable and failure.detected]
        saved = self._current_failures
        self._current_failures = visible
        effects = self._true_effects()
        self._current_failures = saved
        stuck = {failure.failure_type for failure in visible}
        left = effects["left_thrust_available"]
        right = effects["right_thrust_available"]
        return ControlHealth(
            elevator_effectiveness=effects["elevator_effectiveness"],
            aileron_effectiveness=effects["aileron_effectiveness"],
            rudder_effectiveness=effects["rudder_effectiveness"],
            elevator_stuck=FailureType.STUCK_ELEVATOR in stuck,
            aileron_stuck=FailureType.STUCK_AILERON in stuck,
            rudder_stuck=FailureType.STUCK_RUDDER in stuck,
            actuator_delay_ms=effects["actuator_delay_seconds"] * 1000.0,
        ), EngineHealth(
            left_engine_thrust_available=left,
            right_engine_thrust_available=right,
            left_engine_failed=left <= 0.001,
            right_engine_failed=right <= 0.001,
            asymmetric_thrust=abs(left - right) > 0.05,
        )

    def _observed_state(self) -> tuple[dict[str, float], bool]:
        observed = dict(self._state)
        navigation_valid = True
        for failure in self._current_failures:
            intensity, kind = failure.intensity, failure.failure_type
            if kind == FailureType.AIRSPEED_BIAS:
                observed["airspeed_kts"] += _scaled_parameter(failure, "maximum_bias", 30.0)
            elif kind == FailureType.ALTITUDE_BIAS:
                observed["altitude_ft"] += _scaled_parameter(failure, "maximum_bias", 1000.0)
            elif kind == FailureType.HEADING_BIAS:
                observed["heading_deg"] = (observed["heading_deg"] + _scaled_parameter(failure, "maximum_bias", 30.0)) % 360.0
            elif kind == FailureType.FROZEN_SENSOR:
                field = str(failure.parameters.get("field", "airspeed_kts"))
                if field in observed:
                    self._frozen_values.setdefault(failure.failure_id, observed[field])
                    observed[field] = self._frozen_values[failure.failure_id]
            elif kind == FailureType.GAUSSIAN_NOISE:
                field = str(failure.parameters.get("field", "airspeed_kts"))
                if field in observed:
                    observed[field] += self._rng.normal(0.0, _scaled_parameter(failure, "maximum_stddev", 5.0))
            elif kind == FailureType.TEMPORARY_MISSING_SENSOR:
                field = str(failure.parameters.get("field", "airspeed_kts"))
                if self._last_observation and hasattr(self._last_observation.aircraft_state, field):
                    observed[field] = float(getattr(self._last_observation.aircraft_state, field))
                if field in {"latitude_deg", "longitude_deg", "heading_deg"}:
                    navigation_valid = False
            elif kind == FailureType.NAVIGATION_INVALID:
                navigation_valid = False
        observed["altitude_ft"] = _clamp(observed["altitude_ft"], -1000.0, 60000.0)
        observed["airspeed_kts"] = _clamp(observed["airspeed_kts"], 0.0, 600.0)
        observed["heading_deg"] %= 360.0
        return observed, navigation_valid

    def _observation(self) -> ControllerInput:
        observed, navigation_valid = self._observed_state()
        control_health, engine_health = self._visible_health()
        environment_data = dict(self.scenario.environment)
        for failure in self._current_failures:
            if failure.failure_type == FailureType.REDUCED_VISIBILITY:
                environment_data["visibility_nm"] = max(0.0, environment_data["visibility_nm"] * (1.0 - failure.intensity))
            elif failure.failure_type == FailureType.TURBULENCE_INCREASE:
                environment_data["turbulence_intensity"] = _clamp(environment_data["turbulence_intensity"] + failure.intensity, 0.0, 1.0)
            elif failure.failure_type in {FailureType.CROSSWIND_INCREASE, FailureType.HEADWIND_CHANGE}:
                environment_data["wind_speed_kts"] = _clamp(environment_data["wind_speed_kts"] + _scaled_parameter(failure, "maximum_wind_kts", 40.0), 0.0, 250.0)
        airports = [Airport(
            airport_id=item.airport_id,
            distance_nm=self._airport_distances[item.airport_id],
            bearing_deg=item.bearing_deg,
            runway_heading_deg=item.runway_heading_deg,
            runway_length_ft=item.runway_length_ft,
            terrain_clearance_ft=item.terrain_clearance_ft,
            within_glide_range=self._airport_distances[item.airport_id] <= estimate_glide_range_nm(observed["altitude_ft"], item.elevation_ft),
        ) for item in self.scenario.airports]
        aoa = _clamp(observed["pitch_deg"] - observed["vertical_speed_fpm"] / 1000.0, -20.0, 40.0)
        load_factor = _clamp(1.0 / max(math.cos(math.radians(observed["roll_deg"])), 0.12), -3.0, 9.0)
        known = [failure.failure_id for failure in self._current_failures if failure.observable and failure.detected]
        emergency = bool(known) or self.scenario.diversion_requested
        return ControllerInput(
            timestamp_seconds=self._time_seconds,
            aircraft_state=AircraftState(**{key: observed[key] for key in (
                "altitude_ft", "airspeed_kts", "pitch_deg", "roll_deg", "yaw_deg",
                "vertical_speed_fpm", "heading_deg"
            )}),
            flight_envelope=FlightEnvelope(
                angle_of_attack_deg=aoa, load_factor_g=load_factor,
                stall_margin_kts=observed["airspeed_kts"] - self.aircraft.stall_speed_kts,
                overspeed_margin_kts=self.aircraft.overspeed_kts - observed["airspeed_kts"],
            ),
            control_health=control_health,
            engine_health=engine_health,
            navigation=Navigation(
                latitude_deg=observed["latitude_deg"], longitude_deg=observed["longitude_deg"],
                nearest_airports=airports, data_valid=navigation_valid,
            ),
            environment=Environment.model_validate(environment_data),
            mission=Mission(
                phase=MissionPhase.DIVERSION if emergency else MissionPhase.CRUISE,
                emergency_declared=emergency, original_destination="TEST2",
            ),
            active_failures=known,
        )

    def _update_termination(self) -> None:
        state, dt = self._state, self.aircraft.time_step_seconds
        termination = self.scenario.termination
        values = list(state.values())
        if not all(math.isfinite(value) for value in values):
            self._reason = "numerical_instability"
        elif not (
            -1000.0 <= state["altitude_ft"] <= 60000.0
            and 0.0 <= state["airspeed_kts"] <= 600.0
            and -90.0 <= state["pitch_deg"] <= 90.0
            and -180.0 <= state["roll_deg"] <= 180.0
        ):
            self._reason = "invalid_state"
        elif state["altitude_ft"] <= 0.0:
            self._reason = "ground_impact"
        elif float(self.scenario.environment.get("terrain_clearance_ft", 1.0)) <= 0.0:
            self._reason = "terrain_clearance_violation"
        elif abs(state["roll_deg"]) >= float(termination.get("loss_of_control_roll_deg", 120.0)):
            self._reason = "loss_of_control"
        elif self._time_seconds >= float(termination.get("force_unrecoverable_at_seconds", math.inf)):
            self._reason = "unrecoverable_condition"
        else:
            self._stall_seconds = self._stall_seconds + dt if state["airspeed_kts"] < self.aircraft.stall_speed_kts else 0.0
            self._overspeed_seconds = self._overspeed_seconds + dt if state["airspeed_kts"] > self.aircraft.overspeed_kts else 0.0
            stable = (
                state["airspeed_kts"] >= self.aircraft.stall_speed_kts + 15.0
                and abs(state["roll_deg"]) < 10.0 and abs(state["pitch_deg"]) < 8.0
            )
            self._stable_seconds = self._stable_seconds + dt if stable and self._current_failures else 0.0
            if self._stall_seconds >= float(termination.get("stall_timeout_seconds", 10.0)):
                self._reason = "stall_not_recovered"
            elif self._overspeed_seconds >= float(termination.get("overspeed_timeout_seconds", 5.0)):
                self._reason = "overspeed_not_recovered"
            elif termination.get("stable_recovery_enabled", False) and self._stable_seconds >= float(termination.get("stable_duration_seconds", 8.0)):
                self._reason = "stable_recovery_achieved"
            elif termination.get("diversion_reached_enabled", False) and self._airport_distances and min(self._airport_distances.values()) <= float(termination.get("diversion_distance_nm", 0.5)):
                self._reason = "diversion_target_reached"
            elif termination.get("approach_completion_enabled", False) and state["altitude_ft"] <= float(termination.get("approach_completion_altitude_ft", 500.0)):
                self._reason = "approach_completed"
            elif self._time_seconds >= self.scenario.duration_seconds:
                self._reason = "duration_complete"
