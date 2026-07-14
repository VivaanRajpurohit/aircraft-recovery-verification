"""Explainable scoring of synthetic emergency diversion airports."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from aircraft_recovery.config import SyntheticAirportConfig
from aircraft_recovery.models import ControllerInput
from aircraft_recovery.navigation.range_estimation import (
    estimate_glide_range_nm,
    estimate_powered_range_nm,
    turn_altitude_loss_ft,
    wind_components_kts,
    wind_corrected_track_deg,
)


@dataclass(frozen=True)
class CandidateScore:
    airport_id: str
    score: float
    reachable: bool
    feasibility: str
    estimated_range_nm: float
    crosswind_kts: float
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class DiversionPlan:
    selected_airport: str | None
    selected_runway_heading_deg: float | None
    target_heading_deg: float
    estimated_reachable: bool
    feasibility: str
    confidence: float
    candidates: tuple[CandidateScore, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "candidates": [asdict(candidate) for candidate in self.candidates],
        }


class DiversionPlanner:
    """Rank all airports using configurable, explainable simulator factors."""

    def __init__(
        self,
        minimum_runway_ft: float = 3500.0,
        maximum_crosswind_kts: float = 30.0,
        glide_ratio: float = 9.0,
        powered_endurance_minutes: float = 30.0,
    ) -> None:
        self.minimum_runway_ft = minimum_runway_ft
        self.maximum_crosswind_kts = maximum_crosswind_kts
        self.glide_ratio = glide_ratio
        self.powered_endurance_minutes = powered_endurance_minutes

    def plan(
        self,
        observation: ControllerInput,
        airports: list[SyntheticAirportConfig],
    ) -> DiversionPlan:
        state = observation.aircraft_state
        engines = observation.engine_health
        controls = observation.control_health
        environment = observation.environment
        observed_distances = {
            airport.airport_id: airport.distance_nm for airport in observation.navigation.nearest_airports
        }
        engine_availability = (
            engines.left_engine_thrust_available + engines.right_engine_thrust_available
        ) / 2.0
        control_authority = min(
            controls.elevator_effectiveness,
            controls.aileron_effectiveness,
            controls.rudder_effectiveness,
        )
        scores: list[CandidateScore] = []
        for airport in airports:
            distance_nm = observed_distances.get(airport.airport_id, airport.distance_nm)
            glide_range = estimate_glide_range_nm(
                state.altitude_ft, airport.elevation_ft, self.glide_ratio
            )
            powered_range = estimate_powered_range_nm(
                state.airspeed_kts, engine_availability, self.powered_endurance_minutes
            )
            estimated_range = max(glide_range, powered_range if engine_availability > 0.05 else 0.0)
            turn_angle = abs((airport.bearing_deg - state.heading_deg + 180.0) % 360.0 - 180.0)
            turn_loss = turn_altitude_loss_ft(turn_angle, state.airspeed_kts)
            _, crosswind = wind_components_kts(
                environment.wind_speed_kts,
                environment.wind_direction_deg,
                airport.runway_heading_deg,
            )
            reasons: list[str] = []
            if not observation.navigation.data_valid:
                reasons.append("navigation_data_invalid")
            if not airport.available:
                reasons.append("airport_unavailable")
            if not airport.surface_suitable:
                reasons.append("surface_unsuitable")
            if airport.runway_length_ft < self.minimum_runway_ft:
                reasons.append("runway_too_short")
            if crosswind > self.maximum_crosswind_kts:
                reasons.append("crosswind_limit_exceeded")
            if control_authority < airport.minimum_control_authority:
                reasons.append("insufficient_control_authority")
            if distance_nm > estimated_range:
                reasons.append("outside_estimated_range")
            if turn_loss + airport.elevation_ft + 500.0 >= state.altitude_ft:
                reasons.append("insufficient_turn_altitude")
            reachable = not reasons
            distance_score = max(0.0, 1.0 - distance_nm / max(estimated_range, 1.0))
            runway_score = min(1.0, airport.runway_length_ft / 8000.0)
            wind_score = max(0.0, 1.0 - crosswind / max(self.maximum_crosswind_kts, 1.0))
            alignment_score = max(0.0, 1.0 - turn_angle / 180.0)
            score = (
                0.30 * distance_score + 0.20 * runway_score + 0.15 * wind_score
                + 0.15 * alignment_score + 0.10 * control_authority
                + 0.10 * (1.0 - airport.terrain_penalty)
            )
            if reasons:
                score -= 0.2 * len(reasons)
            feasibility = "feasible" if reachable and score >= 0.55 else "marginal" if reachable else "infeasible"
            scores.append(CandidateScore(
                airport.airport_id, score, reachable, feasibility, estimated_range,
                crosswind, tuple(reasons),
            ))
        scores.sort(key=lambda candidate: candidate.score, reverse=True)
        selectable = [candidate for candidate in scores if candidate.reachable]
        selected = selectable[0] if selectable else None
        airport = next((item for item in airports if selected and item.airport_id == selected.airport_id), None)
        target = wind_corrected_track_deg(
            airport.bearing_deg if airport else state.heading_deg,
            state.airspeed_kts,
            environment.wind_speed_kts,
            environment.wind_direction_deg,
        )
        return DiversionPlan(
            selected_airport=selected.airport_id if selected else None,
            selected_runway_heading_deg=airport.runway_heading_deg if airport else None,
            target_heading_deg=target,
            estimated_reachable=selected is not None,
            feasibility=selected.feasibility if selected else "infeasible",
            confidence=max(0.0, min(1.0, selected.score if selected else 0.0)),
            candidates=tuple(scores),
        )
