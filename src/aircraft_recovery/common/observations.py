"""Stable neural observation-vector encoding."""

from __future__ import annotations

import numpy as np

from aircraft_recovery.models import ControllerInput, MissionPhase

MISSION_PHASES = list(MissionPhase)
OBSERVATION_FEATURES = (
    "timestamp_seconds", "altitude_ft", "airspeed_kts", "pitch_deg", "roll_deg",
    "yaw_deg", "vertical_speed_fpm", "heading_deg", "angle_of_attack_deg",
    "load_factor_g", "stall_margin_kts", "overspeed_margin_kts",
    "elevator_effectiveness", "aileron_effectiveness", "rudder_effectiveness",
    "elevator_stuck", "aileron_stuck", "rudder_stuck", "actuator_delay_ms",
    "left_engine_thrust_available", "right_engine_thrust_available",
    "left_engine_failed", "right_engine_failed", "asymmetric_thrust",
    "latitude_deg", "longitude_deg", "navigation_data_valid", "wind_speed_kts",
    "wind_direction_deg", "turbulence_intensity", "visibility_nm",
    "terrain_clearance_ft", "mission_phase_index", "emergency_declared",
    "active_failure_count", "airport_available", "nearest_airport_distance_nm",
    "nearest_airport_bearing_deg", "nearest_runway_heading_deg",
    "nearest_runway_length_ft", "nearest_airport_terrain_clearance_ft",
    "nearest_airport_within_glide_range",
)


def observation_to_vector(observation: ControllerInput) -> np.ndarray:
    """Encode an observation in the documented, stable feature order."""
    s, e = observation.aircraft_state, observation.flight_envelope
    c, h = observation.control_health, observation.engine_health
    n, w, m = observation.navigation, observation.environment, observation.mission
    airport = n.nearest_airports[0] if n.nearest_airports else None
    values = [
        observation.timestamp_seconds, s.altitude_ft, s.airspeed_kts, s.pitch_deg,
        s.roll_deg, s.yaw_deg, s.vertical_speed_fpm, s.heading_deg,
        e.angle_of_attack_deg, e.load_factor_g, e.stall_margin_kts,
        e.overspeed_margin_kts, c.elevator_effectiveness, c.aileron_effectiveness,
        c.rudder_effectiveness, float(c.elevator_stuck), float(c.aileron_stuck),
        float(c.rudder_stuck), c.actuator_delay_ms, h.left_engine_thrust_available,
        h.right_engine_thrust_available, float(h.left_engine_failed),
        float(h.right_engine_failed), float(h.asymmetric_thrust), n.latitude_deg,
        n.longitude_deg, float(n.data_valid), w.wind_speed_kts, w.wind_direction_deg,
        w.turbulence_intensity, w.visibility_nm, w.terrain_clearance_ft,
        float(MISSION_PHASES.index(m.phase)), float(m.emergency_declared),
        float(len(observation.active_failures)), float(airport is not None),
        airport.distance_nm if airport else 0.0, airport.bearing_deg if airport else 0.0,
        airport.runway_heading_deg if airport else 0.0,
        airport.runway_length_ft if airport else 0.0,
        airport.terrain_clearance_ft if airport else 0.0,
        float(airport.within_glide_range) if airport else 0.0,
    ]
    return np.asarray(values, dtype=np.float32)

