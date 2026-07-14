"""Conservative simulator-only diversion range abstractions."""

from __future__ import annotations

import math


def estimate_glide_range_nm(
    altitude_ft: float,
    airport_elevation_ft: float = 0.0,
    glide_ratio: float = 9.0,
    reserve_altitude_ft: float = 1000.0,
) -> float:
    """Estimate still-air range from usable altitude and constant glide ratio."""
    usable_ft = max(0.0, altitude_ft - airport_elevation_ft - reserve_altitude_ft)
    return usable_ft * glide_ratio / 6076.12


def estimate_powered_range_nm(
    airspeed_kts: float,
    engine_availability: float,
    endurance_minutes: float = 30.0,
    reserve_fraction: float = 0.2,
) -> float:
    """Estimate range with constant speed/endurance and conservative authority scaling."""
    usable_hours = endurance_minutes / 60.0 * max(0.0, 1.0 - reserve_fraction)
    authority_factor = 0.35 + 0.65 * max(0.0, min(1.0, engine_availability))
    return max(0.0, airspeed_kts) * usable_hours * authority_factor


def turn_altitude_loss_ft(
    turn_angle_deg: float,
    airspeed_kts: float,
    descent_rate_fpm: float = 700.0,
    bank_angle_deg: float = 25.0,
) -> float:
    """Estimate altitude lost during a coordinated constant-bank turn."""
    speed_mps = max(airspeed_kts * 0.514444, 10.0)
    turn_rate_rad_s = 9.80665 * math.tan(math.radians(bank_angle_deg)) / speed_mps
    duration_minutes = math.radians(abs(turn_angle_deg)) / max(turn_rate_rad_s, 1e-6) / 60.0
    return max(0.0, descent_rate_fpm) * duration_minutes


def wind_components_kts(
    wind_speed_kts: float,
    wind_from_deg: float,
    course_deg: float,
) -> tuple[float, float]:
    """Return signed headwind and absolute crosswind for a desired course."""
    relative = math.radians((wind_from_deg - course_deg + 180.0) % 360.0 - 180.0)
    headwind = wind_speed_kts * math.cos(relative)
    crosswind = abs(wind_speed_kts * math.sin(relative))
    return headwind, crosswind


def wind_corrected_track_deg(
    desired_course_deg: float,
    true_airspeed_kts: float,
    wind_speed_kts: float,
    wind_from_deg: float,
) -> float:
    """Estimate required heading for course using a bounded wind-correction angle."""
    _, crosswind = wind_components_kts(wind_speed_kts, wind_from_deg, desired_course_deg)
    ratio = min(0.99, crosswind / max(true_airspeed_kts, 1.0))
    relative = (wind_from_deg - desired_course_deg + 360.0) % 360.0
    sign = -1.0 if relative < 180.0 else 1.0
    return (desired_course_deg + sign * math.degrees(math.asin(ratio))) % 360.0

