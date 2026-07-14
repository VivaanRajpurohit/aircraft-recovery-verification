"""Explicit unit conversions used by simulator adapters."""

FEET_PER_METER = 3.280839895
KNOTS_PER_MPS = 1.943844492


def feet_to_meters(feet: float) -> float:
    return feet / FEET_PER_METER


def meters_to_feet(meters: float) -> float:
    return meters * FEET_PER_METER


def knots_to_mps(knots: float) -> float:
    return knots / KNOTS_PER_MPS


def mps_to_knots(mps: float) -> float:
    return mps * KNOTS_PER_MPS

