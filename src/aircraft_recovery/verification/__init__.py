"""Bounded Z3 models and structured verification reports."""

from aircraft_recovery.verification.z3_model import SolverCheck, Z3SafetySolver
def run_formal_verification(*args, **kwargs):
    """Load the report generator lazily to avoid controller import cycles."""
    from aircraft_recovery.verification.formal_verifier import run_formal_verification as run

    return run(*args, **kwargs)

__all__ = ["SolverCheck", "Z3SafetySolver", "run_formal_verification"]
