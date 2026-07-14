"""Bounded verification backends and structured reports."""


def __getattr__(name):
    """Load Z3 exports lazily so independent backends do not create import cycles."""
    if name in {"SolverCheck", "Z3SafetySolver"}:
        from aircraft_recovery.verification import z3_model

        return getattr(z3_model, name)
    raise AttributeError(name)


def run_formal_verification(*args, **kwargs):
    """Load the report generator lazily to avoid controller import cycles."""
    from aircraft_recovery.verification.formal_verifier import run_formal_verification as run

    return run(*args, **kwargs)

__all__ = ["SolverCheck", "Z3SafetySolver", "run_formal_verification"]
