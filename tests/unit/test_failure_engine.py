import pytest

from aircraft_recovery.failures import FailureDefinition, FailureEngine, FailureType


def definition(**updates: object) -> FailureDefinition:
    data: dict[str, object] = {
        "failure_id": "failure",
        "failure_type": "left_engine_failure",
        "affected_subsystem": "engine",
        "severity": 0.8,
        "activation": "start",
        "permanent": True,
    }
    data.update(updates)
    return FailureDefinition.model_validate(data)


@pytest.mark.parametrize("failure_type", list(FailureType))
def test_every_failure_type_has_runtime_ground_truth(failure_type: FailureType) -> None:
    item = definition(failure_type=failure_type, failure_id=failure_type.value)
    active = FailureEngine([item]).evaluate(0.0, {"altitude_ft": 1000.0})
    assert active[0].failure_type == failure_type
    assert active[0].intensity == pytest.approx(0.8)


def test_timed_activation() -> None:
    engine = FailureEngine([definition(activation="time", activation_time_seconds=5.0)])
    assert engine.evaluate(4.9, {}) == []
    assert engine.evaluate(5.0, {})[0].activation_time_seconds == 5.0


def test_temporary_failure_deactivates() -> None:
    engine = FailureEngine([definition(permanent=False, duration_seconds=2.0)])
    assert engine.evaluate(0.0, {})
    assert engine.evaluate(1.9, {})
    assert engine.evaluate(2.0, {}) == []


def test_failure_ramps_linearly() -> None:
    engine = FailureEngine([definition(severity=1.0, ramp_duration_seconds=4.0)])
    assert engine.evaluate(0.0, {})[0].intensity == 0.0
    assert engine.evaluate(2.0, {})[0].intensity == pytest.approx(0.5)
    assert engine.evaluate(5.0, {})[0].intensity == 1.0


def test_state_condition_activation() -> None:
    item = definition(
        activation="condition",
        condition={"field": "altitude_ft", "operator": "lt", "value": 500.0},
    )
    engine = FailureEngine([item])
    assert not engine.evaluate(0.0, {"altitude_ft": 600.0})
    assert engine.evaluate(0.1, {"altitude_ft": 400.0})


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        FailureEngine([definition(), definition()])

