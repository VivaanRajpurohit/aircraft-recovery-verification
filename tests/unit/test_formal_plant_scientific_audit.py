import inspect
import json
from pathlib import Path

import pytest
import yaml

from aircraft_recovery.simulator import SimpleAircraftSimulator
from aircraft_recovery.verification import formal_plant as formal_plant_module
from aircraft_recovery.verification import plant_calibration_fit as fit_module
from aircraft_recovery.verification import plant_validation as validation_module
from aircraft_recovery.verification.plant_calibration import (
    PlantDatasetConfig,
    generate_dataset,
    load_records,
)
from aircraft_recovery.verification.plant_calibration_fit import fit_formal_plant
from aircraft_recovery.verification.plant_validation import validate_formal_plant


def _config(tmp_path: Path, split: str, seed: int, samples: int, traces: int) -> PlantDatasetConfig:
    return PlantDatasetConfig(
        protocol_version="scientific-audit-v1",
        split=split,
        seed=seed,
        one_step_samples=samples,
        trace_count=traces,
        trace_length_steps=5,
        pre_roll_steps=2,
        output_directory=str(tmp_path / split),
        protocol_timestamp_utc="2026-07-22T00:00:00Z",
        allow_dirty_tree=True,
        state_ranges={
            "airspeed_kts": (120, 150),
            "pitch_deg": (-5, 5),
            "bank_deg": (-15, 15),
            "terrain_clearance_ft": (1000, 3000),
        },
        failure_ranges={
            "right_thrust_availability": (0.95, 1),
            "aileron_effectiveness": (0.5, 0.8),
            "actuator_delay_seconds": (0, 0.1),
        },
        disturbance_ranges={"crosswind_kts": (-10, 10), "turbulence_intensity": (0, 0.05)},
        sensor_error_ranges={
            "airspeed_kts": (-1, 1),
            "bank_deg": (-0.5, 0.5),
            "terrain_clearance_ft": (-5, 5),
        },
    )


@pytest.fixture()
def audit_case(tmp_path: Path) -> dict[str, Path]:
    calibration = _config(tmp_path, "calibration", 1201, 30, 1)
    validation = _config(tmp_path, "validation", 2201, 15, 2)
    generate_dataset(calibration)
    generate_dataset(validation)
    config_path = tmp_path / "calibration.yaml"
    config_path.write_text(
        yaml.safe_dump(calibration.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    model_path = tmp_path / "model.json"
    fit_formal_plant(calibration.output_directory, config_path, model_path)
    return {
        "calibration": Path(calibration.output_directory),
        "validation": Path(validation.output_directory),
        "config": config_path,
        "model": model_path,
    }


def test_fitting_loads_only_calibration_records(audit_case, monkeypatch) -> None:
    loaded_paths = []
    original = fit_module.load_records

    def recording_loader(path):
        loaded_paths.append(Path(path).resolve())
        return original(path)

    monkeypatch.setattr(fit_module, "load_records", recording_loader)
    fit_formal_plant(
        audit_case["calibration"], audit_case["config"], audit_case["model"].with_name("refit.json")
    )
    assert loaded_paths == [audit_case["calibration"].resolve()]
    assert audit_case["validation"].resolve() not in loaded_paths


def test_validation_uses_stored_next_state_without_calling_simulator_step(
    audit_case, monkeypatch, tmp_path
) -> None:
    def forbidden_step(*_args, **_kwargs):
        raise AssertionError("validation must not regenerate numerical next states")

    monkeypatch.setattr(SimpleAircraftSimulator, "step", forbidden_step)
    original = validation_module.load_records
    validation_path = audit_case["validation"].resolve()

    def altered_stored_record(path):
        records = original(path)
        if Path(path).resolve() == validation_path:
            records = [item.model_copy(deep=True) for item in records]
            record = next(item for item in records if item.record_kind == "one_step")
            record.next_state["airspeed_kts"] += 2.0
        return records

    monkeypatch.setattr(validation_module, "load_records", altered_stored_record)
    report = validate_formal_plant(
        audit_case["model"],
        audit_case["calibration"],
        audit_case["validation"],
        tmp_path / "stored-state-report.json",
        horizons=(5,),
    )
    assert report["one_step"]["metrics"]["airspeed_kts"]["maximum_absolute_error"] > 1.9
    assert report["one_step"]["containment_failures"]
    assert "aircraft_recovery.simulator" not in inspect.getsource(formal_plant_module)


@pytest.mark.parametrize(
    ("perturbation", "state", "minimum_error"),
    [
        ("bank_response", "bank_deg", 0.01),
        ("throttle_response", "airspeed_kts", 0.01),
        ("vertical_speed_scale", "vertical_speed_fpm", 1.0),
        ("clearance_integration", "terrain_clearance_ft", 0.1),
        ("remove_residuals", "airspeed_kts", 0.0),
    ],
)
def test_corrupted_models_are_detected(
    audit_case, tmp_path, perturbation, state, minimum_error
) -> None:
    payload = json.loads(audit_case["model"].read_text(encoding="utf-8"))
    if perturbation == "bank_response":
        payload["coefficients"]["bank_deg"][2] += 0.5
    elif perturbation == "throttle_response":
        payload["coefficients"]["airspeed_kts"][3] += 0.5
    elif perturbation == "vertical_speed_scale":
        payload["coefficients"]["vertical_speed_fpm"][0] += 1.0
    elif perturbation == "clearance_integration":
        payload["coefficients"]["terrain_clearance_ft"][2] += 0.001
    elif perturbation == "remove_residuals":
        payload["residual_bounds"] = {name: [0.0, 0.0] for name in payload["residual_bounds"]}
    corrupted = tmp_path / f"{perturbation}.json"
    corrupted.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_formal_plant(
        corrupted,
        audit_case["calibration"],
        audit_case["validation"],
        tmp_path / f"{perturbation}-report.json",
        horizons=(5,),
    )
    assert report["one_step"]["metrics"][state]["maximum_absolute_error"] > minimum_error
    assert report["one_step"]["containment_failures"], perturbation


def test_manifest_tampering_is_rejected(audit_case) -> None:
    transitions = audit_case["validation"] / "transitions.jsonl"
    content = transitions.read_text(encoding="utf-8")
    transitions.write_text(
        content.replace("validation-sample-", "tampered---sample-", 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="dataset integrity failed"):
        load_records(audit_case["validation"])


def test_report_classifies_fixed_input_rollout_and_initial_box(audit_case, tmp_path) -> None:
    report = validate_formal_plant(
        audit_case["model"],
        audit_case["calibration"],
        audit_case["validation"],
        tmp_path / "classification-report.json",
        horizons=(5,),
    )
    classification = report["scientific_classification"]
    assert classification["closed_loop_controller_validation"] is False
    assert classification["rollout_kind"] == (
        "interval propagation under fixed recorded executed actions and realized disturbances"
    )
    assert classification["initial_interval"] == {"kind": "narrow numerical box", "radius": 1e-9}
