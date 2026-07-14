from pathlib import Path

from aircraft_recovery.verification.plant_calibration import (
    ACTION_NAMES, STATE_NAMES, PlantDatasetConfig, assert_split_isolation,
    generate_dataset, load_records,
)


def _config(tmp_path: Path, split: str, seed: int) -> PlantDatasetConfig:
    return PlantDatasetConfig(
        protocol_version="test-v1", split=split, seed=seed, one_step_samples=6,
        trace_count=1, trace_length_steps=3, pre_roll_steps=2,
        output_directory=str(tmp_path / split), protocol_timestamp_utc="2026-07-14T00:00:00Z",
        allow_dirty_tree=True,
        state_ranges={"airspeed_kts": (120, 150), "pitch_deg": (-5, 5), "bank_deg": (-15, 15), "terrain_clearance_ft": (1000, 3000)},
        failure_ranges={"right_thrust_availability": (0.95, 1), "aileron_effectiveness": (0.5, 0.7), "actuator_delay_seconds": (0.1, 0.1)},
        disturbance_ranges={"crosswind_kts": (-10, 10), "turbulence_intensity": (0, 0.05)},
        sensor_error_ranges={"airspeed_kts": (-1, 1), "bank_deg": (-0.5, 0.5), "terrain_clearance_ft": (-5, 5)},
    )


def test_generation_is_deterministic_schema_valid_and_atomic(tmp_path) -> None:
    config = _config(tmp_path, "calibration", 101)
    first = generate_dataset(config)
    content = (Path(config.output_directory) / "transitions.jsonl").read_bytes()
    second = generate_dataset(config)
    assert content == (Path(config.output_directory) / "transitions.jsonl").read_bytes()
    assert first["transitions_sha256"] == second["transitions_sha256"]
    assert not list(Path(config.output_directory).glob("*.tmp-*"))
    records = load_records(config.output_directory)
    assert len(records) == 9
    assert tuple(records[0].source_state) == STATE_NAMES
    assert tuple(records[0].executed_action) == ACTION_NAMES
    assert all(record.failure_parameters["left_thrust_availability"] == 0 for record in records)
    assert all(record.failure_parameters["aileron_effectiveness"] < 1 for record in records)
    assert any(record.executed_action != record.action_before_degradation for record in records)


def test_calibration_and_validation_are_strictly_isolated(tmp_path) -> None:
    calibration = _config(tmp_path, "calibration", 101)
    validation = _config(tmp_path, "validation", 202)
    calibration_manifest = generate_dataset(calibration)
    validation_manifest = generate_dataset(validation)
    checks = assert_split_isolation(calibration.output_directory, validation.output_directory)
    assert all(checks.values())
    assert calibration_manifest["manifest_sha256"] != validation_manifest["manifest_sha256"]
    assert calibration_manifest["sample_ids_sha256"] != validation_manifest["sample_ids_sha256"]
