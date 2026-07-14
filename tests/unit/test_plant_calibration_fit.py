from pathlib import Path
import numpy as np

from aircraft_recovery.verification.plant_calibration import PlantDatasetConfig,generate_dataset
from aircraft_recovery.verification.plant_calibration_fit import fit_formal_plant,load_formal_plant


def test_fitted_coefficients_and_residuals_are_finite_and_deterministic(tmp_path):
    configuration=PlantDatasetConfig(
        protocol_version="fit-test-v1",split="calibration",seed=303,one_step_samples=20,
        trace_count=0,trace_length_steps=3,pre_roll_steps=2,
        output_directory=str(tmp_path/"calibration"),protocol_timestamp_utc="2026-07-14T00:00:00Z",allow_dirty_tree=True,
        state_ranges={"airspeed_kts":(115,160),"pitch_deg":(-8,8),"bank_deg":(-25,25),"terrain_clearance_ft":(1000,4000)},
        failure_ranges={"right_thrust_availability":(.95,1),"aileron_effectiveness":(.4,.8),"actuator_delay_seconds":(0,.1)},
        disturbance_ranges={"crosswind_kts":(-15,15),"turbulence_intensity":(0,.1)},
        sensor_error_ranges={"airspeed_kts":(-1,1),"bank_deg":(-.5,.5),"terrain_clearance_ft":(-10,10)},
    )
    generate_dataset(configuration)
    config_path=tmp_path/"config.yaml"
    import yaml
    config_path.write_text(yaml.safe_dump(configuration.model_dump(mode="json"),sort_keys=False),encoding="utf-8")
    first=fit_formal_plant(configuration.output_directory,config_path,tmp_path/"model-a.json")
    second=fit_formal_plant(configuration.output_directory,config_path,tmp_path/"model-b.json")
    assert first.coefficients==second.coefficients
    assert first.residual_bounds==second.residual_bounds
    assert first.fit_metadata["validation_used_for_fit"] is False
    assert all(np.all(np.isfinite(values)) for values in first.coefficients.values())
    assert all(lower<=upper for lower,upper in first.residual_bounds.values())
    assert load_formal_plant(tmp_path/"model-a.json")==first
