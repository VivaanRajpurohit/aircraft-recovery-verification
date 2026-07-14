import json
from pathlib import Path
import yaml
from aircraft_recovery.verification.plant_calibration import PlantDatasetConfig,generate_dataset
from aircraft_recovery.verification.plant_calibration_fit import fit_formal_plant
from aircraft_recovery.verification.plant_validation import validate_formal_plant


def _config(tmp_path,split,seed,count,traces):
    return PlantDatasetConfig(protocol_version="validation-test-v1",split=split,seed=seed,one_step_samples=count,trace_count=traces,trace_length_steps=5,pre_roll_steps=2,output_directory=str(tmp_path/split),protocol_timestamp_utc="2026-07-14T00:00:00Z",allow_dirty_tree=True,state_ranges={"airspeed_kts":(120,150),"pitch_deg":(-5,5),"bank_deg":(-15,15),"terrain_clearance_ft":(1000,3000)},failure_ranges={"right_thrust_availability":(.95,1),"aileron_effectiveness":(.5,.8),"actuator_delay_seconds":(0,.1)},disturbance_ranges={"crosswind_kts":(-10,10),"turbulence_intensity":(0,.05)},sensor_error_ranges={"airspeed_kts":(-1,1),"bank_deg":(-.5,.5),"terrain_clearance_ft":(-5,5)})


def test_held_out_validation_reports_one_step_and_rollout_containment(tmp_path):
    calibration=_config(tmp_path,"calibration",501,30,1); validation=_config(tmp_path,"validation",601,15,2)
    generate_dataset(calibration); generate_dataset(validation)
    configuration=tmp_path/"calibration.yaml"; configuration.write_text(yaml.safe_dump(calibration.model_dump(mode="json"),sort_keys=False),encoding="utf-8")
    model_path=tmp_path/"model.json"; fit_formal_plant(calibration.output_directory,configuration,model_path,numerical_tolerance=1e-7)
    output=tmp_path/"validation-report.json"; report=validate_formal_plant(model_path,calibration.output_directory,validation.output_directory,output,horizons=(5,))
    assert report["split_isolation"]=={"sample_ids_disjoint":True,"scenario_ids_disjoint":True,"seeds_disjoint":True}
    assert report["one_step"]["total"]==15
    assert set(report["one_step"]["metrics"])=={"airspeed_kts","pitch_deg","bank_deg","vertical_speed_fpm","terrain_clearance_ft"}
    assert report["rollout"]["trace_count"]==2
    assert len(report["rollout"]["containment_percent_by_step"])==5
    assert json.loads(output.read_text())["report_sha256"]==report["report_sha256"]
