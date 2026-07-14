import json
from aircraft_recovery.experiments.batch import SuccessDefinitions, _extract_metrics

def test_duration_completion_is_not_automatic_stabilization(tmp_path):
    run=tmp_path/"run"; run.mkdir()
    summary={"termination_reason":"duration_complete","minimum_stall_margin_kts":2.0,"maximum_angle_of_attack_deg":20.0,"maximum_absolute_roll_deg":50.0,"minimum_terrain_clearance_ft":1000.0,"controller_latency_ms_mean":1.0}
    (run/"summary.json").write_text(json.dumps(summary))
    row={"next_observation":{"timestamp_seconds":.1,"aircraft_state":{"roll_deg":50,"pitch_deg":20},"flight_envelope":{"stall_margin_kts":2,"overspeed_margin_kts":20,"load_factor_g":1}},"final_action":{"navigation_commands":{"selected_airport":None},"control_commands":{"elevator":0,"aileron":0,"rudder":0,"throttle_left":.5,"throttle_right":.5}},"ground_truth_after":{"airport_distances_nm":{}}}
    (run/"history.jsonl").write_text(json.dumps(row)+"\n")
    d=SuccessDefinitions(minimum_stall_margin_kts=15,minimum_overspeed_margin_kts=15,maximum_absolute_bank_deg=30,maximum_absolute_pitch_deg=15,stabilization_dwell_seconds=5,diversion_arrival_distance_nm=1,diversion_heading_tolerance_deg=15)
    result=_extract_metrics(run,d)
    assert not result["stabilization_success"] and not result["recovery_success"]
    assert result["outcome_classification"] == "duration_complete_without_stabilization"

def test_terminal_diversion_label_does_not_bypass_heading_tolerance(tmp_path):
    run=tmp_path/"run2"; run.mkdir()
    summary={"termination_reason":"diversion_target_reached","minimum_stall_margin_kts":20.0,"maximum_angle_of_attack_deg":2.0,"maximum_absolute_roll_deg":2.0,"minimum_terrain_clearance_ft":1000.0,"controller_latency_ms_mean":1.0}
    (run/"summary.json").write_text(json.dumps(summary))
    row={"next_observation":{"timestamp_seconds":.1,"aircraft_state":{"roll_deg":0,"pitch_deg":0,"heading_deg":180},"flight_envelope":{"stall_margin_kts":20,"overspeed_margin_kts":20,"load_factor_g":1},"navigation":{"nearest_airports":[{"airport_id":"A","bearing_deg":0}]}},"final_action":{"navigation_commands":{"selected_airport":"A"},"control_commands":{"elevator":0,"aileron":0,"rudder":0,"throttle_left":.5,"throttle_right":.5}},"ground_truth_after":{"airport_distances_nm":{"A":.5}}}
    (run/"history.jsonl").write_text(json.dumps(row)+"\n")
    d=SuccessDefinitions(minimum_stall_margin_kts=15,minimum_overspeed_margin_kts=15,maximum_absolute_bank_deg=30,maximum_absolute_pitch_deg=15,stabilization_dwell_seconds=5,diversion_arrival_distance_nm=1,diversion_heading_tolerance_deg=15)
    assert not _extract_metrics(run,d)["diversion_success"]
