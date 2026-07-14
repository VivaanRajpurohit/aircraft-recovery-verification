"""Read-only diversion diagnostics for original and controlled simulator runs."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
from typing import Any

from aircraft_recovery.config import SyntheticAirportConfig
from aircraft_recovery.controllers.fallback import FallbackConfig
from aircraft_recovery.models import ControllerInput
from aircraft_recovery.navigation import DiversionPlanner

def heading_error_deg(target: float, current: float) -> float:
    return abs((target - current + 180.0) % 360.0 - 180.0)

def analyze_diversion_run(run_directory: str | Path, arrival_nm: float = 1.0, heading_tolerance_deg: float = 15.0) -> dict[str, Any]:
    run=Path(run_directory)
    rows=[json.loads(line) for line in (run/"history.jsonl").read_text(encoding="utf-8").splitlines()]
    summary=json.loads((run/"summary.json").read_text(encoding="utf-8"))
    metadata=json.loads((run/"metadata.json").read_text(encoding="utf-8"))
    scenario=metadata["scenario_configuration"]
    selections=[r["final_action"]["navigation_commands"]["selected_airport"] for r in rows]
    selected=next((value for value in selections if value),None)
    airports={a["airport_id"]:a for a in scenario["airports"]}
    observed=ControllerInput.model_validate(rows[0]["observation"])
    planner=DiversionPlanner()
    plan=planner.plan(observed,[SyntheticAirportConfig.model_validate(a) for a in scenario["airports"]])
    feasible=any(c.reachable for c in plan.candidates)
    distances=[]; errors=[]
    if selected:
        bearing=airports[selected]["bearing_deg"]
        for row in rows:
            distances.append(float(row["ground_truth_after"]["airport_distances_nm"][selected]))
            errors.append(heading_error_deg(bearing,float(row["next_observation"]["aircraft_state"]["heading_deg"])))
        initial_distance=float(rows[0]["ground_truth_before"]["airport_distances_nm"][selected])
    else:
        initial_distance=None
    modes=sorted({r["final_action"]["emergency_mode"] for r in rows})
    transitions=[item["new_mode"] for item in summary.get("mode_transitions",[])]
    internal_modes=sorted(set(transitions))
    diversion_mode=bool(set(modes)&{"divert","approach","land"} or set(internal_modes)&{"DIVERT","APPROACH","EMERGENCY_LANDING","GLIDE"})
    approach=bool(set(modes)&{"approach","land"} or set(internal_modes)&{"APPROACH","EMERGENCY_LANDING"})
    landing=bool("land" in modes or "EMERGENCY_LANDING" in internal_modes)
    nav_valid=[r["next_observation"]["navigation"]["data_valid"] for r in rows]
    unstable=any(r["next_observation"]["flight_envelope"]["stall_margin_kts"]<12 or abs(r["next_observation"]["aircraft_state"]["roll_deg"])>45 or abs(r["next_observation"]["aircraft_state"]["pitch_deg"])>20 for r in rows)
    final_distance=distances[-1] if distances else None; minimum_distance=min(distances) if distances else None
    final_error=errors[-1] if errors else None; minimum_error=min(errors) if errors else None
    distance_ok=final_distance is not None and final_distance<=arrival_nm
    heading_ok=final_error is not None and final_error<=heading_tolerance_deg
    strict=bool(selected and distance_ok and heading_ok)
    termination=summary["termination_reason"]
    if strict: category="successful"
    elif scenario.get("expected_recoverability")=="unrecoverable" or termination=="unrecoverable_condition": category="scenario_declared_unrecoverable"
    elif not all(nav_valid) and not selected: category="missing_navigation_data"
    elif not feasible: category="no_feasible_airport"
    elif not selected and not diversion_mode: category="diversion_mode_never_entered"
    elif not selected: category="no_airport_selected"
    elif unstable and termination not in {"duration_complete","diversion_target_reached"}: category="aircraft_became_unstable"
    elif not diversion_mode: category="diversion_mode_never_entered"
    elif initial_distance is not None and not distance_ok and (initial_distance-arrival_nm) > max(r["next_observation"]["aircraft_state"]["airspeed_kts"] for r in rows)*summary["simulated_duration_seconds"]/3600.0: category="insufficient_simulation_duration"
    elif distance_ok and heading_ok and not approach: category="approach_mode_not_reached"
    elif not distance_ok and not heading_ok: category="both_distance_and_heading_failed"
    elif not distance_ok: category="distance_threshold_not_reached"
    else: category="heading_tolerance_not_reached"
    return {
        "run_directory":str(run),"controller":run.name.rsplit("_",1)[-1],"scenario_id":scenario["scenario_id"],
        "selected_airport":selected,"airport_selected":selected is not None,"airport_feasible_at_start":feasible,
        "no_airport_considered_feasible":not feasible,"moved_closer":bool(distances and distances[-1]<initial_distance),
        "initial_distance_nm":initial_distance,"final_distance_nm":final_distance,"minimum_distance_nm":minimum_distance,
        "final_heading_error_deg":final_error,"minimum_heading_error_deg":minimum_error,
        "emergency_modes":modes,"internal_state_machine_modes":internal_modes,"divert_entered":diversion_mode,
        "approach_entered":approach,"emergency_landing_entered":landing,"navigation_valid_ever":any(nav_valid),
        "navigation_valid_throughout":all(nav_valid),"unstable":unstable,"simulated_duration_seconds":summary["simulated_duration_seconds"],
        "strict_diversion_success":strict,"diversion_failure_category":category,"termination_reason":termination,
        "planner_candidates":[c.__dict__ for c in plan.candidates],
    }

def analyze_batch_diversions(batch_directory: str | Path, output_directory: str | Path) -> dict[str, Any]:
    batch=Path(batch_directory); out=Path(output_directory); out.mkdir(parents=True,exist_ok=True)
    records=json.loads((batch/"batch_records.json").read_text(encoding="utf-8"))
    details=[]
    for record in records:
        item=analyze_diversion_run(record["run_directory"]); item["controller"]=record["controller"]; item["scenario_group"]=record["scenario_category"]; item["pair_id"]=record["pair_id"]; details.append(item)
    aggregate={}
    for controller in ("expert","neural","monitored"):
        aggregate[controller]={}
        for group in sorted({d["scenario_group"] for d in details}):
            subset=[d for d in details if d["controller"]==controller and d["scenario_group"]==group]
            aggregate[controller][group]={
                "runs":len(subset),"airport_selected":sum(d["airport_selected"] for d in subset),
                "no_airport_considered_feasible":sum(d["no_airport_considered_feasible"] for d in subset),
                "moved_closer":sum(d["moved_closer"] for d in subset),
                "run_measurements":[{"pair_id":d["pair_id"],"initial_distance_nm":d["initial_distance_nm"],"final_distance_nm":d["final_distance_nm"],"minimum_distance_nm":d["minimum_distance_nm"],"final_heading_error_deg":d["final_heading_error_deg"],"minimum_heading_error_deg":d["minimum_heading_error_deg"],"emergency_modes":d["emergency_modes"],"internal_state_machine_modes":d["internal_state_machine_modes"],"divert_entered":d["divert_entered"],"approach_entered":d["approach_entered"],"emergency_landing_entered":d["emergency_landing_entered"],"simulated_duration_seconds":d["simulated_duration_seconds"],"failure_category":d["diversion_failure_category"],"termination_reason":d["termination_reason"]} for d in subset],
                "failure_categories":dict(Counter(d["diversion_failure_category"] for d in subset))}
    report={"source_batch":str(batch),"source_artifacts_modified":False,"run_count":len(details),"strict_criteria":{"arrival_distance_nm":1.0,"heading_tolerance_deg":15.0},"aggregate":aggregate,"runs":details}
    (out/"original_phase5_diversion_audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report
