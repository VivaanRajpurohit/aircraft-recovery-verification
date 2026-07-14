"""Held-out one-step and short-rollout containment validation."""
from __future__ import annotations
from collections import defaultdict
import json,os
from pathlib import Path
from typing import Any
import numpy as np
from aircraft_recovery.data.hashing import canonical_hash,file_hash
from aircraft_recovery.verification.formal_plant import MODELED_STATES,FormalPlantModel,StateInterval,transition,within_validity
from aircraft_recovery.verification.plant_calibration import ACTION_NAMES,assert_split_isolation,load_records
from aircraft_recovery.verification.plant_calibration_fit import load_formal_plant,predict_point


def _atomic(path:Path,text:str)->None:
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_name(path.name+".tmp")
    temporary.write_text(text,encoding="utf-8",newline="\n"); os.replace(temporary,path)


def _point_bounds(values:dict[str,float])->dict[str,tuple[float,float]]:
    return {name:(float(value),float(value)) for name,value in values.items()}


def _inputs(record):
    actions=_point_bounds(record.executed_action)
    failures=_point_bounds({name:record.failure_parameters[name] for name in ("left_thrust_availability","right_thrust_availability","aileron_effectiveness","actuator_delay_seconds")})
    disturbances=_point_bounds({name:record.disturbance_values[name] for name in ("pitch_delta_deg","roll_delta_deg","crosswind_kts")})
    sensors=_point_bounds(record.sensor_error_values)
    return actions,failures,disturbances,sensors


def _contains(interval:StateInterval,state:dict[str,float],index:int)->bool:
    value=state[MODELED_STATES[index]]; return bool(interval.lower[index]<=value<=interval.upper[index])


def validate_formal_plant(model_path:str|Path,calibration_directory:str|Path,validation_directory:str|Path,output_path:str|Path,horizons:tuple[int,...]=(5,10,20))->dict[str,Any]:
    model=load_formal_plant(model_path); records=load_records(validation_directory); isolation=assert_split_isolation(calibration_directory,validation_directory)
    one_step=[item for item in records if item.record_kind=="one_step"]
    errors={name:[] for name in MODELED_STATES}; contained={name:0 for name in MODELED_STATES}; outside=[]; failures=[]
    for item in one_step:
        actions,failure,disturbance,sensors=_inputs(item); state=StateInterval.point(item.source_state)
        if not within_validity(model,state,failure,disturbance,sensors): outside.append(item.sample_id); continue
        predicted=predict_point(model,item); reachable=transition(model,state,actions,failure,disturbance)
        failed=[]
        for index,name in enumerate(MODELED_STATES):
            error=item.next_state[name]-predicted[name]; errors[name].append(error)
            if _contains(reachable,item.next_state,index): contained[name]+=1
            else: failed.append(name)
        if failed: failures.append({"sample_id":item.sample_id,"states":failed})
    evaluated=len(one_step)-len(outside); metrics={}
    for name,values_list in errors.items():
        values=np.asarray(values_list,dtype=np.float64); absolute=np.abs(values)
        metrics[name]={
            "mean_signed_error":float(np.mean(values)),"minimum_signed_error":float(np.min(values)),"maximum_signed_error":float(np.max(values)),
            "mean_absolute_error":float(np.mean(absolute)),"median_absolute_error":float(np.median(absolute)),"p95_absolute_error":float(np.percentile(absolute,95)),"maximum_absolute_error":float(np.max(absolute)),
            "residual_bound":list(model.residual_bounds[name]),"contained":contained[name],"evaluated":evaluated,"containment_percent":100.0*contained[name]/evaluated if evaluated else 0.0,
        }
    traces=defaultdict(list)
    for item in records:
        if item.record_kind=="trace": traces[item.trace_id].append(item)
    max_horizon=max(horizons); step_total=[0]*max_horizon; step_contained=[0]*max_horizon; first_failures=[]
    width_by_state={name:[[] for _ in range(max_horizon)] for name in MODELED_STATES}
    for trace_id,items in traces.items():
        items.sort(key=lambda item:item.step_index or 0); state=StateInterval.point(items[0].source_state,radius=1e-9); first=None
        for step,item in enumerate(items[:max_horizon]):
            actions,failure,disturbance,_=_inputs(item)
            try: state=transition(model,state,actions,failure,disturbance)
            except ValueError:
                first=first or {"trace_id":trace_id,"step":step+1,"reason":"outside_model_validity"}; break
            step_total[step]+=1
            all_contained=all(_contains(state,item.next_state,index) for index in range(len(MODELED_STATES)))
            if all_contained: step_contained[step]+=1
            elif first is None: first={"trace_id":trace_id,"step":step+1,"reason":"containment_failure"}
            for index,name in enumerate(MODELED_STATES): width_by_state[name][step].append(float(state.upper[index]-state.lower[index]))
        if first: first_failures.append(first)
    rollout={
        "horizons_tested":list(horizons),"trace_count":len(traces),"first_containment_failures":first_failures,
        "containment_percent_by_step":[100.0*yes/total if total else None for yes,total in zip(step_contained,step_total)],
        "interval_width_by_state":{name:{"initial":values[0][0] if values[0] else None,"maximum":max((max(step) for step in values if step),default=None),"at_horizons":{str(h):float(np.mean(values[h-1])) if h<=len(values) and values[h-1] else None for h in horizons}} for name,values in width_by_state.items()},
    }
    validation_manifest=json.loads((Path(validation_directory)/"manifest.json").read_text(encoding="utf-8"))
    report={"schema_version":1,"model_sha256":file_hash(model_path),"validation_manifest_sha256":validation_manifest["manifest_sha256"],"split_isolation":isolation,"one_step":{"total":len(one_step),"evaluated":evaluated,"outside_model_validity":outside,"containment_failures":failures,"metrics":metrics},"rollout":rollout}
    report["report_sha256"]=canonical_hash(report); _atomic(Path(output_path),json.dumps(report,indent=2,sort_keys=True)+"\n"); return report
