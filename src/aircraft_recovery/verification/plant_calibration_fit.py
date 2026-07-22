"""Reproducible fit of the hybrid interval plant and development residuals."""
from __future__ import annotations
import json,os
from pathlib import Path
from typing import Any
import numpy as np
from aircraft_recovery.data.hashing import canonical_hash,file_hash
from aircraft_recovery.verification.formal_plant import FormalPlantModel,MODELED_STATES
from aircraft_recovery.verification.plant_calibration import PlantDatasetConfig,load_dataset_config,load_records


def _atomic(path:Path,text:str)->None:
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_name(path.name+".tmp")
    temp.write_text(text,encoding="utf-8",newline="\n"); os.replace(temp,path)


def _features(record,name:str)->list[float]:
    s,u,f,d=record.source_state,record.executed_action,record.failure_parameters,record.disturbance_values
    if name=="airspeed_kts": return [1,s["airspeed_kts"],u["throttle_left"]*f["left_thrust_availability"],u["throttle_right"]*f["right_thrust_availability"]]
    if name=="pitch_deg": return [1,s["pitch_deg"],u["elevator"],d["pitch_delta_deg"]]
    if name=="bank_deg": return [1,s["bank_deg"],u["aileron"]*f["aileron_effectiveness"],f["left_thrust_availability"]-f["right_thrust_availability"],d["roll_delta_deg"]]
    if name=="terrain_clearance_ft": return [1,s["terrain_clearance_ft"],record.next_state["vertical_speed_fpm"]]
    raise KeyError(name)


def predict_point(model:FormalPlantModel,record)->dict[str,float]:
    result={}
    for name in ("airspeed_kts","pitch_deg","bank_deg"):
        result[name]=float(np.dot(model.coefficients[name],_features(record,name)))
    result["vertical_speed_fpm"]=model.coefficients["vertical_speed_fpm"][0]*result["airspeed_kts"]*np.sin(np.radians(result["pitch_deg"]))
    c=model.coefficients["terrain_clearance_ft"]; result["terrain_clearance_ft"]=c[0]+c[1]*record.source_state["terrain_clearance_ft"]+c[2]*result["vertical_speed_fpm"]
    return result


def fit_formal_plant(calibration_directory:str|Path,configuration_path:str|Path,output_path:str|Path,numerical_tolerance:float=1e-8)->FormalPlantModel:
    records=[item for item in load_records(calibration_directory) if item.record_kind=="one_step"]
    if len(records)<10: raise ValueError("At least ten one-step calibration samples are required")
    split=max(1,int(len(records)*0.8)); fit_records,residual_records=records[:split],records[split:]
    coefficients={}
    for name in ("airspeed_kts","pitch_deg","bank_deg","terrain_clearance_ft"):
        design=np.asarray([_features(item,name) for item in fit_records],dtype=np.float64)
        target=np.asarray([item.next_state[name] for item in fit_records],dtype=np.float64)
        coefficients[name]=np.linalg.lstsq(design,target,rcond=None)[0].tolist()
    ratios=[]
    for item in fit_records:
        denominator=item.next_state["airspeed_kts"]*np.sin(np.radians(item.next_state["pitch_deg"]))
        if abs(denominator)>1e-9: ratios.append(item.next_state["vertical_speed_fpm"]/denominator)
    coefficients["vertical_speed_fpm"]=[float(np.median(ratios))]
    config:PlantDatasetConfig=load_dataset_config(configuration_path)
    validity={
        "state":config.validity_ranges,
        "failures":{"left_thrust_availability":(0.0,0.0),**config.failure_ranges},
        "disturbances":{"pitch_delta_deg":(-1.0,1.0),"roll_delta_deg":(-2.0,2.0),"crosswind_kts":config.disturbance_ranges["crosswind_kts"]},
        "sensor_errors":config.sensor_error_ranges,
    }
    provisional=FormalPlantModel(time_step_seconds=records[0].time_step_seconds,coefficients=coefficients,residual_bounds={name:(0.0,0.0) for name in MODELED_STATES},numerical_tolerance=numerical_tolerance,validity=validity,fit_metadata={})
    errors={name:[] for name in MODELED_STATES}
    for item in residual_records:
        predicted=predict_point(provisional,item)
        for name in MODELED_STATES: errors[name].append(item.next_state[name]-predicted[name])
    residuals={name:(float(np.nextafter(min(values)-numerical_tolerance,-np.inf)),float(np.nextafter(max(values)+numerical_tolerance,np.inf))) for name,values in errors.items()}
    metadata={"fit_sample_count":len(fit_records),"residual_sample_count":len(residual_records),"calibration_manifest_sha256":json.loads((Path(calibration_directory)/"manifest.json").read_text())["manifest_sha256"],"configuration_sha256":canonical_hash(config.model_dump(mode="json")),"configuration_file_sha256":file_hash(configuration_path),"validation_used_for_fit":False}
    model=FormalPlantModel(time_step_seconds=records[0].time_step_seconds,coefficients=coefficients,residual_bounds=residuals,numerical_tolerance=numerical_tolerance,validity=validity,fit_metadata=metadata)
    payload=model.model_dump(mode="json"); payload["model_sha256"]=canonical_hash(payload); _atomic(Path(output_path),json.dumps(payload,indent=2,sort_keys=True)+"\n")
    return model


def load_formal_plant(path:str|Path)->FormalPlantModel:
    data=json.loads(Path(path).read_text(encoding="utf-8")); data.pop("model_sha256",None); return FormalPlantModel.model_validate(data)
