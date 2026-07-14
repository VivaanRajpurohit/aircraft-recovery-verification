"""Conservative fitted interval transition for the selected simulator state."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from aircraft_recovery.verification.plant_calibration import ACTION_NAMES, STATE_NAMES


MODELED_STATES = ("airspeed_kts", "pitch_deg", "bank_deg", "vertical_speed_fpm", "terrain_clearance_ft")


@dataclass(frozen=True)
class StateInterval:
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower=np.asarray(self.lower,dtype=np.float64); upper=np.asarray(self.upper,dtype=np.float64)
        if lower.shape!=(10,) or upper.shape!=(10,): raise ValueError("Formal state intervals must have shape (10,)")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)): raise ValueError("State intervals must be finite")
        if np.any(lower>upper): raise ValueError("State interval lower bounds must not exceed upper bounds")
        object.__setattr__(self,"lower",lower); object.__setattr__(self,"upper",upper)

    @classmethod
    def point(cls, state: dict[str,float], radius: float=0.0) -> "StateInterval":
        values=np.asarray([state[name] for name in STATE_NAMES],dtype=np.float64)
        return cls(np.nextafter(values-radius,-np.inf),np.nextafter(values+radius,np.inf))

    def as_dict(self) -> dict[str,tuple[float,float]]:
        return {name:(float(self.lower[i]),float(self.upper[i])) for i,name in enumerate(STATE_NAMES)}


class FormalPlantModel(BaseModel):
    model_config=ConfigDict(extra="forbid")
    schema_version: int=1
    model_type: str="hybrid_interval_lpv"
    time_step_seconds: float=Field(gt=0)
    coefficients: dict[str,list[float]]
    residual_bounds: dict[str,tuple[float,float]]
    numerical_tolerance: float=Field(gt=0)
    validity: dict[str,dict[str,tuple[float,float]]]
    fit_metadata: dict[str,Any]


def _out(lower: float,upper: float)->tuple[float,float]:
    return float(np.nextafter(lower,-np.inf)),float(np.nextafter(upper,np.inf))


def _mul(a:tuple[float,float],b:tuple[float,float])->tuple[float,float]:
    values=(a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]); return _out(min(values),max(values))


def _linear(coefficients:list[float],features:list[tuple[float,float]])->tuple[float,float]:
    lower=upper=coefficients[0]
    for coefficient,(left,right) in zip(coefficients[1:],features):
        if coefficient>=0: lower+=coefficient*left; upper+=coefficient*right
        else: lower+=coefficient*right; upper+=coefficient*left
    return _out(lower,upper)


def _add(first:tuple[float,float],second:tuple[float,float])->tuple[float,float]:
    return _out(first[0]+second[0],first[1]+second[1])


def _sin_degrees(bounds:tuple[float,float])->tuple[float,float]:
    lower,upper=bounds
    if upper-lower>=360: return (-1.0,1.0)
    candidates=[math.sin(math.radians(lower)),math.sin(math.radians(upper))]
    start=math.ceil((lower-90)/180); end=math.floor((upper-90)/180)
    for index in range(start,end+1): candidates.append(math.sin(math.radians(90+180*index)))
    return _out(min(candidates),max(candidates))


def resolve_delayed_action(queue:list[dict[str,float]],proposed:dict[str,float],delay_steps:int)->dict[str,float]:
    if delay_steps<0: raise ValueError("delay_steps must be nonnegative")
    if delay_steps==0: return {name:float(proposed[name]) for name in ACTION_NAMES}
    if len(queue)<delay_steps: return {name:0.0 for name in ACTION_NAMES}
    return {name:float(queue[-delay_steps][name]) for name in ACTION_NAMES}


def within_validity(model:FormalPlantModel,state:StateInterval,failures:dict[str,tuple[float,float]],disturbances:dict[str,tuple[float,float]],sensor_errors:dict[str,tuple[float,float]]|None=None)->bool:
    state_dict=state.as_dict()
    groups={"state":state_dict,"failures":failures,"disturbances":disturbances,"sensor_errors":sensor_errors or {}}
    for group_name,values in groups.items():
        declared=model.validity.get(group_name,{})
        for name,(lower,upper) in values.items():
            if name in declared and (lower<declared[name][0] or upper>declared[name][1]): return False
    return True


def transition(model:FormalPlantModel,state:StateInterval,actions:dict[str,tuple[float,float]],failures:dict[str,tuple[float,float]],disturbances:dict[str,tuple[float,float]],*,require_validity:bool=True)->StateInterval:
    if require_validity and not within_validity(model,state,failures,disturbances): raise ValueError("Transition is outside the declared formal-model validity region")
    current=state.as_dict(); left=failures["left_thrust_availability"]; right=failures["right_thrust_availability"]
    throttle_left=_mul(actions["throttle_left"],left); throttle_right=_mul(actions["throttle_right"],right)
    speed=_linear(model.coefficients["airspeed_kts"],[current["airspeed_kts"],throttle_left,throttle_right]); speed=_add(speed,model.residual_bounds["airspeed_kts"])
    pitch=_linear(model.coefficients["pitch_deg"],[current["pitch_deg"],actions["elevator"],disturbances["pitch_delta_deg"]]); pitch=_add(pitch,model.residual_bounds["pitch_deg"])
    effective_aileron=_mul(actions["aileron"],failures["aileron_effectiveness"]); asymmetry=(left[0]-right[1],left[1]-right[0])
    bank=_linear(model.coefficients["bank_deg"],[current["bank_deg"],effective_aileron,asymmetry,disturbances["roll_delta_deg"]]); bank=_add(bank,model.residual_bounds["bank_deg"])
    vertical=_mul(speed,_sin_degrees(pitch)); scale=model.coefficients["vertical_speed_fpm"][0]; vertical=_linear([0.0,scale],[vertical]); vertical=_add(vertical,model.residual_bounds["vertical_speed_fpm"])
    clearance=_linear(model.coefficients["terrain_clearance_ft"],[current["terrain_clearance_ft"],vertical]); clearance=_add(clearance,model.residual_bounds["terrain_clearance_ft"])
    outputs=[speed,pitch,bank,vertical,clearance]+[actions[name] for name in ACTION_NAMES]
    return StateInterval(np.asarray([item[0] for item in outputs]),np.asarray([item[1] for item in outputs]))
