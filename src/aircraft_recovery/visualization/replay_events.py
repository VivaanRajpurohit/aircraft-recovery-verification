"""Normalized failure and monitor timeline extracted from immutable replay logs."""
from __future__ import annotations
from bisect import bisect_right
from dataclasses import dataclass,field
import json
from pathlib import Path
from typing import Any

DISPLAY_NAMES={
    "left_engine_failure":"Left engine failure","right_engine_failure":"Right engine failure",
    "reduced_aileron_effectiveness":"Aileron effectiveness reduced",
    "reduced_elevator_effectiveness":"Elevator effectiveness reduced",
    "reduced_rudder_effectiveness":"Rudder effectiveness reduced",
    "navigation_data_invalidation":"Navigation data invalid",
}

@dataclass(frozen=True)
class ReplayFailure:
    failure_id:str; display_name:str; subsystem:str; activation_time_s:float
    deactivation_time_s:float|None; severity:float|None; observable:bool
    controller_known:bool; active:bool=False; details:dict[str,object]=field(default_factory=dict)

@dataclass(frozen=True)
class ReplayMonitorEvent:
    time_s:float; status:str; reason:str; proposed:dict[str,float]; final:dict[str,float]
    violated_constraints:tuple[str,...]; monitor_latency_ms:float|None; solver_latency_ms:float|None
    fallback_activated:bool; timeout:bool; unknown:bool

@dataclass(frozen=True)
class ReplayTimelineEvent:
    event_id:str; time_s:float; kind:str; labels:tuple[str,...]; failure_ids:tuple[str,...]=()
    monitor:ReplayMonitorEvent|None=None

@dataclass(frozen=True)
class ReplayHealthFrame:
    time_s:float; ground_truth_ids:tuple[str,...]; controller_known_ids:tuple[str,...]
    elevator:float; aileron:float; rudder:float; left_engine:float; right_engine:float

@dataclass
class NormalizedReplayData:
    failures:list[ReplayFailure]; health_frames:list[ReplayHealthFrame]
    failure_events:list[ReplayTimelineEvent]; monitor_events:list[ReplayMonitorEvent]
    missing_fields:list[str]
    def state_at(self,time_s:float)->ReplayHealthFrame:
        times=[frame.time_s for frame in self.health_frames]; index=max(0,min(len(times)-1,bisect_right(times,time_s)-1)); return self.health_frames[index]
    def active_failures(self,time_s:float)->list[ReplayFailure]:
        frame=self.state_at(time_s); ids=set(frame.ground_truth_ids)
        return [ReplayFailure(**{**f.__dict__,"active":f.failure_id in ids,"controller_known":f.failure_id in frame.controller_known_ids}) for f in self.failures if f.failure_id in ids]

class ReplayEventState:
    """One-shot event delivery that rearms when time moves backward."""
    def __init__(self,events:list[ReplayTimelineEvent])->None:
        self.events=sorted(events,key=lambda event:(event.time_s,event.event_id)); self.last_time=-1e-9; self.delivered:set[str]=set()
    def advance(self,time_s:float)->list[ReplayTimelineEvent]:
        if time_s<self.last_time:
            self.delivered={event.event_id for event in self.events if event.event_id in self.delivered and event.time_s<=time_s}
            self.last_time=time_s; return []
        result=[event for event in self.events if self.last_time<event.time_s<=time_s and event.event_id not in self.delivered]
        self.delivered.update(event.event_id for event in result); self.last_time=time_s; return result
    def reset(self)->None: self.last_time=-1e-9; self.delivered.clear()

def _truth(row:dict[str,Any])->list[dict[str,Any]]:
    return list((row.get("ground_truth_after") or {}).get("ground_truth_failures") or [])

def normalize_replay_data(run_directory:str|Path)->NormalizedReplayData:
    run=Path(run_directory); rows=[json.loads(line) for line in (run/"history.jsonl").read_text(encoding="utf-8").splitlines()]; metadata=json.loads((run/"metadata.json").read_text(encoding="utf-8")); missing=[]
    configured=((metadata.get("scenario_configuration") or {}).get("failures") or [])
    health_frames=[]
    for row in rows:
        observation=row["next_observation"]; control=observation.get("control_health") or {}; engine=observation.get("engine_health") or {}; audit=row.get("ground_truth_after") or {}
        if "ground_truth_failures" not in audit: missing.append("ground_truth_after.ground_truth_failures")
        truth_ids=tuple(item.get("failure_id","unknown") for item in _truth(row)); known=tuple(audit.get("controller_known_failures") or observation.get("active_failures") or [])
        health_frames.append(ReplayHealthFrame(round(float(observation["timestamp_seconds"]),9),truth_ids,known,float(control.get("elevator_effectiveness",1)),float(control.get("aileron_effectiveness",1)),float(control.get("rudder_effectiveness",1)),float(engine.get("left_engine_thrust_available",1)),float(engine.get("right_engine_thrust_available",1))))
    failures=[]
    all_ids={item.get("failure_id") for item in configured}|{item.get("failure_id") for row in rows for item in _truth(row)}
    for failure_id in sorted(value for value in all_ids if value):
        config=next((item for item in configured if item.get("failure_id")==failure_id),{}); samples=[(health_frames[i].time_s,item) for i,row in enumerate(rows) for item in _truth(row) if item.get("failure_id")==failure_id]
        first=samples[0][1] if samples else {}; activation=round(float(first.get("activation_time_seconds",config.get("activation_time_seconds",0) or 0)),9); active_times={round(time,9) for time,_ in samples}; later=[frame.time_s for frame in health_frames if frame.time_s>activation and frame.time_s not in active_times]; deactivation=min(later) if later else (round(activation+float(config["duration_seconds"]),9) if config.get("duration_seconds") else None)
        failure_type=str(config.get("failure_type",first.get("failure_type",failure_id))); severity=config.get("severity",first.get("severity")); severity=float(severity) if severity is not None else None
        details={"failure_type":failure_type,"ramp_duration_seconds":config.get("ramp_duration_seconds"),"permanent":config.get("permanent"),"parameters":config.get("parameters") or {}}
        if failure_type=="left_engine_failure": details["remaining_thrust_fraction"]=max(0.0,1.0-(severity or 0.0))
        if failure_type=="right_engine_failure": details["remaining_thrust_fraction"]=max(0.0,1.0-(severity or 0.0))
        if "effectiveness" in failure_type: details["remaining_effectiveness_fraction"]=max(0.0,1.0-(severity or 0.0))
        ever_known=any(failure_id in frame.controller_known_ids for frame in health_frames)
        display=DISPLAY_NAMES.get(failure_type,failure_type.replace("_"," ").title())
        if failure_type in {"left_engine_failure","right_engine_failure"} and severity is not None: display+=f" (thrust {max(0,1-severity)*100:.0f}%)"
        if "effectiveness" in failure_type and severity is not None: display+=f" to {max(0,1-severity)*100:.0f}%"
        failures.append(ReplayFailure(failure_id,display,str(config.get("affected_subsystem",first.get("affected_subsystem","unknown"))),activation,deactivation,severity,bool(config.get("observable",first.get("observable",False))),ever_known,False,details))
    grouped:dict[tuple[float,str],list[ReplayFailure]]={}
    for failure in failures:
        grouped.setdefault((failure.activation_time_s,"activated"),[]).append(failure)
        if failure.deactivation_time_s is not None: grouped.setdefault((failure.deactivation_time_s,"cleared"),[]).append(failure)
    failure_events=[ReplayTimelineEvent(f"failure:{kind}:{time}",time,kind,tuple(item.display_name for item in items),tuple(item.failure_id for item in items)) for (time,kind),items in grouped.items()]
    monitor=[]
    for row in rows:
        decision=row.get("monitor_decision")
        if not decision: continue
        monitor.append(ReplayMonitorEvent(round(float(row["next_observation"]["timestamp_seconds"]),9),str(decision.get("status","unknown")),str(decision.get("reason","No reason recorded")),dict(decision.get("original_action") or row.get("proposed_action",{}).get("control_commands") or {}),dict(decision.get("final_action") or row.get("final_action",{}).get("control_commands") or {}),tuple(decision.get("violated_properties") or []),decision.get("monitor_duration_ms"),decision.get("solver_duration_ms"),bool(decision.get("fallback_activated")),bool(decision.get("timeout")),bool(decision.get("unknown"))))
    return NormalizedReplayData(failures,health_frames,sorted(failure_events,key=lambda e:e.time_s),monitor,sorted(set(missing)))
