"""Run six controlled diversion validation cases outside the research batch."""
from __future__ import annotations
import argparse,json,shutil
from collections import Counter
from pathlib import Path
import torch
from aircraft_recovery.analysis.diversion_validation import analyze_diversion_run
from aircraft_recovery.config import AircraftConfig,ScenarioConfig,load_config
from aircraft_recovery.controllers import DeterministicFallbackController,FallbackConfig,MonitoredNeuralController,NeuralBaselineController
from aircraft_recovery.experiments import ScenarioRunner
from aircraft_recovery.safety import SafetyConfig
from aircraft_recovery.simulator import SimpleAircraftSimulator

SCENARIOS=(
"validation_diversion_nominal.yaml","validation_diversion_single_engine.yaml",
"validation_diversion_total_engine.yaml","validation_diversion_crosswind.yaml",
"validation_diversion_infeasible.yaml","validation_diversion_navigation_invalid.yaml")

def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output",default="results/final_validation/controlled_diversion"); p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto"); p.add_argument("--force",action="store_true"); a=p.parse_args()
    out=Path(a.output).resolve(); root=Path("results/final_validation").resolve()
    if a.force and out.exists():
        if root!=out and root not in out.parents: raise ValueError("--force is restricted to results/final_validation")
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Output is not empty: {out}")
    out.mkdir(parents=True,exist_ok=True)
    aircraft=load_config("configs/aircraft/simple_twin.yaml",AircraftConfig); fallback=load_config("configs/controllers/fallback.yaml",FallbackConfig); safety=load_config("configs/safety/default.yaml",SafetyConfig)
    details=[]
    for filename in SCENARIOS:
        scenario=load_config(Path("configs/scenarios")/filename,ScenarioConfig)
        controllers={"expert":DeterministicFallbackController(fallback,scenario.airports),"neural":NeuralBaselineController("checkpoints/phase3_quick/best.pt",a.device),"monitored":MonitoredNeuralController("checkpoints/phase3_quick/best.pt",a.device,fallback,scenario.airports,safety)}
        for name,controller in controllers.items():
            run=out/"runs"/f"{scenario.scenario_id}_{name}"; ScenarioRunner(SimpleAircraftSimulator(aircraft,scenario),controller).run(scenario.seed,run,scenario.scenario_id)
            item=analyze_diversion_run(run); item["controller"]=name; details.append(item)
    aggregate={}
    for name in ("expert","neural","monitored"):
        subset=[d for d in details if d["controller"]==name]; aggregate[name]={"runs":len(subset),"strict_successes":sum(d["strict_diversion_success"] for d in subset),"airport_selections":sum(d["airport_selected"] for d in subset),"approach_entries":sum(d["approach_entered"] for d in subset),"failure_categories":dict(Counter(d["diversion_failure_category"] for d in subset))}
    result={"validation_scope":"controlled software validation; excluded from original Phase 5 results","strict_criteria":{"arrival_distance_nm":1.0,"heading_tolerance_deg":15.0},"device_requested":a.device,"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"aggregate":aggregate,"runs":details}
    (out/"controlled_diversion_results.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(aggregate,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
