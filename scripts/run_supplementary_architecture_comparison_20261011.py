"""Resumable supplementary comparison; original v2.1B files are read-only."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import gzip
import io
import json
import random
import subprocess
import sys
import time
import types
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import msvcrt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aircraft_recovery.training import v2_1a_architecture_screen as train
from aircraft_recovery.evaluation import v2_1b_nonsealed as ev
from aircraft_recovery.data_quality.common import file_hash, object_hash
from aircraft_recovery.config import load_config, load_yaml, AircraftConfig
from aircraft_recovery.controllers import FallbackConfig, DeterministicFallbackController
from aircraft_recovery.safety import RuntimeSafetyMonitor, SafetyConfig

ID = "supplementary_architecture_comparison_20261011"
OUT = ROOT / "results" / ID
ART = ROOT / "artifacts/research_stage" / ID
CONTROLS = dict(train.SCREEN_CONTROLS, batch_size=512)
CONDITIONS = ("unmonitored", "monitored")

def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    with temp.open("xb") as stream:
        stream.write(value); stream.flush(); os.fsync(stream.fileno())
    if path.exists():
        if path.read_bytes() != value:
            raise FileExistsError(f"immutable output differs: {path}")
        return
    os.replace(temp, path)

def publish(path: Path, value: dict) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())

def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def protect_snapshot() -> dict:
    entries = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    paths = [p for p in entries if p.startswith(("src/", "configs/", "artifacts/research_stage/")) and ID not in p]
    paths += [f"checkpoints/final_policy_v1/moderate/final_training_v2_1a/seed_{s}/best.pt" for s in train.FINAL_TRAINING_SEEDS]
    paths += [f"datasets/approved_policy_data/{train.APPROVED_ID}/" + p for p in ("manifests/approved_manifest.json", "preprocessing/train_only_normalization.json", "raw/flights.jsonl")]
    paths += ["results/final_nonsealed_evaluation_v2_1b/challenge/" + p for p in ("summary.json", "flight_records.jsonl")]
    paths += ["results/final_v2_1b_analysis/analysis.json"]
    return {p: file_hash(ROOT / p) for p in sorted(set(paths))}

def verify_snapshot() -> None:
    for p, expected in read(ART / "protected_inputs.json")["files"].items():
        if file_hash(ROOT / p) != expected:
            raise ValueError(f"protected input changed: {p}")

def prepare() -> None:
    if (ART / "plan.json").exists():
        verify_plan(); return
    if ART.exists() and any(ART.iterdir()):
        raise FileExistsError("partial preparation exists; inspect before continuing")
    preflight = train.architecture_screen_preflight(ROOT)
    ev.nonsealed_preflight(ROOT, "challenge")
    snapshot = protect_snapshot()
    for candidate in train.ARCHITECTURE_CANDIDATES:
        assert train.V21AControlPolicyNetwork(candidate).parameter_count == candidate.expected_parameter_count
    plan = {
        "id": ID, "date": "2026-10-11", "status": "fixed_before_supplementary_fitting",
        "evidence_status": "exploratory_after_original_final_test_outcomes_known",
        "architectures": preflight["architectures"], "seed": 17001,
        "training": CONTROLS, "fit_split": "train", "selection_split": "development",
        "evaluation_split": "challenge", "scenario_count_per_model": 144,
        "controller_conditions": list(CONDITIONS),
        "new_training_runs": 2, "new_evaluation_pairs": 288,
        "complete_comparison_pairs": 432, "complete_comparison_condition_records": 864,
        "reuse_model": "moderate", "primary_checkpoint": "checkpoints/final_policy_v1/moderate/final_training_v2_1a/seed_17001/best.pt",
        "primary_checkpoint_sha256": ev.REPRESENTATIVE_CHECKPOINT_HASH,
        "reuse_original_challenge_records": True,
        "dataset_logical_sha256": train.APPROVED_DATASET_LOGICAL_HASH,
        "normalization_logical_sha256": train.NORMALIZATION_LOGICAL_HASH,
        "no_primary_reselection": True, "no_outcome_dependent_tuning": True,
        "ood_access": False, "sealed_access": False,
        "source_git_revision_before_extension": train._git_revision(ROOT),
        "protected_inputs_logical_sha256": object_hash(snapshot),
    }
    plan["plan_logical_sha256"] = object_hash(plan)
    publish(ART / "protected_inputs.json", {"files": snapshot, "logical_sha256": object_hash(snapshot)})
    publish(ART / "preflight.json", preflight)
    publish(ART / "plan.json", plan)
    print("PLAN PREPARED", plan["plan_logical_sha256"], flush=True)

def verify_plan() -> dict:
    plan = read(ART / "plan.json")
    if plan["plan_logical_sha256"] != object_hash({k: v for k, v in plan.items() if k != "plan_logical_sha256"}):
        raise ValueError("plan hash mismatch")
    if plan["training"] != CONTROLS or plan["evaluation_split"] != "challenge" or plan["seed"] != 17001:
        raise ValueError("plan controls mismatch")
    verify_snapshot(); train.assert_access_policy()
    return plan

def tensors():
    approved = ROOT / "datasets/approved_policy_data" / train.APPROVED_ID
    norm = read(approved / "preprocessing/train_only_normalization.json")
    source = train.ArchitectureScreenDataSource(approved)
    fit = train._read_split(source.training_records(), expected_split="train", expected_count=train.TRAIN_COUNT, expected_flights=576)
    dev = train._read_split(source.development_records(), expected_split="development", expected_count=train.DEVELOPMENT_COUNT, expected_flights=144)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return tuple(torch.from_numpy(v).to(device) for v in (train._normalize(fit.observations, norm), fit.controls, train._normalize(dev.observations, norm), dev.controls)), norm

def checkpoint_bytes(state: dict) -> bytes:
    buffer = io.BytesIO(); torch.save(state, buffer); return buffer.getvalue()

def load_epoch(directory: Path):
    markers = sorted(directory.glob("epoch_*.json"))
    if not markers: return None
    for marker in markers:
        item = read(marker)
        if file_hash(directory / item["checkpoint"]) != item["sha256"]:
            raise ValueError(f"committed epoch corrupted: {marker}")
    item = read(markers[-1])
    return torch.load(directory / item["checkpoint"], map_location="cpu", weights_only=False)

def fit(candidate, arrays, norm, plan):
    directory = OUT / "training" / candidate.architecture_id
    complete = directory / "complete.json"
    if complete.exists():
        result = read(complete)
        if file_hash(directory / "best.pt") != result["best_checkpoint_sha256"]: raise ValueError("completed checkpoint changed")
        return result
    x, y, dx, dy = arrays
    train._set_seed(17001)
    model = train.V21AControlPolicyNetwork(candidate).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    generator = torch.Generator(device=x.device.type).manual_seed(17001)
    history = []; best = float("inf"); best_epoch = 0; stale = 0; runtime = 0.0
    saved = load_epoch(directory)
    if saved:
        model.load_state_dict(saved["model_state_dict"]); optimizer.load_state_dict(saved["optimizer_state_dict"])
        generator.set_state(saved["generator_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if x.device.type == "cuda": torch.cuda.set_rng_state_all(saved["cuda_rng_states"])
        random.setstate(saved["python_rng_state"]); np.random.set_state(saved["numpy_rng_state"])
        history=saved["history"]; best=saved["best_mse"]; best_epoch=saved["best_epoch"]; stale=saved["stale_epochs"]; runtime=saved["runtime_seconds"]
    config = {"plan_logical_sha256": plan["plan_logical_sha256"], "architecture": candidate.__dict__, "seed":17001, "training":CONTROLS, "dataset_logical_sha256":train.APPROVED_DATASET_LOGICAL_HASH, "normalization_logical_sha256":train.NORMALIZATION_LOGICAL_HASH, "source_git_revision":train._git_revision(ROOT)}
    if (directory / "config.json").exists():
        config = read(directory / "config.json")
        if config["plan_logical_sha256"] != plan["plan_logical_sha256"]: raise ValueError("training plan mismatch")
    else: publish(directory / "config.json", config)
    for epoch in range(len(history)+1, 51):
        if stale >= 8: break
        start=time.perf_counter(); model.train(); error_sum=0.0; count=0
        permutation = torch.randperm(x.shape[0], device=x.device, generator=generator)
        for offset in range(0,x.shape[0],512):
            idx=permutation[offset:offset+512]; optimizer.zero_grad(set_to_none=True)
            loss=torch.square(model(x[idx])-y[idx]).mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            error_sum += float(loss.detach().cpu())*y[idx].numel(); count += y[idx].numel()
        metric=train._development_metrics(model,dx,dy,512)
        improved=metric["mse"] < best-1e-10
        if improved: best=metric["mse"]; best_epoch=epoch; stale=0
        else: stale+=1
        runtime+=time.perf_counter()-start
        history.append({"epoch":epoch,"train_mse":error_sum/count,"development":metric,"checkpoint_improved":improved})
        state=train._checkpoint(model=model,optimizer=optimizer,candidate=candidate,epoch=epoch,metrics=metric,run_config=config,normalization=norm,environment=train._environment(x.device))
        state.update(history=history,best_mse=best,best_epoch=best_epoch,stale_epochs=stale,runtime_seconds=runtime,generator_state=generator.get_state(),torch_rng_state=torch.get_rng_state(),cuda_rng_states=torch.cuda.get_rng_state_all() if x.device.type=="cuda" else [],python_rng_state=random.getstate(),numpy_rng_state=np.random.get_state())
        name=f"epoch_{epoch:03d}_{os.getpid()}_{time.time_ns()}.pt"; path=directory/name
        atomic_bytes(path,checkpoint_bytes(state))
        publish(directory/f"epoch_{epoch:03d}.json", {"epoch":epoch,"checkpoint":name,"sha256":file_hash(path),"plan_logical_sha256":plan["plan_logical_sha256"]})
        print(candidate.architecture_id,"epoch",epoch,"committed",flush=True)
    best_path=directory/read(directory/f"epoch_{best_epoch:03d}.json")["checkpoint"]; best_saved=torch.load(best_path,map_location=x.device,weights_only=False)
    model.load_state_dict(best_saved["model_state_dict"])
    reproduced=train._development_metrics(model,dx,dy,512)
    if not np.isclose(reproduced["mse"],best,rtol=1e-8,atol=1e-12): raise ValueError("development metric reload mismatch")
    atomic_bytes(directory/"best.pt",best_path.read_bytes())
    last_path=directory/read(directory/f"epoch_{len(history):03d}.json")["checkpoint"]; atomic_bytes(directory/"last.pt",last_path.read_bytes())
    publish(directory/"history.json",{"history":history})
    result={"status":"COMPLETE","architecture_id":candidate.architecture_id,"parameter_count":candidate.expected_parameter_count,"seed":17001,"epochs_completed":len(history),"best_epoch":best_epoch,"development_metrics":reproduced,"runtime_seconds":runtime,"best_checkpoint_path":str((directory/"best.pt").relative_to(ROOT)).replace('\\','/'),"best_checkpoint_sha256":file_hash(directory/"best.pt"),"last_checkpoint_sha256":file_hash(directory/"last.pt"),"configuration_sha256":file_hash(directory/"config.json"),"history_sha256":file_hash(directory/"history.json"),"plan_logical_sha256":plan["plan_logical_sha256"]}
    publish(complete,result);return result

class GenericNeural(ev.V21AControlPolicyController):
    def __init__(self,checkpoint_path,device):
        self.device=device; self.checkpoint_path=checkpoint_path
        saved=torch.load(checkpoint_path,map_location=device,weights_only=False)
        ident=saved["architecture"]["architecture_id"]
        candidate=next(c for c in train.ARCHITECTURE_CANDIDATES if c.architecture_id==ident)
        if saved["normalization_logical_hash"]!=train.NORMALIZATION_LOGICAL_HASH: raise ValueError("normalization changed")
        self.model=train.V21AControlPolicyNetwork(candidate).to(device)
        self.model.load_state_dict(saved["model_state_dict"]);self.model.eval()
        self.normalization=saved["normalization"];self.model_checkpoint_identifier=file_hash(checkpoint_path)
        self.last_inference_latency_ms=0.0;self.last_proposed_action=None

class GenericMonitored(ev.V21AMonitoredController):
    def __init__(self,checkpoint_path,device,fallback_config,scenario,safety_config):
        self.neural=GenericNeural(checkpoint_path,device)
        self.fallback=DeterministicFallbackController(fallback_config,scenario.airports)
        self.monitor=RuntimeSafetyMonitor(safety_config)
        self.model_checkpoint_identifier=self.neural.model_checkpoint_identifier
        self.last_proposed_action=None;self.last_monitor_decision=None

def read_condition(path: Path, expected: dict) -> dict:
    value=json.loads(gzip.decompress(path.read_bytes()))
    for key, target in expected.items():
        if value.get(key)!=target: raise ValueError(f"condition identity mismatch: {path}: {key}")
    row=value["record"]
    if row["step_count"]!=260 or len(value["traces"])!=260 or row["invalid_experiment"]: raise ValueError("incomplete/invalid condition")
    return value

def pair_directory(directory: Path, flight_id: str) -> Path:
    # Frozen IDs contain colons; preserve them in records, not Windows filenames.
    return directory / object_hash(flight_id)

def evaluate(candidate,result,flights,hashes,plan):
    directory=OUT/"challenge"/candidate.architecture_id
    checkpoint=ROOT/result["best_checkpoint_path"]
    context=dict(ev._run_flight.__globals__,V21AControlPolicyController=GenericNeural,V21AMonitoredController=GenericMonitored)
    run=types.FunctionType(ev._run_flight.__code__,context,name="supplementary_run_flight")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aircraft=load_config(ROOT/"configs/aircraft/simple_twin.yaml",AircraftConfig)
    fallback=load_config(ROOT/"configs/research/final_fallback_v1.yaml",FallbackConfig)
    safety=load_config(ROOT/"configs/research/final_monitor_v1.yaml",SafetyConfig)
    metrics=load_yaml(ROOT/"configs/research/final_metrics_v1.yaml")
    rows=[]; lat={c:{n:[] for n in ("inference","monitor","full_controller")} for c in CONDITIONS}
    for flight in flights:
        if flight["split"]!="challenge": raise PermissionError("supplement accepts Challenge only")
        pairdir=pair_directory(directory,flight["flight_id"])
        if pairdir.resolve().parent!=directory.resolve(): raise ValueError("unsafe flight id path")
        scenario=ev._scenario(ROOT,ev._flight_spec(flight));scenario_hash=object_hash(scenario.model_dump(mode="json"))
        if scenario_hash!=hashes[flight["flight_id"]]: raise ValueError("scenario reconstruction mismatch")
        markers={}
        for condition in CONDITIONS:
            identity={"plan_logical_sha256":plan["plan_logical_sha256"],"checkpoint_sha256":result["best_checkpoint_sha256"],"scenario_sha256":scenario_hash,"condition":condition,"flight_id":flight["flight_id"]}
            path=pairdir/(condition+".json.gz")
            if path.exists(): value=read_condition(path,identity)
            else:
                stream=io.StringIO()
                record,latencies=run(root=ROOT,flight=flight,scenario=scenario,condition=condition,checkpoint=checkpoint,device=device,aircraft=aircraft,fallback_config=fallback,safety_config=safety,metrics=metrics,trace_stream=stream)
                value=dict(identity,record=record,latencies=latencies,traces=[json.loads(line) for line in stream.getvalue().splitlines()])
                if record["step_count"]!=260 or len(value["traces"])!=260 or record["invalid_experiment"]: raise ValueError("condition invalid; stop for audit")
                atomic_bytes(path,gzip.compress(json.dumps(value,separators=(',',':'),allow_nan=False).encode(),mtime=0))
                value=read_condition(path,identity)
            markers[condition]=file_hash(path);rows.append(value["record"])
            for name,values in value["latencies"].items():lat[condition][name].extend(values)
        publish(pairdir/"pair_complete.json",{"status":"COMPLETE","flight_id":flight["flight_id"],"condition_sha256":markers,"plan_logical_sha256":plan["plan_logical_sha256"]})
        print(candidate.architecture_id,"paired scenario committed",flight["flight_id"],flush=True)
    if len(rows)!=288 or len({r['flight_id'] for r in rows})!=144:raise ValueError("Challenge incomplete")
    summary={"status":"COMPLETE","evidence_status":"exploratory","architecture_id":candidate.architecture_id,"seed":17001,"checkpoint_sha256":result["best_checkpoint_sha256"],"pair_count":144,"condition_record_count":288,"trace_count":sum(r['step_count'] for r in rows),"overall":{c:ev._aggregate([r for r in rows if r['controller_condition']==c],lat[c]) for c in CONDITIONS},"paired":ev._paired(rows),"grouped":{c:ev._grouped([r for r in rows if r['controller_condition']==c]) for c in CONDITIONS},"plan_logical_sha256":plan["plan_logical_sha256"],"sealed_opened":False,"ood_opened":False}
    atomic_bytes(directory/"flight_records.jsonl",(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows)).encode())
    publish(directory/"summary.json",summary);return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare();return
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"worker.lock").open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        plan=verify_plan()
        implementation=sorted(ART.glob('worker_implementation*.json'))[-1]
        if read(implementation)['sha256']!=file_hash(Path(__file__)):
            raise ValueError('worker source changed; inspect before resuming')
        if (OUT/"complete.json").exists():
            print("Already complete; verified protected inputs; no experiments repeated",flush=True);return
        needed=any(not (OUT/'training'/c.architecture_id/'complete.json').exists() for c in train.ARCHITECTURE_CANDIDATES[:2])
        arrays,norm=tensors() if needed else (None,None)
        results=[fit(c,arrays,norm,plan) for c in train.ARCHITECTURE_CANDIDATES[:2]]
        del arrays
        if torch.cuda.is_available():torch.cuda.empty_cache()
        # No Challenge metadata or records enter optimization above.
        approved=ROOT/'datasets/approved_policy_data'/train.APPROVED_ID
        flights=ev._load_flight_metadata(approved,'challenge','challenge_evaluation')
        hashes=ev._expected_scenario_hashes(approved,'challenge','challenge_evaluation')
        summaries=[evaluate(c,result,flights,hashes,plan) for c,result in zip(train.ARCHITECTURE_CANDIDATES[:2],results)]
        retained=read(ROOT/'results/final_nonsealed_evaluation_v2_1b/challenge/summary.json')
        if retained['status']!='COMPLETE' or retained['representative_checkpoint_sha256']!=ev.REPRESENTATIVE_CHECKPOINT_HASH:raise ValueError('retained original challenge identity mismatch')
        verify_snapshot()
        publish(OUT/'complete.json',{'status':'COMPLETE','plan_logical_sha256':plan['plan_logical_sha256'],'training':results,'new_model_challenge_summaries':summaries,'retained_moderate_challenge_summary_sha256':file_hash(ROOT/'results/final_nonsealed_evaluation_v2_1b/challenge/summary.json'),'model_scenario_pair_count':432,'controller_condition_record_count':864,'sealed_opened':False,'ood_opened':False,'primary_reselected':False})
        print('SUPPLEMENTARY COMPARISON COMPLETE',flush=True)

if __name__=='__main__':main()
