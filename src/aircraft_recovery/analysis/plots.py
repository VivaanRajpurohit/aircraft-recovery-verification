"""Headless, publication-oriented plots from Phase 5 flat records."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"expert": "#4C78A8", "neural": "#F58518", "monitored": "#54A24B"}

def _save(fig: Any, path: Path) -> None:
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)

def generate_plots(batch_directory: str | Path, output_directory: str | Path | None = None) -> list[str]:
    batch = Path(batch_directory); out = Path(output_directory) if output_directory else batch / "plots"; out.mkdir(parents=True, exist_ok=True)
    records = json.loads((batch / "batch_records.json").read_text(encoding="utf-8")); names = ("expert", "neural", "monitored")
    paths: list[str] = []
    def bar_binary(metric: str, title: str, filename: str) -> None:
        rates = [np.mean([r[metric] for r in records if r["controller"] == n]) for n in names]
        fig, ax = plt.subplots(figsize=(6.4, 4)); ax.bar(names, rates, color=[COLORS[n] for n in names]); ax.set_ylim(0, 1); ax.set_ylabel("Proportion"); ax.set_title(f"{title} (n={len(records)//3} paired scenarios)")
        for i,v in enumerate(rates): ax.text(i, v+.02, f"{v:.2f}", ha="center")
        path=out/filename; _save(fig,path); paths.append(str(path))
    bar_binary("recovery_success", "Recovery success", "recovery_success_by_controller.png")
    bar_binary("stabilization_success", "Stabilization success", "stabilization_by_controller.png")
    bar_binary("diversion_success", "Diversion success", "diversion_success.png")
    categories=sorted({r["scenario_category"] for r in records}); x=np.arange(len(categories)); fig,ax=plt.subplots(figsize=(max(8,len(categories)*1.1),4)); width=.25
    for i,n in enumerate(names): ax.bar(x+(i-1)*width,[np.mean([r["recovery_success"] for r in records if r["controller"]==n and r["scenario_category"]==c]) for c in categories],width,label=n,color=COLORS[n])
    ax.set_xticks(x,categories,rotation=25,ha="right"); ax.set_ylim(0,1); ax.set_ylabel("Recovery proportion"); ax.set_title("Recovery success by scenario group (n=2 seeds/group)"); ax.legend(); path=out/"recovery_success_by_group.png"; _save(fig,path); paths.append(str(path))
    for metric, ylabel, filename in (
        ("time_to_stabilization_seconds", "Time (s)", "time_to_stabilization.png"),
        ("minimum_stall_margin_kts", "Minimum stall margin (kt)", "stall_margin.png"),
        ("maximum_absolute_roll_deg", "Maximum absolute bank (deg)", "max_bank.png"),
        ("minimum_terrain_clearance_ft", "Minimum terrain clearance (ft)", "min_terrain.png"),
        ("controller_latency_ms_mean", "Mean latency (ms)", "controller_latency.png"),
    ):
        series=[[r[metric] for r in records if r["controller"]==n and r.get(metric) is not None] for n in names]
        fig,ax=plt.subplots(figsize=(6.4,4)); ax.boxplot(series,tick_labels=names,showmeans=True); ax.set_ylabel(ylabel); ax.set_title(f"{ylabel} by controller"); path=out/filename; _save(fig,path); paths.append(str(path))
    monitored=[r for r in records if r["controller"]=="monitored"]
    fig,ax=plt.subplots(figsize=(6.4,4)); labels=["modified","rejected","fallback","timeout","unknown"]; values=[sum(r[k] for r in monitored) for k in ("neural_action_modification_count","neural_action_rejection_count","fallback_activation_count","solver_timeout_count","solver_unknown_count")]; ax.bar(labels,values,color="#54A24B"); ax.set_ylabel("Action/solver events"); ax.set_title("Runtime monitor interventions"); path=out/"monitor_actions.png"; _save(fig,path); paths.append(str(path))
    fig,ax=plt.subplots(figsize=(6.4,4)); ax.boxplot([[r["monitor_latency_ms_mean"] for r in monitored],[r["solver_latency_ms_mean"] for r in monitored]],tick_labels=["monitor","solver"]); ax.set_ylabel("Mean latency per run (ms)"); ax.set_title("Monitor and solver latency"); path=out/"monitor_latency.png"; _save(fig,path); paths.append(str(path))
    reasons=sorted({r["termination_reason"] for r in records}); x=np.arange(len(reasons)); fig,ax=plt.subplots(figsize=(max(7,len(reasons)*1.2),4)); width=.25
    for i,n in enumerate(names):
        c=Counter(r["termination_reason"] for r in records if r["controller"]==n); ax.bar(x+(i-1)*width,[c[v] for v in reasons],width,label=n,color=COLORS[n])
    ax.set_xticks(x, reasons, rotation=25, ha="right"); ax.set_ylabel("Runs"); ax.set_title("Termination reasons"); ax.legend(); path=out/"termination_reasons.png"; _save(fig,path); paths.append(str(path))
    distributions=("in_distribution","out_of_distribution"); fig,ax=plt.subplots(figsize=(7,4)); width=.25; x=np.arange(2)
    for i,n in enumerate(names): ax.bar(x+(i-1)*width,[np.mean([r["recovery_success"] for r in records if r["controller"]==n and r["distribution"]==d]) for d in distributions],width,label=n,color=COLORS[n])
    ax.set_xticks(x,["ID","OOD"]); ax.set_ylim(0,1); ax.set_ylabel("Recovery proportion"); ax.set_title("ID/OOD simulator performance"); ax.legend(); path=out/"id_ood_performance.png"; _save(fig,path); paths.append(str(path))
    paired=[]
    for metric in ("minimum_stall_margin_kts","maximum_absolute_roll_deg","safety_violation_duration_seconds"):
        m={r["pair_id"]:r[metric] for r in records if r["controller"]=="monitored"}; n={r["pair_id"]:r[metric] for r in records if r["controller"]=="neural"}; paired.append([m[k]-n[k] for k in sorted(m)])
    fig,ax=plt.subplots(figsize=(7,4)); ax.boxplot(paired,tick_labels=["stall margin\n(kt)","max bank\n(deg)","violation duration\n(s)"]); ax.axhline(0,color="black",lw=.8); ax.set_ylabel("Monitored minus neural"); ax.set_title(f"Paired differences (n={len(paired[0])})"); path=out/"paired_differences.png"; _save(fig,path); paths.append(str(path))
    activation_path=Path("results/phase3_activation_comparison/activation_comparison.json")
    if activation_path.exists():
        data=json.loads(activation_path.read_text(encoding="utf-8")); labels=list(data["results"]); values=[data["results"][label]["final_validation_loss"] for label in labels]
        fig,ax=plt.subplots(figsize=(5.5,4)); ax.bar(labels,values,color=["#E45756","#72B7B2"]); ax.set_ylim(0,max(values)*1.2); ax.set_ylabel("Validation total loss"); ax.set_title("Controlled activation comparison"); path=out/"activation_comparison.png"; _save(fig,path); paths.append(str(path))
    return paths
