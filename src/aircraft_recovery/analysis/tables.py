"""Machine-readable and paper-ready Phase 5 tables."""
from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Any

def export_tables(batch_directory: str | Path, output_directory: str | Path | None = None) -> list[str]:
    batch=Path(batch_directory); out=Path(output_directory) if output_directory else batch/"tables"; out.mkdir(parents=True,exist_ok=True)
    analysis=json.loads((batch/"analysis.json").read_text(encoding="utf-8")); records=json.loads((batch/"batch_records.json").read_text(encoding="utf-8"))
    rows=[]
    for name,data in analysis["controllers"].items():
        rows.append({"controller":name,"n":data["n"],"recovery_rate":data["recovery_success"]["rate"],"recovery_ci_low":data["recovery_success"]["wilson_95_ci"][0],"recovery_ci_high":data["recovery_success"]["wilson_95_ci"][1],"stabilization_rate":data["stabilization_success"]["rate"],"mean_stall_margin_kts":data["minimum_stall_margin_kts"]["mean"],"mean_max_bank_deg":data["maximum_absolute_roll_deg"]["mean"],"mean_latency_ms":data["controller_latency_ms_mean"]["mean"]})
    csv_path=out/"controller_summary.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    scenario_counts={category:len({r["pair_id"] for r in records if r["scenario_category"]==category}) for category in sorted({r["scenario_category"] for r in records})}
    terminations={name:{reason:sum(r["controller"]==name and r["termination_reason"]==reason for r in records) for reason in sorted({r["termination_reason"] for r in records})} for name in ("expert","neural","monitored")}
    json_path=out/"research_tables.json"; json_path.write_text(json.dumps({"scenario_group_counts":scenario_counts,"controller_summary":rows,"full_descriptive_statistics":analysis["controllers"],"paired_effects":analysis["paired_effects"],"statistical_tests":analysis["statistical_tests"],"termination_classifications":terminations},indent=2)+"\n",encoding="utf-8")
    headers=list(rows[0]); md=["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|"]+["| "+" | ".join(str(r[h]) for h in headers)+" |" for r in rows]
    md_path=out/"controller_summary.md"; md_path.write_text("\n".join(md)+"\n",encoding="utf-8")
    latex_path=out/"controller_summary.tex"; latex_path.write_text("\\begin{tabular}{lrrrrrrrr}\n\\hline\nController & n & Recovery & CI low & CI high & Stabilized & Stall (kt) & Bank (deg) & Latency (ms) \\\\\n\\hline\n"+"\n".join(f"{r['controller']} & {r['n']} & {r['recovery_rate']:.3f} & {r['recovery_ci_low']:.3f} & {r['recovery_ci_high']:.3f} & {r['stabilization_rate']:.3f} & {r['mean_stall_margin_kts']:.2f} & {r['mean_max_bank_deg']:.2f} & {r['mean_latency_ms']:.2f} \\\\" for r in rows)+"\n\\hline\n\\end{tabular}\n",encoding="utf-8")
    solver={"monitored_runs":sum(r["controller"]=="monitored" for r in records),"timeouts":sum(r["solver_timeout_count"] for r in records if r["controller"]=="monitored"),"unknowns":sum(r["solver_unknown_count"] for r in records if r["controller"]=="monitored")}
    solver_path=out/"solver_statistics.json"; solver_path.write_text(json.dumps(solver,indent=2)+"\n",encoding="utf-8")
    counts_path=out/"scenario_group_counts.csv"
    with counts_path.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=["scenario_group","paired_instances"]); w.writeheader(); w.writerows({"scenario_group":k,"paired_instances":v} for k,v in scenario_counts.items())
    termination_path=out/"termination_classifications.csv"
    with termination_path.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=["controller","termination_reason","count"]); w.writeheader(); w.writerows({"controller":name,"termination_reason":reason,"count":count} for name,values in terminations.items() for reason,count in values.items())
    return [str(p) for p in (csv_path,json_path,md_path,latex_path,solver_path,counts_path,termination_path)]
