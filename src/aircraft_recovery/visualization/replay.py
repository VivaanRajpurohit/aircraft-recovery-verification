"""Static multi-panel replay for one simulator run directory."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

def render_replay(run_directory: str | Path, output: str | Path | None = None, show: bool = False) -> Path:
    if not show: matplotlib.use("Agg", force=True)
    run=Path(run_directory); rows=[json.loads(x) for x in (run/"history.jsonl").read_text(encoding="utf-8").splitlines()]; summary=json.loads((run/"summary.json").read_text(encoding="utf-8"))
    if not rows: raise ValueError("Replay requires at least one history row")
    t=[r["next_observation"]["timestamp_seconds"] for r in rows]; states=[r["next_observation"]["aircraft_state"] for r in rows]; env=[r["next_observation"]["flight_envelope"] for r in rows]
    fig,axs=plt.subplots(4,2,figsize=(14,13))
    axs[0,0].plot(t,[s["altitude_ft"] for s in states]); axs[0,0].set_ylabel("Altitude (ft)")
    axs[0,1].plot(t,[s["airspeed_kts"] for s in states],label="airspeed"); axs[0,1].plot(t,[e["stall_margin_kts"] for e in env],label="stall margin"); axs[0,1].set_ylabel("Speed / margin (kt)"); axs[0,1].legend()
    axs[1,0].plot(t,[s["pitch_deg"] for s in states],label="pitch"); axs[1,0].plot(t,[s["roll_deg"] for s in states],label="roll"); axs[1,0].axhline(30,color="r",ls="--",alpha=.4); axs[1,0].axhline(-30,color="r",ls="--",alpha=.4); axs[1,0].set_ylabel("Attitude (deg)"); axs[1,0].legend()
    axs[1,1].plot(t,[s["heading_deg"] for s in states],label="heading"); axs[1,1].plot(t,[s["vertical_speed_fpm"] for s in states],label="vertical speed"); axs[1,1].set_ylabel("deg / ft min$^{-1}$"); axs[1,1].legend()
    controls=("elevator","aileron","rudder","throttle_left","throttle_right")
    for c in controls:
        axs[2,0].plot(t,[r["final_action"]["control_commands"][c] for r in rows],label=c)
        if any(r["proposed_action"]["control_commands"][c] != r["final_action"]["control_commands"][c] for r in rows): axs[2,0].plot(t,[r["proposed_action"]["control_commands"][c] for r in rows],ls=":",alpha=.5)
    axs[2,0].set_ylabel("Command"); axs[2,0].legend(ncol=2,fontsize=8)
    status=[(r.get("monitor_decision") or {}).get("status","unmonitored") for r in rows]; mapping={v:i for i,v in enumerate(sorted(set(status)))}; axs[2,1].step(t,[mapping[v] for v in status],where="post"); axs[2,1].set_yticks(list(mapping.values()),list(mapping)); axs[2,1].set_ylabel("Monitor status")
    axs[3,0].plot([s.get("longitude_deg",r["next_observation"]["navigation"]["longitude_deg"]) for s,r in zip(states,rows)],[s.get("latitude_deg",r["next_observation"]["navigation"]["latitude_deg"]) for s,r in zip(states,rows)]); axs[3,0].set_xlabel("Longitude (deg)"); axs[3,0].set_ylabel("Latitude (deg)")
    modes=[r["final_action"]["emergency_mode"] for r in rows]; mm={v:i for i,v in enumerate(sorted(set(modes)))}; axs[3,1].step(t,[mm[v] for v in modes],where="post"); axs[3,1].set_yticks(list(mm.values()),list(mm)); axs[3,1].set_xlabel("Time (s)"); axs[3,1].set_ylabel("Emergency mode")
    selected=next((r["final_action"]["navigation_commands"]["selected_airport"] for r in rows if r["final_action"]["navigation_commands"]["selected_airport"]),None)
    failures=sorted({f for r in rows for f in r["next_observation"].get("active_failures",[])})
    fig.suptitle(f"Replay: {run.name} | termination={summary['termination_reason']} | selected airport={selected} | failures={failures or ['none']}")
    destination=Path(output) if output else run/"replay.png"; destination.parent.mkdir(parents=True,exist_ok=True); fig.tight_layout(rect=(0,0,1,.97)); fig.savefig(destination,dpi=180,bbox_inches="tight")
    if show: plt.show()
    plt.close(fig); return destination
