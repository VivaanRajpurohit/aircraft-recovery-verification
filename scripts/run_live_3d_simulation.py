"""Launch a fixed-step real-time aircraft simulation with optional Panda3D UI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter, sleep

from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config
from aircraft_recovery.controllers import (
    DeterministicFallbackController,
    FallbackConfig,
    MonitoredNeuralController,
    NeuralBaselineController,
)
from aircraft_recovery.safety import SafetyConfig
from aircraft_recovery.simulator import SimpleAircraftSimulator
from aircraft_recovery.visualization.live_3d import Live3DViewer, LiveSimulationLoop
from aircraft_recovery.visualization.replay_3d import load_replay_3d_settings


def build_controller(args: argparse.Namespace, scenario: ScenarioConfig):
    fallback_config = load_config(args.fallback_config, FallbackConfig)
    if args.controller == "expert":
        return DeterministicFallbackController(fallback_config, scenario.airports)
    if args.controller == "neural":
        return NeuralBaselineController(args.checkpoint, args.device)
    safety_config = load_config(args.safety_config, SafetyConfig)
    return MonitoredNeuralController(
        args.checkpoint, args.device, fallback_config, scenario.airports, safety_config,
    )


def run_headless_realtime(loop: LiveSimulationLoop, duration_seconds: float) -> dict[str, object]:
    """Run a bounded wall-clock smoke session using the same accumulator."""
    started = perf_counter()
    previous = started
    render_frames = 0
    while perf_counter() - started < duration_seconds and not loop.simulator.is_terminal():
        now = perf_counter()
        loop.advance_wall_time(now - previous)
        previous = now
        render_frames += 1
        sleep(.001)
    wall = perf_counter() - started
    summary = loop.summary()
    summary.update({"render_frames": render_frames, "render_fps": render_frames / wall, "measured_wall_seconds": wall})
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("expert", "neural", "monitored"), default="monitored")
    parser.add_argument("--scenario", default="configs/scenarios/phase2_engine_aileron.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/policy_smoke_v1/best.pt")
    parser.add_argument("--aircraft-config", default="configs/aircraft/simple_twin.yaml")
    parser.add_argument("--fallback-config", default="configs/controllers/fallback.yaml")
    parser.add_argument("--safety-config", default="configs/safety/default.yaml")
    parser.add_argument("--visualization-config", default="configs/visualization/replay_3d.yaml")
    parser.add_argument("--aircraft-model", help="Override only the visualization model path.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--headless", action="store_true", help="Run a bounded real-time loop without creating a window.")
    parser.add_argument("--headless-duration", type=float, default=3.2)
    args = parser.parse_args()
    if args.speed <= 0.0:
        parser.error("--speed must be positive")
    if args.headless_duration <= 0.0:
        parser.error("--headless-duration must be positive")
    return args


def main() -> int:
    args = parse_args()
    scenario = load_config(args.scenario, ScenarioConfig)
    aircraft = load_config(args.aircraft_config, AircraftConfig)
    controller = build_controller(args, scenario)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output or f"results/live_3d_sessions/{scenario.scenario_id}_{args.controller}_{stamp}")
    output.mkdir(parents=True, exist_ok=False)
    loop = LiveSimulationLoop(
        SimpleAircraftSimulator(aircraft, scenario), controller, scenario.seed,
        args.speed, args.start_paused, output / "live_history.jsonl",
    )
    metadata = {
        "interface": "LIVE SIMULATION",
        "evidence_status": "demonstration_and_debugging_only_not_formal_evidence",
        "controller": args.controller,
        "scenario": args.scenario,
        "scenario_seed": scenario.seed,
        "checkpoint": args.checkpoint if args.controller != "expert" else None,
        "physics_timestep_seconds": aircraft.time_step_seconds,
        "controller_frequency_hz": aircraft.controller_frequency_hz,
        "speed": args.speed,
        "aircraft_model": args.aircraft_model,
    }
    (output / "live_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.headless:
        if args.start_paused:
            loop.set_paused(False)
        summary = run_headless_realtime(loop, args.headless_duration)
        summary["viewer_label"] = "LIVE SIMULATION"
        (output / "live_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output_directory": str(output), **summary}, indent=2))
        return 0
    visualization_config = args.visualization_config
    if (
        args.aircraft_model is not None
        and Path(args.aircraft_model).name.lower() == "generated_plane.glb"
        and args.visualization_config == "configs/visualization/replay_3d.yaml"
    ):
        visualization_config = "configs/visualization/generated_plane.yaml"
    settings = load_replay_3d_settings(visualization_config)
    if args.aircraft_model is not None:
        settings.aircraft.model_path = args.aircraft_model
    viewer = Live3DViewer(loop, settings, output)
    print("LIVE SIMULATION - demonstration/debugging only; not formal verification evidence")
    print(viewer.model_message)
    print(f"Session directory: {output}")
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
