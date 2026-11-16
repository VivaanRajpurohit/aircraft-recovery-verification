"""Fixed-step live simulation loop and optional Panda3D presentation.

Importing this module never creates a window. The live interface is a debugging
and demonstration surface, not formal-verification evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any
import warnings

from aircraft_recovery.visualization.replay_3d import (
    Replay3DSettings,
    local_xy_m,
    replay_parent_hpr,
)


CONTROL_NAMES = ("elevator", "aileron", "rudder", "throttle_left", "throttle_right")


def _commands(action: Any) -> dict[str, float]:
    return {
        name: float(getattr(action.control_commands, name))
        for name in CONTROL_NAMES
    }


class LiveSimulationLoop:
    """Advance control and physics at fixed dt independently of render cadence."""

    def __init__(
        self,
        simulator: Any,
        controller: Any,
        seed: int,
        speed: float = 1.0,
        start_paused: bool = False,
        log_path: str | Path | None = None,
    ) -> None:
        if speed <= 0.0:
            raise ValueError("Live simulation speed must be positive")
        self.simulator = simulator
        self.controller = controller
        self.seed = seed
        self.speed = speed
        self.paused = start_paused
        self.dt = float(simulator.aircraft.time_step_seconds)
        self.log_path = Path(log_path) if log_path is not None else None
        self.accumulator_seconds = 0.0
        self.active_wall_seconds = 0.0
        self.physics_steps = 0
        self.restart_count = 0
        self.records: list[dict[str, Any]] = []
        self.controller_latencies_ms: list[float] = []
        self.neural_inference_latencies_ms: list[float] = []
        self.maximum_wall_clock_lag_seconds = 0.0
        self.observation = self._reset_components()

    def _reset_components(self) -> Any:
        reset = getattr(self.controller, "reset", None)
        if callable(reset):
            reset()
        return self.simulator.reset(self.seed)

    @property
    def simulated_elapsed_seconds(self) -> float:
        return self.physics_steps * self.dt

    @property
    def wall_clock_lag_seconds(self) -> float:
        return self.active_wall_seconds * self.speed - self.simulated_elapsed_seconds

    @property
    def realtime_ratio(self) -> float:
        return self.simulated_elapsed_seconds / self.active_wall_seconds if self.active_wall_seconds else 0.0

    @property
    def mean_controller_latency_ms(self) -> float:
        if not self.controller_latencies_ms:
            return 0.0
        return sum(self.controller_latencies_ms) / len(self.controller_latencies_ms)

    @property
    def mean_neural_inference_latency_ms(self) -> float:
        if not self.neural_inference_latencies_ms:
            return 0.0
        return sum(self.neural_inference_latencies_ms) / len(self.neural_inference_latencies_ms)

    def set_paused(self, paused: bool) -> None:
        self.paused = paused

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def restart(self, preserve_pause: bool = True) -> None:
        was_paused = self.paused
        self.accumulator_seconds = 0.0
        self.active_wall_seconds = 0.0
        self.physics_steps = 0
        self.records.clear()
        self.controller_latencies_ms.clear()
        self.neural_inference_latencies_ms.clear()
        self.maximum_wall_clock_lag_seconds = 0.0
        self.observation = self._reset_components()
        self.restart_count += 1
        self.paused = was_paused if preserve_pause else False
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")

    def advance_wall_time(self, render_delta_seconds: float) -> list[dict[str, Any]]:
        """Consume elapsed wall time and run every due fixed physics step."""
        if render_delta_seconds < 0.0:
            raise ValueError("Render delta cannot be negative")
        if self.paused or self.simulator.is_terminal():
            return []
        self.active_wall_seconds += render_delta_seconds
        self.accumulator_seconds += render_delta_seconds * self.speed
        emitted: list[dict[str, Any]] = []
        while self.accumulator_seconds + 1e-12 >= self.dt and not self.simulator.is_terminal():
            emitted.append(self.step_once())
            self.accumulator_seconds -= self.dt
        self.maximum_wall_clock_lag_seconds = max(
            self.maximum_wall_clock_lag_seconds, self.wall_clock_lag_seconds,
        )
        return emitted

    def step_once(self) -> dict[str, Any]:
        """Run exactly one inference and one fixed simulator step."""
        if self.simulator.is_terminal():
            raise RuntimeError("Cannot step a terminal live simulation")
        observation = self.observation
        truth_before = self.simulator.audit_snapshot()
        started = perf_counter()
        final_action = self.controller.act(observation)
        latency_ms = (perf_counter() - started) * 1000.0
        neural_controller = getattr(self.controller, "neural", self.controller)
        neural_latency_ms = float(getattr(neural_controller, "last_inference_latency_ms", latency_ms))
        proposed_action = getattr(self.controller, "last_proposed_action", None) or final_action
        monitor_decision = getattr(self.controller, "last_monitor_decision", None)
        next_observation = self.simulator.step(final_action)
        truth_after = self.simulator.audit_snapshot()
        record = {
            "simulation_timestamp_seconds": float(observation.timestamp_seconds),
            "next_simulation_timestamp_seconds": float(next_observation.timestamp_seconds),
            "wall_clock_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "wall_clock_monotonic_seconds": perf_counter(),
            "controller_mode": final_action.emergency_mode,
            "controller_latency_ms": latency_ms,
            "neural_inference_latency_ms": neural_latency_ms,
            "observation": observation.model_dump(mode="json"),
            "next_observation": next_observation.model_dump(mode="json"),
            "proposed_action": proposed_action.model_dump(mode="json"),
            "final_action": final_action.model_dump(mode="json"),
            "executed_action_after_degradation": dict(truth_after["applied_control_commands"]),
            "monitor_status": monitor_decision.status if monitor_decision is not None else final_action.safety_decision.status,
            "monitor_decision": monitor_decision.to_dict() if monitor_decision is not None else None,
            "active_failures": list(next_observation.active_failures),
            "ground_truth_before": truth_before,
            "ground_truth_after": truth_after,
        }
        self.records.append(record)
        self.controller_latencies_ms.append(latency_ms)
        self.neural_inference_latencies_ms.append(neural_latency_ms)
        self.physics_steps += 1
        self.observation = next_observation
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        return record

    def summary(self) -> dict[str, Any]:
        return {
            "physics_timestep_seconds": self.dt,
            "physics_steps": self.physics_steps,
            "simulated_elapsed_seconds": self.simulated_elapsed_seconds,
            "active_wall_seconds": self.active_wall_seconds,
            "requested_speed": self.speed,
            "realtime_ratio": self.realtime_ratio,
            "wall_clock_lag_seconds": self.wall_clock_lag_seconds,
            "maximum_wall_clock_lag_seconds": self.maximum_wall_clock_lag_seconds,
            "mean_controller_latency_ms": self.mean_controller_latency_ms,
            "mean_neural_inference_latency_ms": self.mean_neural_inference_latency_ms,
            "restart_count": self.restart_count,
            "termination_reason": self.simulator.termination_reason(),
        }


def format_live_hud(record: dict[str, Any] | None, loop: LiveSimulationLoop) -> str:
    if record is None:
        state = loop.observation.aircraft_state
        environment = loop.observation.environment
        failures = loop.observation.active_failures
        mode, monitor, proposed, executed = "waiting", "not_checked", {}, {}
        sim_time = loop.observation.timestamp_seconds
    else:
        state_data = record["next_observation"]["aircraft_state"]
        environment_data = record["next_observation"]["environment"]
        state, environment = type("State", (), state_data), type("Environment", (), environment_data)
        failures = record["active_failures"]
        mode, monitor = record["controller_mode"], record["monitor_status"]
        proposed = record["proposed_action"]["control_commands"]
        executed = record["executed_action_after_degradation"]
        sim_time = record["next_simulation_timestamp_seconds"]

    def compact(values: dict[str, float]) -> str:
        if not values:
            return "waiting"
        return " ".join(f"{name[:3]}={values[name]:+.2f}" for name in CONTROL_NAMES)

    return (
        "LIVE SIMULATION\n"
        f"sim={sim_time:6.2f}s  lag={loop.wall_clock_lag_seconds:+.3f}s  "
        f"mode={mode}  monitor={monitor}\n"
        f"airspeed={state.airspeed_kts:6.1f} kt  pitch={state.pitch_deg:+6.1f} deg  "
        f"bank={state.roll_deg:+6.1f} deg\n"
        f"vertical speed={state.vertical_speed_fpm:+7.0f} fpm  "
        f"clearance={environment.terrain_clearance_ft:7.0f} ft\n"
        f"proposed: {compact(proposed)}\n"
        f"executed: {compact(executed)}\n"
        f"active failures: {', '.join(failures) if failures else 'none'}"
    )


class Live3DViewer:
    """Panda3D shell around :class:`LiveSimulationLoop`."""

    def __init__(
        self,
        loop: LiveSimulationLoop,
        settings: Replay3DSettings,
        output_directory: str | Path,
        window_type: str | None = None,
    ) -> None:
        from panda3d.core import loadPrcFileData

        loadPrcFileData("", "window-title Aircraft Recovery LIVE SIMULATION\nsync-video 1")
        from direct.showbase.ShowBase import ShowBase

        self.base = ShowBase(windowType=window_type)
        self.base.disableMouse()
        if self.base.camera is None:
            from panda3d.core import PandaNode
            self.base.camera = self.base.render.attachNewNode(PandaNode("live-headless-camera"))
        self.loop = loop
        self.settings = settings
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.camera_mode = "chase"
        self.render_frames = 0
        self.render_wall_seconds = 0.0
        self.finalized = False
        self.last_record: dict[str, Any] | None = None
        first = loop.observation.navigation
        self.origin_lat = first.latitude_deg
        self.origin_lon = first.longitude_deg
        airports = loop.simulator.scenario.airports
        self.ground_elevation_ft = float(airports[0].elevation_ft) if airports else 0.0
        self.aircraft_parent = self.base.render.attachNewNode("aircraft_live_transform")
        self.model_node, self.model_loaded, self.model_message = self._load_aircraft_child()
        from aircraft_recovery.visualization.aircraft_animation import AircraftNodeAnimator
        self.aircraft_animator = AircraftNodeAnimator(self.model_node, settings.aircraft.animation)
        self._setup_world()
        self._setup_light()
        self._setup_ui()
        self._update_aircraft()
        self.base.accept("escape", self._quit)
        self.base.accept("space", self._toggle_pause)
        self.base.accept("r", self._restart)
        self.base.accept("1", self._set_camera, ["chase"])
        self.base.accept("2", self._set_camera, ["side"])
        self.base.accept("3", self._set_camera, ["top"])
        self.base.taskMgr.add(self._update, "live-simulation-update")

    def _load_aircraft_child(self) -> tuple[Any, bool, str]:
        from panda3d.core import Filename, Material

        cfg = self.settings.aircraft
        path = Path(cfg.model_path).resolve()
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            model = self.base.loader.loadModel(Filename.fromOsSpecific(str(path)), noCache=True)
            if model.isEmpty():
                raise RuntimeError("Panda3D returned an empty model")
            model.reparentTo(self.aircraft_parent)
            model.setScale(cfg.scale)
            model.setHpr(cfg.yaw_offset_deg, cfg.pitch_offset_deg, cfg.roll_offset_deg)
            model.setPos(*cfg.position_offset_m)
            if model.findAllTextures().getNumTextures() == 0 and not cfg.animation.enabled:
                material = Material("neutral-live-aircraft")
                material.setDiffuse(cfg.neutral_color_rgba)
                model.setMaterial(material, 1)
                model.setColor(*cfg.neutral_color_rgba)
            return model, True, f"Loaded live aircraft model: {path}"
        except Exception as error:
            message = f"Could not load aircraft model '{path}' ({type(error).__name__}: {error}); using primitive fallback aircraft."
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            root = self.aircraft_parent.attachNewNode("primitive_fallback_aircraft")
            for name, pos, scale in (
                ("fuselage", (0, 0, 0), (.65, 4, .65)),
                ("wing", (0, .2, .15), (5.2, .65, .12)),
                ("tail", (0, -3.2, .45), (2.2, .45, .10)),
            ):
                node = self.base.loader.loadModel("models/misc/rgbCube")
                node.setName(name)
                node.reparentTo(root)
                node.setPos(*pos)
                node.setScale(*scale)
                node.setColor(.72, .74, .78, 1)
            return root, False, message

    def _setup_world(self) -> None:
        from panda3d.core import CardMaker

        size = self.settings.world.ground_size_m / 2.0
        maker = CardMaker("live-ground")
        maker.setFrame(-size, size, -size, size)
        ground = self.base.render.attachNewNode(maker.generate())
        ground.setP(-90)
        ground.setColor(.16, .23, .18, 1)
        ground.setTwoSided(True)

    def _setup_light(self) -> None:
        from panda3d.core import AmbientLight, DirectionalLight

        ambient = AmbientLight("live-ambient")
        ambient.setColor((.48, .48, .52, 1))
        self.base.render.setLight(self.base.render.attachNewNode(ambient))
        sun = DirectionalLight("live-sun")
        sun.setColor((.9, .88, .82, 1))
        sun_node = self.base.render.attachNewNode(sun)
        sun_node.setHpr(-30, -55, 0)
        self.base.render.setLight(sun_node)
        self.base.setBackgroundColor(.45, .68, .9, 1)

    def _setup_ui(self) -> None:
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode

        self.live_notice = OnscreenText(
            parent=self.base.aspect2d,
            text="LIVE SIMULATION\nDEMONSTRATION / DEBUGGING - NOT FORMAL EVIDENCE",
            pos=(0, .93), scale=.034, align=TextNode.ACenter,
            fg=(1, .9, .3, 1), shadow=(0, 0, 0, 1),
        )
        self.hud = OnscreenText(
            parent=self.base.a2dTopLeft, text="", pos=(.04, -.08), scale=.032,
            align=TextNode.ALeft, mayChange=True, fg=(.94, .97, 1, 1),
            shadow=(0, 0, 0, 1),
        )
        self.help = OnscreenText(
            parent=self.base.a2dBottomLeft,
            text="Space pause/resume   R restart   1 chase   2 side   3 top   Esc quit",
            pos=(.04, .05), scale=.028, align=TextNode.ALeft, fg=(.9, .93, .96, 1),
        )
        self.hud.setText(format_live_hud(None, self.loop))

    def _position(self) -> tuple[float, float, float]:
        observation = self.loop.observation
        east, north = local_xy_m(
            observation.navigation.latitude_deg, observation.navigation.longitude_deg,
            self.origin_lat, self.origin_lon,
        )
        world = self.settings.world
        altitude = max(
            0.0,
            (observation.aircraft_state.altitude_ft - self.ground_elevation_ft)
            * .3048 * world.vertical_scale,
        )
        return east * world.horizontal_scale, north * world.horizontal_scale, altitude

    def _update_aircraft(self, delta_seconds: float = 0.0) -> None:
        state = self.loop.observation.aircraft_state
        position = self._position()
        self.aircraft_parent.setPos(*position)
        self.aircraft_parent.setHpr(*replay_parent_hpr(state.heading_deg, state.pitch_deg, state.roll_deg))
        executed = self.last_record["executed_action_after_degradation"] if self.last_record else {}
        self.aircraft_animator.apply(executed, 0.0 if self.loop.paused else delta_seconds)
        self._update_camera(position, state.heading_deg)

    def _update_camera(self, position: tuple[float, float, float], heading_deg: float) -> None:
        heading = math.radians(heading_deg)
        forward = (math.sin(heading), math.cos(heading), 0.0)
        right = (math.cos(heading), -math.sin(heading), 0.0)
        if self.camera_mode == "side":
            camera = (position[0] + right[0] * 24, position[1] + right[1] * 24, position[2] + 5)
        elif self.camera_mode == "top":
            camera = (position[0], position[1], position[2] + 55)
        else:
            cfg = self.settings.camera
            camera = (
                position[0] - forward[0] * cfg.chase_distance_m,
                position[1] - forward[1] * cfg.chase_distance_m,
                position[2] + cfg.chase_height_m,
            )
        self.base.camera.setPos(*camera)
        self.base.camera.lookAt(*position)

    def _update(self, task: Any) -> Any:
        from direct.task import Task
        from panda3d.core import ClockObject

        delta = max(0.0, ClockObject.getGlobalClock().getDt())
        emitted = self.loop.advance_wall_time(delta)
        if emitted:
            self.last_record = emitted[-1]
        self.render_frames += 1
        self.render_wall_seconds += delta
        self._update_aircraft(delta)
        self.hud.setText(format_live_hud(self.last_record, self.loop))
        if self.loop.simulator.is_terminal():
            self._finalize()
            return Task.done
        return Task.cont

    def _toggle_pause(self) -> None:
        self.loop.toggle_pause()

    def _restart(self) -> None:
        self.loop.restart()
        self.aircraft_animator.reset()
        self.last_record = None
        first = self.loop.observation.navigation
        self.origin_lat, self.origin_lon = first.latitude_deg, first.longitude_deg
        self._update_aircraft()

    def _set_camera(self, mode: str) -> None:
        self.camera_mode = mode

    def _finalize(self) -> dict[str, Any]:
        if self.finalized:
            return json.loads(
                (self.output_directory / "live_summary.json").read_text(encoding="utf-8")
            )
        summary = {
            **self.loop.summary(),
            "render_frames": self.render_frames,
            "render_fps": self.render_frames / self.render_wall_seconds if self.render_wall_seconds else 0.0,
            "model_loaded": self.model_loaded,
            "viewer_label": "LIVE SIMULATION",
        }
        (self.output_directory / "live_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        self.finalized = True
        return summary

    def _quit(self) -> None:
        self._finalize()
        self.base.userExit()

    def run(self) -> None:
        self.base.run()

    def destroy(self) -> None:
        self._finalize()
        self.base.destroy()
