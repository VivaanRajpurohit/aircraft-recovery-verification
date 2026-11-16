from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from aircraft_recovery.config import AircraftConfig, ScenarioConfig, load_config
from aircraft_recovery.models import (
    ControlCommands,
    ControllerOutput,
    NavigationCommands,
    SafetyDecision,
    StabilizationTargets,
)
from aircraft_recovery.simulator import SimpleAircraftSimulator
from aircraft_recovery.visualization.live_3d import (
    Live3DViewer,
    LiveSimulationLoop,
    format_live_hud,
)
from aircraft_recovery.visualization.replay_3d import Replay3DSettings


class RecordingController:
    name = "recording_constant"

    def __init__(self) -> None:
        self.timestamps: list[float] = []
        self.observations: list[dict] = []

    def reset(self) -> None:
        self.timestamps.clear()
        self.observations.clear()

    def act(self, observation):
        self.timestamps.append(observation.timestamp_seconds)
        self.observations.append(observation.model_dump(mode="json"))
        state = observation.aircraft_state
        elevator = max(-1.0, .7 - .1 * observation.timestamp_seconds)
        return ControllerOutput(
            timestamp_seconds=observation.timestamp_seconds,
            control_commands=ControlCommands(
                elevator=elevator, aileron=.7, rudder=.2,
                throttle_left=.8, throttle_right=.8,
            ),
            stabilization_targets=StabilizationTargets(
                target_pitch_deg=0.0, target_roll_deg=0.0,
                target_airspeed_kts=140.0, target_altitude_ft=state.altitude_ft,
            ),
            navigation_commands=NavigationCommands(target_heading_deg=state.heading_deg),
            emergency_mode="normal",
            safety_decision=SafetyDecision(
                status="not_checked", ai_action_accepted=True,
                fallback_activated=False, explanation="test controller",
            ),
            confidence=1.0,
        )


def make_loop(scenario_path="configs/scenarios/phase2_engine_aileron.yaml", paused=False):
    aircraft = load_config("configs/aircraft/simple_twin.yaml", AircraftConfig)
    scenario = load_config(scenario_path, ScenarioConfig)
    controller = RecordingController()
    return LiveSimulationLoop(
        SimpleAircraftSimulator(aircraft, scenario), controller, scenario.seed,
        start_paused=paused,
    ), controller


def test_fixed_timestep_is_independent_of_render_rate():
    fine, _ = make_loop()
    coarse, _ = make_loop()
    for _ in range(20):
        fine.advance_wall_time(.05)
    coarse.advance_wall_time(1.0)
    assert fine.physics_steps == coarse.physics_steps == 10
    assert fine.observation.timestamp_seconds == pytest.approx(1.0)
    assert fine.observation.model_dump() == coarse.observation.model_dump()


def test_realtime_synchronization_pause_and_resume():
    loop, _ = make_loop(paused=True)
    loop.advance_wall_time(1.0)
    assert loop.physics_steps == 0
    loop.toggle_pause()
    for _ in range(100):
        loop.advance_wall_time(.01)
    assert loop.physics_steps == 10
    assert loop.realtime_ratio == pytest.approx(1.0)
    assert abs(loop.wall_clock_lag_seconds) < 1e-9


def test_restart_resets_physics_and_controller_state():
    loop, controller = make_loop()
    loop.advance_wall_time(.3)
    assert controller.timestamps == pytest.approx([0.0, .1, .2])
    loop.restart(preserve_pause=False)
    assert loop.physics_steps == 0
    assert loop.observation.timestamp_seconds == 0.0
    assert controller.timestamps == []
    loop.advance_wall_time(.1)
    assert controller.timestamps == [0.0]


def test_controller_starts_on_first_step_and_failure_timing_is_exact():
    loop, controller = make_loop()
    loop.advance_wall_time(2.1)
    assert controller.timestamps[0] == 0.0
    failure_records = [
        record for record in loop.records
        if record["observation"]["active_failures"]
    ]
    assert failure_records[0]["simulation_timestamp_seconds"] == pytest.approx(2.0)


def test_post_action_state_is_remeasured_for_every_next_prediction():
    loop, controller = make_loop()
    loop.advance_wall_time(.5)
    assert len(controller.observations) == loop.physics_steps == 5
    for index in range(1, len(controller.observations)):
        assert controller.observations[index] == loop.records[index - 1]["next_observation"]


def test_proposed_and_executed_actions_remain_separate_under_actuator_delay():
    loop, _ = make_loop("configs/scenarios/phase2_sensor_actuator.yaml")
    loop.advance_wall_time(4.0)
    delayed = [
        record for record in loop.records
        if record["observation"]["control_health"]["actuator_delay_ms"] > 0.0
    ]
    assert delayed
    assert any(
        record["proposed_action"]["control_commands"]
        != record["executed_action_after_degradation"]
        for record in delayed
    )


def test_monitor_status_and_live_label_are_formatted():
    loop, _ = make_loop()
    record = loop.step_once()
    record["monitor_status"] = "modify"
    text = format_live_hud(record, loop)
    assert text.startswith("LIVE SIMULATION")
    assert "monitor=modify" in text
    assert "proposed:" in text and "executed:" in text


def test_import_does_not_create_window():
    result = subprocess.run(
        [sys.executable, "-c", "import aircraft_recovery.visualization.live_3d; print('imported')"],
        check=True, capture_output=True, text=True,
    )
    assert result.stdout.strip() == "imported"


def test_headless_live_loop_smoke(tmp_path):
    loop, _ = make_loop()
    emitted = []
    for delta in (.016, .033, .12, .007, .224):
        emitted.extend(loop.advance_wall_time(delta))
    assert emitted
    assert loop.physics_steps == 4
    assert loop.observation.timestamp_seconds == pytest.approx(.4)
    assert all("wall_clock_timestamp_utc" in record for record in emitted)


def test_headless_viewer_identifies_live_simulation(tmp_path):
    loop, _ = make_loop(paused=True)
    settings = Replay3DSettings()
    settings.aircraft.model_path = "assets/aircraft/cessna/does-not-exist.fbx"
    with pytest.warns(RuntimeWarning, match="primitive fallback"):
        viewer = Live3DViewer(loop, settings, tmp_path / "live", window_type="none")
    try:
        assert viewer.live_notice.getText().startswith("LIVE SIMULATION")
        assert "RECORDED REPLAY" not in viewer.live_notice.getText()
        viewer._set_camera("side")
        assert viewer.camera_mode == "side"
    finally:
        viewer.destroy()
