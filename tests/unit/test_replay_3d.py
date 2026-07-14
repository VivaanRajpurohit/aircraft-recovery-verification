from pathlib import Path
import pytest
from panda3d.core import NodePath,PandaNode,Vec3
from aircraft_recovery.visualization.replay_3d import Replay3DSettings,Replay3DViewer,load_replay_3d_settings,replay_parent_hpr,validate_model

def _direction(h,p,r,vector):
    node=NodePath(PandaNode("orientation")); node.setHpr(h,p,r); return node.getQuat().xform(vector)

def test_parent_replay_axes_match_aviation_conventions():
    east=_direction(*replay_parent_hpr(90,0,0),Vec3(0,1,0)); assert east.x>0.99
    nose_up=_direction(*replay_parent_hpr(0,10,0),Vec3(0,1,0)); assert nose_up.z>0
    right_wing=_direction(*replay_parent_hpr(0,0,10),Vec3(1,0,0)); assert right_wing.z<0

def test_default_cessna_profile_has_forward_axes_and_reasonable_scale():
    if not Path("assets/aircraft/cessna/cessna.fbx").is_file():
        pytest.skip("user-supplied Cessna FBX is intentionally excluded")
    settings=load_replay_3d_settings("configs/visualization/replay_3d.yaml"); result=validate_model(settings)
    assert result["loaded"] and result["nose_points_forward"] and result["model_up_is_world_up"]
    assert 9<result["dimensions_m"]["wingspan_m"]<12
    assert 6<result["dimensions_m"]["length_m"]<9
    assert 14<=settings.camera.chase_distance_m<=20
    assert 4<=settings.camera.chase_height_m<=7
    assert settings.camera.chase_distance_m>result["dimensions_m"]["wingspan_m"]

def test_missing_fbx_warns_and_uses_primitive_fallback():
    run=Path("results/phase5_quick/runs/phase2_engine_aileron_5103_monitored")
    if not run.exists(): pytest.skip("quick replay artifact unavailable")
    settings=Replay3DSettings(); settings.aircraft.model_path="assets/aircraft/cessna/does-not-exist.fbx"
    with pytest.warns(RuntimeWarning,match="primitive fallback"):
        viewer=Replay3DViewer(run,settings,window_type="none")
    try:
        assert not viewer.model_loaded
        assert viewer.model_node.getName()=="primitive_fallback_aircraft"
    finally: viewer.destroy()

def test_combined_failure_and_monitor_notifications_are_rendered():
    run=Path("results/phase5_quick/runs/phase2_engine_aileron_5103_monitored")
    if not run.exists(): pytest.skip("quick replay artifact unavailable")
    viewer=Replay3DViewer(run,Replay3DSettings(),window_type="none",show_failures=True,show_controller_known_failures=True,show_ground_truth_failures=True)
    try:
        viewer.event_state.advance(1.9); viewer._update_overlays(2.0)
        assert "MULTIPLE FAILURES ACTIVATED" in viewer.notification.getText()
        assert "LEFT ENGINE FAILURE" in viewer.notification.getText()
        viewer.event_state.advance(2.0); viewer._update_overlays(2.1)
        assert "NEURAL ACTION MODIFIED" in viewer.notification.getText()
        assert "Status: MODIFIED" in viewer.monitor_panel.getText()
        rejection=next(event for event in viewer.replay_data.monitor_events if event.status=="reject_fallback")
        viewer.event_state.advance(rejection.time_s-.1); viewer._update_overlays(rejection.time_s)
        assert "FALLBACK CONTROLLER ACTIVE" in viewer.notification.getText()
    finally: viewer.destroy()

def test_redesigned_hud_uses_scaled_past_trail_and_compact_monitor():
    run=Path("results/phase5_quick/runs/phase2_engine_aileron_5103_monitored")
    if not run.exists(): pytest.skip("quick replay artifact unavailable")
    viewer=Replay3DViewer(run,load_replay_3d_settings("configs/visualization/replay_3d.yaml"),window_type="none",show_path=True,show_failures=True,start_paused=True)
    try:
        first,last=viewer.points[0],viewer.points[-1]
        assert first[:2]==pytest.approx((0,0))
        assert abs(last[0])+abs(last[1])>1000
        assert abs(last[2]-first[2])<1
        assert viewer.paused and viewer.auto_start_at is None
        viewer._update_overlays(1.0)
        assert "Reason:" not in viewer.monitor_panel.getText()
        assert "Proposed:" not in viewer.monitor_panel.getText()
        viewer._toggle_monitor(); viewer._update_overlays(1.0)
        assert "Proposed:" in viewer.monitor_panel.getText()
        assert set(viewer.health_bars)=={"elevator","aileron","rudder","left_engine","right_engine"}
    finally: viewer.destroy()

def test_default_launch_has_two_second_orientation_pause():
    run=Path("results/phase5_quick/runs/phase2_engine_aileron_5103_monitored")
    if not run.exists(): pytest.skip("quick replay artifact unavailable")
    viewer=Replay3DViewer(run,Replay3DSettings(),window_type="none")
    try:
        assert viewer.paused and viewer.auto_start_at is not None
        assert viewer.replay_notice.getText()=="RECORDED SIMULATION REPLAY\nVISUALIZATION ONLY - NOT LIVE CONTROL"
    finally: viewer.destroy()

def test_replay_viewer_source_is_separated_from_live_control_stack():
    source=Path("src/aircraft_recovery/visualization/replay_3d.py").read_text(encoding="utf-8")
    forbidden=("aircraft_recovery.simulator","aircraft_recovery.controllers","aircraft_recovery.safety","import torch","simulator.step(")
    assert all(item not in source for item in forbidden)

def test_chase_zoom_orbit_reset_and_restart_persistence():
    run=Path("results/phase5_quick/runs/phase2_engine_aileron_5103_monitored")
    if not run.exists(): pytest.skip("quick replay artifact unavailable")
    viewer=Replay3DViewer(run,load_replay_3d_settings("configs/visualization/replay_3d.yaml"),window_type="none")
    try:
        viewer._zoom(-1); assert viewer.camera_distance_target==pytest.approx(16)
        viewer._restart(); assert viewer.camera_distance_target==pytest.approx(16)
        for _ in range(100): viewer._zoom(-1)
        assert viewer.camera_distance_target==6
        for _ in range(100): viewer._zoom(1)
        assert viewer.camera_distance_target==40
        viewer._apply_mouse_delta(100,-10000)
        assert viewer.orbit_yaw_deg!=0 and viewer.orbit_pitch_deg==65
        viewer._reset_camera()
        assert viewer.camera_mode=="chase" and viewer.camera_distance_target==18
        assert viewer.orbit_yaw_deg==viewer.orbit_pitch_deg==0
    finally: viewer.destroy()

def test_camera_modes_and_free_camera_controls():
    run=Path("results/phase5_quick/runs/phase2_engine_aileron_5103_monitored")
    if not run.exists(): pytest.skip("quick replay artifact unavailable")
    viewer=Replay3DViewer(run,Replay3DSettings(),window_type="offscreen")
    try:
        if viewer.base.camera is None:
            viewer.base.camera=viewer.base.render.attachNewNode(PandaNode("headless-test-camera"))
        for mode in ("chase","side","top","free","cockpit","fixed"):
            viewer._set_camera_mode(mode); assert viewer.camera_mode==mode
        viewer._set_camera_mode("free"); viewer.base.camera.setPos(0,0,10); viewer.base.camera.setHpr(0,0,0)
        viewer._apply_mouse_delta(10,-10); assert viewer.base.camera.getH()!=0 and viewer.base.camera.getP()>0
        start=viewer.base.camera.getPos(); viewer._set_key("forward",True); viewer._update_free_camera(1.0); viewer._set_key("forward",False)
        assert (viewer.base.camera.getPos()-start).length()==pytest.approx(viewer.settings.camera.free_move_speed_mps)
        initial_speed=viewer.free_move_speed; viewer._zoom(-1); assert viewer.free_move_speed>initial_speed
    finally: viewer.destroy()
