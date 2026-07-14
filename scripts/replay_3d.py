"""Launch the standalone interactive 3D experiment replay viewer."""
from __future__ import annotations
import argparse,json
from aircraft_recovery.visualization.replay_3d import Replay3DViewer,load_replay_3d_settings,validate_model

def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("run_directory"); p.add_argument("--config",default="configs/visualization/replay_3d.yaml"); p.add_argument("--aircraft-model"); p.add_argument("--aircraft-scale",type=float); p.add_argument("--aircraft-yaw-offset",type=float); p.add_argument("--aircraft-pitch-offset",type=float); p.add_argument("--aircraft-roll-offset",type=float); p.add_argument("--camera",choices=("chase","fixed","top"),default="chase"); p.add_argument("--speed",type=float,default=1.0); p.add_argument("--show-path",action="store_true"); p.add_argument("--show-failures",action="store_true"); p.add_argument("--show-controller-known-failures",action="store_true"); p.add_argument("--show-ground-truth-failures",action="store_true"); p.add_argument("--start-paused",action="store_true",help="Remain paused at t=0 instead of auto-starting after two seconds."); p.add_argument("--debug-coordinates",action="store_true",help="Print the first and last scaled replay coordinates."); p.add_argument("--validate-only",action="store_true",help="Headlessly validate model axes, dimensions, and replay input."); a=p.parse_args()
    if a.speed<=0: p.error("--speed must be positive")
    if a.aircraft_scale is not None and a.aircraft_scale<=0: p.error("--aircraft-scale must be positive")
    settings=load_replay_3d_settings(a.config); cfg=settings.aircraft
    if a.aircraft_model is not None: cfg.model_path=a.aircraft_model
    if a.aircraft_scale is not None: cfg.scale=a.aircraft_scale
    if a.aircraft_yaw_offset is not None: cfg.yaw_offset_deg=a.aircraft_yaw_offset
    if a.aircraft_pitch_offset is not None: cfg.pitch_offset_deg=a.aircraft_pitch_offset
    if a.aircraft_roll_offset is not None: cfg.roll_offset_deg=a.aircraft_roll_offset
    print(f"Aircraft model={cfg.model_path} scale={cfg.scale} child offsets yaw={cfg.yaw_offset_deg}° pitch={cfg.pitch_offset_deg}° roll={cfg.roll_offset_deg}° position={cfg.position_offset_m} m")
    if a.validate_only:
        from aircraft_recovery.visualization.replay_3d import load_replay_frames
        frames,summary,_=load_replay_frames(a.run_directory); result=validate_model(settings); result.update({"replay_frames":len(frames),"termination_reason":summary.get("termination_reason"),"parent_transform_mapping":"H=-heading, P=+pitch, R=+roll"}); print(json.dumps(result,indent=2)); return 0 if result["loaded"] else 2
    from panda3d.core import loadPrcFileData
    loadPrcFileData("","win-size 1200 875")
    viewer=Replay3DViewer(a.run_directory,settings,a.camera,a.speed,a.show_path,None,a.show_failures,a.show_controller_known_failures,a.show_ground_truth_failures,a.start_paused,a.debug_coordinates); print(viewer.model_message); print(f"Replay session log: {viewer.session_path}"); viewer.run(); return 0
if __name__=="__main__": raise SystemExit(main())
