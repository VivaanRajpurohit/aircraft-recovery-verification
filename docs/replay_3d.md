# Standalone 3D Replay

This viewer reads existing JSON replay artifacts only. It does not import or modify simulator code, drive experiments, or write research results.

Install the optional renderer dependency:

```powershell
python -m pip install -e ".[replay3d]"
```

Launch with an existing run:

```powershell
python scripts/replay_3d.py `
  "results\phase5_quick\runs\phase2_engine_aileron_5103_monitored" `
  --aircraft-model "assets\aircraft\cessna\cessna.fbx" `
  --camera chase `
  --speed 1.0 `
  --show-path `
  --show-failures `
  --show-controller-known-failures `
  --show-ground-truth-failures
```

The requested example seed `phase2_engine_aileron_3101_monitored` is not present in the quick batch; its two available paired seeds are `5103` and `5104`.

Default FBX profile:

- scale: `0.00056`
- child yaw offset: `180°`
- child pitch offset: `90°`
- child roll offset: `0°`
- child position offset: `(0, 0, 0)` metres
- resulting approximate dimensions: 10.22 m span × 7.30 m length × 2.65 m height

The replay parent receives world position and `H=-heading`, `P=+pitch`, `R=+roll`. Only the imported FBX child receives scale and model-axis offsets. Positive pitch raises the nose, positive roll lowers the right wing, and heading 90° points east. If FBX loading fails, the viewer emits a warning and creates a canonical primitive aircraft. Missing textures cause a neutral gray material to be applied.

Configuration is in `configs/visualization/replay_3d.yaml`. The world profile separately controls `horizontal_scale` and `vertical_scale`; the trail contains only recorded points up to the current replay time. The default chase camera is 18 m behind, 6 m high, looks 6 m ahead, and uses smoothing `8.0`. Zoom is clamped from 6–40 m in configurable 2 m wheel increments. Orbit sensitivity and pitch limits, plus free-camera movement speed, are configurable in the camera profile.

The viewer opens at 1200 x 875 and holds at t=0 for two seconds before playback. Pass `--start-paused` to require manual start, or `--debug-coordinates` to print the first and last scaled world coordinates. The responsive HUD uses four dark corner panels and keeps the aircraft area clear. It has compact monitor information by default and real graphical control-health bars.

Hold the right mouse button and drag to orbit in chase mode or look around in free mode. The wheel smoothly zooms chase view and adjusts free-camera movement speed. `C` resets chase orbit and zoom. Camera modes are `1` chase, `2` side, `3` top-down, `4` free, `5` cockpit, and `6` fixed ground. Free mode uses WASD, Q/E for down/up, Shift for faster travel, and F to focus the aircraft. Switching modes and restarting preserve the last chase zoom unless `C` is pressed.

Use Space to pause, R to restart, Escape to close, H to show controls, F outside free mode to toggle failures and world markers, G to toggle hidden ground truth, M to expand monitor action details, T to toggle the past-flight trail, and Left/Right to step while paused. Each launch writes a separate JSONL session log under `results/replay_3d_sessions/`; source replay files remain read-only.
