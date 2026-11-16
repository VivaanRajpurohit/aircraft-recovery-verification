from __future__ import annotations

import importlib
import json
from pathlib import Path
import struct
import sys

import pytest

from aircraft_recovery.visualization.aircraft_animation import (
    AircraftNodeAnimator,
    animation_angles,
)
from aircraft_recovery.visualization.replay_3d import (
    Replay3DViewer,
    load_replay_3d_settings,
    validate_model,
)


ROOT = Path("assets/aircraft/generated_plane")
MODEL = ROOT / "generated_plane.glb"
MANIFEST = ROOT / "generation_manifest.json"
REQUIRED = {
    "fuselage", "nose", "cockpit_windows", "side_windows", "left_wing",
    "right_wing", "left_aileron", "right_aileron", "horizontal_stabilizer",
    "elevator", "vertical_stabilizer", "rudder", "left_engine", "right_engine",
    "left_propeller", "right_propeller", "left_flap", "right_flap",
}


def _glb_json() -> dict:
    data = MODEL.read_bytes()
    magic, version, _ = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF" and version == 2
    length, kind = struct.unpack_from("<II", data, 12)
    assert kind == 0x4E4F534A
    return json.loads(data[20:20 + length].decode("utf-8"))


def _load_model():
    from panda3d.core import Filename, loadPrcFileData
    loadPrcFileData("", "window-type none")
    from direct.showbase.ShowBase import ShowBase
    base = ShowBase(windowType="none")
    model = base.loader.loadModel(Filename.fromOsSpecific(str(MODEL.resolve())), noCache=True)
    return base, model


def test_generated_manifest_and_geometry_contract() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(manifest["object_names"]) == REQUIRED
    assert manifest["object_names_unique"] is True
    assert 10_000 <= manifest["triangle_count"] <= 50_000
    assert manifest["landing_gear_state"] == "fixed_retracted"
    assert manifest["external_landing_gear_objects"] == []
    assert manifest["external_textures"] == []
    assert manifest["animation_baked"] is False
    assert manifest["axis_convention"]["forward"] == "+Y"
    assert manifest["axis_convention"]["up"] == "+Z"


def test_glb_is_self_contained_and_has_window_materials() -> None:
    gltf = _glb_json()
    assert not gltf.get("images") and not gltf.get("textures")
    materials = {row["name"] for row in gltf["materials"]}
    assert {"window_blue_gray", "airframe_white", "control_surface_gray"} <= materials
    material_names = [row["name"] for row in gltf["materials"]]
    nodes = {row["name"]: row for row in gltf["nodes"]}
    for name in ("cockpit_windows", "side_windows"):
        mesh = gltf["meshes"][nodes[name]["mesh"]]
        used = {material_names[primitive["material"]] for primitive in mesh["primitives"]}
        assert used == {"window_blue_gray"}


def test_panda_loads_all_unique_named_nodes_and_hinge_origins() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    base, model = _load_model()
    try:
        assert not model.isEmpty()
        for name in REQUIRED:
            assert model.findAllMatches(f"**/{name}").getNumPaths() == 1
        for name, blender_hinge in manifest["hinge_locations_m"].items():
            # Blender +Y/+Z is imported by Panda as -Z/+Y before the model offset.
            expected = (blender_hinge[0], blender_hinge[2], -blender_hinge[1])
            assert tuple(model.find(f"**/{name}").getPos(model)) == pytest.approx(expected, abs=2e-3)
        assert not any(token in node.getName().lower() for token in ("gear", "wheel") for node in model.findAllMatches("**"))
    finally:
        base.destroy()


def test_control_mapping_and_independent_propellers() -> None:
    settings = load_replay_3d_settings("configs/visualization/generated_plane.yaml").aircraft.animation
    values = animation_angles({"aileron": 1, "elevator": 1, "rudder": 1}, settings)
    assert values["left_aileron"] == -20 and values["right_aileron"] == 20
    assert values["elevator"] == -25 and values["rudder"] == 25
    base, model = _load_model()
    try:
        animator = AircraftNodeAnimator(model, settings)
        assert animator.available and not animator.missing_nodes
        assert len({node.node() for node in animator.nodes.values()}) == len(animator.nodes)
        animator.apply({"aileron": 1, "elevator": 1, "rudder": 1,
                        "throttle_left": 0, "throttle_right": 1}, .1)
        assert animator.nodes["left_aileron"].getP() == pytest.approx(-20)
        assert animator.nodes["right_aileron"].getP() == pytest.approx(20)
        assert animator.nodes["elevator"].getP() == pytest.approx(-25)
        assert animator.nodes["rudder"].getR() == pytest.approx(25)
        animator.apply({"flap": 1, "throttle_left": 0, "throttle_right": 1}, 0.0)
        assert animator.nodes["left_flap"].getP() == pytest.approx(-30)
        assert animator.nodes["right_flap"].getP() == pytest.approx(-30)
        assert animator.propeller_degrees["left_propeller"] != animator.propeller_degrees["right_propeller"]
        assert animator.nodes["left_propeller"].getH() != animator.nodes["right_propeller"].getH()
        animator.reset()
        assert animator.nodes["left_aileron"].getP() == pytest.approx(0)
    finally:
        base.destroy()


def test_generated_profile_orientation_dimensions_and_replay_contract() -> None:
    settings = load_replay_3d_settings("configs/visualization/generated_plane.yaml")
    result = validate_model(settings)
    assert result["loaded"] and result["nose_points_forward"] and result["model_up_is_world_up"]
    assert result["dimensions_m"]["wingspan_m"] == pytest.approx(14.057, rel=1e-3)
    run = Path("results/phase5_quick/runs/phase2_engine_aileron_5103_monitored")
    if not run.is_dir():
        pytest.skip("recorded replay fixture unavailable")
    viewer = Replay3DViewer(run, settings, window_type="none", start_paused=True)
    try:
        assert viewer.model_loaded and viewer.aircraft_animator.available
        assert "executed_controls" in viewer.frames[0]
    finally:
        viewer.destroy()


def test_generator_is_original_and_animation_test_has_no_import_time_window() -> None:
    source = Path("scripts/generate_animated_aircraft.py").read_text(encoding="utf-8")
    assert "PLANE.obj" not in source and "PLANE.mtl" not in source
    before = set(sys.modules)
    importlib.import_module("scripts.test_generated_aircraft_animation")
    added = set(sys.modules) - before
    assert "direct.showbase.ShowBase" not in added
