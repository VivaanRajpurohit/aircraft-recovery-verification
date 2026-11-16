from __future__ import annotations

import json
from pathlib import Path
import struct

import pytest

from aircraft_recovery.visualization.aircraft_animation import AircraftNodeAnimator
from aircraft_recovery.visualization.replay_3d import load_replay_3d_settings, validate_model


ROOT = Path("assets/aircraft/generated_plane_v2")
MODEL = ROOT / "generated_plane_v2.glb"
MANIFEST = ROOT / "generation_manifest.json"
CONFIG = "configs/visualization/generated_plane_v2.yaml"


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


def test_v2_nose_is_continuous_symmetric_and_part_of_fuselage() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    checks = manifest["nose_geometry_validation"]
    assert manifest["model_version"] == 2
    assert checks["cross_section_count"] == 10
    assert checks["nose_connected_to_fuselage_ring"] is True
    assert checks["single_visible_fuselage_mesh"] is True
    assert checks["separate_visible_nose_mesh_removed"] is True
    assert checks["duplicate_overlapping_faces"] is False
    assert checks["zero_area_faces"] is False
    assert checks["outward_side_normals"] is True
    assert checks["nose_centerline_symmetric"] is True
    assert checks["nose_width_within_fuselage_width"] is True
    assert checks["nose_length_m"] == pytest.approx(3.5)
    assert checks["nose_maximum_width_excluding_interface_m"] == pytest.approx(2.1)
    assert checks["fuselage_maximum_width_m"] == pytest.approx(2.6)

    gltf = _glb_json()
    nodes = {node["name"]: node for node in gltf["nodes"]}
    assert "mesh" in nodes["fuselage"]
    assert "mesh" not in nodes["nose"]


def test_v2_proportions_materials_and_no_external_assets() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert 10_000 <= manifest["triangle_count"] <= 50_000
    assert manifest["external_textures"] == []
    assert manifest["landing_gear_state"] == "fixed_retracted"
    assert manifest["external_landing_gear_objects"] == []
    assert all(manifest["proportion_validation"][name] for name in (
        "fuselage_length_in_target", "wingspan_in_target", "fuselage_width_in_target",
    ))
    assert manifest["proportion_validation"]["engine_intersections_severe"] is False
    assert manifest["proportion_validation"]["propeller_intersections_severe"] is False
    assert manifest["symmetry_and_intersection_validation"]["zero_area_meshes"] == []
    assert manifest["symmetry_and_intersection_validation"]["propeller_dimensions_match"] is True
    gltf = _glb_json()
    assert not gltf.get("images") and not gltf.get("textures")


def test_v2_loads_in_panda_and_preserves_animation_contract() -> None:
    settings = load_replay_3d_settings(CONFIG)
    result = validate_model(settings)
    assert result["loaded"] and result["nose_points_forward"] and result["model_up_is_world_up"]
    assert result["dimensions_m"]["wingspan_m"] == pytest.approx(19.984, rel=1e-3)
    base, model = _load_model()
    try:
        assert not model.isEmpty()
        animator = AircraftNodeAnimator(model, settings.aircraft.animation)
        assert animator.available and not animator.missing_nodes
        animator.apply({"aileron": 1, "elevator": 1, "rudder": 1,
                        "throttle_left": 0, "throttle_right": 1}, .1)
        assert animator.nodes["left_aileron"].getP() == pytest.approx(-20)
        assert animator.nodes["right_aileron"].getP() == pytest.approx(20)
        assert animator.nodes["elevator"].getP() == pytest.approx(-25)
        assert animator.nodes["rudder"].getR() == pytest.approx(25)
        assert animator.nodes["left_propeller"].getH() != animator.nodes["right_propeller"].getH()
        animator.apply({"flap": 1, "throttle_left": .5, "throttle_right": .5}, .1)
        assert animator.nodes["left_flap"].getP() == pytest.approx(-30)
        assert animator.nodes["right_flap"].getP() == pytest.approx(-30)
    finally:
        base.destroy()


def test_v1_artifacts_are_preserved() -> None:
    v1 = Path("assets/aircraft/generated_plane")
    assert (v1 / "generated_plane.blend").is_file()
    assert (v1 / "generated_plane.glb").is_file()
    assert (v1 / "generation_manifest.json").is_file()
