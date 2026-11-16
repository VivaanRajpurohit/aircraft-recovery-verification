from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest

from aircraft_recovery.visualization.aircraft_animation import animation_angles
from aircraft_recovery.visualization.replay_3d import load_replay_3d_settings
from scripts.audit_generated_aircraft_v2 import validate_schema, verify_ledger


ROOT = Path("assets/aircraft/generated_plane_v2")
MANIFEST = ROOT / "generation_manifest.json"
CONFIG = "configs/visualization/generated_plane_v2.yaml"


def data(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_decision_manifest_validates_against_repository_schema() -> None:
    decisions = data("design_decisions.json")
    errors = validate_schema(decisions, data("design_decisions.schema.json"))
    assert errors == []
    ids = [row["decision_id"] for row in decisions["decisions"]]
    assert ids == [f"AIRCRAFT-DESIGN-{index:03d}" for index in range(1, 16)]
    assert all(row["physics_effect"] is False for row in decisions["decisions"])


def test_manifest_has_reproducible_provenance_and_deterministic_contract() -> None:
    manifest = data("generation_manifest.json")
    assert manifest["manifest_version"] == 3
    assert manifest["deterministic_seed"] == 737
    assert manifest["geometry_algorithm_version"] == "elliptical-stations-v2.1"
    assert manifest["material_generation_version"] == "procedural-principled-v1"
    assert manifest["generator_script_sha256"]
    assert manifest["git_branch"] and manifest["git_commit_hash"]
    assert manifest["object_names"] == list(dict.fromkeys(manifest["object_names"]))
    assert manifest["triangle_count"] == sum(manifest["triangle_counts_per_object"].values())
    assert manifest["object_dimensions_m"]["fuselage"] == pytest.approx([2.6, 14.58, 2.68])
    assert manifest["hinge_locations_m"]["left_aileron"] == [-8.1, -.6, .48]
    assert manifest["material_assignments"]["fuselage"] == ["airframe_white"]


def test_geometry_ratio_and_material_audits() -> None:
    geometry, materials = data("geometry_audit.json"), data("material_audit.json")
    manifest = data("generation_manifest.json")
    assert geometry["totals"]["zero_area_face_count"] == 0
    assert geometry["totals"]["duplicate_face_count"] == 0
    assert geometry["severe_intersection_warnings"] == []
    assert geometry["centerline_symmetry_error_m"] <= geometry["tolerances"]["centerline_symmetry_m"]["value"]
    assert geometry["hinge_origin_maximum_error_m"] <= geometry["tolerances"]["hinge_origin_m"]["value"]
    assert manifest["nose_geometry_validation"]["nose_connected_to_fuselage_ring"] is True
    assert manifest["symmetry_and_intersection_validation"]["window_spacing_even"] is True
    assert all(row["status"] == "PASS" for row in manifest["dimensional_ratio_audit"]["checks"].values())
    assert materials["status"] == "PASS"
    assert materials["checks"]["absolute_texture_paths"] == "PASS"
    assert materials["checks"]["missing_external_resources"] == "PASS"
    assert materials["checks"]["materials_embedded_in_glb"] == "PASS"
    assert materials["materials"]["window_blue_gray"]["base_color_rgba"] != materials["materials"]["airframe_white"]["base_color_rgba"]


def test_animation_contract_signs_clamps_neutral_and_no_input_mutation() -> None:
    contract = data("animation_contract.json")
    settings = load_replay_3d_settings(CONFIG).aircraft.animation
    executed = {"aileron": 5.0, "elevator": -5.0, "rudder": 4.0, "flap": 3.0}
    original = copy.deepcopy(executed)
    angles = animation_angles(executed, settings)
    assert executed == original
    assert angles == {"left_aileron": -20, "right_aileron": 20, "elevator": 25,
                      "rudder": 25, "left_flap": -30, "right_flap": -30}
    neutral = animation_angles({}, settings)
    assert all(value == 0 for value in neutral.values())
    assert contract["verified_properties"]["proposed_actions_do_not_animate"] == "PASS"
    assert contract["verified_properties"]["executed_actions_are_source"] == "PASS"
    assert contract["verified_properties"]["animation_writeback_to_simulator"].startswith("PASS")
    assert contract["moving_objects"]["left_propeller"]["command_source"] == "executed_actions.throttle_left"
    assert contract["moving_objects"]["right_propeller"]["command_source"] == "executed_actions.throttle_right"


def test_coordinate_mapping_and_generation_log_are_complete() -> None:
    manifest = data("generation_manifest.json")
    assert manifest["axis_convention"]["forward"] == "+Y"
    assert manifest["axis_convention"]["up"] == "+Z"
    assert manifest["axis_convention"]["viewer_parent_hpr"] == "heading -> -H, positive pitch -> +P, positive right bank -> +R"
    rows = [json.loads(line) for line in (ROOT / "generation_log.jsonl").read_text(encoding="utf-8").splitlines()]
    operations = {row["operation"] for row in rows}
    assert {"scene reset", "material creation", "fuselage and nose generation",
            "wing and roll-surface generation", "tail and pitch/yaw surface generation",
            "engine and propeller generation", "window generation",
            "pivot placement and geometry validation", "Blend save", "GLB export",
            "audit artifact and checksum input generation"} <= operations
    assert all(row["result"] == "PASS" and row["duration_seconds"] >= 0 for row in rows)


def test_checksum_ledger_and_import_time_window_contract() -> None:
    assert verify_ledger() == []
    module = importlib.import_module("scripts.audit_generated_aircraft_v2")
    assert not hasattr(module, "base")
    live = importlib.import_module("aircraft_recovery.visualization.live_3d")
    assert not hasattr(live, "base")


def test_fallback_and_v1_assets_remain_available() -> None:
    assert Path("assets/aircraft/generated_plane/generated_plane.glb").is_file()
    source = Path("src/aircraft_recovery/visualization/replay_3d.py").read_text(encoding="utf-8")
    assert "fallback" in source.lower() and "primitive" in source.lower()
